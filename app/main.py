import os
import uuid
import boto3
import io
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Use a flag to easily switch between Zilliz Cloud and a local Milvus instance
USE_ZILLIZ = os.getenv("USE_ZILLIZ", "false").lower() == "true"

# Zilliz Cloud configs
ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_API_KEY = os.getenv("ZILLIZ_API_KEY")

# Local Milvus configs
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

# Common configs
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
DIMENSION = 384  # Dimension of the "all-MiniLM-L6-v2" model

# AWS configs
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET = os.getenv("S3_BUCKET")

# --- Database Connection ---
print("Connecting to vector database...")
if USE_ZILLIZ:
    if not ZILLIZ_URI or not ZILLIZ_API_KEY:
        raise ValueError("ZILLIZ_URI and ZILLIZ_API_KEY must be set in .env when USE_ZILLIZ is true")
    connections.connect(
        alias="default",
        uri=ZILLIZ_URI,
        token=ZILLIZ_API_KEY,
    )
    print("Connected to Zilliz Cloud.")
else:
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT,
    )
    print(f"Connected to local Milvus at {MILVUS_HOST}:{MILVUS_PORT}.")

# --- Model Loading ---
print("Loading sentence transformer model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Model loaded.")

# --- Milvus Collection and Index Setup ---
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64, auto_id=False),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
    FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=64),   # NEW
]
schema = CollectionSchema(fields, description="Document collection for semantic search")


# Create collection if it doesn't exist
if not utility.has_collection(COLLECTION_NAME):
    print(f"Collection '{COLLECTION_NAME}' does not exist. Creating now...")
    collection = Collection(name=COLLECTION_NAME, schema=schema, using='default')
    print("Collection created.")
else:
    print(f"Using existing collection '{COLLECTION_NAME}'.")
    collection = Collection(name=COLLECTION_NAME)

# Check if an index exists on the 'embedding' field
if not collection.has_index():
    print("No index found on the 'embedding' field. Creating one now...")
    # Using HNSW index for a good balance of performance and accuracy
    index_params = {
        "metric_type": "IP",  # Inner Product for similarity
        "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 200}
    }
    collection.create_index(
        field_name="embedding",
        index_params=index_params
    )
    print("Index created successfully.")
else:
    print("Index already exists.")

# Load the collection into memory for searching
print("Loading collection into memory...")
collection.load()
print("Collection loaded.")


# --- AWS S3 Client ---
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)

# --- FastAPI App ---
app = FastAPI(title="Semantic Search API")

def chunk_text(text: str, max_words: int = 200):
    """Splits text into chunks of a specified number of words."""
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])



@app.post("/ingest", summary="Ingest a document")
async def ingest_document(file: UploadFile, title: str = Form(...), session_id: str = Form(...)):
    if not S3_BUCKET:
        raise ValueError("S3_BUCKET environment variable is not set.")

    content_bytes = await file.read()
    file_id = str(uuid.uuid4())
    s3_key = f"documents/{file_id}_{file.filename}"

    s3_client.upload_fileobj(io.BytesIO(content_bytes), S3_BUCKET, s3_key)
    file_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"

    try:
        content = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return JSONResponse(status_code=400, content={"message": "Failed to decode file. Please use UTF-8 text."})

    chunks = list(chunk_text(content))
    if not chunks:
        return {"message": "Document is empty", "file_url": file_url}

    embeddings = model.encode(chunks).tolist()

    entities = [
        [f"{file_id}_{i}" for i in range(len(chunks))],  # id
        [title] * len(chunks),                           # title
        chunks,                                          # text
        embeddings,                                      # embedding
        [session_id] * len(chunks),                      # session_id
    ]

    collection.insert(entities)
    collection.flush()

    return {"message": "Document ingested successfully", "file_url": file_url, "chunks_ingested": len(chunks)}

@app.get("/search", summary="Perform a semantic search")
async def search(query: str, session_id: str, limit: int = 5):
    query_embedding = model.encode([query]).tolist()

    search_params = {"metric_type": "IP", "params": {"ef": 10}}

    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param=search_params,
        limit=limit,
        output_fields=["title", "text"],
        expr=f"session_id == \"{session_id}\""   # filter only this session's docs
    )

    output = []
    for hits in results:
        for hit in hits:
            output.append({
                "id": hit.id,
                "score": hit.distance,
                "title": hit.entity.get("title"),
                "text": hit.entity.get("text"),
            })

    return {"results": output}
