import os
import uuid
import io
import boto3
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, Form, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sentence_transformers import SentenceTransformer
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

# === LLM backends ===
USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() == "true"
if USE_OPENAI:
    from openai import OpenAI
else:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoConfig,
        pipeline,
    )

# -----------------------
# Environment / Config
# -----------------------
load_dotenv()

USE_ZILLIZ = os.getenv("USE_ZILLIZ", "false").lower() == "true"
ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_API_KEY = os.getenv("ZILLIZ_API_KEY")

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
DIMENSION = 384  # all-MiniLM-L6-v2

# AWS
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET = os.getenv("S3_BUCKET")

# LLM
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "google/flan-t5-base")  # default: fast model
HF_DEVICE = os.getenv("HF_DEVICE", "auto")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

TOP_K_DEFAULT = int(os.getenv("TOP_K", "3"))

# -----------------------
# FastAPI setup
# -----------------------
app = FastAPI(title="Semantic Search + RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# -----------------------
# Vector DB connection
# -----------------------
print("Connecting to vector DB...")
if USE_ZILLIZ:
    connections.connect(alias="default", uri=ZILLIZ_URI, token=ZILLIZ_API_KEY)
    print("Connected to Zilliz.")
else:
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    print(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")

# -----------------------
# Embedding model
# -----------------------
print("Loading embeddings model...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -----------------------
# Collection schema
# -----------------------
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64, auto_id=False),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
    FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=64),
]
schema = CollectionSchema(fields, description="Docs for semantic search & RAG")

if not utility.has_collection(COLLECTION_NAME):
    collection = Collection(name=COLLECTION_NAME, schema=schema, using="default")
else:
    collection = Collection(COLLECTION_NAME)

if not collection.has_index():
    index_params = {"metric_type": "IP", "index_type": "HNSW", "params": {"M": 8, "efConstruction": 200}}
    collection.create_index(field_name="embedding", index_params=index_params)

collection.load()

# -----------------------
# AWS S3
# -----------------------
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)

# -----------------------
# Helper functions
# -----------------------
def chunk_text(text: str, max_words: int = 200):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])

# -----------------------
# LLM Client
# -----------------------
class LLMClient:
    def __init__(self):
        self.use_openai = USE_OPENAI
        if self.use_openai:
            if not OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY missing")
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model = OPENAI_MODEL
            self.mode = "openai"
            print(f"Using OpenAI model: {self.model}")
        else:
            print(f"Loading HF model: {HF_MODEL_ID}")
            config = AutoConfig.from_pretrained(HF_MODEL_ID)
            self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)

            if config.is_encoder_decoder:  # seq2seq (e.g., T5, BART)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_ID)
                self.pipe = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)
                self.mode = "seq2seq"
            else:  # causal LM (e.g., LLaMA, Mistral, GPT)
                self.model = AutoModelForCausalLM.from_pretrained(
                    HF_MODEL_ID,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else None,
                    device_map="auto" if HF_DEVICE == "auto" else None,
                )
                self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
                self.mode = "causal"

    def generate(self, prompt: str) -> str:
        if self.mode == "openai":
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Answer only using the provided context."},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_NEW_TOKENS,
            )
            return resp.choices[0].message.content.strip()

        out = self.pipe(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True if TEMPERATURE > 0 else False,
            temperature=TEMPERATURE,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        # HuggingFace pipelines return different keys for seq2seq vs causal
        return out[0].get("generated_text") or out[0].get("translation_text")

llm = LLMClient()

# -----------------------
# Schemas
# -----------------------
class ChatRequest(BaseModel):
    query: str
    top_k: int = TOP_K_DEFAULT
    session_id: Optional[str] = None

class ChunkOut(BaseModel):
    title: Optional[str]
    text: str
    score: float

class ChatResponse(BaseModel):
    answer: str
    chunks: List[ChunkOut]
    used_top_k: int

# -----------------------
# Endpoints
# -----------------------
@app.post("/ingest")
async def ingest(file: UploadFile, title: str = Form(...), session_id: str = Form(...)):
    content_bytes = await file.read()
    file_id = str(uuid.uuid4())
    s3_key = f"documents/{file_id}_{file.filename}"
    s3_client.upload_fileobj(io.BytesIO(content_bytes), S3_BUCKET, s3_key)
    file_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"

    try:
        content = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return JSONResponse(status_code=400, content={"message": "File must be UTF-8 text."})

    chunks = list(chunk_text(content))
    embeddings = embedder.encode(chunks).tolist()

    entities = [
        [f"{file_id}_{i}" for i in range(len(chunks))],
        [title] * len(chunks),
        chunks,
        embeddings,
        [session_id] * len(chunks),
    ]

    collection.insert(entities)
    collection.flush()

    return {"message": "Ingested", "file_url": file_url, "chunks": len(chunks), "session_id": session_id}

@app.get("/search")
async def search(query: str, limit: int = 5, session_id: Optional[str] = None):
    query_emb = embedder.encode([query]).tolist()
    search_params = {"metric_type": "IP", "params": {"ef": 32}}
    expr = f'session_id == "{session_id}"' if session_id else None

    results = collection.search(
        data=query_emb,
        anns_field="embedding",
        param=search_params,
        limit=limit,
        output_fields=["title", "text"],
        expr=expr,
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

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest = Body(...)):
    q_emb = embedder.encode([req.query]).tolist()
    search_params = {"metric_type": "IP", "params": {"ef": 32}}
    expr = f'session_id == "{req.session_id}"' if req.session_id else None

    results = collection.search(
        data=q_emb,
        anns_field="embedding",
        param=search_params,
        limit=req.top_k,
        output_fields=["title", "text"],
        expr=expr,
    )

    chunks: List[ChunkOut] = []
    for hits in results:
        for h in hits:
            chunks.append(ChunkOut(
                title=h.entity.get("title"),
                text=h.entity.get("text"),
                score=float(h.distance),
            ))

    if not chunks:
        return ChatResponse(answer="No context available. Please ingest documents first.", chunks=[], used_top_k=req.top_k)

    context = "\n\n---\n\n".join([c.text for c in chunks])
    prompt = (
        "Answer only using this context.\n"
        "If not in context, say 'I don't know'.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {req.query}\nAnswer:"
    )

    answer = llm.generate(prompt)
    return ChatResponse(answer=answer, chunks=chunks, used_top_k=req.top_k)
