# 📄 Semantic Search Engine (FastAPI + Zilliz/Milvus + AWS S3 + Streamlit)

A production-ready **semantic search engine** with:
- **FastAPI** backend for document ingestion & search  
- **Sentence-Transformers** (`all-MiniLM-L6-v2`) for embeddings  
- **Zilliz Cloud (managed Milvus)** or self-hosted **Milvus on AWS EC2** for vector DB  
- **AWS S3** to store original documents  
- **Streamlit UI** for uploading & searching documents  

---

## 🚀 Features
- Upload documents → automatically chunked (~200–300 words each)  
- Store embeddings + metadata in Zilliz/Milvus  
- Store original file in AWS S3 → results return S3 link  
- Search across **only the documents uploaded in your session**  
- Streamlit UI for upload & search  

---

## 📦 Project Structure
```
.
├─ app/
│  └─ main.py          # FastAPI backend
├─ streamlit_app.py    # Streamlit frontend
├─ requirements.txt    # Python dependencies
├─ .env.example        # Example environment config
└─ README.md
```

---

## ⚙️ Prerequisites
- Python 3.10+  
- AWS account with:
  - S3 bucket created (e.g. `semantic-docs-bucket`)  
  - IAM user with `s3:PutObject` + `s3:GetObject` permissions  
- Either:
  - **Zilliz Cloud** cluster (recommended)  
  - Or **EC2 instance** running Milvus in Docker  

---

## 🔑 Environment Variables (`.env`)
Copy `.env.example` → `.env` and fill in:

```ini
# --- Toggle ---
USE_ZILLIZ=true          # set false if using self-hosted Milvus

# --- Zilliz Cloud (if USE_ZILLIZ=true) ---
ZILLIZ_URI="grpc+ssl://xxx.api.gcp-zillizcloud.com:19530"
ZILLIZ_API_KEY="your_zilliz_api_key"

# --- Local Milvus (if USE_ZILLIZ=false) ---
MILVUS_HOST=localhost
MILVUS_PORT=19530

# --- AWS S3 ---
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=ap-south-1
S3_BUCKET=semantic-docs-bucket

# --- Common ---
COLLECTION_NAME=documents
```

---

## 🖥️ Backend Setup (FastAPI)

### 1. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run FastAPI
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend runs at `http://localhost:8000`  
Swagger UI: `http://localhost:8000/docs`

---

## 🎨 Frontend Setup (Streamlit)

Run:
```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`.

---

## 📝 Usage Flow
1. Open Streamlit UI (`localhost:8501`)  
2. **Upload** one or more docs → stored in S3, embeddings in Zilliz/Milvus  
3. Switch to **Search tab** → enter query  
4. Get top results (snippet + score + S3 link)  

Each Streamlit session has its own **session_id**, so you only search docs uploaded in that session.  

---

## ☁️ Deployment on AWS EC2

### 1. Launch EC2
- Ubuntu 22.04 / Amazon Linux  
- t3.medium (2 vCPU, 4 GB RAM) minimum  

### 2. Security Group
- Open ports:
  - `22` (SSH)
  - `8000` (FastAPI backend)
  - `8501` (Streamlit frontend)
- Restrict access to your IP for security  

### 3. Install dependencies on EC2
```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git docker.io docker-compose
```

### 4. Clone project
```bash
git clone https://github.com/yourusername/semantic-search.git
cd semantic-search
```

### 5. Setup Python
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 6. Setup Milvus
- **If using Zilliz** → skip this step  
- **If self-hosting Milvus**:
  ```bash
  mkdir milvus && cd milvus
  curl -sLO https://raw.githubusercontent.com/milvus-io/milvus/master/deployments/docker/compose/standalone/docker-compose.yml
  sudo docker compose up -d
  ```
  Milvus listens on port 19530.

### 7. Run backend
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 8. Run frontend
```bash
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

Now accessible via:  
- Backend → `http://<ec2-public-ip>:8000/docs`  
- Streamlit → `http://<ec2-public-ip>:8501`  

---

## 🛡️ Security Notes
- Never expose Milvus directly to the internet → use VPC security group or Zilliz Cloud.  
- Protect `.env` (don’t commit to git).  
- Use HTTPS (e.g. Nginx reverse proxy + certbot).  
- For production: run backend + frontend in **Docker containers** with proper systemd or ECS.  

---

## 📌 Roadmap
- [ ] Support multiple file formats (PDF, DOCX) with text extraction  
- [ ] Add authentication per user  
- [ ] Dockerize backend + frontend  
- [ ] Deploy on ECS / EKS  

