import streamlit as st
import requests
import uuid
import io

import os
BACKEND = os.getenv("BACKEND", "http://localhost:8000")


# Generate one session_id per app run
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

session_id = st.session_state.session_id

st.set_page_config(page_title="Semantic Search & RAG Chatbot", layout="wide")
st.title("📄 Semantic Search & 💬 RAG Chatbot")

tab1, tab2, tab3 = st.tabs(["⬆️ Upload Document", "🔍 Search", "🤖 Chatbot"])

# ------------------------
# Upload tab
# ------------------------
with tab1:
    st.subheader("Upload new documents")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["txt", "md", "pdf", "docx"],
        accept_multiple_files=True
    )
    
    if st.button("Upload to backend") and uploaded_files:
        for f in uploaded_files:
            with st.spinner(f"Uploading {f.name}..."):
                try:
                    files = {"file": (f.name, io.BytesIO(f.getvalue()), f.type)}
                    data = {"title": f.name, "session_id": session_id}
                    res = requests.post(f"{BACKEND}/ingest", files=files, data=data)
                    if res.status_code == 200:
                        st.success(f"✅ Uploaded {f.name}")
                    else:
                        st.error(f"Upload failed: {res.status_code} - {res.text}")
                except Exception as e:
                    st.error(f"Error: {e}")


# ------------------------
# Search tab
# ------------------------
with tab2:
    st.subheader("Search uploaded documents")
    query = st.text_input("Enter your search query")
    top_k = st.slider("Number of results", 1, 10, 5)

    if st.button("Search"):
        with st.spinner("Searching..."):
            try:
                res = requests.get(
                    f"{BACKEND}/search",
                    params={"query": query, "limit": top_k, "session_id": session_id}
                )
                if res.status_code == 200:
                    data = res.json()
                    results = data.get("results", [])
                    if results:
                        for i, r in enumerate(results, 1):
                            st.markdown(f"### {i}. {r['title']} (score: {r['score']:.3f})")
                            st.write(r["text"])
                            st.divider()
                    else:
                        st.warning("No results found for this session.")
                else:
                    st.error(f"Search failed: {res.status_code} - {res.text}")
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")


# ------------------------
# Chatbot tab
# ------------------------
with tab3:
    st.subheader("Chat with your documents")
    chat_query = st.text_input("Ask a question about your uploaded docs")
    chat_top_k = st.slider("Context chunks to use", 1, 10, 3)

    if st.button("Get Answer"):
        with st.spinner("Thinking..."):
            try:
                payload = {"query": chat_query, "top_k": chat_top_k, "session_id": session_id}
                res = requests.post(f"{BACKEND}/chat", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    st.markdown("### 🧾 Answer")
                    st.write(data["answer"])

                    st.markdown("---")
                    st.markdown("### 📑 Supporting Chunks")
                    for i, c in enumerate(data["chunks"], 1):
                        st.markdown(f"**{i}. {c['title']}** (score: {c['score']:.3f})")
                        st.write(c["text"])
                        st.divider()
                else:
                    st.error(f"Chat failed: {res.status_code} - {res.text}")
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")
