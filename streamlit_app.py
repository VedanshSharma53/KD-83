import streamlit as st
import requests
import uuid
import io

BACKEND = "http://localhost:8000"

# Generate one session_id per app run
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

session_id = st.session_state.session_id

st.set_page_config(page_title="Semantic Search", layout="wide")
st.title("📄 Semantic Search")

tab1, tab2 = st.tabs(["⬆️ Upload Document", "🔍 Search"])

# ------------------------
# Upload tab
# ------------------------
with tab1:
    st.subheader("Upload new documents")
    uploaded_files = st.file_uploader("Choose files", type=["txt", "md", "pdf", "docx"], accept_multiple_files=True)
    
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
        if not uploaded_files and "session_id" not in st.session_state:
            st.warning("⚠️ Please upload a document first.")
        else:
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
