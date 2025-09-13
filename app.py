import os
import uuid
import threading
import traceback
from pathlib import Path
from typing import List, Dict

from flask import Flask, request, jsonify, send_file
from git import Repo, GitCommandError

# LangChain + Embeddings + FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI

import pickle
from dotenv import load_dotenv
load_dotenv()

# ========== Config ==========
ALLOWED_EXTENSIONS = {
    '.py', '.js', '.ts', '.java', '.go', '.rs', '.md', '.txt',
    '.json', '.yaml', '.yml', '.html', '.css'
}
IGNORE_DIRS = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', 'dist', 'build'}
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 200
TOP_K = 6

SESSIONS_ROOT = Path("./sessions")
SESSIONS_ROOT.mkdir(exist_ok=True)

sessions_status = {}
vectorstore_cache = {}

# Initialize embeddings
print("Initializing HuggingFaceEmbeddings (this may take a few seconds)...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# LLM
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not set as env var.")

llm = ChatOpenAI(
    temperature=0,
    model="gpt-4",
    openai_api_key=OPENAI_API_KEY
)

# Flask app
app = Flask(__name__)

# ========== Utils ==========
def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_EXTENSIONS

def read_file_safe(path: Path) -> str:
    try:
        return path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return ""

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks, start, length = [], 0, len(text)
    while start < length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def make_session_dir(session_id: str) -> Path:
    path = SESSIONS_ROOT / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_metadata(session_dir: Path, metadata: List[Dict]):
    with open(session_dir / "metadatas.pkl", "wb") as f:
        pickle.dump(metadata, f)

def load_metadata(session_dir: Path) -> List[Dict]:
    with open(session_dir / "metadatas.pkl", "rb") as f:
        return pickle.load(f)

def safe_clone_repo(repo_url: str, dest: Path) -> None:
    Repo.clone_from(repo_url, dest)


# ========== Indexing Worker ==========
def index_repo_worker(session_id: str, repo_url: str):
    session_dir = make_session_dir(session_id)
    try:
        # Step 1: Clone repo
        sessions_status[session_id] = {"status": "indexing", "message": "Cloning repository..."}
        repo_dir = session_dir / "repo"
        try:
            safe_clone_repo(repo_url, repo_dir)
            sessions_status[session_id] = {"status": "indexing", "message": "✅ Cloning completed."}
        except GitCommandError as e:
            sessions_status[session_id] = {"status": "error", "message": f"Git clone failed: {e}"}
            return

        # Step 2: Walk repo
        sessions_status[session_id] = {"status": "indexing", "message": "Scanning repository and collecting files..."}
        texts, metadatas, total_files = [], [], 0

        for root, dirs, files in os.walk(repo_dir):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            for fname in files:
                fpath = Path(root) / fname
                if not is_text_file(fpath):
                    continue
                total_files += 1
                raw = read_file_safe(fpath)
                if not raw.strip():
                    continue
                chunks = chunk_text(raw)
                for idx, chunk in enumerate(chunks):
                    texts.append(chunk)
                    metadatas.append({
                        "file_path": str(fpath.relative_to(repo_dir)),
                        "absolute_path": str(fpath),
                        "chunk_index": idx,
                        "repo_url": repo_url
                    })

        if not texts:
            sessions_status[session_id] = {"status": "error", "message": "No indexable files found."}
            return
        sessions_status[session_id] = {"status": "indexing", "message": f"✅ Collected {total_files} files and {len(texts)} chunks."}

        # Step 3: Create embeddings
        sessions_status[session_id] = {"status": "indexing", "message": f"Creating embeddings for {len(texts)} chunks..."}
        vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        sessions_status[session_id] = {"status": "indexing", "message": "✅ Embeddings created."}

        # Step 4: Save FAISS
        sessions_status[session_id] = {"status": "indexing", "message": "Saving FAISS index and metadata..."}
        vectorstore.save_local(str(session_dir / "vectorstore"))
        save_metadata(session_dir, metadatas)
        sessions_status[session_id] = {"status": "indexing", "message": "✅ Vectorstore saved."}

        # Step 5: Ready
        vectorstore_cache[session_id] = vectorstore
        sessions_status[session_id] = {"status": "ready", "message": f"🎉 Indexing completed. {len(texts)} chunks from {total_files} files."}

    except Exception as e:
        traceback.print_exc()
        sessions_status[session_id] = {"status": "error", "message": f"Indexing error: {str(e)}"}


# ========== Endpoints ==========
@app.route("/")
def index_page():
    # Serve index.html directly from same folder
    return send_file("index.html")

@app.route("/index", methods=["POST"])
def start_index():
    payload = request.get_json()
    repo_url = payload.get("repo_url")
    if not repo_url:
        return jsonify({"error": "repo_url required"}), 400
    session_id = str(uuid.uuid4())
    sessions_status[session_id] = {"status": "indexing", "message": "Queued for indexing."}
    thread = threading.Thread(target=index_repo_worker, args=(session_id, repo_url), daemon=True)
    thread.start()
    return jsonify({"message": "Repository indexing started.", "session_id": session_id}), 200

@app.route("/status/<session_id>", methods=["GET"])
def status(session_id):
    info = sessions_status.get(session_id)
    if not info:
        return jsonify({"status": "error", "message": "Unknown session_id"}), 404
    return jsonify(info), 200

@app.route("/query", methods=["POST"])
def query():
    payload = request.get_json()
    session_id, question = payload.get("session_id"), payload.get("question")
    if not session_id or not question:
        return jsonify({"error": "session_id and question required"}), 400

    info = sessions_status.get(session_id)
    if not info:
        return jsonify({"error": "unknown session_id"}), 404
    if info["status"] != "ready":
        return jsonify({"error": "index not ready", "message": info.get("message", "")}), 400

    try:
        if session_id in vectorstore_cache:
            vectorstore = vectorstore_cache[session_id]
        else:
            session_dir, vs_path = SESSIONS_ROOT / session_id, SESSIONS_ROOT / session_id / "vectorstore"
            if not vs_path.exists():
                return jsonify({"error": "vectorstore files missing"}), 500
            vectorstore = FAISS.load_local(str(vs_path), embeddings)
            vectorstore_cache[session_id] = vectorstore

        docs = vectorstore.similarity_search(question, k=TOP_K)
        retrieved, context_texts = [], []
        for d in docs:
            md, snippet = d.metadata or {}, d.page_content
            file_path = md.get("file_path", md.get("source", "unknown"))
            retrieved.append({"file": file_path, "snippet": snippet[:1000]})
            context_texts.append(f"File: {file_path}\n\n{snippet}")

        system_prompt = (
            "You are an assistant that answers questions about a code repository. "
            "Use ONLY the provided code snippets and file references. "
            "If the answer is not contained, say you cannot find the info. "
            "Provide a concise answer and list sources at the end."
        )
        context_block = "\n\n---\n\n".join(context_texts)
        full_prompt = f"{system_prompt}\n\nRepository question: {question}\n\nContext:\n{context_block}\n\nAnswer concisely."
        llm_response = llm.predict(full_prompt)

        return jsonify({"answer": llm_response, "sources": retrieved}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "query failed", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8800, debug=True)
