"""
FAISS Vector Store — Google Cloud Embeddings Version.

This version uses Google's Generative AI API for embeddings.
- Advantage: Uses 0 MB of server RAM (Perfect for Render Free Tier 512MB).
- Speed: Much faster indexing and retrieval.
- Accuracy: Higher quality semantic search.
"""

import os
import pickle
import numpy as np
import faiss
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List, Optional

_embeddings: Optional[GoogleGenerativeAIEmbeddings] = None

def get_embeddings_model() -> GoogleGenerativeAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            # Fallback to GROQ_API_KEY if they used the same key, but usually they are different
            api_key = os.getenv("GEMINI_API_KEY")
            
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY is not set in environment variables.")
            
        _embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
    return _embeddings

def _index_dir(user_id: str) -> str:
    return os.path.join("uploads", user_id, "faiss_index")

def _load_index(user_id: str):
    """Load FAISS index + metadata list for a user."""
    idx_dir = _index_dir(user_id)
    idx_file = os.path.join(idx_dir, "index.faiss")
    meta_file = os.path.join(idx_dir, "metadata.pkl")

    if not os.path.exists(idx_file):
        return None, []

    index = faiss.read_index(idx_file)
    with open(meta_file, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata

def _save_index(user_id: str, index, metadata: list):
    """Persist FAISS index + metadata to disk."""
    idx_dir = _index_dir(user_id)
    os.makedirs(idx_dir, exist_ok=True)

    faiss.write_index(index, os.path.join(idx_dir, "index.faiss"))
    with open(os.path.join(idx_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

async def add_documents_to_index(user_id: str, doc_id: str, chunks: List[dict]):
    """Embed chunks via Google API and add them to the user's FAISS index."""
    if not chunks:
        return

    embeddings_model = get_embeddings_model()
    texts = [c["content"] for c in chunks]
    
    # Cloud-side embedding generation
    raw_embeddings = embeddings_model.embed_documents(texts)
    embeddings = np.array(raw_embeddings, dtype=np.float32)

    index, metadata = _load_index(user_id)

    if index is None:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    for chunk in chunks:
        metadata.append(
            {
                "doc_id": doc_id,
                "content": chunk["content"],
                "page": chunk["page"],
            }
        )

    _save_index(user_id, index, metadata)

async def search_similar_chunks(
    user_id: str,
    query: str,
    doc_ids: Optional[List[str]] = None,
    top_k: int = 5,
) -> List[dict]:
    """Retrieve top-k most relevant chunks using Google Embeddings."""
    embeddings_model = get_embeddings_model()
    index, metadata = _load_index(user_id)

    if index is None or index.ntotal == 0:
        return []

    # Cloud-side query embedding
    q_emb = embeddings_model.embed_query(query)
    q_emb = np.array([q_emb], dtype=np.float32)

    # Search for more than needed so we can filter by doc_ids
    search_k = min(top_k * 5, index.ntotal)
    distances, indices = index.search(q_emb, search_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        chunk = metadata[idx]
        if doc_ids and chunk["doc_id"] not in doc_ids:
            continue
        results.append(
            {
                "content": chunk["content"],
                "doc_id": chunk["doc_id"],
                "page": chunk["page"],
                "score": float(dist),
            }
        )
        if len(results) >= top_k:
            break

    return results

async def remove_document_from_index(user_id: str, doc_id: str):
    """Remove chunks and rebuild index."""
    index, metadata = _load_index(user_id)
    if index is None:
        return

    remaining = [m for m in metadata if m["doc_id"] != doc_id]

    if not remaining:
        import shutil
        idx_dir = _index_dir(user_id)
        if os.path.exists(idx_dir):
            shutil.rmtree(idx_dir)
        return

    # Re-embed is not needed if we keep the vectors, but for simplicity with IndexFlatL2 we'll re-embed
    # Or better: regenerate from existing metadata vectors if we stored them (but we didn't store vectors in metadata)
    # To save API calls, a production app would store vectors. Here we re-index.
    embeddings_model = get_embeddings_model()
    texts = [m["content"] for m in remaining]
    raw_embeddings = embeddings_model.embed_documents(texts)
    embeddings = np.array(raw_embeddings, dtype=np.float32)

    new_index = faiss.IndexFlatL2(embeddings.shape[1])
    new_index.add(embeddings)

    _save_index(user_id, new_index, remaining)
