"""
backend/vector_store.py
------------------------
ChromaDB vector store for persistent resume embedding storage.
Enables semantic search across previously screened resumes.
"""

import os

# Try to use ChromaDB; fall back to in-memory storage if not installed
_USE_CHROMA = False
_collection = None
_memory_store = []  # Fallback in-memory store

_HERE = os.path.dirname(os.path.abspath(__file__))
_CHROMA_DIR = os.path.join(_HERE, "..", "chroma_db")

try:
    import chromadb
    from chromadb.config import Settings
    _USE_CHROMA = True
except ImportError:
    pass


def _get_collection():
    global _collection
    if _collection is not None:
        return _collection
    if not _USE_CHROMA:
        return None
    try:
        os.makedirs(_CHROMA_DIR, exist_ok=True)
        client = chromadb.PersistentClient(path=_CHROMA_DIR)
        _collection = client.get_or_create_collection(
            name="resumes",
            metadata={"hnsw:space": "cosine"},
        )
        return _collection
    except Exception as e:
        print(f"[vector_store] ChromaDB init failed: {e}")
        return None


def store_resume(
    resume_id: str,
    embedding: list,
    metadata: dict = None,
    document: str = "",
):
    """
    Store a resume embedding in the vector database.

    Args:
        resume_id: Unique identifier (e.g., filename + timestamp)
        embedding: 384-d embedding vector
        metadata: Dict with candidate name, category, skills, etc.
        document: Resume text excerpt for retrieval
    """
    collection = _get_collection()
    if collection is not None:
        try:
            collection.upsert(
                ids=[resume_id],
                embeddings=[embedding if isinstance(embedding, list) else embedding.tolist()],
                metadatas=[metadata or {}],
                documents=[document[:2000]],
            )
        except Exception as e:
            print(f"[vector_store] Store error: {e}")
    else:
        # Fallback: in-memory
        _memory_store.append({
            "id": resume_id,
            "embedding": embedding,
            "metadata": metadata or {},
            "document": document[:2000],
        })


def search_similar(
    query_embedding: list,
    n_results: int = 5,
    filter_metadata: dict = None,
) -> list[dict]:
    """
    Find similar resumes by embedding similarity.

    Args:
        query_embedding: 384-d query vector
        n_results: Number of results to return
        filter_metadata: Optional chromadb where filter

    Returns:
        List of dicts with id, score, metadata, document
    """
    collection = _get_collection()
    if collection is not None:
        try:
            qe = query_embedding if isinstance(query_embedding, list) else query_embedding.tolist()
            kwargs = {
                "query_embeddings": [qe],
                "n_results": n_results,
            }
            if filter_metadata:
                kwargs["where"] = filter_metadata

            results = collection.query(**kwargs)

            output = []
            for i in range(len(results["ids"][0])):
                output.append({
                    "id": results["ids"][0][i],
                    "score": 1 - results["distances"][0][i] if results.get("distances") else 0,
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "document": results["documents"][0][i] if results.get("documents") else "",
                })
            return output
        except Exception as e:
            print(f"[vector_store] Search error: {e}")
            return []
    else:
        # Fallback: simple dot product search
        import numpy as np
        qe = np.array(query_embedding)
        scored = []
        for item in _memory_store:
            emb = np.array(item["embedding"])
            sim = float(np.dot(qe, emb) / (np.linalg.norm(qe) * np.linalg.norm(emb) + 1e-9))
            scored.append({**item, "score": sim})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:n_results]


def get_collection_count() -> int:
    """Return number of stored resumes."""
    collection = _get_collection()
    if collection is not None:
        try:
            return collection.count()
        except Exception:
            return 0
    return len(_memory_store)
