"""Long-term conversation memory on ChromaDB (cosine distance, MiniLM-L6-v2 ONNX).

Privacy rules enforced here:
  * every read and delete is filtered by `user_id`;
  * retrieval excludes the current session (it is already in the chat history);
  * results further than `max_distance` are dropped, so an unrelated question
    recalls nothing;
  * `forget_user` removes everything stored for one user.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

COLLECTION = "user_memories"
MIN_WORDS = 3
REPLY_CHARS = 200


@dataclass
class Memory:
    text: str
    distance: float
    emotion: str
    timestamp: float
    session_id: str
    reply: str = ""


class MemoryStore:
    def __init__(
        self,
        path: Path | None,
        top_k: int = 3,
        max_distance: float = 0.7,
        embedding_function=None,
    ):
        """`path=None` gives an in-memory store (used by tests and evals)."""
        import chromadb

        if path is None:
            self.client = chromadb.EphemeralClient()
        else:
            path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(path))
        kwargs = {"embedding_function": embedding_function} if embedding_function is not None else {}
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION, metadata={"hnsw:space": "cosine"}, **kwargs
        )
        self.top_k = top_k
        self.max_distance = max_distance

    def add(
        self,
        user_id: str,
        session_id: str,
        text: str,
        emotion: str = "",
        reply: str = "",
        timestamp: float | None = None,
    ) -> bool:
        """Store one user message. Returns False if it was too short to keep."""
        text = text.strip()
        if len(text.split()) < MIN_WORDS:
            return False
        self.collection.add(
            ids=[uuid.uuid4().hex],
            documents=[text],
            metadatas=[
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "emotion": emotion,
                    "timestamp": float(timestamp if timestamp is not None else time.time()),
                    "reply": reply.strip()[:REPLY_CHARS],
                }
            ],
        )
        return True

    def search(
        self,
        user_id: str,
        session_id: str,
        query: str,
        top_k: int | None = None,
        max_distance: float | None = None,
    ) -> list[Memory]:
        """This user's memories from other sessions, nearest first, within the threshold."""
        if not query.strip():
            return []
        limit = self.max_distance if max_distance is None else max_distance
        res = self.collection.query(
            query_texts=[query],
            n_results=top_k or self.top_k,
            where={"$and": [{"user_id": {"$eq": user_id}}, {"session_id": {"$ne": session_id}}]},
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            if dist <= limit:
                out.append(
                    Memory(
                        text=doc,
                        distance=float(dist),
                        emotion=meta.get("emotion", ""),
                        timestamp=float(meta.get("timestamp", 0.0)),
                        session_id=meta.get("session_id", ""),
                        reply=meta.get("reply", ""),
                    )
                )
        return out

    def count(self, user_id: str | None = None) -> int:
        if user_id is None:
            return self.collection.count()
        return len(self.collection.get(where={"user_id": {"$eq": user_id}}, include=[])["ids"])

    def forget_user(self, user_id: str) -> int:
        """Delete everything stored for `user_id`. Returns how many items were removed."""
        ids = self.collection.get(where={"user_id": {"$eq": user_id}}, include=[])["ids"]
        if ids:
            self.collection.delete(ids=ids)
        log.info("Deleted %d memories for one user", len(ids))
        return len(ids)


def format_memories(memories: list[Memory]) -> str:
    """The temporary system message that carries recalled memories to the LLM."""
    lines = "\n".join(f'- "{m.text}"' for m in memories)
    return (
        "Things this person told you in earlier conversations, which may or may not "
        "matter now:\n"
        f"{lines}\n"
        "Use one only if it is clearly relevant to what they just said. If one is, connect to it "
        "gently and briefly in your reply (for example, 'last time you mentioned...') so they feel remembered. "
        "If none is relevant, ignore them completely and do not say that you remember anything."
    )


def build_memory(settings) -> MemoryStore | None:
    if not settings.memory_enabled:
        return None
    return MemoryStore(settings.chroma_dir, settings.memory_top_k, settings.memory_max_distance)
