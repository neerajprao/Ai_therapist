"""Optional knowledge base of short coping-skill notes (general wellbeing information).

A separate ChromaDB collection from user memories, filled from
`coping_notes.json` on start-up. Retrieval uses the same cosine-distance
threshold idea as memory. Off unless KNOWLEDGE_ENABLED=true; the notes must be
reviewed by the project owner before switching it on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

NOTES_FILE = Path(__file__).parent / "coping_notes.json"
COLLECTION = "coping_notes"


@dataclass
class Note:
    id: str
    title: str
    text: str
    distance: float


class KnowledgeBase:
    def __init__(self, path: Path | None, max_distance: float = 0.75, top_k: int = 1, embedding_function=None):
        import chromadb

        client = chromadb.EphemeralClient() if path is None else chromadb.PersistentClient(path=str(path))
        kwargs = {"embedding_function": embedding_function} if embedding_function is not None else {}
        self.collection = client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"}, **kwargs)
        self.max_distance = max_distance
        self.top_k = top_k
        self._load()

    def _load(self) -> None:
        notes = json.loads(NOTES_FILE.read_text())["notes"]
        # Embed title + tags + text so short, informal messages can still match.
        self.collection.upsert(
            ids=[n["id"] for n in notes],
            documents=[f"{n['title']}. {n['tags']}. {n['text']}" for n in notes],
            metadatas=[{"title": n["title"], "text": n["text"]} for n in notes],
        )

    def search(self, query: str, top_k: int | None = None, max_distance: float | None = None) -> list[Note]:
        if not query.strip():
            return []
        limit = self.max_distance if max_distance is None else max_distance
        res = self.collection.query(query_texts=[query], n_results=top_k or self.top_k, include=["metadatas", "distances"])
        return [
            Note(i, m["title"], m["text"], float(d))
            for i, m, d in zip(res["ids"][0], res["metadatas"][0], res["distances"][0])
            if d <= limit
        ]


def format_notes(notes: list[Note]) -> str:
    body = "\n".join(f"- {n.text}" for n in notes)
    return (
        "General wellbeing information (not medical advice) that might help here:\n"
        f"{body}\n"
        "Offer it only if it fits what the person needs right now, as a gentle optional suggestion, "
        "and make clear it is general information. Do not offer it if they just want to be heard."
    )


def build_knowledge(settings) -> KnowledgeBase | None:
    if not settings.knowledge_enabled:
        return None
    return KnowledgeBase(settings.data_dir / "knowledge", settings.knowledge_max_distance)
