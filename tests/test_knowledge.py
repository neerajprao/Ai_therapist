import json

from soulsync.chat import build_messages
from soulsync.knowledge import NOTES_FILE, KnowledgeBase, format_notes

from .conftest import HashEmbedding


def kb(max_distance=0.9):
    return KnowledgeBase(None, max_distance=max_distance, embedding_function=HashEmbedding())


def test_notes_file_is_marked_and_complete():
    data = json.loads(NOTES_FILE.read_text())
    assert "not medical advice" in data["status"] and "Reviewed" in data["status"]
    assert len(data["notes"]) >= 8
    assert all({"id", "title", "tags", "text"} <= set(n) for n in data["notes"])


def test_relevant_note_is_retrieved():
    hits = kb().search("I can't sleep, I'm awake at night")
    assert hits and hits[0].id == "sleep_routine"


def test_threshold_blocks_unrelated_queries():
    assert kb(max_distance=0.3).search("what is the capital of Australia") == []


def test_note_is_labelled_general_information():
    note = format_notes(kb().search("panic racing heart"))
    assert "not medical advice" in note and "general information" in note


def test_knowledge_note_is_temporary_and_before_user_message():
    msgs = build_messages([], "hi", None, 8, knowledge_note="TIP")
    assert [m["role"] for m in msgs] == ["system", "system", "user"]
    assert msgs[1]["content"] == "TIP"


def test_knowledge_is_on_by_default_and_can_be_disabled(settings, tmp_path):
    from soulsync.config import load_settings

    assert settings.knowledge_enabled is True
    assert load_settings({"DATA_DIR": str(tmp_path), "KNOWLEDGE_ENABLED": "false"}).knowledge_enabled is False
