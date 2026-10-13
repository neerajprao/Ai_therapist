import uuid

import pytest
from fastapi.testclient import TestClient

from soulsync import safety
from soulsync.api import USER_COOKIE, create_app
from soulsync.chat import EMPTY_INPUT_REPLY
from soulsync.llm import LLMRateLimited
from soulsync.stt import STTRateLimited

from .conftest import make_wav


def new_client(settings, fake_llm, fake_stt, detector, memory):
    app = create_app(settings, llm=fake_llm, stt=fake_stt, emotion=detector, memory=memory)
    return TestClient(app)


@pytest.fixture
def client(settings, fake_llm, fake_stt, detector, memory):
    c = new_client(settings, fake_llm, fake_stt, detector, memory)
    c.get("/")  # sets the user cookie
    return c


def talk(client, session_id=None, wav=None):
    return client.post(
        "/process_audio_stream",
        files={"audio": ("recording.wav", wav or make_wav(), "audio/wav")},
        data={"session_id": session_id or uuid.uuid4().hex},
    )


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["llm"] == "fake:llm"
    assert body["emotion_available"] is True
    assert body["tts_backend"] == "browser"


def test_index_sets_long_lived_httponly_cookie_and_new_session_each_load(settings, fake_llm, fake_stt, detector, memory):
    c = new_client(settings, fake_llm, fake_stt, detector, memory)
    first = c.get("/")
    cookie = first.headers["set-cookie"]
    assert USER_COOKIE in cookie and "HttpOnly" in cookie and "SameSite=lax" in cookie
    assert "Max-Age=31536000" in cookie
    uid = c.cookies[USER_COOKIE]

    second = c.get("/")
    assert c.cookies[USER_COOKIE] == uid, "same browser keeps the same user id"

    def session_of(html):
        return html.split("const SESSION_ID = ")[1].split(";")[0]

    assert session_of(first.text) != session_of(second.text)


def test_page_shows_disclaimer_and_data_flow(client):
    html = client.get("/").text
    assert "not a therapist" in html
    assert "not for emergencies" in html
    assert "14416" in html
    assert "Nothing leaves this computer" in html
    assert "Forget me" in html


def test_stream_starts_with_metadata_then_reply(client):
    res = talk(client)
    assert res.status_code == 200
    assert res.text == "METADATA|I had a rough day at work today|Sad|I hear you. That sounds hard."


def test_pipe_in_transcript_cannot_break_the_header(client, fake_stt):
    fake_stt.text = "yes | no | maybe so"
    parts = talk(client).text.split("|")
    assert parts[0] == "METADATA" and parts[2] == "Sad"
    assert "|" not in parts[1]


def test_user_message_is_tagged_with_tone_and_system_prompt_comes_first(client, fake_llm):
    talk(client)
    messages = fake_llm.calls[0]
    assert messages[0]["role"] == "system"
    assert messages[-1] == {"role": "user", "content": "[User Tone: Sad] I had a rough day at work today"}


def test_requires_cookie(settings, fake_llm, fake_stt, detector, memory):
    c = new_client(settings, fake_llm, fake_stt, detector, memory)
    assert talk(c).status_code == 401
    assert c.delete("/memory").status_code == 401


def test_forged_cookie_is_rejected(client):
    client.cookies.set(USER_COOKIE, "someoneelse.0000")
    assert talk(client).status_code == 401


def test_bad_inputs(client):
    assert talk(client, session_id="../../etc/passwd").status_code == 400
    assert talk(client, wav=b"this is not audio").status_code == 400
    assert client.post("/process_audio_stream", data={"session_id": uuid.uuid4().hex}).status_code == 422


def test_silence_skips_the_llm(client, fake_llm):
    res = talk(client, wav=make_wav(amplitude=0.0))
    assert res.text == f"METADATA||Neutral|{EMPTY_INPUT_REPLY}"
    assert fake_llm.calls == []


def test_crisis_message_bypasses_the_llm_and_memory(client, fake_llm, fake_stt, memory):
    fake_stt.text = "I don't want to live anymore"
    res = talk(client)
    assert res.text.endswith(safety.CRISIS_MESSAGE)
    assert fake_llm.calls == []
    assert memory.count() == 0


def test_history_is_bounded(settings, fake_llm, fake_stt, detector, memory):
    c = new_client(settings, fake_llm, fake_stt, detector, None)
    c.get("/")
    session = uuid.uuid4().hex
    for _ in range(settings.history_max_turns + 5):
        talk(c, session)
    last = fake_llm.calls[-1]
    # system + N past exchanges (2 messages each) + the new user message
    assert len(last) == 1 + 2 * settings.history_max_turns + 1


def test_memory_is_recalled_in_a_later_session_but_never_saved_to_history(client, fake_llm, fake_stt):
    fake_stt.text = "My dog Bruno has been sick and the vet is worried"
    talk(client, "a" * 32)

    fake_stt.text = "The vet called again about my dog Bruno"
    talk(client, "b" * 32)
    recalled = fake_llm.calls[-1]
    notes = [m for m in recalled if m["role"] == "system" and "earlier conversations" in m["content"]]
    assert len(notes) == 1 and "Bruno has been sick" in notes[0]["content"]
    assert recalled[-2] is not None and recalled[-2]["role"] == "system", "note sits just before the new message"

    fake_stt.text = "What should I cook for dinner tonight"
    talk(client, "b" * 32)
    follow_up = fake_llm.calls[-1]
    assert not [m for m in follow_up[1:] if m["role"] == "system"], "old memory notes must not pile up in history"


def test_same_session_is_not_recalled_as_memory(client, fake_llm, fake_stt):
    fake_stt.text = "My dog Bruno has been sick and the vet is worried"
    talk(client, "a" * 32)
    talk(client, "a" * 32)
    assert [m["role"] for m in fake_llm.calls[-1]].count("system") == 1


def test_users_cannot_see_each_others_memories(settings, fake_llm, fake_stt, detector, memory):
    alice = new_client(settings, fake_llm, fake_stt, detector, memory)
    bob = new_client(settings, fake_llm, fake_stt, detector, memory)
    alice.get("/")
    bob.get("/")
    assert alice.cookies[USER_COOKIE] != bob.cookies[USER_COOKIE]

    fake_stt.text = "My dog Bruno has been sick and the vet is worried"
    talk(alice)
    talk(bob)
    assert [m["role"] for m in fake_llm.calls[-1]].count("system") == 1, "bob must not receive alice's memory"


def test_forget_me_deletes_memories_and_stops_recall(client, fake_llm, fake_stt, memory):
    fake_stt.text = "My dog Bruno has been sick and the vet is worried"
    talk(client, "a" * 32)
    assert memory.count() == 1

    res = client.delete("/memory")
    assert res.json() == {"deleted_memories": 1}
    assert memory.count() == 0

    talk(client, "b" * 32)
    assert [m["role"] for m in fake_llm.calls[-1]].count("system") == 1


def test_llm_rate_limit_gives_a_friendly_message_and_saves_nothing(client, fake_llm, memory):
    fake_llm.error = LLMRateLimited("429")
    res = talk(client)
    assert res.status_code == 200
    assert LLMRateLimited.user_message in res.text
    assert memory.count() == 0


def test_stt_rate_limit_gives_a_friendly_message(client, fake_stt, fake_llm):
    fake_stt.error = STTRateLimited("429")
    res = talk(client)
    assert res.text == f"METADATA||Sad|{STTRateLimited.user_message}"
    assert fake_llm.calls == []


def test_stage_directions_are_stripped_across_chunks(client, fake_llm):
    fake_llm.pieces = ["*sig", "hs* I ", "hear [pau", "se] you."]
    assert talk(client).text.endswith("| I hear  you.")


def test_server_tts_is_off_by_default(client):
    assert client.post("/get_audio", json={"text": "hello"}).status_code == 409


def test_upload_size_limit(settings, fake_llm, fake_stt, detector, memory):
    import dataclasses

    small = dataclasses.replace(settings, max_upload_mb=0)
    c = new_client(small, fake_llm, fake_stt, detector, memory)
    c.get("/")
    assert talk(c).status_code == 413
