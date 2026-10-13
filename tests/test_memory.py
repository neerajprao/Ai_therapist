from soulsync.memory import format_memories


def _seed(memory):
    memory.add("alice", "s1", "My dog Bruno has been sick and the vet is worried", emotion="sad")
    memory.add("alice", "s1", "I have a big exam on thermodynamics next week", emotion="anxious")
    memory.add("bob", "s9", "My dog Rocky ran away from the house yesterday", emotion="sad")


def test_retrieves_own_memory_from_an_earlier_session(memory):
    _seed(memory)
    hits = memory.search("alice", "s2", "the vet called about my dog Bruno")
    assert hits and "Bruno" in hits[0].text
    assert hits[0].emotion == "sad"


def test_never_returns_another_users_memories(memory):
    _seed(memory)
    hits = memory.search("alice", "s2", "my dog ran away from the house yesterday", max_distance=2.0)
    assert hits, "expected alice's own memories at a permissive threshold"
    assert all("Rocky" not in h.text for h in hits)
    assert memory.search("carol", "s1", "my dog is sick", max_distance=2.0) == []


def test_current_session_is_excluded(memory):
    _seed(memory)
    assert memory.search("alice", "s1", "my dog Bruno is sick", max_distance=2.0) == []


def test_distance_threshold_filters_unrelated_queries(memory):
    _seed(memory)
    query = "what should I cook for dinner tonight"
    assert memory.search("alice", "s2", query, max_distance=0.3) == []
    assert len(memory.search("alice", "s2", query, max_distance=2.0)) == 2


def test_results_are_sorted_and_limited_by_top_k(memory):
    for i in range(6):
        memory.add("alice", f"old{i}", f"my dog Bruno did trick number {i} today")
    hits = memory.search("alice", "now", "my dog Bruno did trick", max_distance=2.0)
    assert len(hits) == 3
    assert [h.distance for h in hits] == sorted(h.distance for h in hits)


def test_short_messages_are_not_stored(memory):
    assert memory.add("alice", "s1", "ok thanks") is False
    assert memory.add("alice", "s1", "   ") is False
    assert memory.count("alice") == 0
    assert memory.add("alice", "s1", "I feel better") is True


def test_reply_is_truncated_in_metadata(memory):
    memory.add("alice", "s1", "I am worried about my exam", reply="x" * 1000)
    hit = memory.search("alice", "s2", "worried about my exam")[0]
    assert len(hit.reply) == 200


def test_forget_user_deletes_only_that_user(memory):
    _seed(memory)
    assert memory.forget_user("alice") == 2
    assert memory.count("alice") == 0
    assert memory.count("bob") == 1
    assert memory.search("alice", "s2", "my dog Bruno is sick", max_distance=2.0) == []
    assert memory.forget_user("alice") == 0


def test_memory_note_tells_the_model_to_be_selective(memory):
    _seed(memory)
    note = format_memories(memory.search("alice", "s2", "my dog Bruno and the vet"))
    assert "Bruno" in note
    assert "only if" in note and "gently" in note
