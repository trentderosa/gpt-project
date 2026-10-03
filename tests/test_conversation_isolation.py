"""Account isolation: a user must never be able to read, list, or delete another
user's conversations. This is the highest-priority test in the suite per the
upgrade spec's "account isolation" requirement.
"""


def test_user_cannot_read_another_users_conversation(client, make_user):
    owner = make_user("owner")
    intruder = make_user("intruder")

    create_resp = client.post("/conversations", json={}, headers=owner["headers"])
    assert create_resp.status_code == 200
    conversation_id = create_resp.json()["conversation_id"]

    # Owner can read their own empty conversation.
    own_resp = client.get(f"/conversations/{conversation_id}", headers=owner["headers"])
    assert own_resp.status_code == 200
    assert own_resp.json()["conversation_id"] == conversation_id

    # Intruder must be forbidden.
    intruder_resp = client.get(f"/conversations/{conversation_id}", headers=intruder["headers"])
    assert intruder_resp.status_code == 403

    # Anonymous (no auth) must also be forbidden for an owned conversation.
    anon_resp = client.get(f"/conversations/{conversation_id}")
    assert anon_resp.status_code == 403


def test_user_cannot_delete_another_users_conversation(client, make_user):
    owner = make_user("owner")
    intruder = make_user("intruder")

    create_resp = client.post("/conversations", json={}, headers=owner["headers"])
    conversation_id = create_resp.json()["conversation_id"]

    delete_resp = client.delete(f"/conversations/{conversation_id}", headers=intruder["headers"])
    assert delete_resp.status_code == 403

    # Conversation must still exist for the owner afterward.
    still_there = client.get(f"/conversations/{conversation_id}", headers=owner["headers"])
    assert still_there.status_code == 200


def test_conversation_list_only_returns_own_conversations(client, make_user):
    user_a = make_user("alice")
    user_b = make_user("bob")

    a_conv = client.post("/conversations", json={}, headers=user_a["headers"]).json()["conversation_id"]
    b_conv = client.post("/conversations", json={}, headers=user_b["headers"]).json()["conversation_id"]

    a_list = client.get("/conversations", headers=user_a["headers"]).json()["conversations"]
    a_ids = {c["conversation_id"] for c in a_list}
    assert a_conv in a_ids
    assert b_conv not in a_ids

    b_list = client.get("/conversations", headers=user_b["headers"]).json()["conversations"]
    b_ids = {c["conversation_id"] for c in b_list}
    assert b_conv in b_ids
    assert a_conv not in b_ids


def test_anonymous_list_is_empty_not_an_error(client):
    resp = client.get("/conversations")
    assert resp.status_code == 200
    assert resp.json()["conversations"] == []


def test_nonexistent_conversation_returns_404(client, make_user):
    user = make_user("solo")
    resp = client.get("/conversations/does-not-exist", headers=user["headers"])
    assert resp.status_code == 404
