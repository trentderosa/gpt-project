from tests.conftest import DEFAULT_TEST_PASSWORD, unique_email


def test_register_then_me_returns_user(client, make_user):
    user = make_user("regcheck")
    resp = client.get("/auth/me", headers=user["headers"])
    assert resp.status_code == 200
    body = resp.json()
    assert body["user"]["email"] == user["email"]
    assert body["user"]["plan"] == "free"


def test_duplicate_registration_rejected(client, make_user):
    user = make_user("dupe")
    resp = client.post(
        "/auth/register",
        json={"email": user["email"], "password": DEFAULT_TEST_PASSWORD},
    )
    assert resp.status_code == 400


def test_login_with_wrong_password_rejected(client, make_user):
    user = make_user("wrongpw")
    resp = client.post(
        "/auth/login",
        json={"email": user["email"], "password": "definitely-not-the-password-1"},
    )
    assert resp.status_code == 401


def test_login_with_correct_password_succeeds(client, make_user):
    user = make_user("rightpw")
    resp = client.post(
        "/auth/login",
        json={"email": user["email"], "password": user["password"]},
    )
    assert resp.status_code == 200
    assert resp.json()["user"]["email"] == user["email"]


def test_unauthenticated_me_returns_401(client):
    resp = client.get("/auth/me")
    assert resp.status_code == 401


def test_logout_invalidates_session(client, make_user):
    user = make_user("logout")
    resp = client.post("/auth/logout", headers=user["headers"])
    assert resp.status_code == 200

    me_resp = client.get("/auth/me", headers=user["headers"])
    assert me_resp.status_code == 401


def test_nonexistent_login_rejected(client):
    resp = client.post(
        "/auth/login",
        json={"email": unique_email("ghost"), "password": DEFAULT_TEST_PASSWORD},
    )
    assert resp.status_code == 401
