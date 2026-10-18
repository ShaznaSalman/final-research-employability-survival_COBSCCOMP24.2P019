"""Tests for the isolated account and progress-storage service."""
import time
from pathlib import Path

import pytest
import requests
from fastapi.testclient import TestClient

from profile_service import store
from profile_service.main import create_app, normalize_phone


@pytest.fixture
def profile_client(tmp_path):
    db_path = tmp_path / "profiles.sqlite3"
    app = create_app(db_path)
    with TestClient(app) as client:
        yield client, db_path


def signed_in(db_path: Path, phone: str = "+94771234567") -> str:
    user_id = store.create_user(db_path, phone)
    return store.create_session(db_path, user_id)


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_phone_normalization_accepts_local_sri_lankan_mobile_formats():
    assert normalize_phone("+94 77 123 4567") == "+94771234567"
    assert normalize_phone("077-123-4567") == "+94771234567"
    assert normalize_phone("0094771234567") == "+94771234567"
    with pytest.raises(ValueError, match="Sri Lankan mobile"):
        normalize_phone("+1 212 555 0100")


def test_sms_sign_in_is_disabled_until_provider_credentials_are_configured(profile_client, monkeypatch):
    client, db_path = profile_client
    monkeypatch.delenv("SMS_API_KEY", raising=False)
    monkeypatch.delenv("SMS_SENDER_ID", raising=False)
    health = client.get("/health").json()
    assert health == {"status": "ok", "phone_sign_in_enabled": False, "retention_days": 365}

    response = client.post("/auth/request-code", json={"phone": "0771234567"})
    assert response.status_code == 503
    assert "SMS_API_KEY and SMS_SENDER_ID" in response.json()["detail"]
    with store.connect(db_path) as con:
        assert con.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM otp_challenges").fetchone()[0] == 0


def test_sms_otp_creates_account_only_after_valid_code(profile_client, monkeypatch):
    client, db_path = profile_client
    monkeypatch.setenv("SMS_API_KEY", "test-key")
    monkeypatch.setenv("SMS_SENDER_ID", "JobSpark")
    sent = []

    def fake_post(url, **kwargs):
        sent.append((url, kwargs))
        response = requests.Response()
        response.status_code = 202
        return response

    monkeypatch.setattr("profile_service.main.requests.post", fake_post)
    assert client.get("/health").json()["phone_sign_in_enabled"] is True
    requested = client.post("/auth/request-code", json={"phone": "0771234567"})
    assert requested.status_code == 200
    assert requested.json() == {"status": "sent", "expires_in_seconds": 300}
    assert len(sent) == 1
    url, kwargs = sent[0]
    assert url == "https://dashboard.smsapi.lk/api/v3/sms/send"
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert kwargs["json"]["recipient"] == "94771234567"
    assert kwargs["json"]["sender_id"] == "JobSpark"
    assert kwargs["json"]["type"] == "plain"
    code = kwargs["json"]["message"].split(" is ", 1)[1].split(".", 1)[0]
    assert code.isdigit() and len(code) == 6

    wrong_code = "000000" if code != "000000" else "000001"
    assert client.post("/auth/verify", json={"phone": "0771234567", "code": wrong_code}).status_code == 400
    with store.connect(db_path) as con:
        assert con.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
        assert con.execute("SELECT attempts FROM otp_challenges").fetchone()[0] == 1

    verified = client.post("/auth/verify", json={"phone": "0771234567", "code": code})
    assert verified.status_code == 200
    token = verified.json()["access_token"]
    account = client.get("/me", headers=auth(token)).json()
    assert account["phone"] == "+94771234567"
    assert account["display_name"] == ""
    assert account["avatar_emoji"] == "🙂"
    with store.connect(db_path) as con:
        assert con.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM otp_challenges").fetchone()[0] == 0


def test_otp_requests_are_rate_limited_per_phone(profile_client, monkeypatch):
    client, _ = profile_client
    monkeypatch.setenv("SMS_API_KEY", "test-key")
    monkeypatch.setenv("SMS_SENDER_ID", "JobSpark")
    sent = []

    def fake_post(url, **kwargs):
        sent.append(url)
        response = requests.Response()
        response.status_code = 202
        return response

    monkeypatch.setattr("profile_service.main.requests.post", fake_post)
    for _ in range(3):
        assert client.post("/auth/request-code", json={"phone": "0771234567"}).status_code == 200
    limited = client.post("/auth/request-code", json={"phone": "0771234567"})
    assert limited.status_code == 429
    assert len(sent) == 3


def test_saved_data_requires_a_valid_session(profile_client):
    client, _ = profile_client
    assert client.get("/snapshots").status_code == 401
    assert client.get("/goals").status_code == 401
    assert client.get("/actions").status_code == 401
    assert client.get("/snapshots", headers=auth("not-a-token")).status_code == 401


def test_account_profile_identity_is_saved_and_validated(profile_client):
    client, db_path = profile_client
    token = signed_in(db_path)
    headers = auth(token)

    updated = client.patch("/me", headers=headers,
                           json={"display_name": "  SkyCoder  ", "avatar_emoji": "🧑‍💻"})
    assert updated.status_code == 200
    assert updated.json()["display_name"] == "SkyCoder"
    assert updated.json()["avatar_emoji"] == "🧑‍💻"
    assert client.get("/me", headers=headers).json()["display_name"] == "SkyCoder"

    assert client.patch("/me", headers=headers,
                        json={"display_name": " ", "avatar_emoji": "🧑‍💻"}).status_code == 422
    assert client.patch("/me", headers=headers,
                        json={"display_name": "Valid", "avatar_emoji": "📷"}).status_code == 422
    assert client.patch("/me", headers=auth("invalid"),
                        json={"display_name": "Valid", "avatar_emoji": "🙂"}).status_code == 401


def test_profile_snapshots_are_isolated_and_expire_after_one_year(profile_client):
    client, db_path = profile_client
    token = signed_in(db_path)
    first = client.post("/snapshots", headers=auth(token), json={
        "answers": {"p_gpa": 3.2},
        "profile": {"gpa": 3.2},
        "estimate": {"median": {"display": "5"}, "confidence": "High"},
        "outputs": {
            "prediction": {"headline": "Example estimate"},
            "suggestions": {"suggestions": [{"text": "Build a project"}]},
            "career_pathways": {"recommendations": [{"role": "Data & AI", "match_rating": 4.2}]},
            "career_path": {"selected_role": {"role": "Data & AI", "gaps": [{"label": "Python"}]}},
            "improve_chances": {
                "what_if": {
                    "changes": {"internship_months": 6},
                    "result": {"message": "Comparison generated"},
                },
            },
        },
    })
    assert first.status_code == 201
    item = first.json()
    assert item["answers"] == {"p_gpa": 3.2}
    assert item["profile"] == {"gpa": 3.2}
    assert item["estimate"]["confidence"] == "High"
    assert item["outputs"]["prediction"]["headline"] == "Example estimate"
    assert item["outputs"]["suggestions"]["suggestions"][0]["text"] == "Build a project"
    assert item["outputs"]["career_pathways"]["recommendations"][0]["role"] == "Data & AI"
    assert item["outputs"]["career_path"]["selected_role"]["gaps"] == [{"label": "Python"}]
    assert item["outputs"]["improve_chances"]["what_if"]["result"]["message"] == "Comparison generated"

    listed = client.get("/snapshots", headers=auth(token)).json()
    assert [row["id"] for row in listed] == [item["id"]]
    latest = client.get("/snapshots/latest", headers=auth(token)).json()
    assert latest["id"] == item["id"]
    assert latest["outputs"] == item["outputs"]
    assert client.delete(f"/snapshots/{item['id']}", headers=auth(token)).status_code == 204
    assert client.get("/snapshots/latest", headers=auth(token)).status_code == 404

    with store.connect(db_path) as con:
        con.execute(
            "INSERT INTO snapshots(id, user_id, answers_json, profile_json, estimate_json, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("expired", con.execute("SELECT id FROM users").fetchone()[0], "{}", "{}", "{}",
             time.time() - store.RETENTION_SECONDS - 1, time.time() - 1),
        )
    assert client.get("/snapshots", headers=auth(token)).json() == []


def test_existing_snapshot_database_is_migrated_with_empty_outputs(tmp_path):
    db_path = tmp_path / "old_profiles.sqlite3"
    with store.connect(db_path) as con:
        con.executescript("""
            CREATE TABLE users (
                id TEXT PRIMARY KEY, phone TEXT NOT NULL UNIQUE, created_at REAL NOT NULL, expires_at REAL NOT NULL
            );
            CREATE TABLE snapshots (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL, answers_json TEXT NOT NULL,
                profile_json TEXT NOT NULL, estimate_json TEXT NOT NULL, created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            );
        """)
        con.execute("INSERT INTO users VALUES ('user', '+94771234567', ?, ?)", (time.time(), time.time() + 1000))
        con.execute("INSERT INTO snapshots VALUES ('snapshot', 'user', '{}', '{}', '{}', ?, ?)",
                    (time.time(), time.time() + 1000))

    store.initialize(db_path)
    with store.connect(db_path) as con:
        columns = {row["name"] for row in con.execute("PRAGMA table_info(snapshots)")}
        assert "outputs_json" in columns
        assert con.execute("SELECT outputs_json FROM snapshots").fetchone()[0] == "{}"
        user_columns = {row["name"] for row in con.execute("PRAGMA table_info(users)")}
        assert {"display_name", "avatar_emoji"} <= user_columns
        assert con.execute("SELECT display_name, avatar_emoji FROM users WHERE id = 'user'").fetchone()[:] == ("", "🙂")


def test_career_goals_and_action_checklist_can_be_updated(profile_client):
    client, db_path = profile_client
    token = signed_in(db_path)

    goal_response = client.post("/goals", headers=auth(token),
                                json={"role": "Data & AI", "gaps": [{"label": "Python"}]})
    assert goal_response.status_code == 201
    goal = goal_response.json()
    assert goal["role"] == "Data & AI" and goal["status"] == "active"
    assert goal["gaps"] == [{"label": "Python"}]
    updated_goal = client.patch(f"/goals/{goal['id']}", headers=auth(token),
                                json={"status": "paused", "gaps": []})
    assert updated_goal.status_code == 200 and updated_goal.json()["status"] == "paused"
    assert client.get("/goals", headers=auth(token)).json()[0]["gaps"] == []

    action_response = client.post("/actions", headers=auth(token),
                                  json={"title": "Build a Python project", "detail": "Add it to my portfolio"})
    assert action_response.status_code == 201
    action = action_response.json()
    assert action["state"] == "not_started"
    updated_action = client.patch(f"/actions/{action['id']}", headers=auth(token),
                                  json={"state": "in_progress"})
    assert updated_action.status_code == 200 and updated_action.json()["state"] == "in_progress"
    assert client.get("/actions", headers=auth(token)).json()[0]["title"] == "Build a Python project"

    assert client.patch(f"/actions/{action['id']}", headers=auth(token),
                        json={"state": "invalid"}).status_code == 422
    assert client.delete(f"/actions/{action['id']}", headers=auth(token)).status_code == 204
    assert client.delete(f"/goals/{goal['id']}", headers=auth(token)).status_code == 204


def test_account_deletion_removes_all_owned_data_and_token_is_hashed(profile_client):
    client, db_path = profile_client
    token = signed_in(db_path)
    created = client.post("/actions", headers=auth(token), json={"title": "Practice interviews"})
    assert created.status_code == 201
    with store.connect(db_path) as con:
        stored_hashes = [row[0] for row in con.execute("SELECT token_hash FROM sessions")]
    assert token not in stored_hashes and store.hash_secret(token) in stored_hashes
    assert client.delete("/me", headers=auth(token)).status_code == 204
    assert client.get("/me", headers=auth(token)).status_code == 401
    with store.connect(db_path) as con:
        assert con.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM actions").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


def test_logout_revokes_the_session(profile_client):
    client, db_path = profile_client
    token = signed_in(db_path)
    assert client.post("/auth/logout", headers=auth(token)).status_code == 204
    assert client.get("/me", headers=auth(token)).status_code == 401


def test_wrong_user_cannot_delete_another_users_records(profile_client):
    client, db_path = profile_client
    owner_token = signed_in(db_path, "+94771234567")
    other_token = signed_in(db_path, "+94771234568")
    goal = client.post("/goals", headers=auth(owner_token), json={"role": "Software engineering"}).json()
    assert client.delete(f"/goals/{goal['id']}", headers=auth(other_token)).status_code == 404
    assert len(client.get("/goals", headers=auth(owner_token)).json()) == 1
