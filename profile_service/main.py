"""Standalone account/progress API. It does not import or modify the prediction API."""
from __future__ import annotations

import hmac
import json
import os
import re
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
import requests

from profile_service import store

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
_DEFAULT_DB = ROOT / "data" / "profile_store.sqlite3"
RETENTION_SECONDS = store.RETENTION_SECONDS
OTP_SECONDS = 5 * 60
OTP_MAX_ATTEMPTS = 5
OTP_REQUEST_LIMIT = 3
OTP_REQUEST_WINDOW = 15 * 60
SMS_DEFAULT_ENDPOINT = "https://dashboard.smsapi.lk/api/v3/sms/send"
PROFILE_AVATARS = ("🙂", "😀", "😎", "🤓", "🧑‍💻", "👩‍💻", "👨‍💻", "🚀", "🌱", "⭐", "🐼", "🦊")


def sms_config() -> tuple[str, str, str, str] | None:
    endpoint = os.getenv("SMS_API_ENDPOINT", SMS_DEFAULT_ENDPOINT).strip()
    api_key = os.getenv("SMS_API_KEY", "").strip()
    sender_id = os.getenv("SMS_SENDER_ID", "").strip()
    auth_scheme = os.getenv("SMS_API_AUTH_SCHEME", "Bearer").strip()
    if not (endpoint.startswith("https://") and api_key and sender_id):
        return None
    return endpoint, api_key, sender_id, auth_scheme


def normalize_phone(value: str) -> str:
    digits = re.sub(r"[\s().-]", "", value)
    if digits.startswith("00"):
        digits = digits[2:]
    elif digits.startswith("0"):
        digits = "94" + digits[1:]
    elif digits.startswith("+"):
        digits = digits[1:]
    if not re.fullmatch(r"947\d{8}", digits):
        raise ValueError("Enter a Sri Lankan mobile number, such as +94 77 123 4567.")
    return "+" + digits


class PhoneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    phone: str = Field(..., min_length=9, max_length=24)

    @model_validator(mode="after")
    def _normalise(self):
        self.phone = normalize_phone(self.phone)
        return self


class VerifyRequest(PhoneRequest):
    code: str = Field(..., pattern=r"^\d{6}$")


class ProfileIdentityUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    display_name: str = Field(..., min_length=1, max_length=40)
    avatar_emoji: str

    @model_validator(mode="after")
    def _validate_identity(self):
        self.display_name = self.display_name.strip()
        if not self.display_name:
            raise ValueError("Display name cannot be blank.")
        if self.avatar_emoji not in PROFILE_AVATARS:
            raise ValueError("Choose an avatar from the available emoji.")
        return self


class SnapshotCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answers: dict
    profile: dict
    estimate: dict
    outputs: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _limit_payload_size(self):
        if len(self.model_dump_json()) > 256_000:
            raise ValueError("Profile snapshot must be smaller than 256 KB.")
        return self


class GoalCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str = Field(..., min_length=1, max_length=100)
    gaps: list[dict] = Field(default_factory=list, max_length=30)


class GoalUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["active", "paused", "completed"]
    gaps: list[dict] | None = Field(None, max_length=30)


class ActionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(..., min_length=1, max_length=240)
    detail: str = Field("", max_length=2000)


class ActionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: Literal["not_started", "in_progress", "completed"] | None = None
    detail: str | None = Field(None, max_length=2000)

    @model_validator(mode="after")
    def _require_change(self):
        if self.state is None and self.detail is None:
            raise ValueError("Provide an action state or detail to update.")
        return self


def _record(row, json_columns: tuple[str, ...] = ()) -> dict:
    result = dict(row)
    for column in json_columns:
        result[column] = store.json_field(result.pop(f"{column}_json"))
    result["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(result["created_at"]))
    result["expires_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(result["expires_at"]))
    if "updated_at" in result:
        result["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(result["updated_at"]))
    return result


def _owned_row(db_path: Path, table: str, record_id: str, user_id: str):
    if table not in {"snapshots", "career_goals", "actions"}:
        raise ValueError("Unsupported record type.")
    with store.connect(db_path) as con:
        row = con.execute(f"SELECT * FROM {table} WHERE id = ? AND user_id = ?", (record_id, user_id)).fetchone()
    return row


def create_app(db_path: Path | None = None) -> FastAPI:
    database = db_path or Path(os.getenv("PROFILE_STORE_DB", str(_DEFAULT_DB)))
    if not database.is_absolute():
        database = ROOT / database

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        store.initialize(database)
        store.purge_expired(database)
        yield

    app = FastAPI(
        title="JobSpark Profile & Progress Store",
        description="Separate optional account and progress storage. The prediction API is not used or modified.",
        version="1.0",
        lifespan=lifespan,
    )
    app.state.database = database

    def user_id(authorization: str | None = Header(None)) -> str:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Sign in to access saved progress.")
        token = authorization.removeprefix("Bearer ").strip()
        if not token or len(token) > 256:
            raise HTTPException(status_code=401, detail="Sign in to access saved progress.")
        store.purge_expired(database)
        with store.connect(database) as con:
            row = con.execute(
                "SELECT s.user_id FROM sessions s JOIN users u ON u.id = s.user_id "
                "WHERE s.token_hash = ? AND s.expires_at > ? AND u.expires_at > ?",
                (store.hash_secret(token), time.time(), time.time()),
            ).fetchone()
        if row is None:
            raise HTTPException(status_code=401, detail="Your session expired. Sign in again.")
        return row["user_id"]

    @app.get("/health")
    def health():
        store.initialize(database)
        store.purge_expired(database)
        return {"status": "ok", "phone_sign_in_enabled": sms_config() is not None, "retention_days": 365}

    @app.post("/auth/request-code")
    def request_code(body: PhoneRequest):
        config = sms_config()
        if config is None:
            raise HTTPException(
                status_code=503,
                detail="Phone sign-in is not configured. Set SMS_API_KEY and SMS_SENDER_ID in the profile service environment.",
            )

        now = time.time()
        with store.connect(database) as con:
            recent_count = con.execute(
                "SELECT COUNT(*) FROM otp_request_history WHERE phone = ? AND created_at > ?",
                (body.phone, now - OTP_REQUEST_WINDOW),
            ).fetchone()[0]
            if recent_count >= OTP_REQUEST_LIMIT:
                raise HTTPException(status_code=429, detail="Too many code requests. Try again later.")
            con.execute(
                "INSERT INTO otp_request_history(id, phone, created_at) VALUES (?, ?, ?)",
                (str(uuid.uuid4()), body.phone, now),
            )
            con.execute("DELETE FROM otp_challenges WHERE phone = ?", (body.phone,))
            code = f"{secrets.randbelow(1_000_000):06d}"
            challenge_id = str(uuid.uuid4())
            con.execute(
                "INSERT INTO otp_challenges(id, phone, code_hash, created_at, expires_at) VALUES (?, ?, ?, ?, ?)",
                (challenge_id, body.phone, store.hash_secret(f"{body.phone}:{code}"), now, now + OTP_SECONDS),
            )

        endpoint, api_key, sender_id, auth_scheme = config
        if api_key.lower().startswith("bearer "):
            authorization = api_key
        else:
            authorization = f"{auth_scheme} {api_key}" if auth_scheme else api_key
        try:
            response = requests.post(
                endpoint,
                headers={
                    "Authorization": authorization,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json={
                    "recipient": body.phone.removeprefix("+"),
                    "sender_id": sender_id,
                    "type": "plain",
                    "message": f"Your JobSpark sign-in code is {code}. It expires in five minutes.",
                },
                timeout=10,
            )
            response.raise_for_status()
        except requests.RequestException:
            with store.connect(database) as con:
                con.execute("DELETE FROM otp_challenges WHERE id = ?", (challenge_id,))
            raise HTTPException(
                status_code=502,
                detail="The SMS provider could not accept the verification message. Please try again later.",
            ) from None
        return {"status": "sent", "expires_in_seconds": OTP_SECONDS}

    @app.post("/auth/verify")
    def verify_code(body: VerifyRequest):
        if sms_config() is None:
            raise HTTPException(status_code=503, detail="Phone sign-in is not configured.")
        now = time.time()
        invalid_code = False
        with store.connect(database) as con:
            challenge = con.execute(
                "SELECT * FROM otp_challenges WHERE phone = ? ORDER BY created_at DESC LIMIT 1",
                (body.phone,),
            ).fetchone()
            if challenge is None or challenge["expires_at"] <= now:
                raise HTTPException(status_code=400, detail="The code is invalid or expired. Request a new code.")
            if challenge["attempts"] >= OTP_MAX_ATTEMPTS:
                raise HTTPException(status_code=429, detail="Too many incorrect codes. Request a new code later.")
            con.execute("UPDATE otp_challenges SET attempts = attempts + 1 WHERE id = ?", (challenge["id"],))
            expected = store.hash_secret(f"{body.phone}:{body.code}")
            if not hmac.compare_digest(challenge["code_hash"], expected):
                invalid_code = True
            else:
                con.execute("DELETE FROM otp_challenges WHERE phone = ?", (body.phone,))
                account = con.execute("SELECT id FROM users WHERE phone = ?", (body.phone,)).fetchone()
                user = account["id"] if account else secrets.token_urlsafe(18)
                if account is None:
                    con.execute(
                        "INSERT INTO users(id, phone, created_at, expires_at) VALUES (?, ?, ?, ?)",
                        (user, body.phone, now, now + RETENTION_SECONDS),
                    )
                else:
                    con.execute("UPDATE users SET expires_at = ? WHERE id = ?", (now + RETENTION_SECONDS, user))
                token = secrets.token_urlsafe(32)
                con.execute(
                    "INSERT INTO sessions(token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                    (store.hash_secret(token), user, now, now + store.TOKEN_SECONDS),
                )
        if invalid_code:
            raise HTTPException(status_code=400, detail="The code is invalid or expired. Request a new code.")
        return {"access_token": token, "token_type": "bearer", "expires_in_seconds": store.TOKEN_SECONDS}

    @app.post("/auth/logout", status_code=204)
    def logout(authorization: str | None = Header(None), user: str = Depends(user_id)):
        token = authorization.removeprefix("Bearer ").strip()
        with store.connect(database) as con:
            con.execute("DELETE FROM sessions WHERE token_hash = ? AND user_id = ?",
                        (store.hash_secret(token), user))
        return None

    @app.get("/me")
    def me(user: str = Depends(user_id)):
        with store.connect(database) as con:
            row = con.execute(
                "SELECT phone, display_name, avatar_emoji, created_at, expires_at FROM users WHERE id = ?",
                (user,),
            ).fetchone()
        if row is None:
            raise HTTPException(status_code=401, detail="Your account expired. Sign in again.")
        return {
            "phone": row["phone"],
            "display_name": row["display_name"],
            "avatar_emoji": row["avatar_emoji"],
            "created_at": row["created_at"],
            "expires_at": row["expires_at"],
        }

    @app.patch("/me")
    def update_me(body: ProfileIdentityUpdate, user: str = Depends(user_id)):
        now = time.time()
        with store.connect(database) as con:
            con.execute(
                "UPDATE users SET display_name = ?, avatar_emoji = ?, expires_at = ? WHERE id = ?",
                (body.display_name, body.avatar_emoji, now + RETENTION_SECONDS, user),
            )
            row = con.execute(
                "SELECT phone, display_name, avatar_emoji, created_at, expires_at FROM users WHERE id = ?",
                (user,),
            ).fetchone()
        if row is None:
            raise HTTPException(status_code=401, detail="Your account expired. Sign in again.")
        return dict(row)

    @app.delete("/me", status_code=204)
    def delete_account(user: str = Depends(user_id)):
        with store.connect(database) as con:
            con.execute("DELETE FROM users WHERE id = ?", (user,))
        return None

    @app.post("/snapshots", status_code=201)
    def create_snapshot(body: SnapshotCreate, user: str = Depends(user_id)):
        now, item_id = time.time(), str(uuid.uuid4())
        with store.connect(database) as con:
            con.execute(
                "INSERT INTO snapshots(id, user_id, answers_json, profile_json, estimate_json, outputs_json, "
                "created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (item_id, user, json.dumps(body.answers), json.dumps(body.profile), json.dumps(body.estimate),
                 json.dumps(body.outputs), now, now + RETENTION_SECONDS),
            )
            row = con.execute("SELECT * FROM snapshots WHERE id = ?", (item_id,)).fetchone()
        return _record(row, ("answers", "profile", "estimate", "outputs"))

    @app.get("/snapshots")
    def list_snapshots(user: str = Depends(user_id)):
        with store.connect(database) as con:
            rows = con.execute("SELECT * FROM snapshots WHERE user_id = ? ORDER BY created_at DESC", (user,)).fetchall()
        return [_record(row, ("answers", "profile", "estimate", "outputs")) for row in rows]

    @app.get("/snapshots/latest")
    def latest_snapshot(user: str = Depends(user_id)):
        with store.connect(database) as con:
            row = con.execute("SELECT * FROM snapshots WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
                              (user,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="No saved profile is available.")
        return _record(row, ("answers", "profile", "estimate", "outputs"))

    @app.delete("/snapshots/{snapshot_id}", status_code=204)
    def delete_snapshot(snapshot_id: str, user: str = Depends(user_id)):
        with store.connect(database) as con:
            result = con.execute("DELETE FROM snapshots WHERE id = ? AND user_id = ?", (snapshot_id, user))
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Saved profile not found.")
        return None

    @app.post("/goals", status_code=201)
    def create_goal(body: GoalCreate, user: str = Depends(user_id)):
        now, item_id = time.time(), str(uuid.uuid4())
        with store.connect(database) as con:
            con.execute(
                "INSERT INTO career_goals(id, user_id, role, status, gaps_json, created_at, updated_at, expires_at) "
                "VALUES (?, ?, ?, 'active', ?, ?, ?, ?)",
                (item_id, user, body.role.strip(), json.dumps(body.gaps), now, now, now + RETENTION_SECONDS),
            )
            row = con.execute("SELECT * FROM career_goals WHERE id = ?", (item_id,)).fetchone()
        return _record(row, ("gaps",))

    @app.get("/goals")
    def list_goals(user: str = Depends(user_id)):
        with store.connect(database) as con:
            rows = con.execute("SELECT * FROM career_goals WHERE user_id = ? ORDER BY updated_at DESC",
                               (user,)).fetchall()
        return [_record(row, ("gaps",)) for row in rows]

    @app.patch("/goals/{goal_id}")
    def update_goal(goal_id: str, body: GoalUpdate, user: str = Depends(user_id)):
        with store.connect(database) as con:
            row = con.execute("SELECT * FROM career_goals WHERE id = ? AND user_id = ?", (goal_id, user)).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Career goal not found.")
            gaps_json = json.dumps(body.gaps) if body.gaps is not None else row["gaps_json"]
            now = time.time()
            con.execute("UPDATE career_goals SET status = ?, gaps_json = ?, updated_at = ?, expires_at = ? "
                        "WHERE id = ? AND user_id = ?",
                        (body.status, gaps_json, now, now + RETENTION_SECONDS, goal_id, user))
            updated = con.execute("SELECT * FROM career_goals WHERE id = ?", (goal_id,)).fetchone()
        return _record(updated, ("gaps",))

    @app.delete("/goals/{goal_id}", status_code=204)
    def delete_goal(goal_id: str, user: str = Depends(user_id)):
        with store.connect(database) as con:
            result = con.execute("DELETE FROM career_goals WHERE id = ? AND user_id = ?", (goal_id, user))
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Career goal not found.")
        return None

    @app.post("/actions", status_code=201)
    def create_action(body: ActionCreate, user: str = Depends(user_id)):
        now, item_id = time.time(), str(uuid.uuid4())
        with store.connect(database) as con:
            con.execute(
                "INSERT INTO actions(id, user_id, title, detail, state, created_at, updated_at, expires_at) "
                "VALUES (?, ?, ?, ?, 'not_started', ?, ?, ?)",
                (item_id, user, body.title.strip(), body.detail.strip(), now, now, now + RETENTION_SECONDS),
            )
            row = con.execute("SELECT * FROM actions WHERE id = ?", (item_id,)).fetchone()
        return _record(row)

    @app.get("/actions")
    def list_actions(user: str = Depends(user_id)):
        with store.connect(database) as con:
            rows = con.execute("SELECT * FROM actions WHERE user_id = ? ORDER BY updated_at DESC", (user,)).fetchall()
        return [_record(row) for row in rows]

    @app.patch("/actions/{action_id}")
    def update_action(action_id: str, body: ActionUpdate, user: str = Depends(user_id)):
        with store.connect(database) as con:
            row = con.execute("SELECT * FROM actions WHERE id = ? AND user_id = ?", (action_id, user)).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Progress action not found.")
            state = body.state if body.state is not None else row["state"]
            detail = body.detail.strip() if body.detail is not None else row["detail"]
            now = time.time()
            con.execute("UPDATE actions SET state = ?, detail = ?, updated_at = ?, expires_at = ? "
                        "WHERE id = ? AND user_id = ?",
                        (state, detail, now, now + RETENTION_SECONDS, action_id, user))
            updated = con.execute("SELECT * FROM actions WHERE id = ?", (action_id,)).fetchone()
        return _record(updated)

    @app.delete("/actions/{action_id}", status_code=204)
    def delete_action(action_id: str, user: str = Depends(user_id)):
        with store.connect(database) as con:
            result = con.execute("DELETE FROM actions WHERE id = ? AND user_id = ?", (action_id, user))
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Progress action not found.")
        return None

    return app


app = create_app()
