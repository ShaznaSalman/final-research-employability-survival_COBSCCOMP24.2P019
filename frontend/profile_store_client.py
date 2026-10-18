"""Client for the optional profile/progress service on its own port."""
from __future__ import annotations

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
PROFILE_STORE_URL = os.getenv("PROFILE_STORE_URL", "http://127.0.0.1:8001").rstrip("/")
TIMEOUT = 15


class ProfileStoreError(Exception):
    """Readable error from the separate profile/progress service."""


def request(method: str, path: str, payload: dict | None = None, token: str | None = None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        response = requests.request(method, f"{PROFILE_STORE_URL}{path}", json=payload, headers=headers,
                                    timeout=TIMEOUT)
    except requests.ConnectionError:
        raise ProfileStoreError(
            f"The profile storage service at {PROFILE_STORE_URL} is not reachable. "
            "Start it with `uvicorn profile_service.main:app --port 8001`."
        ) from None
    if response.status_code not in (200, 201, 204):
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ProfileStoreError(str(detail))
    if response.status_code == 204:
        return None
    try:
        return response.json()
    except ValueError:
        raise ProfileStoreError("The profile storage service returned an invalid response.") from None


def get(path: str, token: str | None = None):
    return request("GET", path, token=token)


def post(path: str, payload: dict, token: str | None = None):
    return request("POST", path, payload=payload, token=token)


def patch(path: str, payload: dict, token: str):
    return request("PATCH", path, payload=payload, token=token)


def delete(path: str, token: str):
    return request("DELETE", path, token=token)
