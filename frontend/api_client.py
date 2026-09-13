"""Thin HTTP client for the FastAPI backend. The API URL comes from .env (API_URL)."""
from __future__ import annotations

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 30


class APIError(Exception):
    """Readable error for the UI (validation message, or 'API not reachable')."""


def _handle(resp: requests.Response) -> dict:
    if resp.status_code == 200:
        return resp.json()
    try:
        detail = resp.json().get("detail")
    except ValueError:
        detail = resp.text
    if isinstance(detail, list):                       # pydantic validation errors
        detail = "; ".join(f"{'.'.join(map(str, d.get('loc', [])[1:]))}: {d.get('msg')}" for d in detail)
    raise APIError(str(detail))


def get(path: str) -> dict:
    try:
        return _handle(requests.get(f"{API_URL}{path}", timeout=TIMEOUT))
    except requests.ConnectionError:
        raise APIError(f"The prediction service at {API_URL} is not reachable. Start it with "
                       "`uvicorn api.main:app --port 8000`.") from None


def post(path: str, payload: dict) -> dict:
    try:
        return _handle(requests.post(f"{API_URL}{path}", json=payload, timeout=TIMEOUT))
    except requests.ConnectionError:
        raise APIError(f"The prediction service at {API_URL} is not reachable. Start it with "
                       "`uvicorn api.main:app --port 8000`.") from None
