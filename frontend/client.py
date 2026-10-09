"""HTTP client for the FastAPI backend (POST /chat). Independent of backend code."""
import os
from dataclasses import dataclass, field

import requests

DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")


class BackendError(Exception):
    """Raised when the backend is unreachable or returns an unusable response."""


@dataclass
class Citation:
    document_name: str
    page_number: int
    section: str
    sub_section: str | None = None
    category: str | None = None
    document_url: str | None = None


@dataclass
class Reply:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    status: str = "answered"


def send_message(message: str, thread_id: str, api_url: str | None = None,
                 timeout: float = 120) -> Reply:
    url = (api_url or DEFAULT_API_URL).rstrip("/") + "/chat"
    try:
        resp = requests.post(url, json={"message": message, "thread_id": thread_id},
                             timeout=timeout)
    except requests.Timeout as e:
        raise BackendError("The backend timed out. Please try again.") from e
    except requests.RequestException as e:
        raise BackendError(f"Could not reach the backend at {url}.") from e
    if resp.status_code != 200:
        raise BackendError(f"Backend returned HTTP {resp.status_code}.")
    try:
        data = resp.json()
        cites = [
            Citation(
                document_name=c["document_name"],
                page_number=c["page_number"],
                section=c["section"],
                sub_section=c.get("sub_section"),
                category=c.get("category"),
                document_url=c.get("document_url") or None,
            )
            for c in data.get("citations") or []
        ]
        return Reply(answer=data["answer"], citations=cites,
                     categories=[str(x) for x in (data.get("categories") or []) if x],
                     status=data.get("status", "answered"))
    except (ValueError, KeyError, TypeError) as e:
        raise BackendError("Backend returned a malformed response.") from e
