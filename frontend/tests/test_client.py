import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import client  # noqa: E402

OK = {"answer": "Wear gloves.", "categories": ["safety_procedures", "maintenance_manuals"], "status": "answered",
      "citations": [{"document_name": "Safety.pdf", "page_number": 3, "section": "PPE",
                     "sub_section": "Gloves", "category": "safety_procedures",
                     "document_url": "/documents/view?document=Safety.pdf"}]}


def _resp(code=200, data=None, bad_json=False):
    r = MagicMock(status_code=code)
    if bad_json:
        r.json.side_effect = ValueError()
    else:
        r.json.return_value = data
    return r


def test_success():
    with patch("client.requests.post", return_value=_resp(data=OK)) as p:
        rep = client.send_message("hi", "t1", api_url="http://x/")
    assert p.call_args.args[0] == "http://x/chat"
    assert p.call_args.kwargs["json"] == {"message": "hi", "thread_id": "t1"}
    assert rep.answer == "Wear gloves."
    assert rep.citations[0].sub_section == "Gloves"
    assert rep.categories == ["safety_procedures", "maintenance_manuals"]
    assert rep.citations[0].document_url == "/documents/view?document=Safety.pdf"


def test_missing_categories_and_url_tolerated():
    d = {"answer": "a", "citations": [{"document_name": "D", "page_number": 1, "section": "S"}]}
    with patch("client.requests.post", return_value=_resp(data=d)):
        rep = client.send_message("x", "t")
    assert rep.categories == [] and rep.citations[0].document_url is None


def test_blocked_no_citations():
    with patch("client.requests.post", return_value=_resp(data={"answer": "no", "status": "blocked"})):
        rep = client.send_message("x", "t")
    assert rep.status == "blocked" and rep.citations == []


@pytest.mark.parametrize("exc", [requests.ConnectionError(), requests.Timeout()])
def test_network_errors(exc):
    with patch("client.requests.post", side_effect=exc):
        with pytest.raises(client.BackendError):
            client.send_message("x", "t")


def test_http_error():
    with patch("client.requests.post", return_value=_resp(500, {})):
        with pytest.raises(client.BackendError):
            client.send_message("x", "t")


def test_malformed():
    for r in (_resp(bad_json=True), _resp(data={"nope": 1})):
        with patch("client.requests.post", return_value=r):
            with pytest.raises(client.BackendError):
                client.send_message("x", "t")
