import os
import sys
from unittest.mock import patch

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import client  # noqa: E402

APP = os.path.join(os.path.dirname(__file__), "..", "app.py")


def test_smoke_and_chat():
    at = AppTest.from_file(APP).run()
    assert not at.exception
    rep = client.Reply(answer="Out of scope msg", status="out_of_scope")
    with patch("client.send_message", return_value=rep):
        at.chat_input[0].set_value("pizza?").run()
    assert not at.exception
    assert len(at.info) == 1


def test_backend_error():
    at = AppTest.from_file(APP).run()
    with patch("client.send_message", side_effect=client.BackendError("down")):
        at.chat_input[0].set_value("hi").run()
    assert len(at.warning) == 1


def test_categories_and_citation_links():
    at = AppTest.from_file(APP).run()
    cites = [client.Citation("Safety.pdf", 3, "PPE", "Gloves", "safety_procedures",
                             "/documents/view?document=Safety.pdf&page=3"),
             client.Citation("Plain.pdf", 1, "Intro")]
    rep = client.Reply(answer="Wear gloves.", citations=cites,
                       categories=["safety_procedures", "quality_control_standards"])
    with patch("client.send_message", return_value=rep):
        at.chat_input[0].set_value("gloves?").run()
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "Safety procedures" in md and "Quality control standards" in md
    links = at.get("link_button")
    assert len(links) == 1
    assert links[0].proto.url == client.DEFAULT_API_URL.rstrip("/") + "/documents/view?document=Safety.pdf&page=3"
    assert "Plain.pdf" in md
