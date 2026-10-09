"""Streamlit chat UI. Run: streamlit run frontend/app.py"""
import os
import sys
import uuid

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import DEFAULT_API_URL, BackendError, send_message  # noqa: E402

LABELS = {
    "safety_procedures": "Safety procedures",
    "maintenance_manuals": "Maintenance manuals",
    "quality_control_standards": "Quality control standards",
}


def _label(cat):
    return LABELS.get(cat, cat or "n/a")


def _abs_url(rel):
    return DEFAULT_API_URL.rstrip("/") + "/" + rel.lstrip("/")


def render_citations(citations, key_prefix="c"):
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})"):
        for i, c in enumerate(citations):
            where = c["section"] + (f" › {c['sub_section']}" if c.get("sub_section") else "")
            text = f"{c['document_name']}, page {c['page_number']} · {where}"
            if c.get("document_url"):
                # link_button opens the target in a new browser tab
                st.link_button(text, _abs_url(c["document_url"]), icon="📄",
                               key=f"{key_prefix}_{i}")
            else:
                st.markdown(f"- **{c['document_name']}**, page {c['page_number']}  \n  {where}")


def render_message(m, key_prefix="c"):
    status = m.get("status", "answered")
    if status == "out_of_scope":
        st.info("Out of scope: " + m["content"], icon="🧭")
    elif status == "blocked":
        st.error("Blocked by guardrails: " + m["content"], icon="🚫")
    elif status == "error":
        st.warning(m["content"], icon="⚠️")
    else:
        if m.get("categories"):
            st.markdown(" ".join(f":blue-badge[{_label(c)}]" for c in m["categories"]))
        st.markdown(m["content"])
    render_citations(m.get("citations", []), key_prefix=key_prefix)


def new_conversation():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []


st.set_page_config(page_title="Manufacturing Assistant", page_icon="🏭")
if "thread_id" not in st.session_state:
    new_conversation()

st.title("🏭 Manufacturing Assistant")
st.caption("Ask about safety procedures, maintenance manuals, or quality control standards.")
with st.sidebar:
    if st.button("New conversation"):
        new_conversation()
        st.rerun()
    st.caption(f"Thread: {st.session_state.thread_id}")

for idx, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        render_message(m, key_prefix=f"h{idx}")

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documentation..."):
                r = send_message(prompt, st.session_state.thread_id)
            msg = {"role": "assistant", "content": r.answer, "status": r.status,
                   "categories": r.categories,
                   "citations": [c.__dict__ for c in r.citations]}
        except BackendError as e:
            msg = {"role": "assistant", "content": str(e), "status": "error"}
        st.session_state.messages.append(msg)
        render_message(msg, key_prefix=f"n{len(st.session_state.messages)}")
