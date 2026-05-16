"""Kanban-lite Streamlit demo for the triage agent.

Run locally:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import os
import uuid

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Support Triage", page_icon="🎟", layout="wide")

EXAMPLES = [
    "I want a refund for my order #12345, the product arrived broken.",
    "Where is my package, it's been 10 days",
    "Production is DOWN because of your service, this is critical",
    "Please connect me with a human agent",
    "How do I change my shipping address",
    "GDPR — please delete all my data",
]

st.markdown("<h1 style='margin-bottom:0'>🎟 Support Triage</h1>", unsafe_allow_html=True)
st.caption("Intent classification + sentiment/priority + similar-ticket retrieval + auto-resolve / suggest / escalate.")

with st.sidebar:
    st.subheader("Backend")
    try:
        h = httpx.get(f"{API_URL}/health", timeout=2).json()
        st.success(f"online  ·  v{h.get('version')}")
        st.caption(f"classifier: {h.get('classifier_model')}")
        st.caption(f"LLM: {h.get('llm_enabled')}")
    except Exception:
        st.error(f"backend offline at {API_URL}")

    st.subheader("Try these")
    for i, ex in enumerate(EXAMPLES, 1):
        if st.button(f"#{i}", help=ex, use_container_width=True):
            st.session_state.pending = ex

cols = st.columns([2, 3])
with cols[0]:
    text = st.session_state.pop("pending", "") if "pending" in st.session_state else ""
    body = st.text_area("Ticket body", value=text, height=220)
    if st.button("Run triage", type="primary"):
        with st.spinner("Triaging..."):
            r = httpx.post(f"{API_URL}/api/triage", timeout=30,
                            json={"ticket_id": f"st-{uuid.uuid4().hex[:6]}",
                                  "channel": "chat", "body": body}).json()
        st.session_state.result = r

with cols[1]:
    r = st.session_state.get("result")
    if not r:
        st.info("Submit a ticket on the left to see the triage breakdown.")
    else:
        d = r["decision"]
        color = {"auto_resolve": "#34d399", "suggest": "#22d3ee", "escalate": "#f472b6"}.get(d["decision"], "#ccc")
        st.markdown(
            f"<div style='padding:14px; border-radius:10px; background:{color}22; border-left:4px solid {color};'>"
            f"<b>Decision:</b> {d['decision'].upper()} · confidence {d['confidence']:.2f}<br>"
            f"<small>{d['rationale']}</small></div>", unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Intent", r["intent"], f"{r['intent_confidence']:.2f}")
        c2.metric("Priority", r["priority"], r["sentiment"])
        c3.metric("Urgency", f"{r['urgency_score']:.2f}")

        if r.get("top_intents"):
            st.markdown("**Top intents:**")
            for s in r["top_intents"][:3]:
                st.write(f"- `{s['intent']}` ({s['score']:.2f})")

        if r.get("draft_response"):
            st.markdown("**Draft response:**")
            st.info(r["draft_response"])

        if r.get("similar_resolved"):
            with st.expander(f"📚 {len(r['similar_resolved'])} similar resolved tickets"):
                for s in r["similar_resolved"]:
                    st.write(f"**[{s['intent']}]** ({s['similarity']:.2f}) — {s['body_snippet']}")
                    st.caption(f"Resolution: {s['resolution_snippet']}")

        st.caption(f"latency {r['latency_ms']} ms")
