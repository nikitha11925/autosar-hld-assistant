"""
Streamlit frontend.

Two tabs for this build (structured extraction / consistency checks are
deferred — see README known limitations, so the "Explore" tab isn't wired
up yet):
  - Upload/Ingest: upload a PDF, trigger POST /ingest, show ingestion status
  - Ask: chat interface against POST /query, rendering inline citations and
    a grounded/not-grounded badge on every answer

This file is UI-only — it calls the FastAPI backend over HTTP and contains
no business logic of its own.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="AUTOSAR HLD Assistant", layout="wide")
st.title("AUTOSAR HLD Document Analysis Assistant")

if "ingested_docs" not in st.session_state:
    st.session_state.ingested_docs = []

tab_ingest, tab_ask = st.tabs(["Upload / Ingest", "Ask"])

with tab_ingest:
    st.subheader("Upload a PDF")
    uploaded = st.file_uploader("AUTOSAR HLD / SWS-style PDF", type=["pdf"])
    if uploaded is not None and st.button("Ingest document"):
        with st.spinner(f"Parsing, chunking, and embedding '{uploaded.name}'..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/ingest",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                    timeout=300,
                )
            except requests.RequestException as exc:
                st.error(f"Could not reach backend at {BACKEND_URL}: {exc}")
            else:
                if response.ok:
                    data = response.json()
                    st.success(
                        f"Ingested '{data['doc_name']}' — {data['page_count']} pages, "
                        f"{data['chunk_count']} chunks."
                    )
                    if data["warnings"]:
                        for w in data["warnings"]:
                            st.warning(w)
                    if data["doc_name"] not in st.session_state.ingested_docs:
                        st.session_state.ingested_docs.append(data["doc_name"])
                else:
                    st.error(f"Ingestion failed: {response.text}")

    if st.session_state.ingested_docs:
        st.markdown("**Ingested documents this session:**")
        for name in st.session_state.ingested_docs:
            st.markdown(f"- {name}")

with tab_ask:
    st.subheader("Ask a question")
    if not st.session_state.ingested_docs:
        st.info("Ingest a document in the 'Upload / Ingest' tab first.")
    else:
        doc_name = st.selectbox("Document", st.session_state.ingested_docs)
        question = st.text_input("Question")
        if st.button("Ask") and question:
            with st.spinner("Retrieving and generating answer..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/query",
                        json={"doc_name": doc_name, "question": question},
                        timeout=120,
                    )
                except requests.RequestException as exc:
                    st.error(f"Could not reach backend at {BACKEND_URL}: {exc}")
                else:
                    if response.ok:
                        data = response.json()
                        if data["grounded"]:
                            st.success("Grounded answer")
                        else:
                            st.warning("Not grounded — refused")
                        st.markdown(data["answer"])
                        if data["citations"]:
                            st.markdown("**Citations:**")
                            for c in data["citations"]:
                                st.markdown(
                                    f"- {c['doc_name']}, section: {c['section'] or '(none)'}, "
                                    f"page: {c['page_range']}"
                                )
                    else:
                        st.error(f"Query failed: {response.text}")
