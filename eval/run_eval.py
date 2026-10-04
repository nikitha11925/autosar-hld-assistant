"""
Evaluation harness.

Runs the hand-written Q&A test set (eval/qa_test_set.json) against the
running backend's /query endpoint and reports:
  - Citation validity (does every citation in the answer point to the
    expected document/section, for grounded answers?)
  - Refusal correctness (does the system correctly refuse to answer
    questions with no grounding in the ingested documents, and correctly
    answer questions that are grounded?)

Results are printed as a summary table and are meant to be pasted into the
README's eval results section.

Usage:
    python eval/run_eval.py
    (requires the backend running at BACKEND_URL with the test document
    already ingested via POST /ingest)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
TEST_SET_PATH = Path(__file__).parent / "qa_test_set.json"
# Free-tier Gemini allows 5 requests/minute; pace eval calls comfortably under
# that rather than relying solely on the backend's own retry/backoff.
SECONDS_BETWEEN_QUERIES = 20


def load_test_set() -> list[dict]:
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run() -> None:
    cases = load_test_set()
    if not cases:
        print(f"No test cases found in {TEST_SET_PATH}. Nothing to evaluate.")
        return

    total = len(cases)
    refusal_correct = 0
    refusal_cases = 0
    grounded_correct = 0
    grounded_cases = 0
    section_hits = 0
    grounded_with_citations = 0

    for i, case in enumerate(cases):
        if i > 0:
            time.sleep(SECONDS_BETWEEN_QUERIES)

        doc_name = case["doc_name"]
        question = case["question"]
        expect_refusal = case.get("expect_refusal", False)
        expected_section_contains = case.get("expected_section_contains")

        response = requests.post(
            f"{BACKEND_URL}/query",
            json={"doc_name": doc_name, "question": question},
            timeout=120,
        )
        if response.status_code == 503:
            # Transient LLM-provider unavailability (e.g. free-tier rate limit) —
            # wait out a quota window and retry once rather than failing the run.
            print(f"[RETRY] '{question}' got 503, waiting 30s and retrying once...")
            time.sleep(30)
            response = requests.post(
                f"{BACKEND_URL}/query",
                json={"doc_name": doc_name, "question": question},
                timeout=120,
            )
        response.raise_for_status()
        data = response.json()

        if expect_refusal:
            refusal_cases += 1
            if not data["grounded"]:
                refusal_correct += 1
            else:
                print(f"[REFUSAL MISS] '{question}' should have refused but answered.")
        else:
            grounded_cases += 1
            if data["grounded"]:
                grounded_correct += 1
            else:
                print(f"[GROUNDING MISS] '{question}' should have answered but refused.")
                continue

            if data["citations"]:
                grounded_with_citations += 1
                if expected_section_contains:
                    hit = any(
                        expected_section_contains.lower() in (c["section"] or "").lower()
                        for c in data["citations"]
                    )
                    if hit:
                        section_hits += 1
                    else:
                        print(
                            f"[SECTION MISS] '{question}' expected a citation section "
                            f"containing '{expected_section_contains}', got: "
                            f"{[c['section'] for c in data['citations']]}"
                        )

    print("\n--- Eval summary ---")
    print(f"Total cases: {total}")
    if refusal_cases:
        print(
            f"Refusal correctness: {refusal_correct}/{refusal_cases} "
            f"({100 * refusal_correct / refusal_cases:.0f}%)"
        )
    if grounded_cases:
        print(
            f"Groundedness (answered when expected): {grounded_correct}/{grounded_cases} "
            f"({100 * grounded_correct / grounded_cases:.0f}%)"
        )
        print(
            f"Citation presence on grounded answers: {grounded_with_citations}/{grounded_correct} "
            f"({100 * grounded_with_citations / grounded_correct:.0f}%)" if grounded_correct else "n/a"
        )
        expected_section_cases = sum(1 for c in cases if c.get("expected_section_contains"))
        if expected_section_cases:
            print(
                f"Retrieval section-match precision: {section_hits}/{expected_section_cases} "
                f"({100 * section_hits / expected_section_cases:.0f}%)"
            )


if __name__ == "__main__":
    run()
