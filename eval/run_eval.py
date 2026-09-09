"""
Evaluation harness.

Runs the hand-written Q&A test set (eval/qa_test_set.json) against the
running backend's /query endpoint and reports:
  - Retrieval precision@k (did the expected section/page get retrieved?)
  - Citation validity (does every citation in the answer point to a real,
    retrieved chunk?)
  - Refusal correctness (does the system correctly refuse to answer
    questions with no grounding in the ingested documents, and correctly
    answer questions that are grounded?)

Results are printed as a summary table and are meant to be pasted into the
README's eval results section.
"""
