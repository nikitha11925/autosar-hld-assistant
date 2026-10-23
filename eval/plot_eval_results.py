"""
Builds evaluation graphs from the most recent eval/run_eval.py run
(eval/results/eval_run.json).

What "confusion matrix" means for this system: there is no multi-class
text classification task here. The one real binary decision this system
makes on every query is answer vs. refuse (the grounding guardrail,
CLAUDE.md principle 1). Each of the 8 hand-written test questions has a
ground-truth label (should this be answered, or is it genuinely outside
the document?), and the system's grounded:true/false flag is its
prediction — so a standard TP/FP/TN/FN confusion matrix legitimately
applies to that decision, same as it would to any other binary classifier.
"Positive" here means "the system decided to answer."

Usage:
    python eval/run_eval.py          # produces eval/results/eval_run.json
    python eval/plot_eval_results.py # reads it, writes graphs
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
RUN_PATH = RESULTS_DIR / "eval_run.json"


def main() -> None:
    if not RUN_PATH.exists():
        print(f"{RUN_PATH} not found. Run `python eval/run_eval.py` first.")
        return

    with open(RUN_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    # Ground truth: should this question be answered (positive) or refused (negative)?
    # Prediction: did the system actually answer (grounded=True) or refuse?
    tp = sum(1 for c in cases if not c["expect_refusal"] and c["grounded"])
    fn = sum(1 for c in cases if not c["expect_refusal"] and not c["grounded"])
    fp = sum(1 for c in cases if c["expect_refusal"] and c["grounded"])
    tn = sum(1 for c in cases if c["expect_refusal"] and not c["grounded"])

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else float("nan")
    accuracy = (tp + tn) / len(cases) if cases else float("nan")

    print("=== Answer-vs-refuse confusion matrix (grounding guardrail) ===")
    print(f"TP (correctly answered): {tp}")
    print(f"FN (wrongly refused):    {fn}")
    print(f"FP (wrongly answered — hallucination risk): {fp}")
    print(f"TN (correctly refused):  {tn}")
    print()
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1:        {f1:.2f}")
    print(f"Accuracy:  {accuracy:.2f}")

    # --- Confusion matrix heatmap ---
    matrix = np.array([[tp, fn], [fp, tn]])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(matrix, cmap="Blues")
    labels = [["TP", "FN"], ["FP", "TN"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{matrix[i, j]}", ha="center", va="center",
                     fontsize=13, color="black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted:\nAnswered", "Predicted:\nRefused"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual:\nShould answer", "Actual:\nShould refuse"])
    ax.set_title("Grounding guardrail confusion matrix\n(answer vs. refuse decision)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # --- Precision/Recall/F1/Accuracy bar chart ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    metrics = ["Precision", "Recall", "F1", "Accuracy"]
    values = [precision, recall, f1, accuracy]
    bars = ax.bar(metrics, values, color="#2563eb")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Grounding guardrail metrics\n(answer-vs-refuse binary task, n={})".format(len(cases)))
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "grounding_metrics.png", dpi=150)
    plt.close(fig)

    # --- Citation coverage on grounded answers ---
    grounded_cases = [c for c in cases if c["grounded"]]
    with_citations = sum(1 for c in grounded_cases if c["citation_count"] > 0)
    without_citations = len(grounded_cases) - with_citations
    if grounded_cases:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.bar(["With >=1\ncitation", "No citation\n(should not happen)"],
               [with_citations, without_citations], color=["#16a34a", "#dc2626"])
        for i, v in enumerate([with_citations, without_citations]):
            ax.text(i, v + 0.05, str(v), ha="center")
        ax.set_ylabel("Number of grounded answers")
        ax.set_title("Citation coverage on grounded answers\n(1 citation minimum per claim)", fontsize=11)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "citation_coverage.png", dpi=150)
        plt.close(fig)

    print(f"\nSaved plots to {RESULTS_DIR}/")
    print("  - confusion_matrix.png")
    print("  - grounding_metrics.png")
    print("  - citation_coverage.png")


if __name__ == "__main__":
    main()
