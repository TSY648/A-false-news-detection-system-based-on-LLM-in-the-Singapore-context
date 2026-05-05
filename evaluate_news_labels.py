"""
News-level evaluation script.

Inputs:
1. Prediction CSV (default: batch_pipeline_results.csv)
   Required columns:
   - raw_text
   - news_label

2. Manually labeled CSV (default: news_gold_labels.csv)
   Required columns:
   - raw_text
   - news_gold_label

Evaluation granularity:
- News level only, with one final label per news item

Label set:
- Supported
- Refuted
- Not Enough Evidence
"""

import argparse
import csv
import os
from typing import Dict, List, Tuple


LABELS = ["Supported", "Refuted", "Not Enough Evidence"]

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PRED_CSV = os.path.join(_BASE_DIR, "batch_pipeline_results.csv")
DEFAULT_GOLD_CSV = os.path.join(_BASE_DIR, "news_gold_labels.csv")
DEFAULT_MERGED_CSV = os.path.join(_BASE_DIR, "news_eval_merged.csv")


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def load_prediction_news_labels(csv_path: str) -> Dict[str, str]:
    """
    Aggregate claim-level outputs into news-level predictions.

    Rules:
    - Use raw_text as the unique key for each news item
    - For the same raw_text, take the first non-empty news_label
    """
    pred_map: Dict[str, str] = {}

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        required = {"raw_text", "news_label"}
        missing = required - set(fieldnames)
        if missing:
            raise ValueError(
                f"Prediction CSV missing required columns: {sorted(missing)}"
            )

        for row in reader:
            raw_text = (row.get("raw_text") or "").strip()
            news_label = (row.get("news_label") or "").strip()
            if not raw_text:
                continue
            if raw_text not in pred_map and news_label:
                pred_map[raw_text] = news_label

    return pred_map


def load_gold_news_labels(csv_path: str) -> Dict[str, str]:
    gold_map: Dict[str, str] = {}

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        required = {"raw_text", "news_gold_label"}
        missing = required - set(fieldnames)
        if missing:
            raise ValueError(f"Gold CSV missing required columns: {sorted(missing)}")

        for row in reader:
            raw_text = (row.get("raw_text") or "").strip()
            gold_label = (row.get("news_gold_label") or "").strip()
            if not raw_text:
                continue
            if gold_label not in LABELS:
                raise ValueError(
                    f"Invalid news_gold_label '{gold_label}' for raw_text: {raw_text[:80]}"
                )
            gold_map[raw_text] = gold_label

    return gold_map


def build_eval_rows(
    pred_map: Dict[str, str], gold_map: Dict[str, str]
) -> Tuple[List[Dict[str, str]], int, int]:
    rows: List[Dict[str, str]] = []
    missing_pred_count = 0
    extra_pred_count = 0

    for raw_text, gold_label in gold_map.items():
        pred_label = pred_map.get(raw_text, "")
        if not pred_label:
            missing_pred_count += 1
        rows.append(
            {
                "raw_text": raw_text,
                "news_gold_label": gold_label,
                "news_label": pred_label,
                "matched": "1" if pred_label == gold_label and pred_label else "0",
            }
        )

    for raw_text in pred_map:
        if raw_text not in gold_map:
            extra_pred_count += 1

    return rows, missing_pred_count, extra_pred_count


def compute_metrics(eval_rows: List[Dict[str, str]]) -> Dict[str, object]:
    confusion: Dict[str, Dict[str, int]] = {
        gold: {pred: 0 for pred in LABELS} for gold in LABELS
    }

    total = 0
    correct = 0

    for row in eval_rows:
        gold = row["news_gold_label"]
        pred = row["news_label"]
        if pred not in LABELS:
            continue
        confusion[gold][pred] += 1
        total += 1
        if gold == pred:
            correct += 1

    per_class: Dict[str, Dict[str, float]] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[gold][label] for gold in LABELS if gold != label)
        fn = sum(confusion[label][pred] for pred in LABELS if pred != label)

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)

        per_class[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "total_evaluated": total,
        "correct": correct,
        "accuracy": _safe_div(correct, total),
        "macro_precision": sum(precisions) / len(LABELS),
        "macro_recall": sum(recalls) / len(LABELS),
        "macro_f1": sum(f1s) / len(LABELS),
        "per_class": per_class,
        "confusion": confusion,
    }


def write_merged_eval_csv(csv_path: str, rows: List[Dict[str, str]]) -> None:
    fieldnames = ["raw_text", "news_gold_label", "news_label", "matched"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_report(
    pred_csv: str,
    gold_csv: str,
    merged_csv: str,
    row_count: int,
    missing_pred_count: int,
    extra_pred_count: int,
    metrics: Dict[str, object],
) -> None:
    print("=" * 72)
    print("News-Level Evaluation")
    print("=" * 72)
    print(f"Prediction CSV     : {pred_csv}")
    print(f"Gold CSV           : {gold_csv}")
    print(f"Merged Eval CSV    : {merged_csv}")
    print(f"Gold news count    : {row_count}")
    print(f"Missing predictions: {missing_pred_count}")
    print(f"Extra predictions  : {extra_pred_count}")
    print()

    print("Overall Metrics")
    print(f"Accuracy        : {_format_metric(metrics['accuracy'])}")
    print(f"Macro-Precision : {_format_metric(metrics['macro_precision'])}")
    print(f"Macro-Recall    : {_format_metric(metrics['macro_recall'])}")
    print(f"Macro-F1        : {_format_metric(metrics['macro_f1'])}")
    print()

    print("Per-Class Metrics")
    per_class = metrics["per_class"]
    for label in LABELS:
        stats = per_class[label]
        print(f"[{label}]")
        print(f"  TP        : {stats['tp']}")
        print(f"  FP        : {stats['fp']}")
        print(f"  FN        : {stats['fn']}")
        print(f"  Precision : {_format_metric(stats['precision'])}")
        print(f"  Recall    : {_format_metric(stats['recall'])}")
        print(f"  F1        : {_format_metric(stats['f1'])}")
        print()

    print("Confusion Matrix (gold -> predicted)")
    confusion = metrics["confusion"]
    header = "gold\\pred".ljust(24) + "".join(label.ljust(24) for label in LABELS)
    print(header)
    for gold in LABELS:
        line = gold.ljust(24)
        for pred in LABELS:
            line += str(confusion[gold][pred]).ljust(24)
        print(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate news-level labels.")
    parser.add_argument(
        "--pred-csv",
        default=DEFAULT_PRED_CSV,
        help="Path to batch pipeline prediction CSV.",
    )
    parser.add_argument(
        "--gold-csv",
        default=DEFAULT_GOLD_CSV,
        help="Path to manually labeled gold CSV.",
    )
    parser.add_argument(
        "--merged-csv",
        default=DEFAULT_MERGED_CSV,
        help="Path to write merged news-level evaluation CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pred_map = load_prediction_news_labels(args.pred_csv)
    gold_map = load_gold_news_labels(args.gold_csv)
    eval_rows, missing_pred_count, extra_pred_count = build_eval_rows(pred_map, gold_map)
    metrics = compute_metrics(eval_rows)
    write_merged_eval_csv(args.merged_csv, eval_rows)
    print_report(
        pred_csv=args.pred_csv,
        gold_csv=args.gold_csv,
        merged_csv=args.merged_csv,
        row_count=len(eval_rows),
        missing_pred_count=missing_pred_count,
        extra_pred_count=extra_pred_count,
        metrics=metrics,
    )


if __name__ == "__main__":
    main()
