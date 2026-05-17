"""
Evaluation Script - CSC 262 Lab Final
Runs the full gateway on final_eval.csv and produces:
  - results/evaluation_results.csv
  - results/metrics_summary.json
  - Console tables: Rule-only vs Hybrid, Per-language, PII validation, Latency
"""

import os
import sys
import json
import time
import csv
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from app.detectors import rule_detector, semantic_detector
from app.pii import presidio_custom
from app.policy import policy_engine
from app.utils.language import detect_language, is_mixed_language

# ─────────────────────────────────────────────
DATA_PATH = "data/final_eval.csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Map expected_policy to binary: BLOCK=1, others=0 (for injection detection metrics)
ATTACK_LABEL = {"BLOCK": 1, "MASK": 0, "ALLOW": 0}
RISK_LABEL = {"BLOCK": 1, "MASK": 1, "ALLOW": 0}  # for overall risk detection


def run_rule_only(text: str) -> dict:
    """Rule-based detector only pipeline."""
    rule_result = rule_detector.detect(text)
    rule_score = rule_result["rule_score"]
    # Simplified policy: block if rule score >= 0.5
    if rule_score >= 0.5:
        decision = "BLOCK"
    elif rule_score >= 0.2:
        decision = "MASK"
    else:
        decision = "ALLOW"
    return {"rule_score": rule_score, "decision": decision, "latency_ms": rule_result["latency_ms"]}


def run_hybrid(row: dict) -> dict:
    """Full hybrid pipeline."""
    text = row["prompt"]
    start = time.time()

    rule_result = rule_detector.detect(text)
    sem_result = semantic_detector.detect(text)
    pii_result = presidio_custom.analyze(text)
    lang = detect_language(text)
    mixed = is_mixed_language(text)
    all_codes = list(set(rule_result["reason_codes"] + sem_result["reason_codes"]))

    policy_result = policy_engine.decide(
        rule_score=rule_result["rule_score"],
        semantic_score=sem_result["semantic_score"],
        pii_entities=pii_result["pii_entities"],
        pii_score=pii_result["pii_score"],
        reason_codes=all_codes,
        language=lang,
        is_mixed=mixed,
    )

    latency = round((time.time() - start) * 1000, 2)
    return {
        "rule_score": rule_result["rule_score"],
        "semantic_score": sem_result["semantic_score"],
        "pii_score": pii_result["pii_score"],
        "pii_entities": [e["type"] for e in pii_result["pii_entities"]],
        "final_risk": policy_result["final_risk"],
        "decision": policy_result["decision"],
        "reason_codes": policy_result["reason_codes"],
        "detected_language": lang,
        "is_mixed": mixed,
        "latency_ms": latency,
    }


def evaluate():
    print("=" * 70)
    print("  CSC 262 Lab Final - LLM Security Gateway Evaluation")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    print(f"\nLoaded {len(df)} prompts from {DATA_PATH}\n")

    # Run both pipelines
    rule_results = []
    hybrid_results = []

    print("Running evaluation... (this may take a minute)")
    for i, row in df.iterrows():
        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{len(df)}")
        rule_r = run_rule_only(row["prompt"])
        hybrid_r = run_hybrid(row)
        rule_results.append(rule_r)
        hybrid_results.append(hybrid_r)

    df["rule_decision"] = [r["decision"] for r in rule_results]
    df["rule_score"] = [r["rule_score"] for r in rule_results]
    df["rule_latency_ms"] = [r["latency_ms"] for r in rule_results]

    df["hybrid_decision"] = [r["decision"] for r in hybrid_results]
    df["hybrid_final_risk"] = [r["final_risk"] for r in hybrid_results]
    df["hybrid_latency_ms"] = [r["latency_ms"] for r in hybrid_results]
    df["detected_language"] = [r["detected_language"] for r in hybrid_results]
    df["hybrid_rule_score"] = [r["rule_score"] for r in hybrid_results]
    df["hybrid_semantic_score"] = [r["semantic_score"] for r in hybrid_results]
    df["hybrid_pii_score"] = [r["pii_score"] for r in hybrid_results]
    df["hybrid_reason_codes"] = [",".join(r["reason_codes"]) for r in hybrid_results]

    # ─────────────────────────────────────────────
    # TABLE 1: Rule-only vs Hybrid Metrics
    # ─────────────────────────────────────────────
    y_true = df["expected_policy"].apply(lambda x: ATTACK_LABEL.get(x, 0)).values

    rule_pred = df["rule_decision"].apply(lambda x: ATTACK_LABEL.get(x, 0)).values
    hybrid_pred = df["hybrid_decision"].apply(lambda x: ATTACK_LABEL.get(x, 0)).values

    print("\n" + "=" * 70)
    print("TABLE 1: Rule-Only vs Hybrid Detection Metrics (BLOCK vs non-BLOCK)")
    print("=" * 70)
    print(f"{'Metric':<25} {'Rule-Only':>12} {'Hybrid':>12}")
    print("-" * 50)

    metrics = {}
    for name, pred in [("Rule-Only", rule_pred), ("Hybrid", hybrid_pred)]:
        acc = accuracy_score(y_true, pred)
        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        cm = confusion_matrix(y_true, pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (cm[0, 0], 0, 0, cm[0, 0])
        metrics[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                         "fp": int(fp), "fn": int(fn), "tp": int(tp), "tn": int(tn)}

    for metric_name in ["accuracy", "precision", "recall", "f1"]:
        print(f"  {metric_name.capitalize():<23} {metrics['Rule-Only'][metric_name]:>12.4f} {metrics['Hybrid'][metric_name]:>12.4f}")
    print(f"  {'False Positives':<23} {metrics['Rule-Only']['fp']:>12} {metrics['Hybrid']['fp']:>12}")
    print(f"  {'False Negatives':<23} {metrics['Rule-Only']['fn']:>12} {metrics['Hybrid']['fn']:>12}")
    print(f"  {'True Positives':<23} {metrics['Rule-Only']['tp']:>12} {metrics['Hybrid']['tp']:>12}")

    # ─────────────────────────────────────────────
    # TABLE 2: Per-Language Robustness
    # ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TABLE 2: Per-Language Robustness (Recall on attack prompts)")
    print("=" * 70)
    print(f"{'Language':<12} {'Total':>8} {'Attacks':>8} {'Detected':>10} {'Recall':>8}")
    print("-" * 50)

    lang_metrics = {}
    for lang in ["en", "ur", "ko", "mixed"]:
        lang_df = df[df["language"] == lang]
        attack_df = lang_df[lang_df["expected_policy"] == "BLOCK"]
        if len(attack_df) == 0:
            continue
        detected = (attack_df["hybrid_decision"] == "BLOCK").sum()
        recall = detected / len(attack_df)
        lang_metrics[lang] = {"total": len(lang_df), "attacks": len(attack_df),
                               "detected": int(detected), "recall": recall}
        print(f"  {lang:<12} {len(lang_df):>8} {len(attack_df):>8} {int(detected):>10} {recall:>8.4f}")

    # ─────────────────────────────────────────────
    # TABLE 3: PII Validation (sample)
    # ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TABLE 3: Presidio PII Validation (sample)")
    print("=" * 70)
    pii_df = df[df["has_pii"] == True].head(10)
    print(f"{'ID':<6} {'Expected Entity':<20} {'PII Score':>10} {'Decision':>10}")
    print("-" * 50)
    for _, row in pii_df.iterrows():
        print(f"  {row['id']:<6} {str(row['expected_entities']):<20} {row.get('hybrid_pii_score', 0.0):>10.3f} {row['hybrid_decision']:>10}")

    # ─────────────────────────────────────────────
    # TABLE 4: Latency Summary
    # ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TABLE 4: Latency Summary (ms)")
    print("=" * 70)
    print(f"{'Mode':<15} {'Mean':>8} {'Median':>8} {'P95':>8} {'Min':>8} {'Max':>8}")
    print("-" * 55)
    for mode, col in [("Rule-Only", "rule_latency_ms"), ("Hybrid", "hybrid_latency_ms")]:
        vals = df[col].values
        print(f"  {mode:<15} {np.mean(vals):>8.2f} {np.median(vals):>8.2f} "
              f"{np.percentile(vals, 95):>8.2f} {np.min(vals):>8.2f} {np.max(vals):>8.2f}")

    # ─────────────────────────────────────────────
    # TABLE 5: Policy Scenario Summary
    # ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TABLE 5: Decision Distribution")
    print("=" * 70)
    hybrid_counts = df["hybrid_decision"].value_counts()
    expected_counts = df["expected_policy"].value_counts()
    print(f"{'Decision':<12} {'Expected':>10} {'Hybrid Got':>12}")
    print("-" * 36)
    for dec in ["ALLOW", "MASK", "BLOCK"]:
        print(f"  {dec:<12} {expected_counts.get(dec, 0):>10} {hybrid_counts.get(dec, 0):>12}")

    # ─────────────────────────────────────────────
    # TABLE 6: Error Analysis
    # ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TABLE 6: Error Analysis (Incorrect Hybrid Decisions)")
    print("=" * 70)
    errors = df[df["hybrid_decision"] != df["expected_policy"]]
    print(f"Total errors: {len(errors)} / {len(df)}")
    print(f"\n{'ID':<6} {'Expected':>10} {'Got':>10} {'Attack Type':<25} {'Snippet'}")
    print("-" * 80)
    for _, row in errors.head(15).iterrows():
        snippet = str(row["prompt"])[:40].replace("\n", " ")
        print(f"  {row['id']:<6} {row['expected_policy']:>10} {row['hybrid_decision']:>10} "
              f"{str(row['attack_type']):<25} {snippet}...")

    # ─────────────────────────────────────────────
    # Save results
    # ─────────────────────────────────────────────
    df.to_csv(f"{RESULTS_DIR}/evaluation_results.csv", index=False)
    print(f"\nResults saved to {RESULTS_DIR}/evaluation_results.csv")

    summary = {
        "total_prompts": len(df),
        "rule_only_metrics": metrics.get("Rule-Only", {}),
        "hybrid_metrics": metrics.get("Hybrid", {}),
        "per_language_metrics": lang_metrics,
        "decision_distribution": hybrid_counts.to_dict(),
        "expected_distribution": expected_counts.to_dict(),
        "error_count": len(errors),
        "latency": {
            "rule_only": {
                "mean": float(np.mean(df["rule_latency_ms"])),
                "median": float(np.median(df["rule_latency_ms"])),
                "p95": float(np.percentile(df["rule_latency_ms"], 95)),
            },
            "hybrid": {
                "mean": float(np.mean(df["hybrid_latency_ms"])),
                "median": float(np.median(df["hybrid_latency_ms"])),
                "p95": float(np.percentile(df["hybrid_latency_ms"], 95)),
            },
        },
    }

    with open(f"{RESULTS_DIR}/metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Metrics summary saved to {RESULTS_DIR}/metrics_summary.json")
    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    evaluate()
