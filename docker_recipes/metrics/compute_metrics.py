#!/usr/bin/env python3
"""
Compute binary-classification metrics for EuCanImage submissions.

This replaces the old segmentation‐centric *compute_metrics.py*.

**Key points**
──────────────
1. **Inputs**  – a *predictions.csv* (columns: `image`, `predicted_probability`, `predicted_label`) and a *gt.csv* (`image`, `label`).
2. **Metrics** – Sensitivity, Specificity, Precision, NPV, Accuracy, F1, Balanced Accuracy, Cohen’s κ, Weighted κ (quadratic), MCC, ROC-AUC, PR-AUC.
3. **Curve data** – FPR and TPR lists (added directly to the JSON output).
4. **Outputs** – A JSON list of assessment objects matching the format produced by `JSON_templates.write_assessment_dataset`.
5. **Robust paths** – Works whether `-o` is an absolute or a simple filename.
"""
from __future__ import annotations

import io
import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Tuple, List
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_curve,
    precision_recall_curve,
    auc,
)

import JSON_templates  # provided by the evaluation environment

# -----------------------------------------------------------------------------
# CLI argument parsing
# -----------------------------------------------------------------------------
parser = ArgumentParser(
    description="Compute binary-classification metrics for EuCanImage challenges.")
parser.add_argument("-i", "--input", required=True,
                    help="Predictions CSV file.")
parser.add_argument("-g", "--goldstandard_file", required=True,
                    help="Ground-truth CSV file.")
parser.add_argument("-c", "--challenges_ids", nargs='+', required=True,
                    help="Challenge id(s), space-separated.")
parser.add_argument("-p", "--participant_id", required=True,
                    help="Tool / model id (participant).")
parser.add_argument("-com", "--community_id", required=True,
                    help="Benchmarking community id (e.g. 'EuCanImage').")
parser.add_argument("-e", "--event_id", required=True,
                    help="Benchmarking event id.")
parser.add_argument("-o", "--outdir", required=True,
                    help="Path to metrics JSON (other artefacts share the same basename).")

args = parser.parse_args()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def safe_div(num: float, denom: float) -> float:
    """Return *num/denom* or *np.nan* if the denominator is zero."""
    return float(num) / float(denom) if denom else np.nan


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def main(cfg):
    pred_path = Path(cfg.input)
    gt_path = Path(os.path.join(cfg.goldstandard_file, "gt.csv"))

    if not pred_path.is_file():
        sys.exit(f"ERROR: Predictions file '{pred_path}' does not exist.")
    if not gt_path.is_file():
        sys.exit(f"ERROR: Ground-truth file '{gt_path}' does not exist.")

    # 1. Load and align CSVs ------------------------------------------------
    pred_df = pd.read_csv(pred_path)
    gt_df   = pd.read_csv(gt_path)

    required_pred_cols = {"image", "predicted_probability", "predicted_label"}
    required_gt_cols   = {"image", "label"}

    if not required_pred_cols.issubset(pred_df.columns):
        sys.exit(f"ERROR: Predictions CSV missing columns: {required_pred_cols - set(pred_df.columns)}")
    if not required_gt_cols.issubset(gt_df.columns):
        sys.exit(f"ERROR: Ground-truth CSV missing columns: {required_gt_cols - set(gt_df.columns)}")

    pred_df["image"] = pred_df["image"].astype(str).str.strip()
    gt_df["image"]   = gt_df["image"].astype(str).str.strip()

    df = gt_df.merge(pred_df, on="image", how="inner", validate="one_to_one")
    if len(df) != len(gt_df) or len(df) != len(pred_df):
        missing_pred = set(gt_df["image"]) - set(pred_df["image"])
        extra_pred   = set(pred_df["image"]) - set(gt_df["image"])
        if missing_pred:
            print(f"ERROR: {len(missing_pred)} image id(s) present in GT but missing in predictions.")
        if extra_pred:
            print(f"ERROR: {len(extra_pred)} extra image id(s) present in predictions but not in GT.")
        sys.exit(1)

    y_true  = df["label"].astype(int).to_numpy()
    y_pred  = df["predicted_label"].astype(int).to_numpy()
    y_score = df["predicted_probability"].astype(float).to_numpy()

    # 2. Confusion-matrix metrics ------------------------------------------
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    precision   = safe_div(tp, tp + fp)
    npv         = safe_div(tn, tn + fn)

    # 3. Other sklearn metrics ---------------------------------------------
    accuracy       = accuracy_score(y_true, y_pred)
    f1             = f1_score(y_true, y_pred)
    bal_accuracy   = balanced_accuracy_score(y_true, y_pred)
    kappa          = safe_div(cohen_kappa_score(y_true, y_pred), 1)  # ensure float
    weighted_kappa = safe_div(cohen_kappa_score(y_true, y_pred, weights="quadratic"), 1)
    mcc            = matthews_corrcoef(y_true, y_pred)

    # 4. Curves & AUCs ------------------------------------------------------
    if len(np.unique(y_true)) == 2:
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall_curve, precision_curve)

        # Convert to required list formats ---------------------------------
        fpr_list = [float(x) for x in fpr]
        tpr_list = [{"v": float(v), "e": 0.0} for v in tpr]
    else:
        fpr_list = []
        tpr_list = []
        roc_auc = pr_auc = np.nan
        print("WARNING: Only one class present – ROC/PR curves not computed.")

    # 5. Collect metrics ----------------------------------------------------
    metrics: Dict[str, Tuple[float | list, float]] = {
        "sensitivity":          (sensitivity, 0.0),
        "specificity":          (specificity, 0.0),
        "precision":            (precision,   0.0),
        "npv":                  (npv,         0.0),
        "accuracy":             (accuracy,    0.0),
        "f1_score":             (f1,          0.0),
        "balanced_accuracy":    (bal_accuracy, 0.0),
        "cohen_kappa":          (kappa,        0.0),
        "weighted_cohen_kappa": (weighted_kappa, 0.0),
        "matthews_corrcoef":    (mcc,          0.0),
        "roc_auc":              (roc_auc,      0.0),
        "pr_auc":               (pr_auc,       0.0),
        # Added curve data -------------------------------------------------
#        "fpr_curve":            (fpr_list,     0.0),
#        "tpr_curve":            (tpr_list,     0.0),
    }

    # 6. Write assessment JSON ---------------------------------------------
    challenge = cfg.challenges_ids[0] if isinstance(cfg.challenges_ids, list) else cfg.challenges_ids
    base_id   = f"{cfg.community_id}:{cfg.event_id}_{challenge}_{cfg.participant_id}:"

    assessments: List[dict] = []
    for name, (val, std) in metrics.items():
        assessments.append(
            JSON_templates.write_assessment_dataset(
                base_id + name,
                cfg.community_id,
                challenge,
                cfg.participant_id,
                name,
                val,
                std,
            )
        )

    out_json_path = Path(cfg.outdir)
    # If a simple filename was given, place it in the current directory
    if not out_json_path.is_absolute():
        out_json_path = Path.cwd() / out_json_path
    if out_json_path.parent != Path('.'):
        out_json_path.parent.mkdir(parents=True, exist_ok=True)

    with io.open(out_json_path, mode="w", encoding="utf-8") as fp:
        json.dump(assessments, fp, indent=4, sort_keys=True, separators=(",", ": "))

    print(f"INFO: Wrote metrics JSON → {out_json_path}")

    # 7. All done -----------------------------------------------------------
    sys.exit(0)


if __name__ == "__main__":
    main(args)
