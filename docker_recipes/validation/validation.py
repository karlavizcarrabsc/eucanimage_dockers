#!/usr/bin/env python3
"""
"""
from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import numpy as np
import os

import JSON_templates

# -----------------------------------------------------------------------------
# CLI arguments
# -----------------------------------------------------------------------------
parser = ArgumentParser()
parser.add_argument("-i", "--input", required=True,
                    help="CSV file containing participant predictions.")
parser.add_argument("-com", "--community_id", required=True,
                    help="OEB community id or label, e.g. 'EuCanImage'.")
parser.add_argument("-c", "--challenges_ids", nargs='+', required=True,
                    help="Challenge name(s) chosen by the user.")
parser.add_argument("-p", "--participant_id", required=True,
                    help="Tool / model id (participant).")
parser.add_argument("-e", "--event_id", required=True,
                    help="Benchmarking event id or name.")
parser.add_argument("-g", "--goldstandard_file", required=True,
                    help="Ground‑truth CSV containing 'image' and 'label' columns.")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def error(msg: str) -> None:
    """Always exit with status 1 after printing *msg* prefixed by 'ERROR:'"""
    sys.exit(f"ERROR: {msg}")


# -----------------------------------------------------------------------------
# Main validation routine
# -----------------------------------------------------------------------------

def main(cfg):
    # ---------------------------------------------------------------------
    # 1. Load participant predictions
    # ---------------------------------------------------------------------
    pred_path = Path(cfg.input)
    if not pred_path.is_file():
        error(f"Predictions file '{pred_path}' does not exist or is not a file.")

    try:
        pred_df = pd.read_csv(pred_path)
    except Exception as exc:
        error(f"Cannot read predictions CSV: {exc}")

    expected_pred_cols = ["image", "predicted_probability", "predicted_label"]
    missing_cols = [c for c in expected_pred_cols if c not in pred_df.columns]
    extra_cols   = [c for c in pred_df.columns if c not in expected_pred_cols]
    if missing_cols:
        error(f"Missing required column(s) in predictions CSV: {missing_cols}.")
    if extra_cols:
        print(f"WARNING: Ignoring unexpected column(s) in predictions CSV: {extra_cols}")
        pred_df = pred_df[expected_pred_cols]  # drop extras while keeping order

    # Remove any accidental whitespace in image ids
    pred_df["image"] = pred_df["image"].astype(str).str.strip()

    # ---------------------------------------------------------------------
    # 2. Basic checks on predictions
    # ---------------------------------------------------------------------
    if pred_df["image"].duplicated().any():
        dupes = pred_df.loc[pred_df["image"].duplicated(), "image"].unique()
        error(f"Duplicate image id(s) in predictions CSV: {', '.join(dupes)}")

    # Ensure probability is numeric and within [0,1]
    try:
        pred_df["predicted_probability"] = pd.to_numeric(
            pred_df["predicted_probability"], errors="raise")
    except Exception as exc:
        error(f"'predicted_probability' column must be numeric: {exc}")
    if not ((pred_df["predicted_probability"] >= 0) & (pred_df["predicted_probability"] <= 1)).all():
        bad_rows = pred_df.loc[
            ~((pred_df["predicted_probability"] >= 0) & (pred_df["predicted_probability"] <= 1)),
            ["image", "predicted_probability"]]
        error(f"Probability values outside [0,1]:\n{bad_rows.to_string(index=False)}")

    # Ensure label is integer 0/1
    try:
        pred_df["predicted_label"] = pd.to_numeric(pred_df["predicted_label"], downcast="integer", errors="raise")
    except Exception as exc:
        error(f"'predicted_label' column must contain integers 0 or 1: {exc}")
    if not pred_df["predicted_label"].isin([0, 1]).all():
        bad_rows = pred_df.loc[~pred_df["predicted_label"].isin([0, 1]), ["image", "predicted_label"]]
        error(f"Invalid label values (must be 0 or 1):\n{bad_rows.to_string(index=False)}")

    # Optional consistency check – comment out if not desired
    inconsistent = pred_df.query(
        "(predicted_probability >= 0.5 and predicted_label == 0) or (predicted_probability < 0.5 and predicted_label == 1)"
    )
    if not inconsistent.empty:
        print(
            f"WARNING: {len(inconsistent)} rows where hard label != probability threshold 0.5.\n"
            f"         This is *not* an error – only informational.")

    # ---------------------------------------------------------------------
    # 3. Load ground‑truth and check correspondence
    # ---------------------------------------------------------------------
    gt_path = Path(os.path.join(cfg.goldstandard_file, "gt.csv"))
    if not gt_path.is_file():
        error(f"Ground‑truth file '{gt_path}' does not exist or is not a file.")

    try:
        gt_df = pd.read_csv(gt_path)
    except Exception as exc:
        error(f"Cannot read ground‑truth CSV: {exc}")

    expected_gt_cols = ["image", "label"]
    if list(gt_df.columns[:2]) != expected_gt_cols:  # strict but catches common mistakes
        error(f"Ground‑truth CSV must have columns {expected_gt_cols} as the first two columns.")

    gt_df["image"] = gt_df["image"].astype(str).str.strip()

    if gt_df["image"].duplicated().any():
        dupes = gt_df.loc[gt_df["image"].duplicated(), "image"].unique()
        error(f"Duplicate image id(s) in ground‑truth CSV: {', '.join(dupes)}")

    pred_set = set(pred_df["image"])
    gt_set   = set(gt_df["image"])

    missing_in_pred = gt_set - pred_set
    extra_in_pred   = pred_set - gt_set

    if missing_in_pred:
        error(f"{len(missing_in_pred)} image id(s) are present in GT but missing in predictions: {sorted(missing_in_pred)[:5]}…")
    if extra_in_pred:
        error(f"{len(extra_in_pred)} extra image id(s) found in predictions but not in GT: {sorted(extra_in_pred)[:5]}…")

    # ---------------------------------------------------------------------
    # 4. Emit validation JSON
    # ---------------------------------------------------------------------
    output_filename = "validated_result.json"
    data_id = f"{cfg.community_id}:{cfg.event_id}_{cfg.participant_id}"
    validated = True  # reaching this point ⇒ all checks passed

    validation_json = JSON_templates.write_participant_dataset(
        data_id,
        cfg.community_id,
        cfg.challenges_ids,
        cfg.participant_id,
        validated,
    )

    with open(output_filename, "w", encoding="utf-8") as fp:
        json.dump(validation_json, fp, indent=4, sort_keys=True, separators=(",", ": "))

    print(f"INFO: Validation succeeded. JSON written to '{output_filename}'.")
    sys.exit(0)


if __name__ == "__main__":
    main(args)
