"""
predict_trial.py - Run the trained EOG classifier on a trial CSV

Loads the saved model, slides windows over the trial recording, and prints
a timestamped sequence of predicted classes with confidence scores.

Usage:
    python -m src.predict_trial <path_to_trial.csv>

The CSV can be a raw recording file (with or without a '# label:' header).
Columns expected: timestamp, eog1, eog2
"""

import sys
import os
import numpy as np
import joblib

from .signal_processing import process_eog
from .train_classifier import (
    SAMPLE_RATE, WINDOW_SIZE, STRIDE,
    extract_features_from_window,
)

MODEL_DIR = os.path.join("data", "models")


def load_model():
    model_path = os.path.join(MODEL_DIR, "eog_classifier.joblib")
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run 'python -m src.train_classifier' first."
        )

    clf = joblib.load(model_path)
    le = joblib.load(encoder_path)
    return clf, le


def load_trial_csv(filepath: str):
    """
    Load a trial CSV. Does not require a '# label:' header.
    Returns timestamps, eog1, eog2 as numpy arrays.
    """
    data_lines = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#") or line.startswith("timestamp"):
                continue
            elif line.strip():
                data_lines.append(line)

    if not data_lines:
        raise ValueError(f"No data rows found in {filepath}")

    arr = np.array(
        [row.split(",") for row in data_lines],
        dtype=np.float64,
    )

    timestamps = arr[:, 0]
    eog1 = arr[:, 1]
    eog2 = arr[:, 2]

    return timestamps, eog1, eog2


def predict_sequence(clf, le, timestamps, eog1, eog2):
    """
    Slide windows over the trial and return predictions.

    Returns list of dicts with keys:
        window_idx, t_start, t_end, label, confidence, all_probs
    """
    proc1 = process_eog(eog1, sample_rate=SAMPLE_RATE)
    proc2 = process_eog(eog2, sample_rate=SAMPLE_RATE)

    f1, d1 = proc1["filtered"], proc1["derivative"]
    f2, d2 = proc2["filtered"], proc2["derivative"]

    n_samples = len(f1)
    results = []
    window_idx = 0
    start = 0

    while start + WINDOW_SIZE <= n_samples:
        end = start + WINDOW_SIZE

        feats = extract_features_from_window(
            f1[start:end], d1[start:end],
            f2[start:end], d2[start:end],
        )

        probs = clf.predict_proba([feats])[0]
        pred_idx = np.argmax(probs)
        label = le.inverse_transform([pred_idx])[0]
        confidence = probs[pred_idx]

        t_start = timestamps[start]
        t_end = timestamps[min(end - 1, len(timestamps) - 1)]

        results.append({
            "window_idx": window_idx,
            "t_start": t_start,
            "t_end": t_end,
            "label": label,
            "confidence": confidence,
            "all_probs": dict(zip(le.classes_, probs)),
        })

        window_idx += 1
        start += STRIDE

    return results


def print_results(results, le):
    classes = list(le.classes_)
    col_w = max(len(c) for c in classes)

    # Header
    print(f"\n{'Win':>4}  {'t_start(s)':>10}  {'t_end(s)':>8}  {'Prediction':<20}  {'Conf':>6}  " +
          "  ".join(f"{c:>{col_w}}" for c in classes))
    print("-" * (4 + 10 + 8 + 20 + 6 + (col_w + 2) * len(classes) + 20))

    prev_label = None
    for r in results:
        label = r["label"]
        marker = " <--" if label != "baseline" and label != prev_label else ""
        prob_cols = "  ".join(f"{r['all_probs'][c]:>{col_w}.2f}" for c in classes)
        print(
            f"{r['window_idx']:>4}  "
            f"{r['t_start']:>10.3f}  "
            f"{r['t_end']:>8.3f}  "
            f"{label:<20}  "
            f"{r['confidence']:>6.2f}  "
            f"{prob_cols}"
            f"{marker}"
        )
        prev_label = label

    # Summary: collapse consecutive identical predictions
    print("\n=== Prediction Sequence ===")
    if not results:
        print("  (no windows)")
        return

    sequence = []
    cur_label = results[0]["label"]
    cur_start = results[0]["t_start"]
    cur_end = results[0]["t_end"]

    for r in results[1:]:
        if r["label"] == cur_label:
            cur_end = r["t_end"]
        else:
            sequence.append((cur_label, cur_start, cur_end))
            cur_label = r["label"]
            cur_start = r["t_start"]
            cur_end = r["t_end"]
    sequence.append((cur_label, cur_start, cur_end))

    for label, t0, t1 in sequence:
        duration = t1 - t0
        print(f"  [{t0:6.3f}s - {t1:6.3f}s]  ({duration:.2f}s)  {label}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict_trial <path_to_trial.csv>")
        sys.exit(1)

    trial_path = sys.argv[1]

    if not os.path.exists(trial_path):
        print(f"Error: file not found: {trial_path}")
        sys.exit(1)

    print(f"Loading model from {MODEL_DIR} ...")
    clf, le = load_model()
    print(f"Classes: {list(le.classes_)}")

    print(f"\nLoading trial: {trial_path}")
    timestamps, eog1, eog2 = load_trial_csv(trial_path)
    duration = timestamps[-1] - timestamps[0]
    print(f"  {len(timestamps)} samples, {duration:.2f}s @ ~{len(timestamps)/duration:.0f} Hz")

    results = predict_sequence(clf, le, timestamps, eog1, eog2)
    print(f"  {len(results)} windows ({WINDOW_SIZE} samples, {STRIDE}-sample stride)\n")

    print_results(results, le)


if __name__ == "__main__":
    main()
