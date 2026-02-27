"""
train_classifier.py - EOG Feature Extractor + Random Forest Classifier

Loads raw EOG CSV files, preprocesses them, extracts windowed features,
trains a Random Forest classifier, evaluates it, and saves the model.

Usage:
    python -m src.train_classifier
"""

import json
import os
import glob
import warnings

import numpy as np
from scipy import signal as scipy_signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

from .signal_processing import process_eog

warnings.filterwarnings("ignore")

# =====================================================================
# Configuration
# =====================================================================

DATA_DIR = os.path.join("data", "TrainingData", "Traning data")
MODEL_DIR = os.path.join("data", "models")
SAMPLE_RATE = 500.0
WINDOW_SIZE = 250   # 0.5 seconds at 500 Hz
STRIDE = 125        # 50% overlap


# =====================================================================
# Data Loading
# =====================================================================

def load_csv_file(filepath: str):
    """
    Load an EOG CSV file, parsing the label from comment headers.

    Returns:
        label (str), timestamps (np.ndarray), eog1 (np.ndarray), eog2 (np.ndarray)
    """
    label = None
    data_lines = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# label:"):
                label = line.split(":", 1)[1].strip()
            elif line.startswith("#"):
                continue  # skip other comment lines
            elif line.startswith("timestamp"):
                continue  # skip header row
            else:
                data_lines.append(line)

    if label is None:
        raise ValueError(f"No '# label:' found in {filepath}")

    arr = np.array(
        [row.split(",") for row in data_lines if row.strip()],
        dtype=np.float64,
    )

    timestamps = arr[:, 0]
    eog1 = arr[:, 1]
    eog2 = arr[:, 2]

    return label, timestamps, eog1, eog2


def load_all_files(data_dir: str):
    """
    Glob all CSV files in data_dir and load them.

    Returns:
        list of (label, eog1, eog2) tuples
    """
    pattern = os.path.join(data_dir, "*.csv")
    filepaths = sorted(glob.glob(pattern))

    if not filepaths:
        raise FileNotFoundError(f"No CSV files found in: {data_dir}")

    records = []
    for fp in filepaths:
        label, _, eog1, eog2 = load_csv_file(fp)
        records.append((label, eog1, eog2))

    print(f"Found {len(filepaths)} files: {sorted({r[0] for r in records})}")
    return records


# =====================================================================
# Feature Extraction
# =====================================================================

def _zero_crossing_rate(w: np.ndarray) -> float:
    centered = w - np.mean(w)
    return float(np.sum(np.diff(np.sign(centered)) != 0)) / len(w)


def _slope_sign_changes(w: np.ndarray) -> float:
    diff = np.diff(w)
    return float(np.sum(np.diff(np.sign(diff)) != 0)) / len(w)


def extract_channel_features(filtered_w: np.ndarray, deriv_w: np.ndarray) -> np.ndarray:
    """
    Extract 13 features from a single channel window.
    """
    freqs, psd = scipy_signal.welch(filtered_w, fs=SAMPLE_RATE, nperseg=min(256, len(filtered_w)))
    total_power = float(np.sum(psd))
    mean_freq = float(np.sum(freqs * psd) / total_power) if total_power > 0 else 0.0
    peak_freq = float(freqs[np.argmax(psd)])

    features = np.array([
        np.mean(filtered_w),                        # mean
        np.std(filtered_w),                         # std
        np.sqrt(np.mean(filtered_w ** 2)),          # rms
        np.max(np.abs(filtered_w)),                 # max_abs
        np.max(filtered_w) - np.min(filtered_w),   # peak_to_peak
        np.sum(np.abs(np.diff(filtered_w))),        # waveform_length
        _zero_crossing_rate(filtered_w),            # zcr
        _slope_sign_changes(filtered_w),            # ssc
        total_power,                                # total_power
        mean_freq,                                  # mean_freq
        peak_freq,                                  # peak_freq
        np.max(np.abs(deriv_w)),                    # max_velocity
        np.mean(np.abs(deriv_w)),                   # mean_velocity
    ], dtype=np.float64)

    return features


CHANNEL_FEATURE_NAMES = [
    "mean", "std", "rms", "max_abs", "peak_to_peak",
    "waveform_length", "zcr", "ssc",
    "total_power", "mean_freq", "peak_freq",
    "max_velocity", "mean_velocity",
]

FEATURE_NAMES = (
    [f"eog1_{n}" for n in CHANNEL_FEATURE_NAMES]
    + [f"eog2_{n}" for n in CHANNEL_FEATURE_NAMES]
    + ["ch_correlation"]
)


def extract_features_from_window(
    f1_w: np.ndarray, d1_w: np.ndarray,
    f2_w: np.ndarray, d2_w: np.ndarray,
) -> np.ndarray:
    """
    Extract 27 features from a window across both EOG channels.
    """
    ch1_feats = extract_channel_features(f1_w, d1_w)
    ch2_feats = extract_channel_features(f2_w, d2_w)
    corr = np.corrcoef(f1_w, f2_w)[0, 1]
    if np.isnan(corr):
        corr = 0.0
    return np.concatenate([ch1_feats, ch2_feats, [corr]])


# =====================================================================
# Windowing
# =====================================================================

def extract_windows(records):
    """
    Slide windows over each recording and extract features.

    Returns:
        X (np.ndarray): shape (n_windows, 27)
        y (list of str): class labels per window
    """
    X_list = []
    y_list = []
    total_windows = 0

    for label, eog1, eog2 in records:
        # Preprocess both channels
        proc1 = process_eog(eog1, sample_rate=SAMPLE_RATE)
        proc2 = process_eog(eog2, sample_rate=SAMPLE_RATE)

        f1 = proc1["filtered"]
        d1 = proc1["derivative"]
        f2 = proc2["filtered"]
        d2 = proc2["derivative"]

        n_samples = len(f1)
        file_windows = 0

        start = 0
        while start + WINDOW_SIZE <= n_samples:
            end = start + WINDOW_SIZE
            feats = extract_features_from_window(
                f1[start:end], d1[start:end],
                f2[start:end], d2[start:end],
            )
            X_list.append(feats)
            y_list.append(label)
            file_windows += 1
            start += STRIDE

        total_windows += file_windows

    X = np.array(X_list, dtype=np.float64)
    print(f"Loaded {total_windows} windows from {len(records)} files")
    return X, y_list


# =====================================================================
# Training & Evaluation
# =====================================================================

def train_and_evaluate(X: np.ndarray, y_labels: list):
    """
    Encode labels, split, train RF, print evaluation, return model + encoder.
    """
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    class_names = le.classes_

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining on {len(X_train)} windows, testing on {len(X_test)} windows")
    print(f"Classes: {list(class_names)}")

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    print("Confusion Matrix (rows=true, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred)
    # Print with class names as headers
    header = "        " + "  ".join(f"{c:>12}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>12}" for v in row)
        print(f"{class_names[i]:>8}  {row_str}")

    print("\nTop 10 Most Important Features:")
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for rank, idx in enumerate(indices[:10], start=1):
        print(f"  {rank:2d}. {FEATURE_NAMES[idx]:30s}  {importances[idx]:.4f}")

    return clf, le


# =====================================================================
# Model Saving
# =====================================================================

def save_model(clf, le: LabelEncoder):
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "eog_classifier.joblib")
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.joblib")
    features_path = os.path.join(MODEL_DIR, "feature_names.json")

    joblib.dump(clf, model_path)
    joblib.dump(le, encoder_path)
    with open(features_path, "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)

    print(f"\nModel saved to {model_path}")
    print(f"Label encoder saved to {encoder_path}")
    print(f"Feature names saved to {features_path}")


# =====================================================================
# Main
# =====================================================================

def main():
    print("=== EOG Classifier Training ===\n")

    records = load_all_files(DATA_DIR)
    X, y_labels = extract_windows(records)

    clf, le = train_and_evaluate(X, y_labels)
    save_model(clf, le)

    print("\nDone.")


if __name__ == "__main__":
    main()
