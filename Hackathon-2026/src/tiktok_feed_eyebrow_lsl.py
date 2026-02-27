import sys
import time
import json
from pathlib import Path
from collections import deque

import numpy as np
import joblib

from PyQt6.QtCore import Qt, QUrl, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget


# ---------------------------
# USER SETTINGS (EDIT THESE)
# ---------------------------

LSL_STREAM_NAME = "BioRadio"     # must match the outlet name in your GUI
FS = 250                         # sample rate (Hz) - match your device setting
WINDOW_SECONDS = 0.40            # 400 ms is a common start for EOG gesture windows

# Detection logic (tune later)
PROBA_THRESHOLD = 0.85
CONSECUTIVE_HITS = 3
COOLDOWN_SECONDS = 1.0

# Label string in your label encoder that corresponds to eyebrow raise
# If you’re not sure, run the small debug snippet at the bottom.
EYEBROW_LABEL_STRING = "raise_eyebrows"


# ---------------------------
# FEATURE FUNCTIONS
# ---------------------------

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)))

def _waveform_length(x: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(x))))

def _zcr(x: np.ndarray) -> float:
    # zero crossing rate = sign changes / (n-1)
    if len(x) < 2:
        return 0.0
    s = np.sign(x)
    return float(np.sum(s[:-1] * s[1:] < 0) / (len(x) - 1))

def _ssc(x: np.ndarray) -> float:
    # slope sign changes: count sign changes in the first derivative
    if len(x) < 3:
        return 0.0
    dx = np.diff(x)
    s = np.sign(dx)
    return float(np.sum(s[:-1] * s[1:] < 0))

def _power_features(x: np.ndarray, fs: int) -> tuple[float, float, float]:
    """
    total_power, mean_freq, peak_freq using simple FFT power spectrum.
    """
    n = len(x)
    if n < 4:
        return 0.0, 0.0, 0.0

    # remove DC
    x0 = x - np.mean(x)

    # rfft power
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    X = np.fft.rfft(x0)
    Pxx = (np.abs(X) ** 2) / n  # simple power

    total_power = float(np.sum(Pxx))

    if total_power <= 1e-12:
        return total_power, 0.0, 0.0

    mean_freq = float(np.sum(freqs * Pxx) / total_power)
    peak_freq = float(freqs[int(np.argmax(Pxx))])
    return total_power, mean_freq, peak_freq

def _velocity_features(x: np.ndarray, fs: int) -> tuple[float, float]:
    """
    max_velocity, mean_velocity based on |dx/dt|
    """
    if len(x) < 2:
        return 0.0, 0.0
    vel = np.diff(x) * fs
    abs_vel = np.abs(vel)
    return float(np.max(abs_vel)), float(np.mean(abs_vel))


def compute_feature_dict(eog1: np.ndarray, eog2: np.ndarray, fs: int) -> dict[str, float]:
    """
    Compute ALL features listed in feature_names.json.
    Returns a dict mapping feature_name -> value
    """
    feats = {}

    def per_channel(prefix: str, x: np.ndarray):
        feats[f"{prefix}_mean"] = float(np.mean(x))
        feats[f"{prefix}_std"] = float(np.std(x, ddof=0))
        feats[f"{prefix}_rms"] = _rms(x)
        feats[f"{prefix}_max_abs"] = float(np.max(np.abs(x))) if len(x) else 0.0
        feats[f"{prefix}_peak_to_peak"] = float(np.ptp(x)) if len(x) else 0.0
        feats[f"{prefix}_waveform_length"] = _waveform_length(x)
        feats[f"{prefix}_zcr"] = _zcr(x)
        feats[f"{prefix}_ssc"] = _ssc(x)

        total_power, mean_freq, peak_freq = _power_features(x, fs)
        feats[f"{prefix}_total_power"] = total_power
        feats[f"{prefix}_mean_freq"] = mean_freq
        feats[f"{prefix}_peak_freq"] = peak_freq

        max_v, mean_v = _velocity_features(x, fs)
        feats[f"{prefix}_max_velocity"] = max_v
        feats[f"{prefix}_mean_velocity"] = mean_v

    per_channel("eog1", eog1)
    per_channel("eog2", eog2)

    # channel correlation
    if len(eog1) > 1 and len(eog2) > 1:
        c = np.corrcoef(eog1, eog2)[0, 1]
        feats["ch_correlation"] = float(c) if np.isfinite(c) else 0.0
    else:
        feats["ch_correlation"] = 0.0

    return feats


# ---------------------------
# LSL INFERENCE THREAD
# ---------------------------

class LSLInferenceThread(QThread):
    eyebrow_detected = pyqtSignal(float, float)  # (timestamp, proba)

    def __init__(self, model_path: Path, labelenc_path: Path, feature_names_path: Path):
        super().__init__()

        self.model = joblib.load(model_path)
        self.le = joblib.load(labelenc_path)
        self.feature_names = json.loads(feature_names_path.read_text())

        # map eyebrow label string -> class index
        if EYEBROW_LABEL_STRING not in list(self.le.classes_):
            raise ValueError(
                f"'{EYEBROW_LABEL_STRING}' not found in label encoder classes: {list(self.le.classes_)}"
            )
        self.eyebrow_class = EYEBROW_LABEL_STRING

        self.window_size = max(10, int(WINDOW_SECONDS * FS))
        self.buf1 = deque(maxlen=self.window_size)
        self.buf2 = deque(maxlen=self.window_size)

        self.hit_streak = 0
        self.last_trigger = 0.0
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        from pylsl import resolve_byprop, StreamInlet

        streams = resolve_byprop("name", LSL_STREAM_NAME, timeout=5)
        if not streams:
            print(f"[LSL] No stream found with name='{LSL_STREAM_NAME}'")
            return

        inlet = StreamInlet(streams[0], max_buflen=2)
        print(f"[LSL] Connected to stream: {LSL_STREAM_NAME}")

        while self._running:
            sample, ts = inlet.pull_sample(timeout=0.2)
            if sample is None:
                continue

            # Use first two channels as eog1/eog2
            # If you have more channels, just make sure eyebrow EOG channels are first two.
            if len(sample) < 2:
                continue

            self.buf1.append(float(sample[0]))
            self.buf2.append(float(sample[1]))

            if len(self.buf1) < self.window_size:
                continue

            eog1 = np.array(self.buf1, dtype=np.float32)
            eog2 = np.array(self.buf2, dtype=np.float32)

            feat_dict = compute_feature_dict(eog1, eog2, FS)

            # Build feature vector in EXACT order required by training
            x = np.array([feat_dict.get(name, 0.0) for name in self.feature_names], dtype=np.float32).reshape(1, -1)

            proba = self.model.predict_proba(x)[0]
            class_labels = self.le.inverse_transform(np.arange(len(proba)))

            # probability of eyebrow class
            try:
                idx = list(class_labels).index(self.eyebrow_class)
                eyebrow_p = float(proba[idx])
            except ValueError:
                eyebrow_p = 0.0

            now = time.time()

            # cooldown gate
            if (now - self.last_trigger) < COOLDOWN_SECONDS:
                self.hit_streak = 0
                continue

            if eyebrow_p >= PROBA_THRESHOLD:
                self.hit_streak += 1
            else:
                self.hit_streak = 0

            if self.hit_streak >= CONSECUTIVE_HITS:
                self.hit_streak = 0
                self.last_trigger = now
                self.eyebrow_detected.emit(now, eyebrow_p)


# ---------------------------
# TIKTOK-LIKE VIDEO PLAYER
# ---------------------------

class FeedPlayer(QMainWindow):
    def __init__(self, video_paths: list[Path]):
        super().__init__()
        self.setWindowTitle("Scary Feed — Eyebrow Raise = Next")
        self.setMinimumSize(450, 800)

        self.video_paths = video_paths
        self.index = 0

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.video_widget = QVideoWidget()
        layout.addWidget(self.video_widget, stretch=1)

        self.status = QLabel("Status: Ready")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self.next_video)
        layout.addWidget(self.next_btn)

        self.player = QMediaPlayer()
        self.audio = QAudioOutput()
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video_widget)
        self.audio.setVolume(0.5)

        self.load_video(self.index)

    def load_video(self, idx: int):
        if not self.video_paths:
            self.status.setText("No videos found. Put MP4 files in clips/.")
            return

        idx = idx % len(self.video_paths)
        path = self.video_paths[idx]

        self.status.setText(f"Playing {idx+1}/{len(self.video_paths)}: {path.name}")
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(str(path.resolve())))
        self.player.play()

    def next_video(self):
        self.index = (self.index + 1) % len(self.video_paths)
        self.load_video(self.index)

    def on_eyebrow(self, t: float, p: float):
        self.status.setText(f"Eyebrow raise detected (p={p:.2f}) → Next")
        self.next_video()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Down, Qt.Key.Key_Space):
            self.next_video()
        else:
            super().keyPressEvent(event)


def main():
    base = Path(__file__).parent
    clips_dir = base / "clips"
    model_dir = base / "models"

    video_paths = sorted(list(clips_dir.glob("*.mp4")))
    model_path = model_dir / "eog_classifier.joblib"
    le_path = model_dir / "label_encoder.joblib"
    feat_path = model_dir / "feature_names.json"

    app = QApplication(sys.argv)
    window = FeedPlayer(video_paths)
    window.show()

    # Start inference thread
    thread = LSLInferenceThread(model_path, le_path, feat_path)
    thread.eyebrow_detected.connect(window.on_eyebrow)
    thread.start()

    exit_code = app.exec()
    thread.stop()
    thread.wait(2000)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()