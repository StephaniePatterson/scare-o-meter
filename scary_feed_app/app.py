import sys
from pathlib import Path

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget


class FeedPlayer(QMainWindow):
    def __init__(self, video_paths):
        super().__init__()
        self.setWindowTitle("Scary Feed (TikTok-style)")
        self.setMinimumSize(450, 800)

        self.video_paths = video_paths
        self.index = 0

        # --- Central UI ---
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Video area
        self.video_widget = QVideoWidget()
        layout.addWidget(self.video_widget, stretch=1)

        # Overlay-ish info (simple label for now)
        self.caption = QLabel("")
        self.caption.setWordWrap(True)
        self.caption.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.caption)

        # Controls
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self.next_video)
        layout.addWidget(self.next_btn)

        # --- Media player ---
        self.player = QMediaPlayer()
        self.audio = QAudioOutput()
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video_widget)
        self.audio.setVolume(0.5)

        # Click anywhere on the video to go next
        self.video_widget.installEventFilter(self)

        # Load first video
        self.load_video(self.index)

    def eventFilter(self, source, event):
        if source is self.video_widget and event.type() == event.Type.MouseButtonPress:
            self.next_video()
            return True
        return super().eventFilter(source, event)

    def load_video(self, idx):
        if not self.video_paths:
            self.caption.setText("No videos found. Put MP4 files in the clips/ folder.")
            return

        idx = idx % len(self.video_paths)
        path = self.video_paths[idx]

        self.caption.setText(f"Clip {idx+1}/{len(self.video_paths)}: {path.name}")

        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(str(path.resolve())))
        self.player.play()

    def next_video(self):
        self.index = (self.index + 1) % len(self.video_paths)
        self.load_video(self.index)

    def keyPressEvent(self, event):
        # Optional: use down arrow or space to go next
        if event.key() in (Qt.Key.Key_Down, Qt.Key.Key_Space):
            self.next_video()
        else:
            super().keyPressEvent(event)


def main():
    # Look for videos in ./clips
    clips_dir = Path(__file__).parent / "clips"
    video_paths = sorted([p for p in clips_dir.glob("*.mp4")])

    app = QApplication(sys.argv)
    window = FeedPlayer(video_paths)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()