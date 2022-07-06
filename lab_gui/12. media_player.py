from PyQt5.QtCore import Qt, QUrl, QFileInfo
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QSlider, QStyle, QGridLayout
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
import sys


class RoadObjectDetectionAI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("도로 객체 검출 AI")

        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        video_widget = QVideoWidget()
        self.media_player.setVideoOutput(video_widget)
        self.media_player.stateChanged.connect(self.media_state_changed)
        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.durationChanged.connect(self.duration_changed)
        self.media_player.error.connect(self.handle_error)

        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.play_video)

        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.set_position)

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.position_slider)

        media_player_layout = QVBoxLayout()
        media_player_layout.addWidget(video_widget)
        media_player_layout.addLayout(control_layout)

        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(QFileInfo('../outputs/output.wmv').absoluteFilePath())))

        mainLayout = QGridLayout()
        mainLayout.addLayout(media_player_layout, 0, 0, 1, 1)
        self.setLayout(mainLayout)

    def play_video(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def media_state_changed(self, state):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def position_changed(self, position):
        self.position_slider.setValue(position)

    def duration_changed(self, duration):
        self.position_slider.setRange(0, duration)

    def set_position(self, position):
        self.media_player.setPosition(position)

    def handle_error(self):
        print("Error: " + self.media_player.errorString())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    rod_ai = RoadObjectDetectionAI()
    rod_ai.show()
    sys.exit(app.exec_())
