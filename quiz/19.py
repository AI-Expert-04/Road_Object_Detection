from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl, QFileInfo
from PyQt5.QtWidgets import QGridLayout, QGroupBox, QHBoxLayout, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import *

import os
import yolov3
import main
import sys
import cv2


class PredictWorker(QThread):
    def __init__(self, model):
        super(PredictWorker, self).__init__()
        self.model = model

    def set_image_path(self, path):
        self.image_path = path

    def run(self):
        image = cv2.imread(self.image_path)
        result = self.model.predict(image)
        if not os.path.exists('../outputs'):
            os.mkdir('../outputs')
        cv2.imwrite('../outputs/output.jpg', result)
        self.finished.emit()


class RoadObjectDetectionImage(QWidget):
    def __init__(self):
        super().__init__()
        # self.setWindowTitle("도로 객체 검출 AI")

        self.model = yolov3.YOLO_V3()
        self.model.build()
        self.model.load()

        menu_group_box = QGroupBox("메뉴")

        self.predict_button = QPushButton("이미지 선택")
        self.predict_button.clicked.connect(self.predict)

        self.predict_worker = PredictWorker(self.model)
        self.predict_worker.finished.connect(self.predict_finished)

        menu_layout = QHBoxLayout()
        menu_layout.addWidget(self.predict_button)
        menu_layout.addStretch(1)
        menu_group_box.setLayout(menu_layout)

        image_group_box = QGroupBox("원본 이미지")

        self.image_label = QLabel(self)

        imageLayout = QVBoxLayout()
        imageLayout.addWidget(self.image_label)
        imageLayout.addStretch(1)
        image_group_box.setLayout(imageLayout)

        predictGroupBox = QGroupBox("사물 검출 결과")

        self.predict_label = QLabel(self)
        predictLayout = QVBoxLayout()
        predictLayout.addWidget(self.predict_label)
        predictLayout.addStretch(1)
        predictGroupBox.setLayout(predictLayout)

        mainLayout = QGridLayout()
        mainLayout.addWidget(menu_group_box, 0, 0, 1, 3)
        mainLayout.addWidget(image_group_box, 1, 0, 2, 2)
        mainLayout.addWidget(predictGroupBox, 1, 2, 2, 1)
        mainLayout.setRowStretch(1, 1)
        self.setLayout(mainLayout)

    def predict(self):
        path, _ = QFileDialog.getOpenFileName(self, '이미지 선택', "../classification_data/", "Image Files (*.png *.jpg)")
        if path != '':
            pixmap = QPixmap(path)
            height = pixmap.height()
            width = pixmap.width()
            ratio = 0.5
            pixmap = pixmap.scaled(width * ratio, height * ratio, Qt.IgnoreAspectRatio)
            self.image_label.setPixmap(pixmap)

            self.predict_worker.set_image_path(path)
            self.predict_button.setEnabled(False)
            self.predict_button.setText('객체 검출 중')
            self.predict_worker.start()

    def predict_finished(self):
        pixmap = QPixmap('../outputs/output.jpg')
        height = pixmap.height()
        width = pixmap.width()
        ratio = 0.5
        pixmap = pixmap.scaled(width * ratio, height * ratio, Qt.IgnoreAspectRatio)
        self.predict_label.setPixmap(pixmap)

        self.predict_button.setText('이미지 선택')
        self.predict_button.setEnabled(True)


class VideoProcessWorker(QThread):
    finished = pyqtSignal()

    def set_video_path(self, video_path):
        self.video_path = video_path

    def run(self):
        main.video_processing(self.video_path, True)
        self.finished.emit()


class RoadObjectDetectionVideo(QWidget):
    def __init__(self):
        super().__init__()
        # self.setWindowTitle('도로 객체 검출 AI')

        menu_group_box = QGroupBox("메뉴")

        self.load_video_button = QPushButton('영상 선택')
        self.load_video_button.clicked.connect(self.video_process)

        self.load_video_worker = VideoProcessWorker()
        self.load_video_worker.finished.connect(self.video_process_finished)

        menu_layout = QHBoxLayout()
        menu_layout.addWidget(self.load_video_button)
        menu_layout.addStretch(1)
        menu_group_box.setLayout(menu_layout)

        media_player_box = QGroupBox("영상")
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        video_widget = QVideoWidget()
        self.media_player.setVideoOutput(video_widget)
        self.media_player.stateChanged.connect(self.media_state_changed)
        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.durationChanged.connect(self.duration_changed)
        self.media_player.error.connect(self.handle_error)

        self.play_button = QPushButton()
        self.play_button.setEnabled(False)
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

        media_player_box.setLayout(media_player_layout)

        mainLayout = QGridLayout()
        mainLayout.addWidget(menu_group_box, 0, 0, 1, 1)
        mainLayout.addWidget(media_player_box, 1, 0, 2, 1)
        mainLayout.setRowStretch(1, 1)
        self.setLayout(mainLayout)

    def video_process(self):
        video_path, _ = QFileDialog.getOpenFileName(self, '', ".", "Video Files (*.mp4 *.mov *.avi *.wmv)")
        if video_path != '':
            self.load_video_worker.set_video_path(video_path)
            self.load_video_button.setEnabled(False)
            self.load_video_button.setText('영상 처리 중')
            self.load_video_worker.start()

    def video_process_finished(self):
        self.load_video_button.setText('영상 처리 완료')
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(QFileInfo('../outputs/output.wmv').absoluteFilePath())))
        self.play_button.setEnabled(True)

    def play_video(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def media_state_changed(self, state):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.play_button.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.play_button.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))

    def position_changed(self, position):
        self.position_slider.setValue(position)

    def duration_changed(self, duration):
        self.position_slider.setRange(0, duration)

    def set_position(self, position):
        self.media_player.setPosition(position)

    def handle_error(self):
        print("Error: " + self.media_player.errorString())


class RoadObjectDetectionAI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("도로 객체 검출 AI")

        image_tab = RoadObjectDetectionImage()
        video_tab = RoadObjectDetectionVideo()

        tabs = QTabWidget()
        tabs.addTab(image_tab, '이미지')
        tabs.addTab(video_tab, '동영상')

        vbox = QVBoxLayout()
        vbox.addWidget(tabs)

        self.setLayout(vbox)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    rod_ai = RoadObjectDetectionAI()
    rod_ai.show()
    sys.exit(app.exec_())
