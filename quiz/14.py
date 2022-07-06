from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QGridLayout, QGroupBox, QHBoxLayout, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap
import sys
import model


class ClassificationAI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("이미지 분류 AI")

        self.model = model.Model()

        menu_group_box = QGroupBox("메뉴")

        self.load_data_button = QPushButton("데이터 불러오기")
        self.load_data_button.clicked.connect(self.load_data)

        self.train_model_button = QPushButton("모델 학습")
        self.train_model_button.setEnabled(False)
        self.train_model_button.clicked.connect(self.train_model)

        self.save_model_button = QPushButton("모델 저장")
        self.save_model_button.setEnabled(False)
        self.save_model_button.clicked.connect(self.save_model)

        self.load_model_button = QPushButton("모델 불러오기")
        self.load_model_button.setEnabled(False)
        self.load_model_button.clicked.connect(self.load_model)

        self.predict_button = QPushButton("이미지 분류")
        self.predict_button.setEnabled(False)
        self.predict_button.clicked.connect(self.predict)

        menu_layout = QHBoxLayout()
        menu_layout.addWidget(self.load_data_button)
        menu_layout.addWidget(self.train_model_button)
        menu_layout.addWidget(self.save_model_button)
        menu_layout.addWidget(self.load_model_button)
        menu_layout.addWidget(self.predict_button)
        menu_layout.addStretch(1)
        menu_group_box.setLayout(menu_layout)

        image_group_box = QGroupBox("이미지")

        self.image_label = QLabel(self)

        imageLayout = QVBoxLayout()
        imageLayout.addWidget(self.image_label)
        imageLayout.addStretch(1)
        image_group_box.setLayout(imageLayout)

        predictGroupBox = QGroupBox("분류 예측")

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

    def load_data(self):
        self.load_data_button.setEnabled(False)
        self.model.load_data()
        self.load_data_button.setText('데이터 불러오기 완료')
        self.train_model_button.setEnabled(True)
        self.load_model_button.setEnabled(True)

    def train_model(self):
        self.train_model_button.setEnabled(False)
        self.load_model_button.setEnabled(False)
        self.model.build()
        self.model.train()
        self.train_model_button.setText('모델 학습 완료')
        self.save_model_button.setEnabled(True)
        self.predict_button.setEnabled(True)

    def save_model(self):
        self.save_model_button.setEnabled(False)
        self.model.save()
        self.save_model_button.setText('모델 저장 완료')

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, '모델 파일 선택', "../models/", "Model Files (*.h5)")
        if path != '':
            self.load_model_button.setEnabled(False)
            self.train_model_button.setEnabled(False)
            self.model.load(path)
            self.load_model_button.setText('모델 불러오기 완료')
            self.predict_button.setEnabled(True)

    def predict(self):
        path, _ = QFileDialog.getOpenFileName(self, '이미지 선택', "../classification_data/", "Image Files (*.png *.jpg)")
        if path != '':
            self.predict_button.setEnabled(False)
            pixmap = QPixmap(path)
            height = pixmap.height()
            width = pixmap.width()
            ratio = 200 / height
            pixmap = pixmap.scaled(width * ratio, height * ratio, Qt.IgnoreAspectRatio)
            self.image_label.setPixmap(pixmap)
            result = self.model.predict_detail(path)
            t = '\t'
            result_text = '\n'.join([f'{r[0]}{t * (1 if len(r[0]) > 8 else 2)}: {r[1]:.2f}%' for r in result])
            self.predict_label.setText(result_text)
            self.predict_button.setText('이미지 분류 완료')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    classification_ai = ClassificationAI()
    classification_ai.show()
    sys.exit(app.exec_())
