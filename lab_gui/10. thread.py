from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QGridLayout
import sys


class Worker(QThread):
    finished = pyqtSignal()

    def run(self):
        cnt = 0
        for i in range(100000000):
            cnt += 1
        self.finished.emit()


class ClassificationAI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('이미지 분류 AI')

        self.button1 = QPushButton('버튼 1')
        self.button2 = QPushButton('버튼 2')

        self.button1.clicked.connect(self.button1_click)
        self.button2.clicked.connect(self.button2_click)

        self.worker = Worker()
        self.worker.finished.connect(self.button2_finished)

        self.main_layout = QGridLayout()
        self.main_layout.addWidget(self.button1, 0, 0, 1, 2)
        self.main_layout.addWidget(self.button2, 1, 0, 1, 2)

        self.setLayout(self.main_layout)

    def button1_click(self):
        self.button1.setEnabled(False)
        self.button1.setText('버튼 1 처리 중')
        cnt = 0
        for i in range(100000000):
            cnt += 1
        self.button1.setText('버튼 1 완료')

    def button2_click(self):
        self.button2.setEnabled(False)
        self.button2.setText('버튼 2 처리 중')
        self.worker.start()

    def button2_finished(self):
        self.button2.setText('버튼 2 완료')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    classification_ai = ClassificationAI()
    classification_ai.show()
    sys.exit(app.exec())
