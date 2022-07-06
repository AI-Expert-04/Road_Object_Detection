from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QGridLayout, QFileDialog
import sys


class ClassificationAI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('제목')

        self.button1 = QPushButton('파일 열기')
        self.button1.clicked.connect(self.button1_click)

        self.hbox_layout = QHBoxLayout()
        self.hbox_layout.addWidget(self.button1)

        self.main_layout = QGridLayout()
        self.main_layout.addLayout(self.hbox_layout, 0, 0, 1, 1)

        self.setLayout(self.main_layout)


    def button1_click(self):
        path, _ = QFileDialog.getOpenFileName(self, 'ABCD', '../data/images', 'Image Files (*.jpg *.png)')
        if path == '':
            print('취소')
        else:
            print('PATH:', path)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    classification_ai = ClassificationAI()
    classification_ai.show()
    sys.exit(app.exec())
