from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QHBoxLayout, QPushButton, QGroupBox
import sys


class ClassificationAI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('제목')

        self.group_box = QGroupBox('그룹')

        self.button1 = QPushButton('버튼 1')
        self.button2 = QPushButton('버튼 2')

        self.hbox_layout = QHBoxLayout()
        self.hbox_layout.addWidget(self.button1)
        self.hbox_layout.addWidget(self.button2)

        self.group_box.setLayout(self.hbox_layout)

        self.main_layout = QGridLayout()
        self.main_layout.addWidget(self.group_box, 0, 0, 1, 1)

        self.setLayout(self.main_layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    classification_ai = ClassificationAI()
    classification_ai.show()
    sys.exit(app.exec())
