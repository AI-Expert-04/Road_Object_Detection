from PyQt5.QtWidgets import QApplication, QWidget, QTabWidget, QPushButton, QVBoxLayout, QLabel, QGridLayout
import sys


class ClassificationAI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('제목')

        self.text_label = QLabel(self)
        self.text_label.setText('텍스트')

        self.vbox_layout = QVBoxLayout()
        self.vbox_layout.addWidget(self.text_label)

        self.main_layout = QGridLayout()
        self.main_layout.addLayout(self.vbox_layout, 0, 0, 1, 1)

        self.setLayout(self.main_layout)


class RoadObjectDetectionAI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('도로 객체 검출 AI')

        button1_tab = QPushButton('버튼 1')
        button2_tab = QPushButton('버튼 2')
        classification_ai_tab = ClassificationAI()

        tabs = QTabWidget()
        tabs.addTab(button1_tab, '탭 1')
        tabs.addTab(button2_tab, '탭 2')
        tabs.addTab(classification_ai_tab, '이미지 분류 탭')

        vbox = QVBoxLayout()
        vbox.addWidget(tabs)

        self.setLayout(vbox)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    rod_ai = RoadObjectDetectionAI()
    rod_ai.show()
    sys.exit(app.exec_())
