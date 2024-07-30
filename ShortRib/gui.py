# gui.py
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import torch
from pathlib import Path
from yolov5 import detect
import cv2

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv5 Lean Detector")
        self.setGeometry(100, 100, 800, 600)
        
        self.image_label = QLabel(self)
        self.image_label.setGeometry(50, 50, 700, 400)
        self.image_label.setAlignment(Qt.AlignCenter)
        
        self.upload_button = QPushButton("이미지 업로드", self)
        self.upload_button.setGeometry(50, 500, 200, 50)
        self.upload_button.clicked.connect(self.upload_image)
        
        self.result_button = QPushButton("결과 확인", self)
        self.result_button.setGeometry(550, 500, 200, 50)
        self.result_button.clicked.connect(self.detect_lean)
        
        self.image_path = None
    
    def upload_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "이미지 파일 선택", "", "이미지 파일 (*.jpg *.png)", options=options)
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))
    
    def detect_lean(self):
        if self.image_path:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='C://ShortRib/weights/best.pt', force_reload=True)
            results = model(self.image_path)
            results.save()

            result_image_path = Path(results.save_dir) / Path(self.image_path).name
            pixmap = QPixmap(str(result_image_path))
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
