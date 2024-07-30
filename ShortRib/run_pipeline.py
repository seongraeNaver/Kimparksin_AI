import os
import subprocess
import json
# 수정본 kim
# Step 1: JSON to YOLO format conversion 
def convert_annotations():
    def convert_json_to_yolo(json_path, output_path):
        with open(json_path) as f:
            data = json.load(f)
        
        yolo_data = []
        for shape in data['shapes']:
            if shape['label'] == 'lean':
                # Get polygon points
                points = shape['points']
                # Calculate bounding box from polygon points
                xmin = min(point[0] for point in points)
                ymin = min(point[1] for point in points)
                xmax = max(point[0] for point in points)
                ymax = max(point[1] for point in points)
                # Convert bbox to YOLO format (x_center, y_center, width, height)
                x_center = (xmin + xmax) / 2.0
                y_center = (ymin + ymax) / 2.0
                width = xmax - xmin
                height = ymax - ymin
                yolo_data.append(f"0 {x_center} {y_center} {width} {height}")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(yolo_data))

    def process_dataset(json_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for json_file in os.listdir(json_dir):
            if json_file.endswith('.json'):
                json_path = os.path.join(json_dir, json_file)
                output_path = os.path.join(output_dir, json_file.replace('.json', '.txt'))
                convert_json_to_yolo(json_path, output_path)

    # Convert annotations
    process_dataset('C:/ShortRib/annotations/train', 'C:/ShortRib/labels/train')

# Step 2: Train YOLOv5 model
def train_model():
    # Ensure YOLOv5 directory exists
    yolov5_dir = 'C:/ShortRib/yolov5'
    if not os.path.exists(yolov5_dir):
        raise FileNotFoundError("YOLOv5 directory not found. Clone YOLOv5 repository into C:/ShortRib.")

    # Set up directories
    data_dir = 'C:/ShortRib'
    data_file = os.path.join(data_dir, 'data.yaml')
    weights_dir = os.path.join(data_dir, 'weights')

    # Ensure weights directory exists
    os.makedirs(weights_dir, exist_ok=True)

    # Training parameters
    epochs = 50
    batch_size = 16

    # Train the model using subprocess to call train.py
    subprocess.run([
        'python', os.path.join(yolov5_dir, 'train.py'),
        '--data', data_file,
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--weights', os.path.join(weights_dir, 'yolov5s.pt')
    ], check=True)

# Step 3: Run PyQt5 GUI
def run_gui():
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtCore import Qt
    import torch
    from pathlib import Path

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
                model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/ShortRib/weights/best.pt', force_reload=True)
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

if __name__ == "__main__":
    convert_annotations()
    train_model()
    run_gui()
