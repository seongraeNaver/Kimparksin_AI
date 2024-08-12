import json
import os
from glob import glob
from shutil import copyfile
from sklearn.model_selection import train_test_split
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# 설정
input_dir = 'D:/LA_dataset/final/final'
output_dir = 'D:/LA_dataset/yolov5_final'
image_extension = '.JPG'
annotation_extension = '.json'
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# 디렉토리 생성
os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'images', 'test'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels', 'test'), exist_ok=True)

# 이미지와 JSON 파일 리스트 생성
image_files = glob(os.path.join(input_dir, '*' + image_extension))
annotation_files = glob(os.path.join(input_dir, '*' + annotation_extension))

# 파일 쌍 확인
file_pairs = []
for img_file in image_files:
    base_name = os.path.splitext(os.path.basename(img_file))[0]
    json_file = os.path.join(input_dir, base_name + annotation_extension)
    if os.path.exists(json_file):
        file_pairs.append((img_file, json_file))

# 파일 쌍을 train, val, test로 분할
train_files, test_files = train_test_split(file_pairs, test_size=test_ratio, random_state=42)
train_files, val_files = train_test_split(train_files, test_size=val_ratio/(train_ratio + val_ratio), random_state=42)

def convert_and_save(json_path, output_label_path, img_width, img_height):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(output_label_path, 'w', encoding='utf-8') as f:
        for shape in data['shapes']:
            label = shape['label']
            # 클래스 인덱스 (여기서는 label을 숫자로 변환, 필요시 매핑 필요)
            class_index = 0 if label == 'lean' else 1
            points = shape['points']
            
            # YOLOv5 형식으로 좌표 변환 및 정규화
            normalized_points = []
            for point in points:
                x, y = point
                normalized_x = float(x) / img_width
                normalized_y = float(y) / img_height
                normalized_points.append((normalized_x, normalized_y))
            
            # 변환된 데이터 쓰기
            points_str = ' '.join([f"{pt[0]:.6f} {pt[1]:.6f}" for pt in normalized_points])
            f.write(f"{class_index} {points_str}\n")

def process_files(file_list, split):
    for img_file, json_file in tqdm(file_list, desc=f'Processing {split} files'):
        base_name = os.path.splitext(os.path.basename(img_file))[0]
        output_image_path = os.path.join(output_dir, 'images', split, base_name + image_extension)
        output_label_path = os.path.join(output_dir, 'labels', split, base_name + '.txt')
        
        copyfile(img_file, output_image_path)
        
        image = Image.open(img_file)
        img_height, img_width = image.size
        convert_and_save(json_file, output_label_path, img_width, img_height)

# 파일 처리
process_files(train_files, 'train')
process_files(val_files, 'val')
process_files(test_files, 'test')

print("처리가 완료되었습니다!")

################################################################
######################## 어노테이션 테스트#######################
################################################################
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from PIL import Image, ExifTags
# import numpy as np

# def correct_orientation(image):
#     try:
#         for orientation in ExifTags.TAGS.keys():
#             if ExifTags.TAGS[orientation] == 'Orientation':
#                 break
#         exif = dict(image._getexif().items())
#         if exif[orientation] == 3:
#             image = image.rotate(180, expand=True)
#         elif exif[orientation] == 6:
#             image = image.rotate(270, expand=True)
#         elif exif[orientation] == 8:
#             image = image.rotate(90, expand=True)
#     except (AttributeError, KeyError, IndexError):
#         # Cases: image don't have getexif
#         pass
#     return image

# def visualize_yolov5_segmentation(image_path, annotation_path):
#     # 이미지 열기
#     image = Image.open(image_path)
#     image = correct_orientation(image)  # 방향 수정
#     img_width, img_height = image.size

#     # 어노테이션 파일 열기
#     with open(annotation_path, 'r') as f:
#         annotations = f.readlines()

#     fig, ax = plt.subplots(1)
#     ax.imshow(image)

#     for annotation in annotations:
#         parts = annotation.strip().split()
#         class_id = int(parts[0])
#         polygon = np.array([float(p) for p in parts[1:]]).reshape(-1, 2)
        
#         # 원래 크기로 되돌리기
#         polygon[:, 0] *= img_width
#         polygon[:, 1] *= img_height

#         # 폴리곤 그리기 및 내부 채우기
#         poly = patches.Polygon(polygon, closed=True, edgecolor='red', facecolor='red', alpha=0.4)
#         ax.add_patch(poly)
#         # 클래스 라벨 추가 (optional)
#         plt.text(polygon[0, 0], polygon[0, 1], str(class_id), color='red', fontsize=12)

#     plt.axis('off')
#     plt.show()

# # 이미지 및 어노테이션 경로
# image_path = 'D:/LA_dataset/yolov5/images/train/IMG_3158.JPG'
# annotation_path = 'D:/LA_dataset/yolov5/labels/train/IMG_3158.txt'

# # 시각화 함수 호출
# visualize_yolov5_segmentation(image_path, annotation_path)
