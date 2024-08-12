import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ExifTags
import numpy as np

def correct_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())
        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # Cases: image don't have getexif
        pass
    return image

def visualize_yolov5_segmentation(image_path, annotation_path):
    # 이미지 열기
    image = Image.open(image_path)
    image = correct_orientation(image)  # 방향 수정
    img_width, img_height = image.size

    # 어노테이션 파일 열기
    with open(annotation_path, 'r') as f:
        annotations = f.readlines()

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for annotation in annotations:
        parts = annotation.strip().split()
        class_id = int(parts[0])
        polygon = np.array([float(p) for p in parts[1:]]).reshape(-1, 2)
        
        # 원래 크기로 되돌리기
        polygon[:, 0] *= img_width
        polygon[:, 1] *= img_height

        # 폴리곤 그리기 및 내부 채우기
        poly = patches.Polygon(polygon, closed=True, edgecolor='red', facecolor='red', alpha=0.4)
        ax.add_patch(poly)
        # 클래스 라벨 추가 (optional)
        plt.text(polygon[0, 0], polygon[0, 1], str(class_id), color='red', fontsize=12)

    plt.axis('off')
    plt.show()

# 이미지 및 어노테이션 경로
image_path = 'D:/LA_dataset/yolov5/images/train/IMG_3158.JPG'
annotation_path = 'D:/LA_dataset/yolov5/labels/train/IMG_3158.txt'

# 시각화 함수 호출
visualize_yolov5_segmentation(image_path, annotation_path)
