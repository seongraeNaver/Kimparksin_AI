import os
import json

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
