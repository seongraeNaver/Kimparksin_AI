import os
import json

def convert_json_to_yolo(json_path, output_path):
    with open(json_path) as f:
        data = json.load(f)
    
    yolo_data = []
    for obj in data['objects']:
        if obj['label'] == 'lean':
            # Convert bbox to YOLO format (x_center, y_center, width, height)
            xmin, ymin, xmax, ymax = obj['bbox']
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
process_dataset('C://ShortRib/annotations/train', 'C://ShortRib/labels/train')
