import os
import xml.etree.ElementTree as ET

# Customize these paths
XML_DIR = 'datasets/RDD_SPLIT/val/labels/labelxml'
IMG_DIR = 'datasets/RDD_SPLIT/val/images'
OUTPUT_DIR = 'datasets/RDD_SPLIT/val/labels/'

# Make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define class mapping: class name → class ID
class_mapping = {
    'P00': 0,  # <-- add this
    'D00': 1,
    'D10': 2,
    'D20': 3,
    'D40': 4,
    'F00': 5
}

def convert(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    yolo_lines = []

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in class_mapping:
            continue  # skip unknown classes

        cls_id = class_mapping[cls]

        xml_box = obj.find('bndbox')
        xmin = int(xml_box.find('xmin').text)
        xmax = int(xml_box.find('xmax').text)
        ymin = int(xml_box.find('ymin').text)
        ymax = int(xml_box.find('ymax').text)

        # YOLO format: class x_center y_center width height (all normalized)
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        yolo_line = f"{cls_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        yolo_lines.append(yolo_line)

    # Write to .txt file
    base_filename = os.path.splitext(os.path.basename(xml_file))[0]
    txt_path = os.path.join(OUTPUT_DIR, base_filename + '.txt')
    with open(txt_path, 'w') as out_file:
        out_file.write('\n'.join(yolo_lines))


# Convert all XML files
for filename in os.listdir(XML_DIR):
    if filename.endswith('.xml'):
        convert(os.path.join(XML_DIR, filename))

print("✅ Conversion complete. YOLO labels saved to:", OUTPUT_DIR)
