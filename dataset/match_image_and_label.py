import argparse
import os
from pathlib import Path

from tqdm import tqdm

from tools.file_tools import get_files_by_ext, get_file_name

def make_parser():
    parser = argparse.ArgumentParser("MATCH IMAGE AND LABELS")
    parser.add_argument(
        "-ii", 
        "--input_image", 
        required=True, 
        type=str, 
        help="input your image dir path"
    )
    parser.add_argument(
        "-il", 
        "--input_label", 
        required=True, 
        type=str, 
        help="input your label dir path"
    )
    return parser

def mathch_image_and_label(image_dir_path: str, label_dir_path: str):
    print(f"mathch_image_and_label -> image path: {image_dir_path}")
    if not os.path.isdir(image_dir_path):
        print(f"mathch_image_and_label ERROR: '{image_dir_path}' 경로를 찾을 수 없습니다.")
        return

    print(f"mathch_image_and_label -> label path: {label_dir_path}")
    if not os.path.isdir(label_dir_path):
        print(f"mathch_image_and_label ERROR: '{label_dir_path}' 경로를 찾을 수 없습니다.")
        return

    image_files = get_files_by_ext(image_dir_path, [".jpg", ".jpeg", ".webp", ".bmp", ".png"])
    image_names = dict()
    for file in image_files:
        name = get_file_name(file)
        image_names[name] = file
    image_name_keys = image_names.keys()

    label_files = get_files_by_ext(label_dir_path, [".txt"])
    label_names = dict()
    for file in label_files:
        name = get_file_name(file)
        label_names[name] = file
    label_name_keys = label_names.keys()

    images_without_label = []
    for image_name in image_name_keys - label_name_keys:
        images_without_label.append(image_names[image_name])
    print(f"images_without_label: {images_without_label}")

    labels_without_image = []
    for image_name in label_name_keys - image_name_keys:
        labels_without_image.append(label_names[image_name])
    print(f"labels_without_image: {labels_without_image}")

    print("-------------------------------------")
    print(f"Input Counts -> Image: {len(image_names)}, Label: {len(label_names)}, Match: {len(image_name_keys & label_name_keys)}")
    print("-------------------------------------")
    print(f"Images without label -> Counts: {len(images_without_label)}")
    for path in images_without_label:
         print(path)
    print("-------------------------------------")
    print(f"Label without image -> Counts: {len(labels_without_image)}")
    for path in labels_without_image:
         print(path)
    print("-------------------------------------")

if __name__ == "__main__":
    args = make_parser().parse_args()

    mathch_image_and_label(
        image_dir_path=args.input_image, 
        label_dir_path=args.input_label
    )


