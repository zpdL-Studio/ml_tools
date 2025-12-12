import argparse
import os

from tqdm import tqdm

from tools.file_tools import get_files_by_ext

def make_parser():
    parser = argparse.ArgumentParser("COUNT LABELS")
    parser.add_argument(
        "-i", 
        "--input", 
        required=True, 
        type=str, 
        help="input your label dir path"
    )
    return parser

def count_labels(label_dir_path: str):
    print(f"count_labels path: {label_dir_path}")
    label_files = get_files_by_ext(label_dir_path, [".txt"])
    
    label_counts = {}
    for label_path in tqdm(label_files):
        with open(os.path.join(label_path), "r") as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(
                    float, line.strip().split()
                )
                class_id = int(class_id)
                if class_id in label_counts:
                    label_counts[class_id] += 1
                else:
                    label_counts[class_id] = 1    

        
    for key, value in sorted(label_counts.items()):
        print(f"class: {key}, count: {value}")

# python count_labels.py -i /media/DATASET/molu/molu/rdd/train/labels
if __name__ == "__main__":
    args = make_parser().parse_args()

    count_labels(label_dir_path=args.input)


