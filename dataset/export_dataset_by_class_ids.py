import argparse
import os
import shutil

from tqdm import tqdm

def make_parser():
    parser = argparse.ArgumentParser("EXPORT DATASET BY CATEGORY")
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
    parser.add_argument(
        "-c", 
        "--class_ids", 
        required=True, 
        type=str, 
        help="input export class_ids, separate ',', and only int"
    )
    parser.add_argument(
        "-o", 
        "--output", 
        type=str, 
        default="./export_dataset",
        help="output dir"
    )
    parser.add_argument(
        '--label_only_contain_class_ids', 
        action='store_true', 
        help='Label is contain only class ids'
    )
    return parser

def export(
        input_image_directory: str, 
        input_label_directory: str, 
        output_image_directory: str, 
        output_label_directory: str, 
        class_ids: set,
        label_only_contain_class_ids: bool,
        ):
    if not os.path.isdir(input_image_directory):
        print(f"input_image_directory ERROR -> '{input_image_directory}' 경로를 찾을 수 없습니다.")
        return
    if not os.path.isdir(input_label_directory):
        print(f"input_label_directory ERROR -> '{input_label_directory}' 경로를 찾을 수 없습니다.")
        return

    os.makedirs(output_image_directory, exist_ok=True)
    os.makedirs(output_label_directory, exist_ok=True)
    print(f"output_image_directory -> {output_image_directory}")
    print(f"output_label_directory -> {output_label_directory}")

    input_image_files = [f for f in os.listdir(input_image_directory) if os.path.isfile(os.path.join(input_image_directory, f))]
    # input_image_files = input_image_files[:1]
    
    output_count = 0
    skip_count = 0
    for input_image_file in tqdm(input_image_files):
        image_name, _ = os.path.splitext(os.path.basename(input_image_file))
        
        label_path = os.path.join(input_label_directory, f"{image_name}.txt")
        label_path = label_path if os.path.exists(label_path) else None

        if label_path is None:
            continue
        
        new_labels = []
        with open(label_path, "r") as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(
                    float, line.strip().split()
                )
                class_id = int(class_id)
                if class_id not in class_ids:
                    continue
                new_labels.append((class_id, x_center, y_center, width, height))

        if len(new_labels) == 0:
            print(f"SKIP : label is removed, {input_image_file}")
            skip_count += 1
            continue

        shutil.copy2(
            os.path.join(input_image_directory, input_image_file), 
            os.path.join(output_image_directory, input_image_file))
        
        output_label_file = os.path.join(output_label_directory, f"{image_name}.txt")
        if label_only_contain_class_ids:
            with open(output_label_file, 'w') as f:
                for row in new_labels:
                    line = ' '.join(map(str, row))
                    f.write(line + '\n')
        else:
            shutil.copy2(
                label_path, 
                output_label_file)
        output_count += 1
    
    print(f"------------------------------------")
    print(f"Input count -> {len(input_image_files)}")
    print(f"Output count -> {output_count}")
    print(f"Skip count -> {skip_count}")
    print(f"------------------------------------")

#python export_dataset_by_class_ids.py -ii /media/DATASET/molu/molu/rdd/train/images -il /media/DATASET/molu/molu/rdd/train/labels -c 3 --label_only_contain_class_ids
if __name__ == "__main__":
    args = make_parser().parse_args()
    image_path = args.input_image
    print(f"image path : {image_path}")
    label_path = args.input_label
    print(f"label path : {label_path}")

    class_ids = {int(x) for x in args.class_ids.split(',')}    
    print(f"class ids: {class_ids}")
    label_only_contain_class_ids = args.label_only_contain_class_ids
    print(f"label_only_contain_class_ids: {label_only_contain_class_ids}")

    export(
        input_image_directory=image_path, 
        input_label_directory=label_path, 
        output_image_directory=os.path.join(args.output, "image"), 
        output_label_directory=os.path.join(args.output, "label"), 
        class_ids=class_ids,
        label_only_contain_class_ids=label_only_contain_class_ids
    )
    