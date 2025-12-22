import os
from pathlib import Path

def get_file_name_and_ext(file_path: str):
    path = Path(file_path)

    return path.stem, path.suffix

def get_files_by_ext(file_path: str, exts: list):
    files = []
    for maindir, subdir, file_name_list in os.walk(file_path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in exts:
                files.append(apath)
    return files


def read_file_lines(file_path: str):
    lines = []
    with open(file_path, "r") as f:
        for line in f:
            print(f"type(line): {type(line)}, {line}")
            lines.append(line.strip())
    return lines

if __name__ == "__main__":
    results = read_file_lines("./labels/rdd_label.txt")
    print(f"read_file_lines -> {results}")