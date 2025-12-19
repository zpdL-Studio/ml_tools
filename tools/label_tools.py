from dataclasses import dataclass
import json
import os

def get_labels_from_file(label_path: str):
    labels = []
    with open(label_path, "r") as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(
                float, line.strip().split()
            )
            labels.append((int(class_id), x_center, y_center, width, height))
    return labels
