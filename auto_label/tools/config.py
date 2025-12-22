from typing import Self
import argparse
from dataclasses import dataclass, asdict
import pprint

import yaml

@dataclass
class LabelConfig:
    class_id: int
    name: str
    color_str: str

    @property
    def color_rgb(self) -> tuple[int, int, int]:
        return tuple(map(int, self.color_str.split(',')))
    
    @classmethod
    def from_json(cls, json) -> Self:
        return cls(
            class_id=json["class_id"],
            name=json["name"], 
            color_str=json["color"]
        )
    
    @staticmethod
    def from_json_list(json_list) -> list[Self]:
        results = []
        for json in json_list:
            results.append(LabelConfig.from_json(json))
        return results

@dataclass
class Sam3TextPrompt:
    class_id: int
    text_prompt: str

    @classmethod
    def from_json(cls, json) -> Self:
        return cls(
            class_id=json["class_id"],
            text_prompt=json["text_prompt"]
        )
    
    @staticmethod
    def from_json_list(json_list) -> list[Self]:
        results = []
        for json in json_list:
            results.append(Sam3TextPrompt.from_json(json))
        return results

@dataclass
class Sam3NmsClassIds:
    class_ids: list[int]
    nms_threshold: float

    @classmethod
    def from_json(cls, json) -> Self:
        return cls(
            class_ids=json["class_ids"],
            nms_threshold=json["nms_threshold"]
        )
    
    @staticmethod
    def from_json_list(json_list) -> list[Self]:
        if json_list is None:
            return []

        results = []
        for json in json_list:
            results.append(Sam3NmsClassIds.from_json(json))
        return results

@dataclass
class Sam3Config:
    check_point_path: str
    input_path: str
    output_dir_path: str
    confidence_threshold: float
    text_prompts: list[Sam3TextPrompt]
    nms_class_ids: list[Sam3NmsClassIds]
    
    @classmethod
    def from_json(cls, json) -> Self:
        return cls(
            check_point_path=json["check_point_path"],
            input_path=json["input_path"],
            output_dir_path=json["output_dir_path"],
            confidence_threshold=json["confidence_threshold"],
            text_prompts=Sam3TextPrompt.from_json_list(json["text_prompts"]),
            nms_class_ids=Sam3NmsClassIds.from_json_list(json.get("nms_class_ids"))
        )
    
@dataclass
class AutoLabelingConfig:
    labels: list[LabelConfig]
    sam3: Sam3Config

    @classmethod
    def from_json(cls, json) -> Self:
        return cls(
            labels=LabelConfig.from_json_list(json["label"]),
            sam3=Sam3Config.from_json(json["sam3"])
        )
    
    def pprint(self):
        pprint.pprint(asdict(self), width=20, indent=2)

def load_config(json_path) -> AutoLabelingConfig:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
        return AutoLabelingConfig.from_json(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("AUTO LABELING CONFIG")
    parser.add_argument(
        "-i", 
        "--input", 
        required=True, 
        type=str, 
        help="input your config yaml file path"
    )
    args = parser.parse_args()
    config = load_config(args.input)
    pprint.pprint(asdict(config), width=20, indent=2)
