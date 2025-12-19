import json
from dataclasses import dataclass

@dataclass
class LabelInfo:
    class_id: int
    name: str
    text_prompt: str
    color_str: str

    @property
    def color_rgb(self) -> tuple[int, int, int]:
        return tuple(map(int, self.color_str.split(',')))
    
def load_label_infos(json_path: str) -> list[LabelInfo]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

        label_list = []
        for item in data:
            label = LabelInfo(
                class_id=item["class_id"],
                name=item["name"], 
                text_prompt=item["text_prompt"],
                color_str=item["color"]
            )
            label_list.append(label)
            
        return label_list
    raise ValueError(f"load_label_infos -> not open path : {json_path}")

def load_label_infos_with_class_ids(json_path: str, class_ids: list[int]) -> list[LabelInfo]:
    label_infos = load_label_infos(json_path)
    results = []
    for label in label_infos:
        if label.class_id in class_ids:
            results.append(label)
    return results
    