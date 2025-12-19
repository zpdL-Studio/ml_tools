import argparse
import os

from PIL import Image
import torch
from tqdm import tqdm
from lable_info import LabelInfo, load_label_infos_with_class_ids
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from test_sam3_vis import test_sam3_vis, test_sam3_vis2
from tools.nms_tools import nms_across_all_classes

def make_parser():
    parser = argparse.ArgumentParser("TEST IMAGES TEXT PROMPT")
    parser.add_argument(
        "-pt", 
        "--checkpoint", 
        required=True, 
        type=str, 
        help="input sam3 checkpoint path"
    )
    parser.add_argument(
        "-i", 
        "--input", 
        required=True, 
        type=str, 
        help="input your image path"
    )
    parser.add_argument(
        "-o", 
        "--output", 
        type=str, 
        default=None,
        help="output your label dir path"
    )
    parser.add_argument(
        "--label_info", 
        required=True,
        type=str,
        help="input label info"
    )
    parser.add_argument(
        "--class_ids", 
        required=True,
        type=str,
        help="Specific class IDs to select from the label info (comma-separated, e.g., '1,2,5')."
    )
    parser.add_argument(
        "--confidence_threshold", 
        type=float,
        default=0.3,
        help="input text prompt, divder ,"
    )
    return parser

def test_image_text_prompt(
    processor: Sam3Processor,
    input_file_path: str,
    label_infos: list[LabelInfo],
) -> Image.Image:
    image = Image.open(input_file_path)
    inference_state = processor.set_image(image)

    processor.reset_all_prompts(inference_state)
    outputs = []
    for label_info in label_infos:
        output = processor.set_text_prompt(state=inference_state, prompt=label_info.text_prompt)

        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
        for (mask, box, score) in zip(masks, boxes, scores):
            # print(f"mask : {type(mask)}, box: {type(box)}, score: {type(score)}")
            outputs.append({
                "label": label_info,
                "box": box.tolist(),
                "score": score.item(),
                "mask": mask
            })
    print(f"outputs 1: {outputs}")
    outputs = nms_across_all_classes(outputs, iou_threshold=0.5)

    if len(outputs) <= 0:
        return None

    for d in outputs:
        label = d['label']
        print(f"label: {label}")
        print(f"label color_rgb: {label.color_rgb}")
    boxes_tensor = torch.tensor([d['box'] for d in outputs], dtype=torch.float32)
    scores_tensor = torch.tensor([d['score'] for d in outputs], dtype=torch.float32)
    mask_tensor = torch.stack([d['mask'] for d in outputs])
    texts = [d['label'].name for d in outputs]
    colors = [d['label'].color_rgb for d in outputs]

    output_image = test_sam3_vis2(
        image=image,
        masks=mask_tensor,
        boxes=boxes_tensor,
        scores=scores_tensor,
        texts=texts,
        colors=colors,
    )

    return output_image
    # Get the masks, bounding boxes, and scores
    # masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    # if scores.shape[0] <= 0:
    #     return

    # output_image = test_sam3_vis(
    #     image=image,
    #     masks=masks,
    #     boxes=boxes,
    #     scores=scores,
    #     text=text_prompts,
    # )

    # return output_image

def test_images_text_prompt(
    processor: Sam3Processor,
    input_dir_path: str,
    output_dir_path: str,
    label_infos: list[LabelInfo],
):
    for f in tqdm(os.listdir(input_dir_path)):
        image_file = os.path.join(input_dir_path, f)
        if not os.path.isfile(image_file):
            continue

        output_image = test_image_text_prompt(
            processor=processor,
            input_file_path=image_file,
            label_infos=label_infos
        )

        if output_image is not None:
            output_image.save(os.path.join(output_dir_path, f))

# python test_images_text_prompt.py -i ./export_dataset/image -o ./test_images_text_prompt -t pothole        
# python test_images_text_prompt.py -i ./export_dataset/image/0b45b12e6b6d4fba881ff7d441d1b0dd.jpg -o ./test_images_text_prompt_output.jpg -t "road crack. asphalt fracture. pavement damage."
# python test_images_text_prompt.py -i ./export_dataset/image/0b45b12e6b6d4fba881ff7d441d1b0dd.jpg -o ./test_images_text_prompt_output.jpg -t manhole
# python test_images_text_prompt.py -i ./export_dataset/image/0b45b12e6b6d4fba881ff7d441d1b0dd.jpg -o ./test_images_text_prompt_output.jpg -t "pothole. not manhole"
if __name__ == "__main__":
    args = make_parser().parse_args()
    checkpoint_path = args.checkpoint
    confidence_threshold = args.confidence_threshold
    print(f"checkpoint_path: {checkpoint_path}, confidence_threshold: {confidence_threshold}")
    input_path = args.input
    print(f"input path : {input_path}")
    output_path = args.output
    print(f"output path : {output_path}")

    label_infos = load_label_infos_with_class_ids(
        args.label_info, 
        [int(x) for x in args.class_ids.split(',')])
    print(f"label_infos : {label_infos}")

    if len(label_infos) <= 0:
        raise ValueError("label_infos is zero")
        
    model = build_sam3_image_model(checkpoint_path=checkpoint_path)
    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)

    if os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)

        test_images_text_prompt(
            processor=processor,
            input_dir_path=input_path,
            output_dir_path=output_path,
            label_infos=label_infos,
        )
    else:
        output_image = test_image_text_prompt(
            processor=processor,
            input_file_path=input_path,
            label_infos=label_infos
        )

        if output_path is not None and output_image is not None:
            output_image.save(output_path)
