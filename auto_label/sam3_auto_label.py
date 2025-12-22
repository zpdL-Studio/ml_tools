import argparse
import json
import os

from PIL import Image
import torch
from tqdm import tqdm

from tools.coco_mask_tools import encode_mask_to_rle
from tools.file_tools import get_file_name_and_ext
from tools.config import Sam3Config, load_config
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# from test_sam3_vis import test_sam3_vis, test_sam3_vis2
# from tools.nms_tools import nms_across_all_classes

def sam3_auto_label_image(
    processor: Sam3Processor,
    config: Sam3Config,
    image_file_path: str
) -> Image.Image:
    image = Image.open(image_file_path)
    inference_state = processor.set_image(image)

    processor.reset_all_prompts(inference_state)
    outputs = []
    for text_prompt in config.text_prompts:
        output = processor.set_text_prompt(state=inference_state, prompt=text_prompt.text_prompt)

        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
        for (mask, box, score) in zip(masks, boxes, scores):
            # print(f"mask : {type(mask)}, box: {type(box)}, score: {type(score)}")
            segmentation, area = encode_mask_to_rle(mask.squeeze(0))

            outputs.append({
                "category_id": text_prompt.class_id,
                "score": score.item(),
                "segmentation": segmentation,
                "area": area,
                "iscrowd": 1,
                "bbox": box.tolist(),
            })

    # outputs = nms_across_all_classes(outputs, iou_threshold=0.5)
    return outputs

    # if len(outputs) <= 0:
    #     return None

    # for d in outputs:
    #     label = d['label']
    #     print(f"label: {label}")
    #     print(f"label color_rgb: {label.color_rgb}")
    # boxes_tensor = torch.tensor([d['box'] for d in outputs], dtype=torch.float32)
    # scores_tensor = torch.tensor([d['score'] for d in outputs], dtype=torch.float32)
    # mask_tensor = torch.stack([d['mask'] for d in outputs])
    # texts = [d['label'].name for d in outputs]
    # colors = [d['label'].color_rgb for d in outputs]

    # output_image = test_sam3_vis2(
    #     image=image,
    #     masks=mask_tensor,
    #     boxes=boxes_tensor,
    #     scores=scores_tensor,
    #     texts=texts,
    #     colors=colors,
    # )

    # return output_image
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

# def test_images_text_prompt(
#     processor: Sam3Processor,
#     input_dir_path: str,
#     output_dir_path: str,
#     label_infos: list[LabelInfo],
# ):
#     for f in tqdm(os.listdir(input_dir_path)):
#         image_file = os.path.join(input_dir_path, f)
#         if not os.path.isfile(image_file):
#             continue

#         output_image = test_image_text_prompt(
#             processor=processor,
#             input_file_path=image_file,
#             label_infos=label_infos
#         )

#         if output_image is not None:
#             output_image.save(os.path.join(output_dir_path, f))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("AUTO LABELING WITH SAM3")
    parser.add_argument(
        "-i", 
        "--input", 
        required=True, 
        type=str, 
        help="input your config yaml file path"
    )
    args = parser.parse_args()
    config = load_config(args.input)
    print("--------- CONFIG ------------")
    config.pprint()
    print("")

    labels = config.labels
    config = config.sam3
    
    # print(f"checkpoint_path: {checkpoint_path}, confidence_threshold: {confidence_threshold}")
    # input_path = args.input
    # print(f"input path : {input_path}")
    # output_path = args.output
    # print(f"output path : {output_path}")

    # label_infos = load_label_infos_with_class_ids(
    #     args.label_info, 
    #     [int(x) for x in args.class_ids.split(',')])
    # print(f"label_infos : {label_infos}")

    # if len(label_infos) <= 0:
    #     raise ValueError("label_infos is zero")
        
    model = build_sam3_image_model(checkpoint_path=config.check_point_path)
    processor = Sam3Processor(model, confidence_threshold=config.confidence_threshold)

    if os.path.isdir(config.input_path):
        print("DIR")
        # os.makedirs(config.output_dir_path, exist_ok=True)

        # test_images_text_prompt(
        #     processor=processor,
        #     input_dir_path=input_path,
        #     output_dir_path=output_path,
        #     label_infos=label_infos,
        # )
    else:
        output = sam3_auto_label_image(
            processor=processor,
            config=config,
            image_file_path=config.input_path,
        )
        print(f"output: {output}")
        name, ext = get_file_name_and_ext(config.input_path)

        output_path = os.path.join(config.output_dir_path, f"{name}.json")
        os.makedirs(config.output_dir_path, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f)
    #     if output_path is not None and output_image is not None:
    #         output_image.save(output_path)
