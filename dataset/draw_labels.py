import argparse
import os

from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from tools.file_tools import read_file_lines
from tools.label_tools import get_labels_from_file
from tools.vis_tools import draw_bbox

def make_parser():
    parser = argparse.ArgumentParser("DRAW LABELS")
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
        help="input your image dir path"
    )
    parser.add_argument(
        "-o", 
        "--output", 
        type=str, 
        default="./draw_labels",
        help="output your label dir path"
    )
    parser.add_argument(
        "-l", 
        "--label", 
        type=str,
        default=None,
        help="input your label name file"
    )
    return parser

def draw_label(
    image_file: str, 
    label_file: str,
    label_names: list,
) -> Image.Image:
    labels = get_labels_from_file(label_file)

    image = Image.open(image_file)
    width, height = image.size

    draw = ImageDraw.Draw(image)

    for label in labels:
        color = (0, 255, 0)
        class_id = label[0]
        bbox = label[1:5]
        text = label_names[class_id] if label_names is not None else f'{class_id}'

        draw_bbox(
            draw=draw,
            box=[bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height],
            box_format="CxCyWH",
            color=color,
            width=3,
            text=text,
        )
    
    return image

def draw_labels(
    input_image_path: str,
    input_label_path: str,
    output_path: str,
    label_names: list,
):
    if not os.path.isdir(input_image_path):
        print(f"input_image_path ERROR -> '{input_image_path}' 경로를 찾을 수 없습니다.")
        return
    if not os.path.isdir(input_label_path):
        print(f"input_label_path ERROR -> '{input_label_path}' 경로를 찾을 수 없습니다.")
        return
    os.makedirs(output_path, exist_ok=True)

    for f in tqdm(os.listdir(input_image_path)):
        image_file = os.path.join(input_image_path, f)
        if not os.path.isfile(image_file):
            continue
        
        image_name, _ = os.path.splitext(os.path.basename(image_file))

        label_path = os.path.join(input_label_path, f"{image_name}.txt")
        label_path = label_path if os.path.exists(label_path) else None

        if label_path is None:
            continue

        output = draw_label(
            image_file=image_file,
            label_file=label_path,
            label_names=label_names,
        )
        output.save(os.path.join(output_path, f))
        
# python draw_labels.py -ii ./export_dataset/image -il ./export_dataset/label -l ./labels/rdd_label.txt
if __name__ == "__main__":
    args = make_parser().parse_args()

    label_names = None
    if args.label is not None:
        label_names = read_file_lines(args.label)
    print(f"label_names -> {label_names}")

    draw_labels(
        input_image_path=args.input_image,
        input_label_path=args.input_label,
        output_path=args.output,
        label_names=label_names
    )

# # Load the model
# # /home/khkim/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt
# # model = build_sam3_image_model()
# # processor = Sam3Processor(model)
# # Load an image
# image = Image.open("/home/khkim/projects/SAM3/tests/export_dataset/image/0a1cab49ec4f4a00ad80f9d7790b67ee.jpg")
# width, height = image.size

# labels = get_labels_from_file("/home/khkim/projects/SAM3/tests/export_dataset/label/0a1cab49ec4f4a00ad80f9d7790b67ee.txt")
# label_image = Image.open("/home/khkim/projects/SAM3/tests/export_dataset/image/0a1cab49ec4f4a00ad80f9d7790b67ee.jpg")
# label_image_draw = ImageDraw.Draw(label_image)
# width, height = label_image.size

# for label in labels:
#     color = (0, 255, 0)
#     bbox = label[1:5]
#     draw_bbox(
#         draw=label_image_draw,
#         box=[label[1] * width, label[2] * height, label[3] * width, label[4] * height],
#         box_format="CxCyWH",
#         color=color,
#         width=3,
#         text="EXAMPLEgg",
#     )
# label_image.save("output_label_image.jpg")


# inference_state = processor.set_image(image)
# print(f"inference_state: {inference_state.keys()}")
# # Prompt the model with text
# processor.reset_all_prompts(inference_state)
# output = processor.set_text_prompt(state=inference_state, prompt="pothole")

# # plot_results(image, inference_state)
# # plt.savefig("output.png")
# # plt.close()

# # Get the masks, bounding boxes, and scores
# masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

# # print(f"masks: {masks}")
# print(f"boxes: {boxes}")
# print(f"scores: {scores}")
# output_image = vis(image, masks, boxes, scores)
# output_image.save("output_image.jpg")



# plt.figure(figsize=(12, 8))
# plt.imshow(image)
# nb_objects = len(output["scores"])
# print(f"found {nb_objects} object(s)")
# for i in range(nb_objects):
#     color = COLORS[i % len(COLORS)]
#     plot_mask(output["masks"][i].squeeze(0).cpu(), color=color)
#     w, h = image.size
#     prob = output["scores"][i].item()
#     plot_bbox(
#         h,
#         w,
#         output["boxes"][i].cpu(),
#         text=f"(id={i}, {prob=:.2f})",
#         box_format="XYXY",
#         color=color,
#         relative_coords=False,
#     )
# plt.savefig("output.png")
# print(f"Plot saved to tests/output.png")
# plt.close()