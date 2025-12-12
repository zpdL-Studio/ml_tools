from matplotlib import pyplot as plt
import torch
#################################### For Image ####################################
from PIL import Image, ImageDraw, ImageFont
from sam3.model_builder import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results

from tools.vis_tools import draw_mask, draw_bbox

def vis(
    image: Image.Image,
    masks: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
) -> Image.Image:
    for mask, bbox, score in zip(masks, boxes, scores):
        color = (0, 255, 0)
        image = draw_mask(
            image=image, 
            mask=mask.squeeze(0).cpu(), 
            color=color
        )
        draw_bbox(
            draw=ImageDraw.Draw(image),
            box=bbox,
            color=color,
            width=3,
            text="EXAMPLEgg",
        )

    return image.convert("RGB")

# Load the model
# /home/khkim/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
image = Image.open("/home/khkim/projects/SAM3/tests/export_dataset/image/0b45b12e6b6d4fba881ff7d441d1b0dd.jpg")
width, height = image.size

# labels = get_labels_from_file("/home/khkim/projects/SAM3/tests/export_dataset/label/0b45b12e6b6d4fba881ff7d441d1b0dd.txt")
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


inference_state = processor.set_image(image)
print(f"inference_state: {inference_state.keys()}")
# Prompt the model with text
processor.reset_all_prompts(inference_state)
output = processor.set_text_prompt(state=inference_state, prompt="road damage")

# plot_results(image, inference_state)
# plt.savefig("output.png")
# plt.close()

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

# print(f"masks: {masks}")
print(f"boxes: {boxes}")
print(f"scores: {scores}")
output_image = vis(image, masks, boxes, scores)
output_image.save("output_image.jpg")

# box_input_xywh = boxes #torch.tensor([480.0, 290.0, 110.0, 360.0]).view(-1, 4)
# box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh)

# norm_box_cxcywh = normalize_bbox(box_input_cxcywh, width, height).flatten().tolist()
# print("Normalized box input:", norm_box_cxcywh)

# processor.reset_all_prompts(inference_state)
# inference_state = processor.add_geometric_prompt(
#     state=inference_state, box=norm_box_cxcywh, label=True
# )
# print(f"inference_state: {inference_state}")
# print(f"inference_state.keys(): {inference_state.keys()}")
# print(f"inference_state[geometric_prompt]: {inference_state["geometric_prompt"]}")
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