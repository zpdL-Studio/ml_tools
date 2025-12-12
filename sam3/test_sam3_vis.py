import torch

from PIL import Image, ImageDraw
from tools.vis_tools import draw_mask, draw_bbox

def test_sam3_vis(
    image: Image.Image,
    masks: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    text: str,
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
            text=f"{text}:{score}",
        )

    return image.convert("RGB")