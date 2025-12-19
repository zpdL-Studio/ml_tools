from PIL import Image, ImageDraw, ImageFont
import numpy as np

def draw_mask(image: Image.Image, mask, color: tuple, opacity=0.1):
    im_h, im_w = mask.shape
    print(f"im_h : {im_h}, im_w: {im_w}")
    print(f"mask : {mask}")
    mask_img = np.zeros((im_h, im_w, 4), dtype=np.uint8)
    r, g, b = color
    a = int(255 * opacity)
    mask_img[mask] = [r, g, b, a]
    # mask_img[..., :3] = to_rgb(color)
    # mask_img[..., 3] = mask * 0.5

    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    mask_pil = Image.fromarray(mask_img, mode="RGBA")
    image.alpha_composite(mask_pil)

    return image

def draw_bbox_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    text_color = (0, 0, 0),
    font_size: float = 12,
    padding: float = 4,
    fill_color = (255, 255, 255),
    outline_color = (0, 0, 0),
    outline_width = 1,
):
    font = ImageFont.load_default(font_size)
    ascent, descent = font.getmetrics()
    text_height = ascent + descent
    text_width = font.getlength(text)
    x, y = xy
    # x1, y1, x2, y2 = draw.textbbox(xy, text, font=font, anchor='lt')
    bg_bbox = (x, y - text_height - padding * 2, x + text_width + padding * 2, y)
    draw.rectangle(
        bg_bbox,
        fill=fill_color,
        outline=outline_color,
        width=outline_width
    )
    draw.text(
        (bg_bbox[0] + padding, bg_bbox[1] + padding), 
        text, 
        font=font, 
        fill=text_color,
        stroke_width=0.3,
        anchor='la')

def draw_bbox(
    draw: ImageDraw.ImageDraw,
    box,
    box_format="XYXY",
    color = tuple[int, int, int],
    width: int = 1,
    text: str = None,
):
    if box_format == "XYXY":
        x1, y1, x2, y2 = box
    elif box_format == "XYWH":
        x1, y1, w, h = box
        x2 = x1 + w
        y2 = y1 + h
    elif box_format == "CxCyWH":
        cx, cy, w, h = box
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
    else:
        raise RuntimeError(f"Invalid box_format {box_format}")

    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if text is not None:
        draw_bbox_text(
            draw=draw,
            xy=(x1, y1),
            text=text,
            text_color=color,
        )
