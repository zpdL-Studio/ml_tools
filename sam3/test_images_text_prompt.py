import argparse
import os

from PIL import Image
from tqdm import tqdm
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from test_sam3_vis import test_sam3_vis

def make_parser():
    parser = argparse.ArgumentParser("TEST IMAGES TEXT PROMPT")
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
        "-t", 
        "--text_prompt", 
        type=str,
        default=None,
        help="input text prompt"
    )
    return parser

def test_image_text_prompt(
    processor: Sam3Processor,
    input_file_path: str,
    text_prompt: str,
) -> Image.Image:
    image = Image.open(input_file_path)
    inference_state = processor.set_image(image)

    processor.reset_all_prompts(inference_state)
    output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)

    # Get the masks, bounding boxes, and scores
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    if scores.shape[0] <= 0:
        return

    output = test_sam3_vis(
        image=image,
        masks=masks,
        boxes=boxes,
        scores=scores,
        text=text_prompt,
    )

    return output

def test_images_text_prompt(
    processor: Sam3Processor,
    input_dir_path: str,
    output_dir_path: str,
    text_prompt: str,
):
    for f in tqdm(os.listdir(input_dir_path)):
        image_file = os.path.join(input_dir_path, f)
        if not os.path.isfile(image_file):
            continue

        output_image = test_image_text_prompt(
            processor=processor,
            input_file_path=image_file,
            text_prompt=text_prompt
        )

        if output_image is not None:
            output_image.save(os.path.join(output_dir_path, f))

# python test_images_text_prompt.py -i ./export_dataset/image -o ./test_images_text_prompt -t pothole        
# python test_images_text_prompt.py -i ./export_dataset/image/0b45b12e6b6d4fba881ff7d441d1b0dd.jpg -o ./test_images_text_prompt_output.jpg -t "road crack. asphalt fracture. pavement damage."
# python test_images_text_prompt.py -i ./export_dataset/image/0b45b12e6b6d4fba881ff7d441d1b0dd.jpg -o ./test_images_text_prompt_output.jpg -t manhole
# python test_images_text_prompt.py -i ./export_dataset/image/0b45b12e6b6d4fba881ff7d441d1b0dd.jpg -o ./test_images_text_prompt_output.jpg -t "pothole, not manhole"
if __name__ == "__main__":
    args = make_parser().parse_args()

    input_path = args.input
    print(f"input path : {input_path}")
    output_path = args.output
    print(f"output path : {output_path}")

    text_prompt = args.text_prompt
    print(f"text prompt : {text_prompt}")

    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=0.3)

    if os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)

        test_images_text_prompt(
            processor=processor,
            input_dir_path=input_path,
            output_dir_path=output_path,
            text_prompt=text_prompt,
        )
    else:
        output_image = test_image_text_prompt(
            processor=processor,
            input_file_path=input_path,
            text_prompt=text_prompt
        )

        if output_path is not None:
            output_image.save(output_path)
