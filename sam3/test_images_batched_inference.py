from matplotlib import pyplot as plt
import torch
import torchvision

from PIL import Image
import requests
from io import BytesIO
import sam3
from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device
import os

from sam3.visualization_utils import plot_results

from sam3.train.data.sam3_image_dataset import InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint
from typing import List

GLOBAL_COUNTER = 1
def create_empty_datapoint():
    """ A datapoint is a single image on which we can apply several queries at once. """
    return Datapoint(find_queries=[], images=[])

def set_image(datapoint, pil_image):
    """ Add the image to be processed to the datapoint """
    w,h = pil_image.size
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h,w])]

def add_text_prompt(datapoint, text_query):
    """ Add a text query to the datapoint """

    global GLOBAL_COUNTER
    # in this function, we require that the image is already set.
    # that's because we'll get its size to figure out what dimension to resize masks and boxes
    # In practice you're free to set any size you want, just edit the rest of the function
    assert len(datapoint.images) == 1, "please set the image first"

    w, h = datapoint.images[0].size
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_query,
            image_id=0,
            object_ids_output=[], # unused for inference
            is_exhaustive=True, # unused for inference
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=GLOBAL_COUNTER,
                original_image_id=GLOBAL_COUNTER,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            )
        )
    )
    GLOBAL_COUNTER += 1
    return GLOBAL_COUNTER - 1

from sam3 import build_sam3_image_model

model = build_sam3_image_model(checkpoint_path="sam3/sam3.pt")

from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI

from sam3.model.position_encoding import PositionEmbeddingSine
transform = ComposeAPI(
    transforms=[
        RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

from sam3.eval.postprocessors import PostProcessImage
postprocessor = PostProcessImage(
    max_dets_per_img=-1,       # if this number is positive, the processor will return topk. For this demo we instead limit by confidence, see below
    iou_type="segm",           # we want masks
    use_original_sizes_box=True,   # our boxes should be resized to the image size
    use_original_sizes_mask=True,   # our masks should be resized to the image size
    convert_mask_to_rle=False, # the postprocessor supports efficient conversion to RLE format. In this demo we prefer the binary format for easy plotting
    detection_threshold=0.5,   # Only return confident detections
    to_cpu=False,
)

# Image 1, we'll use two text prompts

img1 = Image.open("/var/khkim/projects/SAM3/export_dataset/image/0b45b12e6b6d4fba881ff7d441d1b0dd.jpg")
datapoint1 = create_empty_datapoint()
set_image(datapoint1, img1)
id1 = add_text_prompt(datapoint1, "pothole")
id2 = add_text_prompt(datapoint1, "manhole")

datapoint1 = transform(datapoint1)

# Collate then move to cuda
# batch = collate([datapoint1, datapoint2], dict_key="dummy")["dummy"]
batch = collate([datapoint1], dict_key="dummy")["dummy"]
# batch = collate([datapoint2], dict_key="dummy")["dummy"]
batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)

# Forward. Note that the first forward will be very slow due to compilation
output = model(batch)

processed_results = postprocessor.process_results(output, batch.find_metadatas)

plot_results(img1, processed_results[id1])
plt.savefig("output_processed_results_id1.png")

plot_results(img1, processed_results[id2])
plt.savefig("output_processed_results_id2.png")

# this is the prompt "pot"
# plot_results(img2, processed_results[id3])
# plt.savefig("output_processed_results_id3.png")

# # This is the result of the visual prompt. We prompted for the left-most dial, the model correctly found all of them.
# plot_results(img2, processed_results[id4])
# plt.savefig("output_processed_results_id4.png")

# # This is the same as above, but we also added a prompt for the on/off switch
# plot_results(img2, processed_results[id5])
# plt.savefig("output_processed_results_id5.png")

# # this is the prompt "handle". Notice the oven handles that we want to remove
# plot_results(img2, processed_results[id6])
# plt.savefig("output_processed_results_id6.png")

# # This time we added the negative prompt for the oven handle and the unwanted boxes are gone
# plot_results(img2, processed_results[id7])
# plt.savefig("output_processed_results_id7.png")
