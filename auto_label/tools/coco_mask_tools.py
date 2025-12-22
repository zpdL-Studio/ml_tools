import torch
import numpy as np
from pycocotools import mask as mask_utils

def encode_mask_to_rle(mask: torch.Tensor):
    mask_np = mask.cpu().numpy().astype(np.uint8)

    mask_fortran = np.asfortranarray(mask_np)
    rle = mask_utils.encode(mask_fortran)
    area = mask_utils.area(rle)

    rle['counts'] = rle['counts'].decode('utf-8')

    print(f"rle :{rle}, area: {area}")

    return rle, int(area)
