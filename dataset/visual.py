import os

import numpy as np
import torch
import cv2 as cv
from torchvision.transforms import functional as T

from face_dataset import MXFaceDataset
from palm_dataset import PalmDataset

def gen_random_masking(grid_size=7, image_size=112, mask_ratio=0.4):
    """Generate a coarse grid mask (grid_size x grid_size) and expand to image_size."""
    grid_mask = torch.rand((grid_size, grid_size)) > mask_ratio
    patch_size = image_size // grid_size
    full_mask = grid_mask.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)
    return grid_mask, full_mask


def denorm(img_tensor):
    return img_tensor * 0.5 + 0.5


def to_bgr(img_tensor):
    img = (denorm(img_tensor).numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
    return cv.cvtColor(img, cv.COLOR_RGB2BGR)


if __name__ == "__main__":
    os.makedirs("vis_outputs", exist_ok=True)

    # dataset = MXFaceDataset("/data/tx/MS1MV3")
    dataset = PalmDataset(
            "/data/tx/palm_data/IITD/resized",
            "/data/tx/palm_data/IITD/annotations/fourk_train.json",
            transform=None,
        )
    img = dataset[102][0]
    img = T.resize(img, (112, 112))

    grid_mask, mask = gen_random_masking()

    masked_img = img.clone()
    masked_img[:, ~mask] = 0

    original_bgr = to_bgr(img)
    mask_gray = (mask.float().numpy() * 255).astype(np.uint8)
    mask_bgr = cv.cvtColor(mask_gray, cv.COLOR_GRAY2BGR)
    masked_bgr = to_bgr(masked_img)

    cv.imwrite(os.path.join("vis_outputs", "original.png"), original_bgr, [cv.IMWRITE_PNG_COMPRESSION, 0])
    cv.imwrite(os.path.join("vis_outputs", "mask.png"), mask_bgr, [cv.IMWRITE_PNG_COMPRESSION, 0])
    cv.imwrite(os.path.join("vis_outputs", "masked.png"), masked_bgr, [cv.IMWRITE_PNG_COMPRESSION, 0])



