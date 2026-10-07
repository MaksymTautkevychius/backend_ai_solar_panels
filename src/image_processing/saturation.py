import cv2
import numpy as np
from PIL import Image
from pathlib import Path

def boost_saturation(image: Image.Image, scale: float = 1.8) -> Image.Image:
    """Return a copy of an RGB image with saturation multiplied by scale."""
    arr = np.array(image, dtype=np.uint8).copy()
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)
    boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return Image.fromarray(boosted)




def pink_to_bw_mask(
    image: str | np.ndarray,
    hue_range: tuple[int, int] = (140, 170),
    sat_min: int = 80,
    val_min: int = 80,
    remove_small_blobs: bool = True,
    min_blob_area: int = 500,) -> np.ndarray:
    """
    Return a 0/255 mask of the pink pixels in a BGR image or image file,
    optionally dropping blobs smaller than min_blob_area.
    """
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image))
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {image}")
    else:
        bgr = image.copy()


    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([hue_range[0], sat_min, val_min], dtype=np.uint8)
    upper = np.array([hue_range[1], 255, 255],         dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)


    if remove_small_blobs:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned = np.zeros_like(mask)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= min_blob_area:
                cleaned[labels == label] = 255
        mask = cleaned

    return mask