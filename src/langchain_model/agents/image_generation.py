import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def remove_small_regions(
    binary: np.ndarray,
    min_region_size: int = 1000,
) -> np.ndarray:
    """Remove white regions smaller than min_region_size pixels from a binary mask."""
    try:
        from scipy import ndimage as ndi
        structure = np.ones((3, 3), dtype=np.uint8)
        labeled, n_labels = ndi.label(binary > 0, structure=structure)
        sizes = ndi.sum(binary > 0, labeled, range(1, n_labels + 1))
        keep = np.array(sizes) >= min_region_size
        keep_mask = keep[labeled - 1]
        keep_mask[labeled == 0] = False
        cleaned = np.where(keep_mask, 255, 0).astype(np.uint8)
    except ImportError:
        import cv2
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        cleaned = np.zeros_like(binary)
        for label_id in range(1, num):
            if stats[label_id, cv2.CC_STAT_AREA] >= min_region_size:
                cleaned[labels == label_id] = 255

    removed = int((binary > 0).sum()) - int((cleaned > 0).sum())
    print(f"  remove_small_regions: kept >= {min_region_size} px "
          f"| removed {removed} white pixels")
    return cleaned


def create_pink_region_mask(
    image: str | bytes,
    output_path: str | None = None,
    hue_min: int = 285,
    hue_max: int = 320,
    saturation_min: float = 0.5,
    value_min: float = 0.5,
    min_region_size: int = 200,
) -> bytes:
    """
    Build a black and white mask of the pink area in an image (path or bytes).

    Pixels are kept when they fall inside the given HSV hue, saturation and
    value range; noise and small regions are then removed. Returns PNG bytes
    and also writes them to output_path if given.
    """
    if isinstance(image, (str, bytes.__class__)) and not isinstance(image, bytes):
        img = Image.open(image).convert("RGB")
    else:
        img = Image.open(io.BytesIO(image)).convert("RGB")

    rgb = np.array(img, dtype=np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    cmax  = np.max(rgb, axis=2)
    cmin  = np.min(rgb, axis=2)
    delta = cmax - cmin

    hue = np.zeros_like(cmax)
    mask_r = (cmax == r) & (delta != 0)
    mask_g = (cmax == g) & (delta != 0)
    mask_b = (cmax == b) & (delta != 0)
    hue[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / delta[mask_r])) % 360
    hue[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120) % 360
    hue[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240) % 360

    with np.errstate(invalid="ignore"):
        saturation = np.where(cmax == 0, 0.0, delta / cmax)
    value = cmax

    pink_mask = (
        (hue >= hue_min) & (hue <= hue_max) &
        (saturation >= saturation_min) &
        (value >= value_min)
    )

    binary = np.where(pink_mask, 255, 0).astype(np.uint8)
    print(f"  After colour threshold : {(binary > 0).sum()} white pixels")

    try:
        import cv2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        print(f"  After morph opening   : {(binary > 0).sum()} white pixels")
    except ImportError:
        pass

    binary = remove_small_regions(binary, min_region_size=min_region_size)

    mask_img = Image.fromarray(binary, mode="L")
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    mask_bytes = buf.getvalue()

    if output_path:
        with open(output_path, "wb") as f:
            f.write(mask_bytes)
        print(f"  Mask saved → {output_path}  (final white pixels: {(binary > 0).sum()})")

    return mask_bytes


def show_image(image_bytes: bytes, title: str = "Image", figsize: tuple = (8, 6)) -> None:
    """Display an encoded image in a matplotlib window."""
    image = Image.open(io.BytesIO(image_bytes))
    img_array = np.array(image)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img_array)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()