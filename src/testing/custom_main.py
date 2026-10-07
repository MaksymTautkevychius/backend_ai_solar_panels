import torch
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import numpy as np

from PIL import Image
from huggingface_hub import login

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results

login(token=os.environ.get("HF_TOKEN"))

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

def load_image_rgb(image_path: str) -> np.ndarray:
    """Load an image as a contiguous RGB array."""
    img = Image.open(image_path).convert("RGB")
    arr = np.ascontiguousarray(np.array(img))
    print(f"[load_image_rgb] {image_path} → shape={arr.shape}, dtype={arr.dtype}")
    return arr

def show_image(image_path: str):
    """Display an image file in a matplotlib window."""
    img = load_image_rgb(image_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(image_path.split('/')[-1])
    plt.tight_layout()
    plt.show()

def enhance_contrast_color(image_path: str) -> dict:
    """
    Plot the image next to histogram equalization, CLAHE and saturation boost versions.
    Returns all four images as RGB arrays.
    """
    img_rgb = np.ascontiguousarray(load_image_rgb(image_path))

    def histogram_equalization(img: np.ndarray) -> np.ndarray:
        channels = cv2.split(img)
        eq_channels = [cv2.equalizeHist(c) for c in channels]
        return cv2.merge(eq_channels)

    def apply_clahe(img: np.ndarray, clip_limit=3.0, tile_grid=(4, 4)) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        channels = cv2.split(img)
        clahe_channels = [clahe.apply(c) for c in channels]
        return cv2.merge(clahe_channels)

    def boost_saturation(img: np.ndarray, scale=1.8) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    hist_eq   = histogram_equalization(img_rgb)
    clahe     = apply_clahe(img_rgb)
    sat_boost = boost_saturation(img_rgb)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Contrast & Color Enhancement", fontsize=14, fontweight='bold')

    for ax, (data, title) in zip(axes, [
        (img_rgb,   "Original"),
        (hist_eq,   "Histogram Equalization"),
        (clahe,     "CLAHE"),
        (sat_boost, "Saturation Boost"),
    ]):
        ax.imshow(data)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    return {
        "original":   img_rgb,
        "hist_eq":    hist_eq,
        "clahe":      clahe,
        "sat_boost":  sat_boost,
    }

def show_spectral_analysis(image_path: str):
    """Plot the RGB channels, a simulated NIR channel and the HSV channels of an image."""
    img = load_image_rgb(image_path).astype(np.float32) / 255.0

    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    nir_simulated = np.clip(2 * r - g - b, 0, 1)

    hsv        = mcolors.rgb_to_hsv(img[:, :, :3])
    hue        = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value      = hsv[:, :, 2]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Spectral & Channel Analysis", fontsize=14, fontweight='bold')

    for ax, (data, title, cmap) in zip(axes.flatten(), [
        (img,           "Original RGB",      None),
        (r,             "R Channel",         'Reds'),
        (g,             "G Channel",         'Greens'),
        (b,             "B Channel",         'Blues'),
        (nir_simulated, "Simulated NIR",     'RdYlGn'),
        (hue,           "HSV — Hue",         'hsv'),
        (saturation,    "HSV — Saturation",  'plasma'),
        (value,         "HSV — Value",       'gray'),
    ]):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def run_sam3_inference(
    image_path: str,
    prompt: str = "roof segment",
    output_path: str = "./examples/results/result.png",
    confidence_threshold: float = 0.5,
):
    """Segment an image with a SAM3 text prompt, save the plotted result and return the inference state."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    bpe_path = "./sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    model    = build_sam3_image_model(bpe_path=bpe_path)

    image        = Image.open(image_path).convert("RGB")
    width, height = image.size
    print(f"[run_sam3_inference] Image size: {width}x{height}, prompt: '{prompt}'")

    processor       = Sam3Processor(model, confidence_threshold=confidence_threshold)
    inference_state = processor.set_image(image)

    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)

    plot_results(image, inference_state)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.show()
    print(f"[run_sam3_inference] Result saved to: {output_path}")

    scores = inference_state.get("scores", [])
    labels = inference_state.get("labels", [])
    print(f"[run_sam3_inference] Found {len(labels)} object(s)")
    print(f"[run_sam3_inference] Confidence scores: {scores}")
    print(f"[run_sam3_inference] Labels: {labels}")

    return inference_state


def run_sam3_inference2(
    image_path: str,
    output_path: str = "./examples/results/result.png",
    confidence_threshold: float = 0.5,
    box: tuple[int, int, int, int] | None = None,
    negative_box: tuple[int, int, int, int] | None = None,
    text_prompt: str | None = "roof",
):
    """
    Segment an 800x600 image with SAM3 using a box prompt (x1, y1, x2, y2 in pixels)
    and an optional text prompt. Saves the plotted result and returns the inference state.
    """
    IMAGE_W, IMAGE_H = 800, 600

    if box is None:
        box = (425, 290, 570, 435)

    if negative_box is None:
        negative_box = (200, 350, 650, 560)

    def to_cxcywh_norm(b: tuple[int, int, int, int]) -> list[float]:
        """Convert a pixel box (x1, y1, x2, y2) to normalized [cx, cy, w, h]."""
        x1, y1, x2, y2 = b
        return [
            ((x1 + x2) / 2) / IMAGE_W,
            ((y1 + y2) / 2) / IMAGE_H,
            (x2 - x1) / IMAGE_W,
            (y2 - y1) / IMAGE_H,
        ]

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    bpe_path = "./sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    model    = build_sam3_image_model(bpe_path=bpe_path)

    image = Image.open(image_path).convert("RGB")
    #w, h  = image.size
    #if (w, h) != (IMAGE_W, IMAGE_H):
    #    raise ValueError(
    #        f"Expected {IMAGE_W}×{IMAGE_H} image, got {w}×{h}: {image_path}"
      #  )

    print(f"[run_sam3_inference2] Image: {w}×{h}")
    print(f"[run_sam3_inference2] Positive box (pixels): {box}")
    print(f"[run_sam3_inference2] Negative box (pixels): {negative_box}")
    print(f"[run_sam3_inference2] Text prompt: {text_prompt!r}")

    processor       = Sam3Processor(model, confidence_threshold=confidence_threshold)
    inference_state = processor.set_image(image)
    processor.reset_all_prompts(inference_state)

    if text_prompt is not None:
        inference_state = processor.set_text_prompt(
            state=inference_state,
            prompt=text_prompt,
        )

    inference_state = processor.add_geometric_prompt(
        box=to_cxcywh_norm(box),
        label=True,
        state=inference_state,
    )

  # inference_state = processor.add_geometric_prompt(
  #     box=to_cxcywh_norm(negative_box),
  #     label=False,
  #     state=inference_state,
  # )

    plot_results(image, inference_state)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.show()
    print(f"[run_sam3_inference2] Result saved to: {output_path}")

    scores = inference_state.get("scores", [])
    labels = inference_state.get("labels", [])
    print(f"[run_sam3_inference2] Found {len(labels)} object(s)")
    print(f"[run_sam3_inference2] Confidence scores: {scores}")
    print(f"[run_sam3_inference2] Labels: {labels}")

    return inference_state

def boost_saturation(image_url: str, output_path: str, scale: float = 1.8) -> str:
    """Boost the saturation of an image (URL or local path), save it and return the absolute path."""
    import requests
    from io import BytesIO

    if image_url.startswith("http://") or image_url.startswith("https://"):
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_url).convert("RGB")

    arr = np.array(image, dtype=np.uint8).copy()
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)
    boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(boosted).save(output_path)

    return os.path.abspath(output_path)



if __name__ == "__main__":
    image = boost_saturation("./uploads/2134.png","./examples/results/resultsature.png")
    inference_state = run_sam3_inference(
        image_path=image,
        output_path="./examples/results/result.png",
        confidence_threshold=0.7,
    )