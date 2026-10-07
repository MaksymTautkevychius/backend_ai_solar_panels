import requests
from PIL import Image

import os
import io
import requests
from dotenv import load_dotenv
from PIL import Image, ImageFilter

load_dotenv()
API_KEY = os.environ.get("STABILITY_KEY")

PROMPT =(
    "Fill the entire masked roof slope with a dark blue-black photovoltaic solar panel array. "
    "The entire masked region should become solar panels, edge to edge. "
    "Panels are rectangular, arranged in a neat grid, installed flat on the roof, "
    "following the roof perspective and angle. "
    "Photorealistic aerial satellite image, same blur, same lighting, same shadows. "
    "Only modify the masked area."
)
NEGATIVE_PROMPT = (
    "grass, lawn, green area, roof shingles, single small panel, tiny object, "
    "changed building, changed street, cartoon, illustration, floating object"
)


def binarize_mask(mask):
    """Convert a mask to pure black and white."""
    mask = mask.convert("L")
    return mask.point(lambda p: 255 if p > 127 else 0)


def get_mask_bbox(mask, padding=80):
    """Return the bounding box of the mask's white area, padded and clamped to the image."""
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("Mask is empty. No white pixels found.")

    left, top, right, bottom = bbox

    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(mask.width, right + padding)
    bottom = min(mask.height, bottom + padding)

    return left, top, right, bottom


def upscale_to_target(img, mask, target_long_side=1024):
    """Resize the image and mask so their longer side equals target_long_side."""
    w, h = img.size
    scale = target_long_side / max(w, h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    img_up = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    mask_up = mask.resize((new_w, new_h), Image.Resampling.NEAREST)

    return img_up, mask_up


def call_stability_inpaint(image, mask):
    """Inpaint solar panels into the white area of the mask with the Stability AI API."""
    image_bytes = io.BytesIO()
    mask_bytes = io.BytesIO()

    image.save(image_bytes, format="PNG")
    mask.save(mask_bytes, format="PNG")

    image_bytes.seek(0)
    mask_bytes.seek(0)

    response = requests.post(
        "https://api.stability.ai/v2beta/stable-image/edit/inpaint",
        headers={
            "authorization": f"Bearer {API_KEY}",
            "accept": "image/*"
        },
        files={
            "image": ("image.png", image_bytes, "image/png"),
            "mask": ("mask.png", mask_bytes, "image/png"),
        },
        data={
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "output_format": "png",
            "grow_mask": "0",
        },
    )

    if response.status_code != 200:
        try:
            print(response.json())
        except Exception:
            print(response.text)
        raise Exception(f"Stability API error: {response.status_code}")

    return Image.open(io.BytesIO(response.content)).convert("RGB")

def generate_solar_panels_full_image_stability(
    image_bytes: bytes,
    mask_bytes: bytes,
) -> bytes:
    """
    Add solar panels to the masked area of an image and return the result as PNG bytes.

    Only a padded crop around the mask is upscaled and sent to Stability AI; the
    edited crop is then blended back into the original image.
    """
    original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    mask     = Image.open(io.BytesIO(mask_bytes))


    mask = mask.resize(original.size, Image.Resampling.NEAREST)
    mask = binarize_mask(mask)


    bbox = get_mask_bbox(mask, padding=90)
    crop      = original.crop(bbox)
    crop_mask = mask.crop(bbox)

    crop_up, mask_up = upscale_to_target(crop, crop_mask, target_long_side=1024)


    edited_up = call_stability_inpaint(crop_up, mask_up)


    edited_crop = edited_up.resize(crop.size, Image.Resampling.LANCZOS)


    paste_mask   = crop_mask.filter(ImageFilter.GaussianBlur(radius=1.2))
    blended_crop = Image.composite(edited_crop, crop, paste_mask)


    final = original.copy()
    final.paste(blended_crop, bbox)


    buf = io.BytesIO()
    final.save(buf, format="PNG")
    return buf.getvalue()