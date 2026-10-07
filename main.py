import base64

from sam3.visualization_utils import plot_results
import io,math,os,shutil,traceback,requests,torch 
from contextlib import nullcontext
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from src.image_processing.saturation import boost_saturation
import numpy as np
import matplotlib.pyplot as plt
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import JSONResponse
from huggingface_hub import login
from PIL import Image, ImageDraw
from pydantic import BaseModel, HttpUrl
from src.langchain_model.chains.image_generation_chain import run_pipeline
import sam3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from dotenv import load_dotenv
from src.langchain_model.agents.openai import call_gpt_image
from src.langchain_model.agents.image_generation import create_pink_region_mask
from src.langchain_model.agents.stability_ai import generate_solar_panels_full_image_stability

def debug_show(image_bytes: bytes, title: str = ""):
    """Display an encoded image in a matplotlib window for debugging."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()

load_dotenv()
class TaskAcceptedResponse(BaseModel):
    """Response returned when a task is queued."""
    status: str
    taskId: str


class ErrorResponse(BaseModel):
    """Response returned when a request cannot be processed."""
    status: str
    error: str


class HealthResponse(BaseModel):
    """Response returned by the health check endpoint."""
    status: str
    service: str
    device: str

class TaskRequest(BaseModel):
    """Input for a roof processing task."""
    task_id: str
    image_url: HttpUrl
    lat: float
    callback_url: Optional[HttpUrl] = None
    street: str
    callback_url_second: Optional[HttpUrl] = None
    is_masked: bool

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "123",
                "image_url": "https://example.com/roof.png",
                "lat": 52.2297,
                "callback_url": "https://example.com/api/v1/roof/callback",
            }
        }







DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = DEVICE == "cuda" and torch.cuda.is_bf16_supported()
PROMPT="Task: You have a house image, the image of a mask of the roof of the house, and its address, generate prompt to create square solar panels on the side of the roof image "

print(f"Using device: {DEVICE}")
print(f"BF16 supported: {USE_BF16}")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True




app = FastAPI(
    title="Roof AI API",
    version="1.0.0",
    docs_url="/swagger",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "Roof")
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

login(token=os.environ.get("HF_TOKEN"))
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
BPE_PATH = os.path.join(
    os.path.dirname(sam3.__file__),
    "assets",
    "bpe_simple_vocab_16e6.txt.gz",
)


ZOOM_LEVEL: int = 20
PANEL_AREA_M2: float = 2.2
WATTS_PER_PANEL: int = 450
IMAGE_CENTER = (300, 400)


def upload_to_supabase(data: bytes, filename: str) -> str:
    """Upload bytes to the Supabase bucket and return the file's public URL."""
    path = f"uploads/{filename}"
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{path}"

    headers = {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "apikey": SUPABASE_KEY,
        "x-upsert": "true",
        "Content-Type": "application/octet-stream",
    }

    resp = requests.post(url, headers=headers, data=data, timeout=30)
    resp.raise_for_status()

    public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{path}"
    return public_url


def calculate_gsd(latitude: float, zoom: int) -> float:
    """Return the ground sample distance (metres per pixel) of a Web Mercator tile."""
    lat_rad = latitude * math.pi / 180
    return 156543.03392 * math.cos(lat_rad) / (2 ** zoom)


def build_masked_image(original_image: Image.Image, mask_np: np.ndarray) -> Image.Image:
    """Return the image with the mask drawn on top as a semi-transparent green overlay."""
    rgba = original_image.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    mask_bool = mask_np > 0
    for y in range(mask_bool.shape[0]):
        for x in range(mask_bool.shape[1]):
            if mask_bool[y, x]:
                draw.point((x, y), fill=(0, 200, 80, 140))

    return Image.alpha_composite(rgba, overlay).convert("RGB")


def mask_to_base64_png(original_image: Image.Image, mask_np: np.ndarray) -> str:
    """Return the mask overlay image as a base64 PNG data URL."""
    composite = build_masked_image(original_image, mask_np)
    buf = io.BytesIO()
    composite.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def save_masked_image(original_image: Image.Image, mask_np: np.ndarray, output_path) -> None:
    """Save the image with a green mask overlay as a PNG file."""
    output_path = Path(output_path)

    rgba = original_image.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    mask_bool = mask_np > 0
    for y in range(mask_bool.shape[0]):
        for x in range(mask_bool.shape[1]):
            if mask_bool[y, x]:
                draw.point((x, y), fill=(0, 200, 80, 140))

    composite = Image.alpha_composite(rgba, overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(output_path, format="PNG")


def scores_to_numpy(scores) -> np.ndarray:
    """Convert SAM3 confidence scores to a flat float32 array."""
    if scores is None:
        return np.array([], dtype=np.float32)
    if isinstance(scores, torch.Tensor):
        return scores.detach().float().cpu().numpy().reshape(-1)
    return np.array(scores, dtype=np.float32).reshape(-1)


def mask_centroid(mask_np: np.ndarray):
    """Return the (y, x) centroid of a mask, or None if the mask is empty."""
    mask_2d = mask_np.squeeze()
    if mask_2d.ndim > 2:
        mask_2d = mask_2d[0]
    ys, xs = np.where(mask_2d > 0)
    if len(xs) == 0:
        return None
    return (float(np.mean(ys)), float(np.mean(xs)))


def euclidean_to_center(mask_np: np.ndarray) -> float:
    """Return the distance from the mask centroid to IMAGE_CENTER, or inf for an empty mask."""
    centroid = mask_centroid(mask_np)
    if centroid is None:
        return float("inf")
    return math.sqrt(
        (centroid[0] - IMAGE_CENTER[0]) ** 2 + (centroid[1] - IMAGE_CENTER[1]) ** 2
    )


def normalize_masks(masks) -> np.ndarray:
    """
    Convert SAM3 masks (tensor, list or array) to a float32 array of shape (N, H, W).
    Raises ValueError if there are no masks.
    """
    if masks is None:
        raise ValueError("SAM3 returned no masks for the given image.")

    if isinstance(masks, torch.Tensor):
        masks = masks.detach().float().cpu().numpy()
    elif isinstance(masks, (list, tuple)):
        converted = []
        for m in masks:
            if isinstance(m, torch.Tensor):
                m = m.detach().float().cpu().numpy()
            converted.append(np.array(m, dtype=np.float32).squeeze())
        masks = np.stack(converted) if converted else np.array([], dtype=np.float32)
    else:
        masks = np.array(masks, dtype=np.float32)

    if masks.size == 0:
        raise ValueError("SAM3 returned no masks for the given image.")

    if masks.ndim == 2:
        masks = masks[np.newaxis]

    if masks.shape[0] == 0:
        raise ValueError("SAM3 returned no masks for the given image.")

    return masks


def select_closest_mask(masks_np: np.ndarray, scores_np: np.ndarray):
    """
    Pick the mask whose centroid is closest to IMAGE_CENTER.
    Returns (index, mask, score, distance).
    """
    distances = [euclidean_to_center(masks_np[i]) for i in range(masks_np.shape[0])]
    best_idx = int(np.argmin(distances))
    best_mask = masks_np[best_idx]
    best_score = float(scores_np[best_idx]) if len(scores_np) > best_idx else 0.0
    best_distance = distances[best_idx]
    return best_idx, best_mask, best_score, best_distance




def process_roof(image_path: str, task_id: str, lat: float, callback_url: str | None, callback_url_second: str | None, street: str | None) -> None:
    """
    Run the full roof pipeline for one task.

    Step 1 segments the roof with SAM3, estimates area, panel count and power,
    uploads the mask overlay and posts the result to callback_url.
    Step 2 has GPT mark the sunniest roof section, inpaints solar panels there
    with Stability AI, uploads the result and posts it to callback_url_second.
    """
    masked_bytes = None

    try:
        print(f"[{task_id}] Starting processing")
        print(f"[{task_id}] Image path: {image_path}")

        gsd = calculate_gsd(lat, ZOOM_LEVEL)
        print(f"[{task_id}] GSD calculated: {gsd}")

        image = Image.open(image_path).convert("RGB")
        print(f"[{task_id}] Image loaded: {image.size}")

        image = boost_saturation(image, scale=1.8)
        print(f"[{task_id}] Saturation boosted")

        model = build_sam3_image_model(bpe_path=BPE_PATH)
        print(f"[{task_id}] Model built")

        processor = Sam3Processor(model, confidence_threshold=0.5)
        print(f"[{task_id}] Processor created")

        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if torch.cuda.is_available()
            else nullcontext()
        )

        with torch.inference_mode():
            with autocast_ctx:
                inference_state = processor.set_image(image)
                print(f"[{task_id}] Image set in processor")

                processor.reset_all_prompts(inference_state)
                inference_state = processor.set_text_prompt(
                    state=inference_state,
                    prompt="roof",
                )
                print(f"[{task_id}] Text prompt applied")

        scores = inference_state.get("scores", [])
        labels = inference_state.get("labels", [])
        masks  = inference_state.get("masks", [])

        print(f"[{task_id}] Confidence scores: {scores}")
        print(f"[{task_id}] Found objects: {labels}")

        masks_np  = normalize_masks(masks)
        scores_np = scores_to_numpy(scores)

        print(f"[{task_id}] Total masks found: {masks_np.shape[0]}")

        best_idx, best_mask, best_score, best_distance = select_closest_mask(masks_np, scores_np)
        mask_np = best_mask.squeeze()

        print(f"[{task_id}] Selected mask {best_idx} — centroid distance: {best_distance:.2f}px — score: {best_score:.4f}")
        print(f"[{task_id}] Mask shape: {mask_np.shape}")

        pixel_count = int(np.sum(mask_np > 0))
        area_m2     = pixel_count * (gsd ** 2)
        panels      = int(area_m2 // PANEL_AREA_M2)
        watts       = panels * WATTS_PER_PANEL
        print(f"[{task_id}] Area={area_m2:.2f}m², panels={panels}, watts={watts}")

        saved_image_path = os.path.join(UPLOAD_DIR, f"{task_id}_result.png")
        save_masked_image(image, mask_np, saved_image_path)
        print(f"[{task_id}] Saved result image: {saved_image_path}")

        with open(saved_image_path, "rb") as f:
            masked_bytes = f.read()

        imageUrl = upload_to_supabase(masked_bytes, f"{task_id}_masked.png")

        payload1 = {
            "status": "done",
            "taskId": task_id,
            "imageUrl": imageUrl,
            "ResultsItem": {
                "areaM2": round(area_m2, 2),
                "panels": panels,
                "generatedWatts": watts,
            },
        }

    except Exception as exc:
        print(f"[{task_id}] Step 1 failed: {exc}")
        traceback.print_exc()
        payload1 = {
            "status": "error",
            "taskId": task_id,
            "error": str(exc),
        }

    if callback_url and callback_url.startswith(("http://", "https://")):
        try:
            resp = requests.post(callback_url, json=payload1, timeout=30)
            print(f"[{task_id}] Callback 1 → {resp.status_code}")
        except Exception as exc:
            print(f"[{task_id}] Callback 1 failed: {exc}")
    else:
        print(f"[{task_id}] Skipping callback 1 — invalid or missing callback_url")
    if masked_bytes is None:
        print(f"[{task_id}] Skipping step 2 — no masked image available (step 1 failed)")
        return

    try:

        print(f"[{task_id}] Step 2: asking GPT to highlight best solar region")
        gpt_highlighted_bytes = call_gpt_image(
            "You have photo of the map with the mask of the roof and its address: "
            f"{street}"
            "Get the part of the roof that directed to the side with the biggest amount of the sun for the solar panels and colour it in pink #FF00FF colour, NOTE: This cant be full roof, PART ONLY",
            masked_bytes,
        )
        

        


        with open(image_path, "rb") as f:
            original_image = f.read()

        print(f"[{task_id}] Step 2: creating pink region mask")

        print(f"[{task_id}] Step 2: generating solar panels via Stability AI")


        pink1 = os.path.join(UPLOAD_DIR, f"pink1.png")
        with open(pink1, "wb") as f:
            f.write(gpt_highlighted_bytes)

        pink = os.path.join(UPLOAD_DIR, f"pink.png")
        with open(pink, "wb") as f:
            f.write(create_pink_region_mask(gpt_highlighted_bytes))

        orig = os.path.join(UPLOAD_DIR, f"orig.png")
        with open(orig, "wb") as f:
            f.write(original_image)

        solar_bytes = generate_solar_panels_full_image_stability(
            image_bytes=gpt_highlighted_bytes,
            mask_bytes=create_pink_region_mask(gpt_highlighted_bytes),
        )
        solar = os.path.join(UPLOAD_DIR, f"solar.png")
        with open(solar, "wb") as f:
            f.write(solar_bytes)
        

        print(f"[{task_id}] Step 2: GPT image improvement pass")
        solar_improved_bytes = call_gpt_image(
            "Improve the picture quality",
            solar_bytes,
        )

        maskedImageUrl = upload_to_supabase(solar_improved_bytes, f"{task_id}_pinkmasked.png")
        print(f"[{task_id}] Step 2: uploaded solar image  {maskedImageUrl}")

        payload2 = {
            "maskedImageUrl": maskedImageUrl,
            "resultItemDto": {
                "areaM2": round(area_m2, 2),
                "panels": panels,
                "generatedWatts": watts,
            },
            "taskId": task_id
        }

    except Exception as exc:
        print(f"[{task_id}] Step 2 failed: {exc}")
        traceback.print_exc()
        payload2 = {
            "status": "error",
            "taskId": task_id,
            "error": str(exc),
        }

    if callback_url_second and callback_url_second.startswith(("http://", "https://")):
        try:
            resp = requests.post(callback_url_second, json=payload2, timeout=30)
            print(f"[{task_id}] Callback 2 → {resp.status_code} — {resp.text[:500]}")
        except Exception as exc:
            print(f"[{task_id}] Callback 2 failed: {exc}")
    else:
        print(f"[{task_id}] Skipping callback 2 — invalid or missing callback_url_second")



@app.get(
    "/health",
    tags=["System"],
    summary="Health check",
    response_model=HealthResponse,
)

async def health():
    """Report service status and the device used for inference."""
    return {
        "status": "ok",
        "service": "roof-ai",
        "device": DEVICE,
    }
"""
                "area_ft2": round(area_m2 * 10.764, 2), 
"""


@app.post(
    "/tasks",
    status_code=202,
    tags=["Tasks"],
    summary="Queue a roof segmentation task",
    response_model=TaskAcceptedResponse,
    responses={
        202: {
            "description": "Task accepted for background processing",
            "model": TaskAcceptedResponse,
        },
        400: {
            "description": "Image download failed",
            "model": ErrorResponse,
        },
    },
)

async def create_task(data: TaskRequest, background_tasks: BackgroundTasks):
    """Download the task image, queue process_roof in the background and return 202."""
    image_url = str(data.image_url)

    try:
        response = requests.get(image_url, timeout=15, stream=True)
        response.raise_for_status()
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "error": f"Failed to download image: {exc}"},
        )

    ext = os.path.splitext(image_url.split("?")[0])[-1] or ".png"
    image_path = os.path.join(UPLOAD_DIR, f"{data.task_id}{ext}")

    with open(image_path, "wb") as f:
        shutil.copyfileobj(response.raw, f)

    background_tasks.add_task(
        process_roof,
        image_path=str(image_path),
        task_id=data.task_id,
        lat=data.lat,
        callback_url=str(data.callback_url) if data.callback_url else None,
        callback_url_second=str(data.callback_url_second) if data.callback_url_second else None,
        street=str(data.street)
    )

    return JSONResponse(
        status_code=202,
        content={"status": "accepted", "taskId": data.task_id},
    )



def main():
    """Run a single task from the command line without the API server."""
    import argparse

    parser = argparse.ArgumentParser(description="Roof AI — run a single task locally")
    parser.add_argument("--image",    required=True,  help="Path or URL to the roof image")
    parser.add_argument("--task-id",  default="local_test", help="Task ID (default: local_test)")
    parser.add_argument("--lat",      type=float, required=True, help="Latitude of the property")
    parser.add_argument("--callback", default=None, help="Optional callback URL")
    parser.add_argument("--street", default=None, help="Rosewoods USA")
    args = parser.parse_args()

    image_path = args.image

    if image_path.startswith(("http://", "https://")):
        import shutil, requests
        ext = os.path.splitext(image_path.split("?")[0])[-1] or ".png"
        dest = os.path.join(UPLOAD_DIR, f"{args.task_id}{ext}")
        print(f"Downloading image → {dest}")
        with requests.get(image_path, stream=True, timeout=15) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                shutil.copyfileobj(r.raw, f)
        image_path = dest

    process_roof(
        image_path=image_path,
        task_id=args.task_id,
        lat=args.lat,
        callback_url=args.callback,
        callback_url_second=args.callback,
        street=args.street
    )


if __name__ == "__main__":
    main()