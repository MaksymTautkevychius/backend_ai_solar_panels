import base64
import io
import tempfile
import os
from pathlib import Path
from PIL import Image
from src.langchain_model.agents.openai import call_gpt_image
from src.langchain_model.agents.stability_ai import call_stability_inpaint
from src.langchain_model.agents.image_generation import create_pink_region_mask,

GPT_HIGHLIGHT_PROMPT = (
    "You are analyzing an aerial/satellite image of a building rooftop. "
    "Identify all roof slope surfaces that are suitable for solar panel installation. "
    "Paint those surfaces solid magenta/pink (RGB ~255, 0, 255). "
    "Leave everything else (streets, gardens, other buildings) completely unchanged. "
    "Return a photorealistic aerial image with only the target slopes painted pink."
)

GPT_FINALIZE_PROMPT = (
    "This is an aerial satellite image of a building where solar panels have just been "
    "installed on the roof. Make the result look fully photorealistic: blend the panel "
    "edges naturally with the surrounding roof, match the lighting, add subtle shadows "
    "under panel rows, and ensure color consistency with the rest of the image. "
    "Do not add or remove panels — only improve realism and blending."
)


def _bytes_to_temp_png(data: bytes, suffix: str = ".png") -> str:
    """Write bytes to a temporary file and return its path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return tmp.name


def _pil_to_bytes(img: Image.Image) -> bytes:
    """Encode a PIL image as PNG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate_panels_image(
    image_path: str,
    output_path: str = "solar_final_enhanced.png",
    openai_api_key: str | None = None,
    stability_api_key: str | None = None,
    hue_min: int = 290,
    hue_max: int = 330,
    saturation_min: float = 0.4,
    value_min: float = 0.3,
    min_region_size: int = 1000,
    gpt_output_size: str = "1024x1024",
) -> Image.Image:
    """
    Add solar panels to a roof image and save the result to output_path.

    GPT paints suitable roof slopes pink, the pink area becomes an inpainting
    mask, Stability AI fills it with panels, and a final GPT pass blends the
    result. Temporary files are removed afterwards.
    """
    temp_files: list[str] = []

    try:
        print("[1/4] GPT: highlighting roof slopes in pink …")
        original_bytes = Path(image_path).read_bytes()

        highlighted_bytes = call_gpt_image(
            prompt=GPT_HIGHLIGHT_PROMPT,
            image1=original_bytes,
            image2=original_bytes,
            api_key=openai_api_key,
            size=gpt_output_size,
        )

        highlighted_path = _bytes_to_temp_png(highlighted_bytes, "_highlighted.png")
        temp_files.append(highlighted_path)
        print(f"   → highlighted image saved to temp: {highlighted_path}")

        print("[2/4] Creating pink → binary mask …")
        mask_path = _bytes_to_temp_png(b"", "_mask.png")
        os.unlink(mask_path)
        temp_files.append(mask_path)

        create_pink_region_mask(
            image_path=highlighted_path,
            output_path=mask_path,
            hue_min=hue_min,
            hue_max=hue_max,
            saturation_min=saturation_min,
            value_min=value_min,
            min_region_size=min_region_size,
        )
        print(f"   → mask saved to temp: {mask_path}")

        print("[3/4] Stability AI: inpainting solar panels …")

        if stability_api_key:
            import stability_inpaint_module as _sim
            import sys
            current_module = sys.modules[__name__]
            if hasattr(current_module, "API_KEY"):
                current_module.API_KEY = stability_api_key

        stability_output = "solar_roof_final.png"
        generate_solar_panels_full_image_stability(
            image_path=image_path,
            mask_path=mask_path,
        )
        temp_files.append(stability_output)
        print(f"   → stability result: {stability_output}")

        print("[4/4] GPT: final realism / blending pass …")
        stability_bytes = Path(stability_output).read_bytes()

        final_bytes = call_gpt_image(
            prompt=GPT_FINALIZE_PROMPT,
            image1=original_bytes,
            image2=stability_bytes,
            api_key=openai_api_key,
            size=gpt_output_size,
        )

        final_image = Image.open(io.BytesIO(final_bytes)).convert("RGB")
        final_image.save(output_path)
        print(f"   → final image saved: {output_path}")

        return final_image

    finally:
        for path in temp_files:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "88.png"
    dst = sys.argv[2] if len(sys.argv) > 2 else "solar_final_enhanced.png"

    result = generate_panels_image(
        image_path=src,
        output_path=dst,
    )
    print(f"Done — {dst}  ({result.size[0]}×{result.size[1]} px)")