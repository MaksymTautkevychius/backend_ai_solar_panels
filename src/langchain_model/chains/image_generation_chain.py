"""
Two-step image pipeline: Claude writes positive and negative image prompts for
an address, then Gemini uses those prompts and two roof images to render the
roof covered with solar panels.
"""

import base64
import json
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

def load_image_as_base64(image_path: str) -> tuple[str, str]:
    """Read a local image and return (base64_data, mime_type)."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    mime_type = mime_map.get(suffix, "image/png")
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, mime_type


def save_gemini_image(ai_message, out_path: str = "generated_roof.png") -> str:
    """
    Save the first image in a Gemini response to out_path and return the saved path.
    The extension is adjusted to match the returned image type.
    """
    content = ai_message.content

    if isinstance(content, str):
        raise ValueError(
            f"Gemini returned text instead of an image. "
            f"First 200 chars: {content[:200]!r}"
        )

    for part in content:
        if isinstance(part, dict) and part.get("type") == "image_url":
            url = part["image_url"]["url"]
            if not url.startswith("data:"):
                raise ValueError(f"Unexpected image URL format: {url[:80]!r}")

            header, b64_data = url.split(",", 1)
            mime = header.split(";")[0].split(":", 1)[1]

            img_bytes = base64.b64decode(b64_data)
            ext = ".png" if "png" in mime else ".jpg"
            path = Path(out_path).with_suffix(ext)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "wb") as f:
                f.write(img_bytes)

            return str(path)

    raise ValueError("No image_url part found in Gemini AIMessage content.")

claude_model = ChatAnthropic(
    model="claude-sonnet-4-6",
)

CLAUDE_SYSTEM = """You are an expert in solar energy assessment and AI image generation.
Given a property address and a user instruction, you must output ONLY a valid JSON object
with exactly these two keys:

{
  "Image-Generation-Prompt": "<detailed positive prompt>",
  "Image-Generation-Negative-Prompt": "<detailed negative prompt>"
}

Rules:
- "Image-Generation-Prompt": a rich, photorealistic aerial image generation prompt describing
  the property roof covered with square solar panels. Include: overhead satellite perspective,
  dark navy blue monocrystalline panels in a regular grid, thin silver aluminium frames,
  panels flush with the roof surface, realistic lighting, surrounding neighbourhood context,
  and the specific address location.
- "Image-Generation-Negative-Prompt": everything the image generator must avoid, such as:
  tilted panels, sloped mounting racks, cartoon style, blurry output, people, construction
  equipment, unrealistic perspective, green mask overlay, shadows obscuring panels, incorrect
  panel shapes.
- Output ONLY the JSON. No preamble, no explanation, no markdown fences."""


def build_claude_messages(inputs: dict) -> list:
    """Build the Claude prompt from the address and user instruction."""
    address = inputs["address"]
    user_prompt = inputs["prompt"]
    return [
        SystemMessage(content=CLAUDE_SYSTEM),
        HumanMessage(
            content=(
                f"Property address: {address}\n\n"
                f"User instruction: {user_prompt}\n\n"
                "Output the JSON now."
            )
        ),
    ]


def parse_claude_json(raw: str) -> dict:
    """Parse Claude's JSON reply, stripping code fences, and check both prompt keys exist."""
    cleaned = (
        raw.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    parsed = json.loads(cleaned)
    required = {"Image-Generation-Prompt", "Image-Generation-Negative-Prompt"}
    missing = required - parsed.keys()
    if missing:
        raise ValueError(f"Claude JSON missing keys: {missing}")
    return parsed


claude_chain = (
    RunnableLambda(build_claude_messages)
    | claude_model
    | StrOutputParser()
    | RunnableLambda(parse_claude_json)
)

gemini_model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-image-preview",
)


def build_gemini_messages(inputs: dict) -> list:
    """
    Build the Gemini request: the Claude prompts, the address, the aerial photo
    (image1_path) and the roof mask overlay (image2_path).
    """
    img1_data, img1_mime = load_image_as_base64(inputs["image1_path"])
    img2_data, img2_mime = load_image_as_base64(inputs["image2_path"])

    claude = inputs["claude_output"]
    pos_prompt = claude["Image-Generation-Prompt"]
    neg_prompt = claude["Image-Generation-Negative-Prompt"]

    text_instruction = (
        f"Property address: {inputs['address']}\n\n"
        f"Image-Generation-Prompt:\n{pos_prompt}\n\n"
        f"Image-Generation-Negative-Prompt:\n{neg_prompt}\n\n"
        "Using the above prompts and the two reference images, generate ONE "
        "photorealistic overhead image of this property with the roof fully "
        "covered by a realistic grid of square solar panels.\n"
        "- Use an overhead satellite-like perspective.\n"
        "- Match the orientation and context of the input aerial photo.\n"
        "- Do NOT return any text in your response, only the rendered image."
    )

    content = [
        {"type": "text", "text": text_instruction},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{img1_mime};base64,{img1_data}"},
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:{img2_mime};base64,{img2_data}"},
        },
    ]
    return [HumanMessage(content=content)]


gemini_chain = RunnableLambda(build_gemini_messages) | gemini_model

def run_pipeline(image1_path: str, image2_path: str, address: str, prompt: str) -> dict:
    """
    Generate prompts with Claude, then render the roof with panels with Gemini.
    Returns {"claude_output": prompts, "gemini_image_path": saved image path}.
    """
    base_inputs = {
        "address": address,
        "prompt": prompt,
        "image1_path": image1_path,
        "image2_path": image2_path,
    }

    print("→ Claude: generating JSON prompts...")
    claude_output = claude_chain.invoke(base_inputs)
    print(f"  Claude output keys: {list(claude_output.keys())}\n")

    print("→ Gemini: generating image...")
    gemini_inputs = {**base_inputs, "claude_output": claude_output}
    gemini_message = gemini_chain.invoke(gemini_inputs)

    generated_image_path = save_gemini_image(
        gemini_message, "outputs/roof_with_panels.png"
    )
    print(f"  Gemini image saved to: {generated_image_path}\n")

    return {
        "claude_output": claude_output,
        "gemini_image_path": generated_image_path,
    }

if __name__ == "__main__":
    result = run_pipeline(
        image1_path="./uploads/123.png",
        image2_path="./uploads/123_result.png",
        address="37 Fife Road, East Sheen, SW14 8BJ, London, UK",
        prompt=(
            "Assess how many square solar panels can fit on this roof, "
            "estimate total kWp capacity assuming 400W panels, "
            "and flag any installation challenges."
        ),
    )

    print("=" * 60)
    print("CLAUDE JSON OUTPUT:\n")
    print(json.dumps(result["claude_output"], indent=2))
    print("\n" + "=" * 60)
    print("GENERATED IMAGE PATH:\n")
    print(result["gemini_image_path"])