import base64
import openai
from pathlib import Path
import io


def call_gpt_image(
    prompt: str,
    image1: str | bytes,
    api_key: str | None = None,
    size: str = "1024x1024",
) -> bytes:
    """Edit an image (path or bytes) with gpt-image-2 using the prompt and return the result as bytes."""
    client = openai.OpenAI(api_key=api_key)

    def to_bytes(img: str | bytes) -> bytes:
        if isinstance(img, (str, Path)):
            return Path(img).read_bytes()
        return img

    img1_bytes = to_bytes(image1)

    buf = io.BytesIO(img1_bytes)
    buf.name = "image.png"

    response = client.images.edit(
        model="gpt-image-2",
        image=buf,
        prompt=prompt,
        size=size,
        n=1,
    )

    b64 = response.data[0].b64_json
    if b64 is None:
        import requests
        return requests.get(response.data[0].url).content

    return base64.b64decode(b64)