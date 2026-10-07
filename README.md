# Roof AI: solar panel backend

A backend service that looks at a satellite image of a house, finds its roof,
estimates how many solar panels fit on it and how much power they would
generate, and produces a realistic image of the house with panels installed.

Roof segmentation uses Meta's [SAM 3](https://github.com/facebookresearch/sam3)
model, which is vendored in the `sam3/` package. The solar panel images are
generated with OpenAI and Stability AI.

## How it works

A client sends a task with a satellite image URL, the latitude and the street
address. The API answers `202 Accepted` right away and processes the task in
the background in two steps, reporting each one to a callback URL.

### Step 1: roof analysis

1. Download the image and boost its saturation so the roof stands out.
2. Run SAM 3 with the text prompt `"roof"` and keep the mask whose centre is
   closest to the middle of the image (the target house).
3. Compute the ground sample distance (metres per pixel) from the latitude
   and map zoom level, then turn the mask's pixel count into:
   - roof area in m²
   - number of panels (2.2 m² each)
   - generated power (450 W per panel)
4. Upload the image with the roof highlighted in green to Supabase Storage.
5. POST the image URL and the estimates to `callback_url`.

### Step 2: solar panel visualization

1. GPT (`gpt-image-2`) gets the highlighted image and the address and paints
   the part of the roof facing the sun pink.
2. The pink area is turned into a black and white inpainting mask.
3. Stability AI inpaints solar panels into that area. Only a crop around the
   mask is sent, then blended back into the full image.
4. GPT does a final quality pass.
5. The result is uploaded to Supabase and POSTed to `callback_url_second`.

## API

| Method | Path       | Description                         |
| ------ | ---------- | ----------------------------------- |
| GET    | `/health`  | Service status and inference device |
| POST   | `/tasks`   | Queue a roof task                   |
| GET    | `/swagger` | Interactive API docs                |

Example `POST /tasks` body:

```json
{
  "task_id": "123",
  "image_url": "https://example.com/roof.png",
  "lat": 52.2297,
  "street": "37 Fife Road, London, UK",
  "is_masked": false,
  "callback_url": "https://example.com/api/v1/roof/callback",
  "callback_url_second": "https://example.com/api/v1/roof/callback-image"
}
```

Step 1 callback:

```json
{
  "status": "done",
  "taskId": "123",
  "imageUrl": "https://.../123_masked.png",
  "ResultsItem": { "areaM2": 84.5, "panels": 38, "generatedWatts": 17100 }
}
```

Step 2 callback:

```json
{
  "taskId": "123",
  "maskedImageUrl": "https://.../123_pinkmasked.png",
  "resultItemDto": { "areaM2": 84.5, "panels": 38, "generatedWatts": 17100 }
}
```

If a step fails, its callback gets `{"status": "error", "taskId": ..., "error": ...}`.

## Project layout

```
main.py                     FastAPI app and the roof processing pipeline
src/
  image_processing/         saturation boost and pink mask helpers
  langchain_model/
    agents/                 OpenAI image edits, Stability AI inpainting, pink mask extraction
    chains/                 Claude + Gemini prompt and image generation pipeline (experimental)
  testing/                  scripts for trying SAM 3 prompts and image preprocessing
sam3/                       SAM 3 model code
manifest.json               Azure Web App deployment template
```

## Setup

Requires Python 3.12 and a GPU is strongly recommended (CUDA with bf16 is used
when available).

```bash
pip install -e .
pip install fastapi uvicorn python-dotenv requests openai \
    langchain-anthropic langchain-google-genai scipy
```

Create a `.env` file in the project root:

```bash
HF_TOKEN=...            # Hugging Face token with access to the SAM 3 weights
OPENAI_API_KEY=...
STABILITY_KEY=...
ANTHROPIC_API_KEY=...   # only for the Claude + Gemini chain
GOOGLE_API_KEY=...      # only for the Claude + Gemini chain
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_BUCKET=Roof
SUPABASE_KEY=...
```

## Running

Start the API:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Or process a single image from the command line:

```bash
python main.py --image path/or/url/to/roof.png --lat 52.2297 --street "37 Fife Road, London, UK"
```

Downloaded images and intermediate results are written to `uploads/`.

## Credits

Built on [SAM 3: Segment Anything with Concepts](https://ai.meta.com/sam3) by
Meta Superintelligence Labs
([paper](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)).
