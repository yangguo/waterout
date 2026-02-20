# waterout

Remove watermarks and enhance photos using AI image-editing APIs.

Supports multiple backends:

| Backend | Model family | Notes |
|---|---|---|
| **ModelScope** | Qwen-Image-Edit | Async polling, free tier available |
| **Ark / Volcengine** | Doubao-Seedream | Sync, 4 K output |
| **Gemini** | gemini-2.5-flash-image | Google AI Studio |
| **Hunyuan** | hunyuan-image-3.0 | Tencent Cloud AIART SDK |
| **xAI** | grok-imagine-image | Grok image-edit endpoint |

---

## Requirements

- Python 3.10+
- `openai` (all backends except Hunyuan native)
- `python-dotenv` (optional, auto-loads `.env`)
- `tencentcloud-sdk-python` (Hunyuan native SDK only)

Install:

```bash
pip install openai python-dotenv          # core
pip install tencentcloud-sdk-python       # Hunyuan only
```

---

## Quick start

### 1. Configure credentials

Copy `.env.example` to `.env` and fill in your API key:

```bash
cp .env.example .env
# edit .env
```

### 2. Enhance a whole folder (batch)

```bash
python waterout.py --input-dir ./photos
# output written to ./photos_enhanced_2k/
```

Override backend / model / size:

```bash
# Ark (Volcengine)
python waterout.py --input-dir ./photos \
    --base-url https://ark.cn-beijing.volces.com/api/v3 \
    --model doubao-seedream-4-5-251128 \
    --size 4K

# Gemini
python waterout.py --input-dir ./photos \
    --model gemini-2.5-flash-image

# xAI
python waterout.py --input-dir ./photos \
    --base-url https://api.x.ai/v1 \
    --model grok-imagine-image
```

### 3. Enhance a single image

```bash
python enhance_one.py --input-file photo.jpg --output-file out.jpg
```

---

## CLI reference

### `waterout.py` (batch)

| Flag | Env var | Default |
|---|---|---|
| `--input-dir` | — | *required* |
| `--output-dir` | — | `<input-dir>_enhanced_2k` |
| `--recursive` | — | off |
| `--base-url` | `ENHANCE_BASE_URL` | ModelScope |
| `--model` | `ENHANCE_MODEL` | auto per backend |
| `--size` | `ENHANCE_SIZE` | auto per backend |
| `--prompt` | `ENHANCE_PROMPT` | built-in default |
| `--api-key` | `ENHANCE_API_KEY` | — |
| `--watermark` / `--no-watermark` | — | off |

### `enhance_one.py` (single image)

Same flags as above, replacing `--input-dir` / `--output-dir` with:

| Flag | Default |
|---|---|
| `--input-file` | *required* |
| `--output-file` | `./enhanced_output.jpg` |

---

## Environment variables

See `.env.example` for a full list. The backend is auto-detected from the model name or base URL; you only need to set the relevant key.

| Variable | Used by |
|---|---|
| `ENHANCE_API_KEY` | all backends (generic) |
| `ARK_API_KEY` | Ark / Volcengine |
| `GEMINI_API_KEY` | Gemini |
| `XAI_API_KEY` | xAI |
| `HUNYUAN_SECRET_ID` | Hunyuan (native SDK) |
| `HUNYUAN_SECRET_KEY` | Hunyuan (native SDK) |

---

## Running tests

```bash
python -m pytest tests/
# or
python -m unittest discover tests
```

---

## Project structure

```
waterout.py          # batch processor (main CLI)
enhance_one.py       # single-image helper CLI
tests/
  test_waterout.py
  test_enhance_one.py
.env.example         # credential template
```

---

## License

MIT
