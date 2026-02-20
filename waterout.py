#!/usr/bin/env python3
import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable, Iterable

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime for CLI usage
    OpenAI = None

try:
    from tencentcloud.common import credential as _tc_cred
    from tencentcloud.aiart.v20221229 import aiart_client as _tc_aiart, models as _tc_aiart_models
except ImportError:  # pragma: no cover
    _tc_cred = None
    _tc_aiart = None
    _tc_aiart_models = None


def _load_dotenv() -> None:
    """Load .env from the script's directory (if python-dotenv is available)."""
    if load_dotenv is None:
        return
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)


_load_dotenv()


# ---------------------------------------------------------------------------
# Backend presets  (pick via ENHANCE_BASE_URL / ENHANCE_MODEL env or CLI)
# ---------------------------------------------------------------------------
# ModelScope
MODELSCOPE_BASE_URL = "https://api-inference.modelscope.cn/v1"
MODELSCOPE_MODEL = "Qwen/Qwen-Image-Edit-2511"
MODELSCOPE_SIZE = "1024x1536"
# Ark (Volcengine)
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
ARK_MODEL = "doubao-seedream-4-5-251128"
ARK_SIZE = "4K"
# Gemini (Google)
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MODEL = "gemini-2.5-flash-image"
# Hunyuan (Tencent)
HUNYUAN_BASE_URL = "https://api.hunyuan.cloud.tencent.com/v1"
HUNYUAN_MODEL = "hunyuan-image-3.0-instruct"
HUNYUAN_SIZE = "1024x1024"
# xAI (Grok Imagine)
XAI_BASE_URL = "https://api.x.ai/v1"
XAI_MODEL = "grok-imagine-image"
XAI_RESOLUTION = "2k"

# Defaults (env vars override these, CLI args override env vars)
DEFAULT_BASE_URL = MODELSCOPE_BASE_URL
DEFAULT_MODEL = MODELSCOPE_MODEL
DEFAULT_SIZE = MODELSCOPE_SIZE

MODELSCOPE_POLL_INTERVAL = 5  # seconds between status checks
MODELSCOPE_MAX_WAIT = 600  # max seconds to wait for a task
HUNYUAN_POLL_INTERVAL = 5
HUNYUAN_MAX_WAIT = 600
DEFAULT_PROMPT = (
    "【最高优先级】绝对不允许修改、重绘或捏造任何人脸。"
    "人脸区域必须像素级保留原图，不得对五官、轮廓、肤色、发型做任何改动或美化。"
    "如果人脸因水印遮挡而模糊，仅做最小程度的修复，宁可保留模糊也不要生成新的面部特征。"
    "在保持人物身份、姿态、表情、服饰、场景和构图完全不变的前提下，"
    "仅对非人脸区域进行轻度降噪与轻锐化，提升背景和服饰的清晰度，"
    "避免过度磨皮、过锐边缘或风格化重绘，保持自然观感。"
    "去除图片上的所有水印文字和水印标识，修复水印区域使其与周围背景自然融合。"
)
SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def iter_image_files(input_dir: Path, recursive: bool) -> Iterable[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    files = [path for path in iterator if path.is_file() and is_image_file(path)]
    files.sort()
    return files


def encode_image_as_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type:
        mime_type = "application/octet-stream"
    raw = image_path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def download_image_from_url(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={
            # Some CDNs behind Cloudflare block default urllib user-agent.
            "User-Agent": "waterout-enhancer/1.0",
        },
        method="GET",
    )
    with urllib.request.urlopen(req) as response:
        output_path.write_bytes(response.read())


def _modelscope_async_submit(base_url: str, api_key: str, payload: dict) -> str:
    """Submit an async image generation task to ModelScope, return request_id."""
    url = f"{base_url.rstrip('/')}/images/generations"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "X-ModelScope-Async-Mode": "true",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {error_body}") from e
    request_id = body.get("task_id") or body.get("request_id")
    if not request_id:
        raise RuntimeError(f"No request_id/task_id in async response: {body}")
    return request_id


def _modelscope_poll_result(base_url: str, api_key: str, request_id: str) -> str:
    """Poll ModelScope task until done, return the output image URL."""
    url = f"{base_url.rstrip('/')}/tasks/{request_id}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "X-ModelScope-Task-Type": "image_generation",
        },
        method="GET",
    )
    deadline = time.monotonic() + MODELSCOPE_MAX_WAIT
    while time.monotonic() < deadline:
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read())
        output = body.get("output", {}) if isinstance(body.get("output"), dict) else {}
        status = (output.get("task_status") or body.get("task_status") or "").upper()
        if status in {"SUCCEEDED", "SUCCEED", "SUCCESS"}:
            # Sample response shape: {"task_status":"SUCCEED","output_images":[...]}
            output_images = body.get("output_images") or output.get("output_images") or []
            if output_images and isinstance(output_images, list):
                return output_images[0]
            # Try multiple known response shapes
            results = (
                output.get("results")
                or output.get("result")
                or body.get("results")
                or body.get("result")
                or []
            )
            if results and isinstance(results, list):
                first = results[0]
                if isinstance(first, dict):
                    return first.get("url") or first.get("b64_json", "")
                if isinstance(first, str):
                    return first
            # Fallback: check data array (OpenAI-compatible)
            data_list = body.get("data") or []
            if data_list:
                return data_list[0].get("url", "")
            raise RuntimeError(f"Task succeeded but no image URL found: {body}")
        if status in {"FAILED", "FAIL", "CANCELED", "CANCELLED"}:
            msg = output.get("message") or output.get("code") or str(body)
            raise RuntimeError(f"Task {status}: {msg}")
        time.sleep(MODELSCOPE_POLL_INTERVAL)
    raise RuntimeError(f"Task {request_id} timed out after {MODELSCOPE_MAX_WAIT}s")


def _is_modelscope(base_url: str) -> bool:
    return "modelscope" in base_url.lower()


def _is_gemini(base_url: str = "", model: str = "") -> bool:
    return "googleapis.com" in base_url.lower() or model.lower().startswith("gemini")


def _is_hunyuan(base_url: str = "", model: str = "") -> bool:
    return "hunyuan.cloud.tencent.com" in base_url.lower() or model.lower().startswith("hunyuan")


def _is_xai(base_url: str = "", model: str = "") -> bool:
    return "x.ai" in base_url.lower() or model.lower().startswith("grok-imagine")


def _hunyuan_native_enhance(
    image_path: Path,
    prompt: str,
    size: str,
    output_path: Path,
    secret_id: str,
    secret_key: str,
    watermark: bool,
    download_image: Callable[[str, Path], None] = download_image_from_url,
) -> str:
    """Use Tencent Cloud AIART ImageToImage API to edit the input photo."""
    if _tc_aiart is None:
        raise RuntimeError(
            "tencentcloud-sdk-python is not installed. Run: pip install tencentcloud-sdk-python"
        )
    if not secret_id or not secret_key:
        raise RuntimeError(
            "HUNYUAN_SECRET_ID and HUNYUAN_SECRET_KEY are required. "
            "Get them from https://console.cloud.tencent.com/cam/capi"
        )

    raw_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")

    cred = _tc_cred.Credential(secret_id, secret_key)
    client = _tc_aiart.AiartClient(cred, "ap-guangzhou")

    req = _tc_aiart_models.ImageToImageRequest()
    req.InputImage = raw_b64
    req.Prompt = prompt
    req.NegativePrompt = "watermark, text overlay, logo, low quality, blurry, noise"
    req.Strength = 0.35  # stay close to the original (0=identical, 1=fully redrawn)
    req.Styles = []       # empty list avoids the default anime style
    req.EnhanceImage = 1  # enable built-in quality enhancement
    req.LogoAdd = 1 if watermark else 0
    req.RspImgType = "base64"

    resp = client.ImageToImage(req)
    img_data = base64.b64decode(resp.ResultImage)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(img_data)
    return str(output_path)


def _gemini_enhance(
    base_url: str,
    api_key: str,
    model: str,
    image_path: Path,
    prompt: str,
    output_path: Path,
) -> str:
    """Use Google Gemini API to enhance an image."""
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type:
        mime_type = "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")

    url = f"{base_url.rstrip('/')}/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": encoded,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "responseModalities": ["IMAGE", "TEXT"],
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini HTTP {e.code}: {error_body}") from e

    candidates = body.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"No candidates in Gemini response: {body}")

    parts = candidates[0].get("content", {}).get("parts", [])
    for part in parts:
        inline_data = part.get("inlineData") or part.get("inline_data")
        if inline_data:
            img_data = base64.b64decode(inline_data["data"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Determine output suffix from response mime type
            resp_mime = inline_data.get("mimeType") or inline_data.get("mime_type", "")
            if resp_mime and resp_mime != mime_type:
                ext_map = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}
                new_ext = ext_map.get(resp_mime)
                if new_ext:
                    output_path = output_path.with_suffix(new_ext)
            output_path.write_bytes(img_data)
            return str(output_path)

    raise RuntimeError(f"No image data in Gemini response parts: {parts}")


def _xai_edit_image(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    image_url: str,
    size: str,
) -> str:
    """Use xAI image edit endpoint with JSON payload."""
    payload = {
        "model": model or XAI_MODEL,
        "prompt": prompt,
        "image": {"url": image_url},
        "response_format": "url",
    }
    if size:
        payload["resolution"] = size

    url = f"{(base_url or XAI_BASE_URL).rstrip('/')}/images/edits"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            # xAI/Cloudflare may block urllib default user-agent (HTTP 403 code 1010).
            "User-Agent": "waterout-enhancer/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"xAI HTTP {e.code}: {error_body}") from e

    data_items = body.get("data") or []
    if not data_items:
        raise RuntimeError(f"No output returned from xAI image edit: {body}")

    first = data_items[0]
    if isinstance(first, dict) and first.get("url"):
        return first["url"]
    raise RuntimeError(f"No output url returned from xAI image edit: {body}")


def enhance_one_image(
    client,
    image_path: Path,
    output_path: Path,
    model: str,
    prompt: str,
    size: str,
    watermark: bool,
    download_image: Callable[[str, Path], None] = download_image_from_url,
    base_url: str = "",
    api_key: str = "",
) -> str:
    payload_image = encode_image_as_data_url(image_path)

    if _is_gemini(base_url=base_url, model=model):
        return _gemini_enhance(
            base_url=base_url,
            api_key=api_key,
            model=model,
            image_path=image_path,
            prompt=prompt,
            output_path=output_path,
        )

    if _is_hunyuan(base_url=base_url, model=model):
        return _hunyuan_native_enhance(
            image_path=image_path,
            prompt=prompt,
            size=size,
            output_path=output_path,
            secret_id=os.environ.get("HUNYUAN_SECRET_ID", ""),
            secret_key=os.environ.get("HUNYUAN_SECRET_KEY", ""),
            watermark=watermark,
            download_image=download_image,
        )

    if _is_xai(base_url=base_url, model=model):
        output_url = _xai_edit_image(
            base_url=base_url,
            api_key=api_key,
            model=model,
            prompt=prompt,
            image_url=payload_image,
            size=size,
        )
        download_image(output_url, output_path)
        return output_url

    if _is_modelscope(base_url):
        # Use direct HTTP for ModelScope async mode
        payload = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "response_format": "url",
            "image_url": [payload_image],
            "watermark": watermark,
        }
        request_id = _modelscope_async_submit(base_url, api_key, payload)
        output_url = _modelscope_poll_result(base_url, api_key, request_id)
        download_image(output_url, output_path)
        return output_url

    # Standard OpenAI-compatible sync path
    response = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        response_format="url",
        extra_body={"image": payload_image, "watermark": watermark},
    )
    if not response.data or not getattr(response.data[0], "url", None):
        raise RuntimeError(f"No output url returned for {image_path.name}")
    output_url = response.data[0].url
    download_image(output_url, output_path)
    return output_url


def create_client(base_url: str, api_key: str):
    if OpenAI is None:
        raise RuntimeError("openai package is not installed. Run: pip install openai")
    return OpenAI(base_url=base_url, api_key=api_key)


def process_folder(
    client,
    input_dir: Path,
    output_dir: Path,
    recursive: bool,
    model: str,
    prompt: str,
    size: str,
    watermark: bool,
    base_url: str = "",
    api_key: str = "",
) -> int:
    images = list(iter_image_files(input_dir, recursive=recursive))
    if not images:
        print(f"[WARN] No image files found in {input_dir}")
        return 0

    ok = 0
    for index, image_path in enumerate(images, start=1):
        rel_path = image_path.relative_to(input_dir)
        output_path = output_dir / rel_path
        try:
            enhance_one_image(
                client=client,
                image_path=image_path,
                output_path=output_path,
                model=model,
                prompt=prompt,
                size=size,
                watermark=watermark,
                base_url=base_url,
                api_key=api_key,
            )
            ok += 1
            print(f"[{index}/{len(images)}] OK {image_path} -> {output_path}")
        except Exception as exc:  # noqa: BLE001 - continue on per-file failure
            print(f"[{index}/{len(images)}] FAIL {image_path}: {exc}")
    return ok


def _env(name: str, fallback: str = "") -> str:
    """Read an environment variable with a fallback."""
    return os.environ.get(name, fallback)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch enhance images with denoise + mild sharpening. "
        "Supports ModelScope, Ark (Volcengine), and Gemini (Google) backends.",
    )
    parser.add_argument("--input-dir", required=True, help="Input folder containing images.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output folder. Default: <input-dir>_enhanced_2k",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan input directory recursively.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name. Env: ENHANCE_MODEL. "
        f"(default ModelScope: {MODELSCOPE_MODEL}, Ark: {ARK_MODEL})",
    )
    parser.add_argument(
        "--size",
        default=None,
        help="Output size. Env: ENHANCE_SIZE. "
        f"(default ModelScope: {MODELSCOPE_SIZE}, Ark: {ARK_SIZE})",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Enhancement prompt. Env: ENHANCE_PROMPT.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="API base URL. Env: ENHANCE_BASE_URL. "
        f"(default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (prefer env ENHANCE_API_KEY, ARK_API_KEY, GEMINI_API_KEY, or XAI_API_KEY).",
    )
    parser.add_argument(
        "--watermark",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to request watermark in result (default: false).",
    )
    return parser.parse_args()


def _resolve_config(args: argparse.Namespace):
    """Resolve final config from CLI args -> env vars -> built-in defaults."""
    base_url = args.base_url or _env("ENHANCE_BASE_URL") or ""
    model = args.model or _env("ENHANCE_MODEL") or ""

    # Detect backend from model name or base URL
    is_gemini = _is_gemini(base_url=base_url, model=model)
    is_hunyuan = not is_gemini and _is_hunyuan(base_url=base_url, model=model)
    is_xai = not is_gemini and not is_hunyuan and _is_xai(base_url=base_url, model=model)
    is_ark = (
        not is_gemini
        and not is_hunyuan
        and not is_xai
        and ("volces.com" in base_url or "ark" in base_url.lower())
    )

    if is_gemini:
        base_url = base_url or GEMINI_BASE_URL
        model = model or GEMINI_MODEL
        size = args.size or _env("ENHANCE_SIZE") or ""
    elif is_hunyuan:
        base_url = base_url or HUNYUAN_BASE_URL
        model = model or HUNYUAN_MODEL
        size = args.size or _env("ENHANCE_SIZE") or HUNYUAN_SIZE
    elif is_xai:
        base_url = base_url or XAI_BASE_URL
        model = model or XAI_MODEL
        size = args.size or _env("ENHANCE_SIZE") or XAI_RESOLUTION
    elif is_ark:
        base_url = base_url or ARK_BASE_URL
        model = model or ARK_MODEL
        size = args.size or _env("ENHANCE_SIZE") or ARK_SIZE
    else:
        base_url = base_url or MODELSCOPE_BASE_URL
        model = model or MODELSCOPE_MODEL
        size = args.size or _env("ENHANCE_SIZE") or MODELSCOPE_SIZE

    prompt = args.prompt or _env("ENHANCE_PROMPT", DEFAULT_PROMPT)
    api_key = (
        args.api_key
        or _env("ENHANCE_API_KEY")
        or _env("ARK_API_KEY")
        or _env("GEMINI_API_KEY")
        or _env("XAI_API_KEY")
        or _env("HUNYUAN_API_KEY")
        or _env("HUNYUAN_SECRET_ID")  # native SDK mode: SecretId used as sentinel
    )
    return base_url, model, size, prompt, api_key


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] Input dir does not exist: {input_dir}")
        return 1

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_dir.parent / f"{input_dir.name}_enhanced_2k"
    )

    base_url, model, size, prompt, api_key = _resolve_config(args)
    if not api_key:
        print("[ERROR] No API key. Set ENHANCE_API_KEY/XAI_API_KEY env, or use --api-key.")
        return 1

    print(f"Backend : {base_url}")
    print(f"Model   : {model}")
    print(f"Size    : {size}")
    print(f"Input   : {input_dir}")
    print(f"Output  : {output_dir}")
    print()

    try:
        client = create_client(base_url=base_url, api_key=api_key)
        ok = process_folder(
            client=client,
            input_dir=input_dir,
            output_dir=output_dir,
            recursive=args.recursive,
            model=model,
            prompt=prompt,
            size=size,
            watermark=args.watermark,
            base_url=base_url,
            api_key=api_key,
        )
    except Exception as exc:  # noqa: BLE001 - CLI level fallback
        print(f"[ERROR] {exc}")
        return 1

    print(
        f"Done. Success: {ok} / {len(list(iter_image_files(input_dir, recursive=args.recursive)))}"
    )
    print(f"Output dir: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
