#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from waterout import (
    DEFAULT_PROMPT,
    _is_gemini,
    _is_xai,
    _load_dotenv,
    _resolve_config,
    create_client,
    enhance_one_image,
)

DEFAULT_OUTPUT_FILE = "./enhanced_output.jpg"


def _load_dotenv_with_fallback(env_path: Path | None = None) -> None:
    _load_dotenv()
    if any(
        (
            os.environ.get("ENHANCE_API_KEY"),
            os.environ.get("ARK_API_KEY"),
            os.environ.get("GEMINI_API_KEY"),
            os.environ.get("XAI_API_KEY"),
            os.environ.get("HUNYUAN_API_KEY"),
            os.environ.get("HUNYUAN_SECRET_ID"),
        )
    ):
        return

    path = env_path or (Path(__file__).resolve().parent / ".env")
    if not path.is_file():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhance one image using configured backend.")
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to the input image file.",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output image path (default: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Enhancement prompt.",
    )
    parser.add_argument("--base-url", default=None, help="API base URL. Env: ENHANCE_BASE_URL")
    parser.add_argument("--model", default=None, help="Model id. Env: ENHANCE_MODEL")
    parser.add_argument("--size", default=None, help="Output size. Env: ENHANCE_SIZE")
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key. Env: ENHANCE_API_KEY / ARK_API_KEY / GEMINI_API_KEY / XAI_API_KEY",
    )
    parser.add_argument(
        "--watermark",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to request watermark in generated result.",
    )
    return parser.parse_args(argv)


def run_once(args: argparse.Namespace) -> int:
    base_url, model, size, prompt, api_key = _resolve_config(args)
    image_path = Path(args.input_file).expanduser()
    output_path = Path(args.output_file).expanduser()

    if not image_path.is_file():
        print(f"FAIL: input file not found: {image_path}")
        return 1
    if not api_key:
        print("FAIL: missing API key. Set ENHANCE_API_KEY/XAI_API_KEY or pass --api-key.")
        return 1

    client = None
    if not _is_gemini(base_url=base_url, model=model) and not _is_xai(base_url=base_url, model=model):
        client = create_client(base_url=base_url, api_key=api_key)

    try:
        enhance_one_image(
            client=client,
            image_path=image_path,
            output_path=output_path,
            model=model,
            prompt=prompt,
            size=size,
            watermark=args.watermark,
            base_url=base_url,
            api_key=api_key,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {exc}")
        return 1

    print(f"OK {image_path} -> {output_path}")
    return 0


def main() -> int:
    _load_dotenv_with_fallback()
    args = parse_args()
    return run_once(args)


if __name__ == "__main__":
    raise SystemExit(main())
