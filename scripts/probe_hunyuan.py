#!/usr/bin/env python3
"""Development script: probe the Hunyuan API to find the correct endpoint for hunyuan-image-3.0-instruct.

Run from the project root:
    python scripts/probe_hunyuan.py
"""
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Allow importing from the project root when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from waterout import _load_dotenv  # noqa: E402

_load_dotenv()

API_KEY = os.environ["HUNYUAN_API_KEY"]
BASE = "https://api.hunyuan.cloud.tencent.com/v1"
MODEL = "hunyuan-image-3.0-instruct"
HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

# Sample image (tiny 1x1 red PNG, base64)
TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
)
DATA_URL = f"data:image/png;base64,{TINY_PNG_B64}"

PROMPT = "enhance this image"


def post(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=HEADERS, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:600]


print("--- 1) chat/completions text-only ---")
code, resp = post(f"{BASE}/chat/completions", {
    "model": MODEL, "messages": [{"role": "user", "content": PROMPT}]
})
print(code, resp if isinstance(resp, str) else json.dumps(resp)[:400])

print()
print("--- 2) chat/completions with image_url ---")
code, resp = post(f"{BASE}/chat/completions", {
    "model": MODEL,
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": DATA_URL}},
        ],
    }]
})
print(code, resp if isinstance(resp, str) else json.dumps(resp)[:400])

print()
print("--- 3) images/generations text-only ---")
code, resp = post(f"{BASE}/images/generations", {
    "model": MODEL, "prompt": PROMPT
})
print(code, resp if isinstance(resp, str) else json.dumps(resp)[:400])

print()
print("--- 4) images/generations with image (extra_body style via raw HTTP) ---")
code, resp = post(f"{BASE}/images/generations", {
    "model": MODEL, "prompt": PROMPT, "image": DATA_URL
})
print(code, resp if isinstance(resp, str) else json.dumps(resp)[:400])

print()
print("--- 5) images/generations with image_url ---")
code, resp = post(f"{BASE}/images/generations", {
    "model": MODEL, "prompt": PROMPT, "image_url": DATA_URL
})
print(code, resp if isinstance(resp, str) else json.dumps(resp)[:400])
