import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import waterout as script


class FakeResponseItem:
    def __init__(self, url: str) -> None:
        self.url = url


class FakeResponse:
    def __init__(self, url: str) -> None:
        self.data = [FakeResponseItem(url)]


class FakeImagesAPI:
    def __init__(self, out_url: str) -> None:
        self.out_url = out_url
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self.out_url)


class FakeClient:
    def __init__(self, out_url: str = "https://example.com/enhanced.png") -> None:
        self.images = FakeImagesAPI(out_url)


class BatchEnhanceArkTests(unittest.TestCase):
    def test_is_image_file_supports_common_extensions(self):
        self.assertTrue(script.is_image_file(Path("a.JPG")))
        self.assertTrue(script.is_image_file(Path("b.webp")))
        self.assertFalse(script.is_image_file(Path("c.txt")))

    def test_encode_image_as_data_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "sample.png"
            raw = b"\x89PNG\r\n"
            image_path.write_bytes(raw)

            data_url = script.encode_image_as_data_url(image_path)

            self.assertTrue(data_url.startswith("data:image/png;base64,"))
            self.assertEqual(data_url.split(",", 1)[1], base64.b64encode(raw).decode("ascii"))

    def test_download_image_from_url_includes_user_agent_header(self):
        captured = {}

        def fake_urlopen(req):
            captured["headers"] = {k.lower(): v for k, v in req.header_items()}
            fake_resp = mock.MagicMock()
            fake_resp.read.return_value = b"img-bytes"
            fake_cm = mock.MagicMock()
            fake_cm.__enter__.return_value = fake_resp
            fake_cm.__exit__.return_value = False
            return fake_cm

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.png"
            with mock.patch("waterout.urllib.request.urlopen", side_effect=fake_urlopen):
                script.download_image_from_url("https://imgen.x.ai/fake.png", out)

            self.assertEqual(out.read_bytes(), b"img-bytes")

        self.assertIn("user-agent", captured["headers"])
        self.assertTrue(captured["headers"]["user-agent"])

    def test_enhance_one_image_calls_api_and_writes_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.jpg"
            dst = Path(tmp) / "out.jpg"
            src.write_bytes(b"test-image")
            client = FakeClient()

            captured = {}

            def fake_download(url: str, out_path: Path):
                captured["url"] = url
                out_path.write_bytes(b"enhanced-image")

            script.enhance_one_image(
                client=client,
                image_path=src,
                output_path=dst,
                model="doubao-seedream-4-5-251128",
                prompt="do not change people",
                size="2K",
                watermark=True,
                download_image=fake_download,
            )

            self.assertTrue(dst.exists())
            self.assertEqual(dst.read_bytes(), b"enhanced-image")
            self.assertEqual(captured["url"], "https://example.com/enhanced.png")

            call = client.images.calls[0]
            self.assertEqual(call["model"], "doubao-seedream-4-5-251128")
            self.assertEqual(call["size"], "2K")
            self.assertEqual(call["response_format"], "url")
            self.assertTrue(call["extra_body"]["image"].startswith("data:image/jpeg;base64,"))
            self.assertTrue(call["extra_body"]["watermark"])

    def test_modelscope_async_submit_accepts_task_id_field(self):
        fake_resp = mock.MagicMock()
        fake_resp.read.return_value = json.dumps({"task_id": "task-123"}).encode("utf-8")
        fake_cm = mock.MagicMock()
        fake_cm.__enter__.return_value = fake_resp
        fake_cm.__exit__.return_value = False

        with mock.patch("waterout.urllib.request.urlopen", return_value=fake_cm):
            task_id = script._modelscope_async_submit(
                base_url="https://api-inference.modelscope.cn/v1",
                api_key="ms-token",
                payload={"model": "Qwen/Qwen-Image-Edit-2511"},
            )

        self.assertEqual(task_id, "task-123")

    def test_modelscope_async_submit_prefers_task_id_over_request_id(self):
        fake_resp = mock.MagicMock()
        fake_resp.read.return_value = json.dumps(
            {"task_id": "task-123", "request_id": "request-xyz"}
        ).encode("utf-8")
        fake_cm = mock.MagicMock()
        fake_cm.__enter__.return_value = fake_resp
        fake_cm.__exit__.return_value = False

        with mock.patch("waterout.urllib.request.urlopen", return_value=fake_cm):
            task_id = script._modelscope_async_submit(
                base_url="https://api-inference.modelscope.cn/v1",
                api_key="ms-token",
                payload={"model": "Qwen/Qwen-Image-Edit-2511"},
            )

        self.assertEqual(task_id, "task-123")

    def test_modelscope_poll_result_supports_sample_status_and_output_images(self):
        payload = {
            "task_status": "SUCCEED",
            "output_images": ["https://example.com/result.jpg"],
        }
        fake_resp = mock.MagicMock()
        fake_resp.read.return_value = json.dumps(payload).encode("utf-8")
        fake_cm = mock.MagicMock()
        fake_cm.__enter__.return_value = fake_resp
        fake_cm.__exit__.return_value = False

        with mock.patch("waterout.urllib.request.urlopen", return_value=fake_cm), mock.patch(
            "waterout.time.sleep", side_effect=AssertionError("should not sleep when task already succeeded")
        ):
            output_url = script._modelscope_poll_result(
                base_url="https://api-inference.modelscope.cn/v1",
                api_key="ms-token",
                request_id="task-123",
            )

        self.assertEqual(output_url, "https://example.com/result.jpg")

    def test_enhance_one_image_modelscope_uses_image_url_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.jpg"
            dst = Path(tmp) / "out.jpg"
            src.write_bytes(b"test-image")

            captured = {}

            def fake_submit(base_url: str, api_key: str, payload: dict) -> str:
                captured["payload"] = payload
                return "task-123"

            with mock.patch("waterout._modelscope_async_submit", side_effect=fake_submit), mock.patch(
                "waterout._modelscope_poll_result",
                return_value="https://example.com/result.jpg",
            ):
                script.enhance_one_image(
                    client=None,
                    image_path=src,
                    output_path=dst,
                    model="Qwen/Qwen-Image-Edit-2511",
                    prompt="prompt",
                    size="1024x1536",
                    watermark=False,
                    download_image=lambda url, output_path: output_path.write_bytes(b"ok"),
                    base_url="https://api-inference.modelscope.cn/v1",
                    api_key="token",
                )

        payload = captured["payload"]
        self.assertIn("image_url", payload)
        self.assertIsInstance(payload["image_url"], list)
        self.assertEqual(len(payload["image_url"]), 1)
        self.assertTrue(payload["image_url"][0].startswith("data:image/jpeg;base64,"))

    def test_enhance_one_image_xai_uses_image_edit_endpoint_not_generate(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.jpg"
            dst = Path(tmp) / "out.jpg"
            src.write_bytes(b"test-image")
            client = FakeClient()

            captured = {}

            def fake_xai_edit(*, base_url: str, api_key: str, model: str, prompt: str, image_url: str, size: str):
                captured["base_url"] = base_url
                captured["api_key"] = api_key
                captured["model"] = model
                captured["prompt"] = prompt
                captured["image_url"] = image_url
                captured["size"] = size
                return "https://example.com/xai-edited.jpg"

            def fake_download(url: str, out_path: Path):
                captured["download_url"] = url
                out_path.write_bytes(b"enhanced-image")

            with mock.patch("waterout._xai_edit_image", side_effect=fake_xai_edit):
                script.enhance_one_image(
                    client=client,
                    image_path=src,
                    output_path=dst,
                    model="grok-imagine-image",
                    prompt="edit this image",
                    size="2k",
                    watermark=False,
                    download_image=fake_download,
                    base_url="https://api.x.ai/v1",
                    api_key="xai-token",
                )

        self.assertEqual(client.images.calls, [])
        self.assertEqual(captured["base_url"], "https://api.x.ai/v1")
        self.assertEqual(captured["api_key"], "xai-token")
        self.assertEqual(captured["model"], "grok-imagine-image")
        self.assertEqual(captured["prompt"], "edit this image")
        self.assertEqual(captured["size"], "2k")
        self.assertTrue(captured["image_url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(captured["download_url"], "https://example.com/xai-edited.jpg")

    def test_resolve_config_accepts_xai_api_key(self):
        args = mock.MagicMock()
        args.base_url = "https://api.x.ai/v1"
        args.model = "grok-imagine-image"
        args.size = None
        args.prompt = None
        args.api_key = None

        with mock.patch.dict("waterout.os.environ", {"XAI_API_KEY": "xai-key"}, clear=True):
            _, _, _, _, api_key = script._resolve_config(args)

        self.assertEqual(api_key, "xai-key")

    def test_xai_edit_image_includes_user_agent_header(self):
        captured = {}

        def fake_urlopen(req, timeout=300):
            captured["headers"] = {k.lower(): v for k, v in req.header_items()}
            fake_resp = mock.MagicMock()
            fake_resp.read.return_value = json.dumps(
                {"data": [{"url": "https://example.com/edited.png"}]}
            ).encode("utf-8")
            fake_cm = mock.MagicMock()
            fake_cm.__enter__.return_value = fake_resp
            fake_cm.__exit__.return_value = False
            return fake_cm

        with mock.patch("waterout.urllib.request.urlopen", side_effect=fake_urlopen):
            output_url = script._xai_edit_image(
                base_url="https://api.x.ai/v1",
                api_key="xai-token",
                model="grok-imagine-image",
                prompt="edit this image",
                image_url="data:image/jpeg;base64,AA==",
                size="2k",
            )

        self.assertEqual(output_url, "https://example.com/edited.png")
        self.assertIn("user-agent", captured["headers"])
        self.assertTrue(captured["headers"]["user-agent"])


if __name__ == "__main__":
    unittest.main()
