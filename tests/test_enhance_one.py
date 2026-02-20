import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import enhance_one as script


class EnhanceOneTests(unittest.TestCase):
    def test_parse_args_defaults_match_output_path_and_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "sample.jpg"
            img.write_bytes(b"x")
            args = script.parse_args(["--input-file", str(img)])

        self.assertEqual(args.output_file, script.DEFAULT_OUTPUT_FILE)
        self.assertEqual(args.prompt, script.DEFAULT_PROMPT)

    def test_run_once_calls_enhance_with_resolved_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "input.jpg"
            output_path = Path(tmp) / "output.jpg"
            image_path.write_bytes(b"raw-image")

            args = argparse.Namespace(
                input_file=str(image_path),
                output_file=str(output_path),
                prompt="ignored",
                base_url=None,
                model=None,
                size=None,
                api_key=None,
                watermark=False,
            )

            with (
                mock.patch(
                    "enhance_one._resolve_config",
                    return_value=(
                        "https://api-inference.modelscope.cn/v1",
                        "Qwen/Qwen-Image-Edit-2511",
                        "1024x1536",
                        script.DEFAULT_PROMPT,
                        "token",
                    ),
                ),
                mock.patch("enhance_one._is_gemini", return_value=False),
                mock.patch("enhance_one.create_client", return_value=object()) as mock_create_client,
                mock.patch("enhance_one.enhance_one_image") as mock_enhance_one_image,
            ):
                rc = script.run_once(args)

        self.assertEqual(rc, 0)
        mock_create_client.assert_called_once_with(
            base_url="https://api-inference.modelscope.cn/v1",
            api_key="token",
        )
        kwargs = mock_enhance_one_image.call_args.kwargs
        self.assertEqual(kwargs["image_path"], image_path)
        self.assertEqual(kwargs["output_path"], output_path)
        self.assertEqual(kwargs["prompt"], script.DEFAULT_PROMPT)

    def test_run_once_returns_error_when_input_file_is_missing(self):
        args = argparse.Namespace(
            input_file="/tmp/__missing__/input.jpg",
            output_file="/tmp/out.jpg",
            prompt=script.DEFAULT_PROMPT,
            base_url=None,
            model=None,
            size=None,
            api_key=None,
            watermark=False,
        )

        with mock.patch(
            "enhance_one._resolve_config",
            return_value=(
                "https://api-inference.modelscope.cn/v1",
                "Qwen/Qwen-Image-Edit-2511",
                "1024x1536",
                script.DEFAULT_PROMPT,
                "token",
            ),
        ):
            rc = script.run_once(args)

        self.assertEqual(rc, 1)

    def test_load_dotenv_with_fallback_parses_env_file_without_python_dotenv(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("ENHANCE_API_KEY=test-token\n", encoding="utf-8")

            with mock.patch("enhance_one._load_dotenv"), mock.patch.dict(
                "enhance_one.os.environ",
                {},
                clear=True,
            ):
                script._load_dotenv_with_fallback(env_path)
                self.assertEqual(script.os.environ.get("ENHANCE_API_KEY"), "test-token")

    def test_load_dotenv_with_fallback_respects_existing_xai_api_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("ENHANCE_API_KEY=from-dotenv\n", encoding="utf-8")

            with mock.patch("enhance_one._load_dotenv"), mock.patch.dict(
                "enhance_one.os.environ",
                {"XAI_API_KEY": "already-set"},
                clear=True,
            ):
                script._load_dotenv_with_fallback(env_path)
                self.assertIsNone(script.os.environ.get("ENHANCE_API_KEY"))


if __name__ == "__main__":
    unittest.main()
