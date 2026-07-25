import base64
import hashlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from utils.api.base import OpenAIChatAPI, OpenAIResponsesAPI
from utils.api.judge_api import JudgeAPI


class _FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class OpenAIResponsesAPITest(unittest.TestCase):
    def test_text_file_uses_utf8_text_mime_data_url(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            text_path = Path(temp_dir) / "material.md"
            text_path.write_text("中文材料", encoding="utf-8")
            session = _FakeSession([
                _FakeResponse(
                    200,
                    {"output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]},
                )
            ])
            api = OpenAIResponsesAPI(api_key="test")
            api.session = session

            api.generate_content("model", "prompt", [str(text_path)])

            file_data = session.calls[0][1]["json"]["input"][0]["content"][0]["file_data"]
            prefix, payload = file_data.split(",", 1)
            self.assertEqual(prefix, "data:text/markdown;charset=utf-8;base64")
            self.assertEqual(base64.b64decode(payload).decode("utf-8"), "中文材料")

    def test_extracts_reasoning_and_message_text(self):
        result = {
            "output": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "摘要"}],
                    "content": [{"type": "reasoning_text", "text": "推理"}],
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "最终答案"}],
                },
            ]
        }
        session = _FakeSession([_FakeResponse(200, result)])
        api = OpenAIResponsesAPI(api_key="test")
        api.session = session

        text, returned_result = api.generate_content("model", "prompt")

        self.assertEqual(
            text,
            "--- reasoning ---\n[summary] 摘要\n推理\n\n--- message ---\n最终答案",
        )
        self.assertIs(returned_result, result)

    @patch("utils.api.base.time.sleep", return_value=None)
    def test_retries_transient_responses(self, _sleep):
        success = {
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]
        }
        session = _FakeSession([
            _FakeResponse(500, text="server error"),
            _FakeResponse(429, text="rate limited"),
            _FakeResponse(200, success),
        ])
        api = OpenAIResponsesAPI(api_key="test")
        api.session = session

        text, _ = api.generate_content("model", "prompt")

        self.assertEqual(text, "--- message ---\nok")
        self.assertEqual(len(session.calls), 3)

    @patch("utils.api.base.time.sleep", return_value=None)
    def test_does_not_sleep_after_last_failed_attempt(self, sleep):
        session = _FakeSession([_FakeResponse(500, text="server error") for _ in range(6)])
        api = OpenAIResponsesAPI(api_key="test")
        api.session = session

        result = api.generate_content("model", "prompt")

        self.assertEqual(result, (None, None))
        self.assertEqual(len(session.calls), 6)
        self.assertEqual(sleep.call_count, 5)

    @patch("utils.api.base.time.sleep", return_value=None)
    def test_retries_timeouts_and_connection_errors(self, _sleep):
        success = {
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]
        }
        session = _FakeSession([
            requests.Timeout("timed out"),
            requests.ConnectionError("disconnected"),
            _FakeResponse(200, success),
        ])
        api = OpenAIResponsesAPI(api_key="test")
        api.session = session

        text, _ = api.generate_content("model", "prompt")

        self.assertEqual(text, "--- message ---\nok")
        self.assertEqual(len(session.calls), 3)

    @patch("utils.api.base.time.sleep", return_value=None)
    def test_does_not_retry_non_transient_request_errors(self, sleep):
        session = _FakeSession([requests.exceptions.InvalidURL("invalid URL")])
        api = OpenAIResponsesAPI(api_key="test")
        api.session = session

        result = api.generate_content("model", "prompt")

        self.assertEqual(result, (None, None))
        self.assertEqual(len(session.calls), 1)
        sleep.assert_not_called()


class OpenAIChatAPITest(unittest.TestCase):
    @staticmethod
    def _pdf_cache_path(
        temp_dir: str,
        pdf_path: Path,
        max_pages: int = 30,
        dpi: int = 72,
    ) -> Path:
        cache_key = hashlib.sha1(
            f"{pdf_path}|{pdf_path.stat().st_mtime}|{dpi}|{max_pages}".encode()
        ).hexdigest()[:16]
        return Path(temp_dir) / f"openai_chat_pdf_v2_{cache_key}.tsv"

    def test_legacy_pdf_cache_is_ignored(self):
        class FakePixmap:
            def tobytes(self, _output_format):
                return b"fresh-png"

        class FakePage:
            def get_pixmap(self, matrix, alpha):
                self.matrix = matrix
                self.alpha = alpha
                return FakePixmap()

        class FakeDocument:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def __len__(self):
                return 1

            def load_page(self, page_index):
                self.page_index = page_index
                return FakePage()

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "slides.pdf"
            pdf_path.write_bytes(b"pdf")
            current_cache_path = self._pdf_cache_path(temp_dir, pdf_path)
            legacy_cache_path = current_cache_path.with_name(
                current_cache_path.name.replace(
                    "openai_chat_pdf_v2_", "openai_chat_pdf_", 1
                )
            )
            legacy_cache_path.write_text(
                base64.b64encode(b"stale-partial-png").decode("utf-8") + "\n",
                encoding="utf-8",
            )
            fitz_module = types.SimpleNamespace(
                open=lambda _path: FakeDocument(),
                Matrix=lambda *_args: object(),
            )

            with (
                patch.dict(sys.modules, {"fitz": fitz_module}),
                patch("utils.api.base.tempfile.gettempdir", return_value=temp_dir),
            ):
                contents = OpenAIChatAPI._pdf_to_image_contents(str(pdf_path))

            self.assertEqual(
                base64.b64decode(contents[0]["image_url"]["url"].split(",", 1)[1]),
                b"fresh-png",
            )
            self.assertTrue(current_cache_path.exists())

    def test_pdf_cache_hit_skips_rendering(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "slides.pdf"
            pdf_path.write_bytes(b"pdf")
            cache_path = self._pdf_cache_path(temp_dir, pdf_path)
            cache_path.write_text(
                base64.b64encode(b"cached-png").decode("utf-8") + "\n",
                encoding="utf-8",
            )
            fitz_open = Mock(side_effect=AssertionError("不应重新渲染"))
            fitz_module = types.SimpleNamespace(open=fitz_open, Matrix=Mock())

            with (
                patch.dict(sys.modules, {"fitz": fitz_module}),
                patch("utils.api.base.tempfile.gettempdir", return_value=temp_dir),
            ):
                contents = OpenAIChatAPI._pdf_to_image_contents(str(pdf_path))

            fitz_open.assert_not_called()
            self.assertEqual(
                contents[0]["image_url"]["url"],
                "data:image/png;base64,"
                + base64.b64encode(b"cached-png").decode("utf-8"),
            )

    def test_corrupt_pdf_cache_is_rebuilt_and_published_atomically(self):
        class FakePixmap:
            def tobytes(self, output_format):
                self.output_format = output_format
                return b"fresh-png"

        class FakePage:
            def get_pixmap(self, matrix, alpha):
                self.matrix = matrix
                self.alpha = alpha
                return FakePixmap()

        class FakeDocument:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def __len__(self):
                return 1

            def load_page(self, page_index):
                self.page_index = page_index
                return FakePage()

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "slides.pdf"
            pdf_path.write_bytes(b"pdf")
            cache_path = self._pdf_cache_path(temp_dir, pdf_path)
            cache_path.write_text("not-valid-base64\n", encoding="utf-8")
            fitz_module = types.SimpleNamespace(
                open=lambda _path: FakeDocument(),
                Matrix=lambda *_args: object(),
            )

            with (
                patch.dict(sys.modules, {"fitz": fitz_module}),
                patch("utils.api.base.tempfile.gettempdir", return_value=temp_dir),
                patch("utils.api.base.os.replace", wraps=os.replace) as replace,
            ):
                contents = OpenAIChatAPI._pdf_to_image_contents(str(pdf_path))

            replace.assert_called_once()
            self.assertEqual(
                base64.b64decode(contents[0]["image_url"]["url"].split(",", 1)[1]),
                b"fresh-png",
            )
            self.assertEqual(
                base64.b64decode(cache_path.read_text(encoding="utf-8").strip()),
                b"fresh-png",
            )

    def test_reasoning_split_is_opt_in(self):
        session = _FakeSession([
            _FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]})
        ])
        api = OpenAIChatAPI(api_key="test", base_url="https://example.test")
        api.session = session

        api.generate_content("model", "prompt")

        payload = session.calls[0][1]["json"]
        self.assertNotIn("reasoning_split", payload)

    def test_judge_api_forwards_thinking_level_to_chat_api(self):
        session = _FakeSession([
            _FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]})
        ])
        api = OpenAIChatAPI(
            api_key="test",
            base_url="https://example.test",
            reasoning_split=True,
        )
        api.session = session

        JudgeAPI(api).generate_content(
            model="model",
            prompt="prompt",
            thinking_level="disabled",
        )

        payload = session.calls[0][1]["json"]
        self.assertTrue(payload["reasoning_split"])
        self.assertEqual(payload["thinking"], {"type": "disabled"})


if __name__ == "__main__":
    unittest.main()
