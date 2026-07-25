import argparse
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import judge
from utils.api.base import OpenAIChatAPI
from utils.score_utils import find_single_score_yaml


class _FakeJudgeAPI:
    def __init__(self, responses):
        self.responses = list(responses)

    def generate_content(self, **_kwargs):
        text = self.responses.pop(0)
        return text, {"text": text}


class JudgeParsingTest(unittest.TestCase):
    def _context(self, response):
        return judge.JudgeContext(
            judge_api=_FakeJudgeAPI([response]),
            model="model",
            thinking_level=None,
            slides_obj=None,
            material_objs=None,
            slides_path="slides.pdf",
            material_paths=None,
            max_retries=1,
        )

    def test_normalizes_latex_wrapped_boxed_answer(self):
        _, _, result = judge.process_single_item(
            (0, 0, "prompt"),
            self._context(r"结论：\boxed{\text{no}}"),
        )

        self.assertEqual(result["answer"], "no")

    def test_uses_last_valid_boxed_verdict(self):
        _, _, result = judge.process_single_item(
            (0, 0, "prompt"),
            self._context(r"示例：\boxed{maybe}。最终：\boxed{yes}"),
        )

        self.assertEqual(result["answer"], "yes")


class JudgeFilenameTest(unittest.TestCase):
    def test_model_filename_prefix_removes_namespace(self):
        helper = getattr(judge, "model_filename_prefix", None)

        self.assertIsNotNone(helper)
        self.assertEqual(helper("org/model", "high"), "model_high")
        self.assertEqual(helper(r"org\model", "high"), "model_high")

    def test_namespaced_model_matches_sanitized_score_filename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            score_path = Path(temp_dir) / "model_2026-07-25_12-00-00_score.yaml"
            score_path.touch()

            selected = find_single_score_yaml(
                Path(temp_dir),
                judge_model="org/model",
            )

            self.assertEqual(selected, score_path)

    def test_zero_score_output_stays_in_result_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = argparse.Namespace(
                output=None,
                slides=str(root / "slides.pdf"),
                model="org/model",
                thinking_level=None,
                weights_path=None,
                zero_score=True,
            )

            judge.main(args)

            score_files = list(root.glob("*_score.yaml"))
            self.assertEqual(len(score_files), 1)
            self.assertFalse((root / "org").exists())

    def test_main_does_not_reconfigure_callers_logging(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = argparse.Namespace(
                output=str(root / "score.yaml"),
                slides=str(root / "slides.pdf"),
                model="model",
                thinking_level=None,
                weights_path=None,
                zero_score=True,
            )

            with patch("judge.logging.basicConfig") as basic_config:
                judge.main(args)

            basic_config.assert_not_called()


class JudgeApiFactoryTest(unittest.TestCase):
    def test_minimax_backend_uses_opt_in_reasoning_split(self):
        environment = {
            "MINIMAX_API_KEY": "test-key",
            "MINIMAX_BASE_URL": "https://example.test/v1/chat/completions",
        }
        with patch.dict(os.environ, environment, clear=False):
            adapter = judge.create_judge_api("minimax")

        self.assertIsInstance(adapter.api_client, OpenAIChatAPI)
        self.assertEqual(
            adapter.api_client.base_url,
            "https://example.test/v1/chat/completions",
        )
        self.assertTrue(adapter.api_client.reasoning_split)


if __name__ == "__main__":
    unittest.main()
