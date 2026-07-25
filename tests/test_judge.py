import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import judge


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


if __name__ == "__main__":
    unittest.main()
