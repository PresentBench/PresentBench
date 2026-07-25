import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils import score_utils
from utils.score_utils import find_single_score_yaml
from utils.statistics import calculate_average_scores


class ScoreUtilsTest(unittest.TestCase):
    def test_find_single_score_yaml_defaults_to_newest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            older = root / "judge_2026-01-01_00-00-00_score.yaml"
            newer = root / "judge_2026-01-02_00-00-00_score.yaml"
            older.touch()
            newer.touch()

            self.assertEqual(find_single_score_yaml(root), newer)

    def test_namespaced_model_matches_sanitized_score_filename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            score_file = root / "model_2026-01-02_00-00-00_score.yaml"
            score_file.touch()

            result = find_single_score_yaml(root, judge_model="org/model")

            self.assertEqual(result, score_file)

    def test_model_filename_prefix_sanitizes_cross_platform_separators(self):
        helper = getattr(score_utils, "model_filename_prefix", None)

        self.assertIsNotNone(helper)
        self.assertEqual(helper(r"org\model:preview"), "model-preview")


class CalculateAverageScoresTest(unittest.TestCase):
    def test_cli_keeps_judge_model_in_default_output_filename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            argv = [
                "calculate_average_scores",
                "--result_root_dir",
                str(root),
                "--judge_model",
                "org/model",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    calculate_average_scores,
                    "collect_all_scores",
                    return_value={"source": [{"score": 1}]},
                ),
                patch.object(calculate_average_scores, "build_result_tree", return_value={}),
                patch.object(
                    calculate_average_scores,
                    "format_output",
                    return_value={"average": None},
                ),
            ):
                calculate_average_scores.main()

            self.assertTrue((root / "average_scores__model.yaml").exists())
            self.assertFalse((root / "average_scores.yaml").exists())


if __name__ == "__main__":
    unittest.main()
