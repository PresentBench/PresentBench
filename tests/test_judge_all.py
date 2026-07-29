import tempfile
import unittest
from pathlib import Path
from unittest import mock

import judge_all


class JudgeAllCliTest(unittest.TestCase):
    def test_run_once_forwards_retry_to_judge(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            result_root = root / "results"
            data_item_dir = data_root / "academic" / "source" / "case"
            generation_task = data_item_dir / "generation_task"
            agent_results = (
                result_root
                / "academic"
                / "source"
                / "case"
                / "generation_task"
                / "results"
            )
            generation_task.mkdir(parents=True)
            agent_results.mkdir(parents=True)
            (data_item_dir / "material.md").write_text("material", encoding="utf-8")
            (generation_task / "judge_prompt.json").write_text("{}", encoding="utf-8")
            (agent_results / "slides_generation_failed.txt").write_text(
                "failed", encoding="utf-8"
            )

            with mock.patch.object(judge_all.judge, "main") as judge_main:
                result = judge_all.run_once(
                    api_type="gemini",
                    model="test-model",
                    thinking_level=None,
                    type_name="academic",
                    data_item_dir=data_item_dir,
                    result_root=result_root,
                    data_root=data_root,
                    retry=9,
                    temperature=0.25,
                    seed=7,
                )

            self.assertIsNone(result[2])
            self.assertEqual(judge_main.call_count, 1)
            judge_args = judge_main.call_args.kwargs["args"]
            self.assertEqual(judge_args.retry, 9)
            self.assertEqual(judge_args.temperature, 0.25)
            self.assertEqual(judge_args.seed, 7)


if __name__ == "__main__":
    unittest.main()
