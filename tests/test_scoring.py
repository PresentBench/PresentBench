import unittest

from scoring import score_section


class ScoringTest(unittest.TestCase):
    def test_unanswered_and_not_applicable_items_keep_historical_denominator(self):
        answers = {
            "1": {
                "1.1": {"answer": "yes"},
                "1.2": {"answer": "no"},
                "1.3": {"answer": "not applicable"},
                "1.4": {"answer": None},
            }
        }

        total, class_info = score_section(answers, {"1": 10})

        self.assertEqual(total, 2.5)
        self.assertEqual(class_info["1"]["yes_count"], 1)
        self.assertEqual(class_info["1"]["valid_count"], 4)
        self.assertEqual(class_info["1"]["not_applicable_count"], 1)


if __name__ == "__main__":
    unittest.main()
