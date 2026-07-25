import tempfile
import unittest
from pathlib import Path

from utils.paths import get_valid_item_dirs


class PathsTest(unittest.TestCase):
    def test_get_valid_item_dirs_default_skips_pycache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "case-a").mkdir()
            (root / "__pycache__").mkdir()

            result = get_valid_item_dirs(root)

            self.assertEqual(result, [root / "case-a"])


if __name__ == "__main__":
    unittest.main()
