from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from dronevision.dialog import _start_dir


class DialogPathsTest(unittest.TestCase):
    def test_missing_child_starts_in_nearest_existing_parent(self):
        with TemporaryDirectory(prefix="DroneVision путь ") as folder:
            root = Path(folder)
            self.assertEqual(_start_dir(str(root / "ещё" / "нет")), root)

    def test_file_starts_in_its_parent(self):
        with TemporaryDirectory(prefix="DroneVision путь ") as folder:
            file = Path(folder) / "сессия.dvsession.json"
            file.write_text("{}", encoding="utf-8")
            self.assertEqual(_start_dir(str(file)), file.parent)


if __name__ == "__main__":
    unittest.main()
