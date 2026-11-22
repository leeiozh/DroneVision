import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dronevision.discovery import _find_near, _is_mount


class DiscoveryPathTests(unittest.TestCase):
    def test_unsupported_mount_check_is_not_fatal(self):
        with patch.object(Path, "is_mount", side_effect=NotImplementedError):
            self.assertFalse(_is_mount(Path("C:/DroneData")))

    def test_find_near_continues_when_mount_check_is_unsupported(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "DroneData" / "media" / "videos"
            root.mkdir(parents=True)
            video = root / "DJI_0001.MP4"
            video.touch()
            srt = root / "DJI_0001.SRT"
            srt.touch()

            with patch.object(Path, "is_mount", side_effect=NotImplementedError):
                found = _find_near(video, "DJI_0001.SRT")

            self.assertIn(srt.resolve(), found)


if __name__ == "__main__":
    unittest.main()
