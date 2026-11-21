import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from dronevision import session


class SessionPathsTest(unittest.TestCase):
    def test_moved_data_root_is_used_on_load_and_save(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            old_root = root / "old disk" / "DRONE"
            new_root = root / "новый диск" / "DRONE"
            new_root.mkdir(parents=True)
            (new_root / "DJI_0001.MP4").write_bytes(b"video")
            last = root / "profile" / "last_session.json"
            previous = last.with_name("previous_session.json")
            raw = {
                "format": session.FORMAT,
                "version": session.VERSION,
                "form": {
                    "video_path": str(old_root / "DJI_0001.MP4"),
                    "project_dir": str(old_root / "work" / "dji_0001"),
                },
                "selected_frames": [{"time_s": 1.25}],
                "results": {"note": "kept"},
            }
            last.parent.mkdir(parents=True)
            last.write_text(json.dumps(raw), encoding="utf-8")
            with patch("dronevision.session.LAST_SESSION", last), \
                 patch("dronevision.session.PREVIOUS_SESSION", previous), \
                 patch("dronevision.paths.data_root", return_value=new_root):
                loaded = session.load_session()
                self.assertTrue(loaded["video_exists"])
                self.assertEqual(loaded["session"]["form"]["video_path"], str(new_root / "DJI_0001.MP4"))
                self.assertEqual(loaded["session"]["form"]["project_dir"], str(new_root / "work" / "dji_0001"))

                # Simulate an old browser tab posting its stale path without bulky results.
                stale = {**raw, "results": None}
                stale.pop("results")
                session.save_session(stale, keep_results=True)
                stored = json.loads(last.read_text(encoding="utf-8"))
                self.assertEqual(stored["form"]["video_path"], str(new_root / "DJI_0001.MP4"))
                self.assertEqual(stored["results"], {"note": "kept"})


if __name__ == "__main__":
    unittest.main()
