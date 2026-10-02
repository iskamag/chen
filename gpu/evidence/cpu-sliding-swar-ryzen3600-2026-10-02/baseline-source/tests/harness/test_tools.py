import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts.fetch_match_tools import extract, fetch


class MatchToolArchiveTests(unittest.TestCase):
    def test_archive_cannot_escape_or_create_links(self):
        for name, link in [("../escape", False), ("/absolute", False), ("link", True)]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                archive = root / "input.tar"
                with tarfile.open(archive, "w") as stream:
                    member = tarfile.TarInfo(name)
                    if link:
                        member.type = tarfile.SYMTYPE
                        member.linkname = "../escape"
                    else:
                        member.size = 4
                    stream.addfile(member, None if link else io.BytesIO(b"test"))
                with self.assertRaises(ValueError):
                    extract(archive, root / "output")
                self.assertFalse((root / "escape").exists())
                self.assertFalse((root / "output" / "link").exists())

    def test_release_digest_checked_before_extraction_or_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "fastchess.tar").write_bytes(b"corrupt-release")
            with patch("platform.system", return_value="Linux"), patch("platform.machine", return_value="x86_64"):
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    fetch(root / "installation", root)
            self.assertFalse((root / "installation").exists())


if __name__ == "__main__":
    unittest.main()
