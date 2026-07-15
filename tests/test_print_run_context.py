import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.print_run_context import sha256_file


class PrintRunContextTests(unittest.TestCase):
    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.bin"
            path.write_bytes(b"fastmri")
            self.assertEqual(
                sha256_file(path), hashlib.sha256(b"fastmri").hexdigest()
            )


if __name__ == "__main__":
    unittest.main()
