import subprocess
import sys
import unittest
from pathlib import Path


class AnalyticCodebookChannelTests(unittest.TestCase):
    def test_runs_as_script_outside_repo_root(self):
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(repo_root / "scripts/analytic_codebook_channel.py"), "--help"],
            cwd=repo_root.parent,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--samples-per-bin", result.stdout)
