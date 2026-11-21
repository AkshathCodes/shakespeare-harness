import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class CommandLineTest(unittest.TestCase):
    def test_runs_real_training_and_writes_a_paired_report(self):
        with TemporaryDirectory() as folder:
            corpus = Path(folder) / "text.txt"
            corpus.write_bytes(b"Learning takes practice. " * 24)
            report = Path(folder) / "runs" / "smoke.json"
            completed = subprocess.run(
                [sys.executable, "run.py", str(corpus), "--target-steps", "4",
                 "--seeds", "1", "--output", str(report)],
                cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Mean held-out delta", completed.stdout)
            data = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(data["dataset"]["bytes"], 600)
        self.assertEqual(len(data["dataset"]["sha256"]), 64)
        run, = data["repeats"]
        self.assertEqual(len(run["proxy_trials"]), 8)
        self.assertEqual(len(run["direct"]["trials"]), 4)
        self.assertEqual(len(run["screened"]["trials"]), 3)
        self.assertGreater(run["screened"]["held_out_loss"], 0)
        self.assertLessEqual(run["screened"]["estimated_flops"], run["direct"]["estimated_flops"])
        self.assertIn("learning_rate", run["screened"]["winner"]["recipe"])
