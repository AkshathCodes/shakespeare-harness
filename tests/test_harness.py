import io
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class HarnessTest(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.corpus = self.folder / "training text.txt"
        self.corpus.write_bytes(b"Learning takes practice. " * 24)
        self.output = self.folder / "runs" / "example.json"

    def cli(self, *args, without_torch=False):
        python = [sys.executable, "-S"] if without_torch else [sys.executable]
        return subprocess.run(
            [*python, "-m", "harness", *map(str, args)],
            cwd=ROOT, capture_output=True, text=True, timeout=90,
        )

    def test_help_and_planning_do_not_require_torch(self):
        for command in ((), ("doctor",), ("plan",), ("run",), ("report",)):
            with self.subTest(command=command):
                help_text = self.cli(*command, "--help", without_torch=True)
                self.assertEqual(help_text.returncode, 0, help_text.stderr)
                self.assertIn("Example:", help_text.stdout)
        plan = self.cli("plan", self.corpus, "--device", "cuda", without_torch=True)
        self.assertEqual(plan.returncode, 0, plan.stderr)
        self.assertIn("45 total", plan.stdout)
        self.assertIn("1,698,416,640", plan.stdout)
        self.assertIn("1,699,430,400", plan.stdout)
        self.assertIn("1,013,760", plan.stdout)
        self.assertFalse(self.output.parent.exists())

    def test_doctor_checks_cpu_and_explains_missing_torch(self):
        doctor = self.cli("doctor", "--device", "cpu")
        self.assertEqual(doctor.returncode, 0, doctor.stderr)
        self.assertIn("CPU ready", doctor.stdout)
        self.assertIn("PyTorch", doctor.stdout)
        missing = self.cli("doctor", without_torch=True)
        self.assertEqual(missing.returncode, 1, missing.stderr)
        self.assertIn("requirements.txt", missing.stderr)

    def test_plan_rejects_invalid_experiment_options_before_importing_torch(self):
        for flags, error in (
            (("--seeds", "7", "7"), "distinct"),
            (("--seeds", str(2**63)), "2**63"),
            (("--proxy-width", "16", "--target-width", "8"), "smaller"),
            (("--proxy-width", "0"), "positive"),
            (("--target-steps", "0"), "positive"),
            (("--target-steps", "1"), "zero proxy steps"),
        ):
            with self.subTest(flags=flags):
                result = self.cli("plan", self.corpus, *flags, without_torch=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(error, result.stderr)

        self.corpus.write_bytes(b"too small")
        result = self.cli("plan", self.corpus, without_torch=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("splits each need", result.stderr)

    def test_run_agrees_with_plan_and_only_replaces_output_when_requested(self):
        options = ("--proxy-width", "7", "--target-width", "20", "--target-steps", "4",
                   "--seeds", "5")
        plan = self.cli("plan", self.corpus, *options, without_torch=True)
        self.assertEqual(plan.returncode, 0, plan.stderr)
        self.output.parent.mkdir()
        self.output.write_text("keep this", encoding="utf-8")
        blocked = self.cli("run", self.corpus, *options, "--output", self.output,
                           without_torch=True)
        self.assertEqual(blocked.returncode, 1, blocked.stderr)
        self.assertIn("--overwrite", blocked.stderr)
        self.assertEqual(self.output.read_text(encoding="utf-8"), "keep this")
        ran = self.cli("run", self.corpus, *options, "--output", self.output, "--overwrite")
        self.assertEqual(ran.returncode, 0, ran.stderr)
        self.assertIn("[1/1] Seed 5", ran.stdout)
        saved = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(saved["model_widths"], {"proxy": 7, "target": 20})
        self.assertIn(saved["dataset"]["sha256"], plan.stdout)
        self.assertEqual(saved["budget"]["target_steps"], 4)
        row, = saved["repeats"]
        self.assertEqual(row["seed"], 5)
        self.assertEqual(sum(len(trials) for trials in (
            row["proxy_trials"], row["screened"]["trials"], row["direct"]["trials"]
        )), 15)
        for strategy in ("screened", "direct"):
            self.assertIn(f"{row[strategy]['estimated_flops']:,}", plan.stdout)
        summary = self.cli("report", self.output, without_torch=True)
        self.assertEqual(summary.returncode, 0, summary.stderr)
        self.assertIn("Seed 5", summary.stdout)
        self.assertIn("Winning recipes", summary.stdout)

    def test_report_reads_published_results_and_rejects_malformed_data(self):
        old = self.cli("report", ROOT / "results" / "tinyshakespeare.json", without_torch=True)
        self.assertEqual(old.returncode, 0, old.stderr)
        self.assertIn("-0.1696", old.stdout)
        self.assertIn("1 win, 2 ties, 0 losses", old.stdout)
        malformed = self.folder / "bad.json"
        for content in ("{", '{"repeats": []}', '{"repeats": [null]}'):
            with self.subTest(content=content):
                malformed.write_text(content, encoding="utf-8")
                result = self.cli("report", malformed, without_torch=True)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("bad.json", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_run_rejects_corpus_as_output_before_training(self):
        alias = self.folder / "corpus-alias.txt"
        os.link(self.corpus, alias)
        for output in (self.corpus, alias):
            with self.subTest(output=output):
                result = self.cli("run", self.corpus, "--output", output, "--overwrite",
                                  without_torch=True)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("different file than the corpus", result.stderr)
        self.assertEqual(self.corpus.read_bytes(), b"Learning takes practice. " * 24)

    def test_run_rejects_output_beneath_a_file_before_training(self):
        blocked = self.folder / "not-a-directory"
        blocked.write_text("keep this", encoding="utf-8")
        result = self.cli("run", self.corpus, "--output", blocked / "report.json",
                          without_torch=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("not a directory", result.stderr)
        self.assertIn(blocked.name, result.stderr)

    def test_atomic_publication_preserves_existing_results_and_cleans_temp_files(self):
        from report_io import save_report

        self.output.parent.mkdir()
        self.output.write_text("previous report", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            save_report(self.output, {"ready": True}, overwrite=False)
        with patch("report_io.json.dump", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                save_report(self.output, {"ready": True}, overwrite=True)
        self.assertEqual(self.output.read_text(encoding="utf-8"), "previous report")
        self.assertEqual(list(self.output.parent.iterdir()), [self.output])
        fresh = self.output.with_name("fresh.json")
        save_report(fresh, {"ready": True}, overwrite=False)
        self.assertEqual(json.loads(fresh.read_text(encoding="utf-8")), {"ready": True})

    def test_interrupted_run_returns_130_and_preserves_report(self):
        import harness

        self.output.parent.mkdir()
        self.output.write_text("previous report", encoding="utf-8")
        with (patch("harness.check_device"), patch("run.execute", side_effect=KeyboardInterrupt),
              redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as stderr):
            code = harness.main(["run", str(self.corpus), "--output", str(self.output), "--overwrite"])
        self.assertEqual(code, 130)
        self.assertIn("interrupted", stderr.getvalue())
        self.assertEqual(self.output.read_text(encoding="utf-8"), "previous report")
