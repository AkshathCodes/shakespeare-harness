from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from corpus import load_corpus


class CorpusTest(unittest.TestCase):
    def test_splits_are_contiguous_and_fingerprinted(self):
        raw = bytes(range(200))
        with TemporaryDirectory() as folder:
            path = Path(folder) / "corpus.txt"
            path.write_bytes(raw)
            corpus = load_corpus(path, context=10)
        self.assertEqual(corpus.train, raw[:160])
        self.assertEqual(corpus.validation, raw[160:180])
        self.assertEqual(corpus.test, raw[180:])
        self.assertEqual(corpus.sha256, sha256(raw).hexdigest())

    def test_every_split_must_hold_a_full_context_and_target(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "corpus.txt"
            path.write_bytes(b"a" * 100)
            with self.assertRaisesRegex(ValueError, "validation and test"):
                load_corpus(path, context=10)
