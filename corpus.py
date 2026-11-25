"""Use bytes as tokens and keep contiguous train, validation, and test splits."""

from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True)
class Corpus:
    train: bytes
    validation: bytes
    test: bytes
    sha256: str

    def metadata(self, name):
        split_bytes = {
            "train": len(self.train), "validation": len(self.validation), "test": len(self.test),
        }
        return {
            "name": name, "sha256": self.sha256,
            "bytes": sum(split_bytes.values()), "split_bytes": split_bytes,
        }


def load_corpus(path, context):
    raw = path.read_bytes()
    train_end, validation_end = len(raw) * 8 // 10, len(raw) * 9 // 10
    corpus = Corpus(raw[:train_end], raw[train_end:validation_end], raw[validation_end:],
                    sha256(raw).hexdigest())
    if min(len(corpus.train), len(corpus.validation), len(corpus.test)) <= context:
        raise ValueError(
            f"Training, validation and test splits each need at least {context + 1} bytes."
        )
    return corpus
