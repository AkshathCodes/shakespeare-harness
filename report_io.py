"""Publish a complete JSON report without exposing an incomplete file."""

from dataclasses import asdict
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def save_report(path, report, *, overwrite):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".harness-", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, default=asdict, indent=2, allow_nan=False)
            stream.write("\n")
        if overwrite:
            os.replace(temporary, path)
        else:
            # Link the finished file without replacing a concurrent writer's result.
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"{path} already exists. Choose a new --output or use --overwrite."
                ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
