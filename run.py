"""Run the training transfer benchmark and write its complete experiment record."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import random
from time import perf_counter

import torch
from torch.nn import functional as F

from corpus import load_corpus
from experiment import Measurement, compare, plan_budget, summarize
from model import ByteTransformer, estimated_step_flops


WIDTHS = {"proxy": 8, "target": 16}
CONTEXT = 12
BATCH_SIZE = 2
EVAL_WINDOWS = 32


def batch(data, offsets, device):
    windows = torch.tensor(
        [list(data[start:start + CONTEXT + 1]) for start in offsets],
        dtype=torch.long, device=device,
    )
    return windows[:, :-1], windows[:, 1:]


def loss(model, inputs, targets):
    return F.cross_entropy(model(inputs).flatten(0, 1), targets.reshape(-1))


@torch.inference_mode()
def evaluate(model, data, device):
    blocks = (len(data) - 1) // CONTEXT
    windows = min(EVAL_WINDOWS, blocks)
    # Use a fixed sample of disjoint target windows spread across the split.
    offsets = [i * blocks // windows * CONTEXT for i in range(windows)]
    model.eval()
    return loss(model, *batch(data, offsets, device)).item() / math.log(2)


class Trainer:
    def __init__(self, corpus, device):
        self.corpus = corpus
        self.device = device

    def train(self, recipe, scale, steps, seed):
        cuda = self.device.type == "cuda"
        if cuda:
            torch.cuda.synchronize(self.device)
            baseline = torch.cuda.memory_allocated(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        start = perf_counter()
        torch.manual_seed(seed)
        model = ByteTransformer(WIDTHS[scale], CONTEXT).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=recipe.learning_rate, foreach=False)
        sampler = random.Random(seed ^ 0xBAD5EED)
        data = self.corpus.train
        for step in range(steps):
            offsets = [sampler.randrange(len(data) - CONTEXT) for _ in range(BATCH_SIZE)]
            inputs, targets = batch(data, offsets, self.device)
            if recipe.schedule == "cosine":
                factor = 0.1 + 0.9 * (1 + math.cos(math.pi * step / steps)) / 2
                optimizer.param_groups[0]["lr"] = recipe.learning_rate * factor
            optimizer.zero_grad(set_to_none=True)
            loss(model, inputs, targets).backward()
            optimizer.step()

        # Discard the final gradients before measuring validation.
        optimizer.zero_grad(set_to_none=True)
        validation = evaluate(model, self.corpus.validation, self.device)
        peak = (torch.cuda.max_memory_allocated(self.device) - baseline) / 2**20 if cuda else None
        scores = Measurement(validation if math.isfinite(validation) else None,
                             perf_counter() - start, peak)
        return scores, model

    def held_out_loss(self, checkpoint):
        return evaluate(checkpoint, self.corpus.test, self.device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="Local text file, read as raw bytes")
    parser.add_argument("--target-steps", type=int, default=400)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 41])
    parser.add_argument("--output", type=Path, default=Path("runs/results.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    try:
        if len(set(args.seeds)) != len(args.seeds):
            raise ValueError("Use distinct seeds for each repeat.")
        if any(not 0 <= seed < 2**63 for seed in args.seeds):
            raise ValueError("Seeds must be between 0 and 2**63 - 1.")
        if args.output.resolve() == args.corpus.resolve():
            raise ValueError("Write the output to a different file than the corpus.")
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable. Use --device cpu or install a CUDA-enabled PyTorch build.")
        torch.set_num_threads(1)
        device = torch.device(args.device)
        corpus = load_corpus(args.corpus, CONTEXT)
        budget = plan_budget(
            estimated_step_flops(WIDTHS["proxy"], CONTEXT, BATCH_SIZE),
            estimated_step_flops(WIDTHS["target"], CONTEXT, BATCH_SIZE),
            args.target_steps,
        )
        trainer = Trainer(corpus, device)
        repeats = []
        for seed in args.seeds:
            result = compare(trainer, budget, seed)
            repeats.append(result)
            screened = result["screened"]["held_out_loss"]
            direct = result["direct"]["held_out_loss"]
            print(f"Seed {seed}: screened {screened:.4f} | direct {direct:.4f} held-out bits/byte",
                  flush=True)
        summary = summarize(repeats)
        report = {
            "dataset": {
                "name": args.corpus.name, "sha256": corpus.sha256,
                "bytes": sum(map(len, (corpus.train, corpus.validation, corpus.test))),
                "split_bytes": {name: len(getattr(corpus, name))
                                for name in ("train", "validation", "test")},
            },
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": args.device,
            "device_name": torch.cuda.get_device_name(device) if args.device == "cuda" else None,
            "model_widths": WIDTHS,
            "context": CONTEXT,
            "batch_size": BATCH_SIZE,
            "max_eval_windows": EVAL_WINDOWS,
            "loss_unit": "bits per byte",
            "budget": {**asdict(budget), "cap": budget.cap},
            "repeats": repeats,
            "summary": summary,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, default=asdict, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Screened/direct estimated FLOPs: {budget.screening_cost:,} / {budget.cap:,} per seed")
    print(f"Mean held-out delta (screened - direct): {summary['mean_delta']:+.4f} bits/byte")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
