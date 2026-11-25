"""Run the training transfer benchmark and write its complete experiment record."""

import argparse
from dataclasses import asdict
import math
from pathlib import Path
import platform
import random
from time import perf_counter

import torch
from torch.nn import functional as F

from corpus import load_corpus
from experiment import (
    BATCH_SIZE, CONTEXT, DEFAULT_SEEDS, DEFAULT_TARGET_STEPS, EVAL_WINDOWS, LOSS_UNIT,
    WIDTHS, Measurement, compare, experiment_budget, summarize, validate_seeds,
)
from model import ByteTransformer
from report_io import save_report


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
    def __init__(self, corpus, device, widths=WIDTHS):
        self.corpus = corpus
        self.device = device
        self.widths = widths

    def train(self, recipe, scale, steps, seed):
        cuda = self.device.type == "cuda"
        if cuda:
            torch.cuda.synchronize(self.device)
            baseline = torch.cuda.memory_allocated(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        start = perf_counter()
        torch.manual_seed(seed)
        model = ByteTransformer(self.widths[scale], CONTEXT).to(self.device)
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


def execute(corpus, corpus_name, widths, budget, seeds, device, on_seed):
    torch.set_num_threads(1)
    trainer = Trainer(corpus, device, widths)
    repeats = []
    for index, seed in enumerate(seeds, start=1):
        result = compare(trainer, budget, seed)
        repeats.append(result)
        on_seed(index, result)
    return {
        "dataset": corpus.metadata(corpus_name),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "model_widths": widths,
        "context": CONTEXT,
        "batch_size": BATCH_SIZE,
        "max_eval_windows": EVAL_WINDOWS,
        "loss_unit": LOSS_UNIT,
        "budget": {**asdict(budget), "cap": budget.cap},
        "repeats": repeats,
        "summary": summarize(repeats),
    }


def print_seed(result, prefix=""):
    screened = result["screened"]["held_out_loss"]
    direct = result["direct"]["held_out_loss"]
    print(f"{prefix}Seed {result['seed']}: screened {screened:.4f} | direct {direct:.4f}"
          " held-out bits/byte", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="Local text file, read as raw bytes")
    parser.add_argument("--target-steps", type=int, default=DEFAULT_TARGET_STEPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--output", type=Path, default=Path("runs/results.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    try:
        validate_seeds(args.seeds)
        if args.output.resolve() == args.corpus.resolve():
            raise ValueError("Write the output to a different file than the corpus.")
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable. Use --device cpu or install a CUDA-enabled PyTorch build.")
        device = torch.device(args.device)
        corpus = load_corpus(args.corpus, CONTEXT)
        budget = experiment_budget(WIDTHS, args.target_steps)
        report = execute(
            corpus, args.corpus.name, WIDTHS, budget, args.seeds, device,
            on_seed=lambda _, result: print_seed(result),
        )
        save_report(args.output, report, overwrite=True)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Screened/direct estimated FLOPs: {budget.screening_cost:,} / {budget.cap:,} per seed")
    delta = report["summary"]["mean_delta"]
    print(f"Mean held-out delta (screened - direct): {delta:+.4f} bits/byte")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
