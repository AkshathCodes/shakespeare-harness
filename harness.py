"""Shakespeare Harness: plan, run, and inspect matched-compute training experiments."""

import argparse
import json
import math
import os
from pathlib import Path
import platform
import sys

from corpus import load_corpus
from experiment import (
    CONTEXT, DEFAULT_SEEDS, DEFAULT_TARGET_STEPS, DIRECT_TRIALS, LOSS_UNIT,
    PROMOTIONS, RECIPES, WIDTHS, experiment_budget, summarize, validate_seeds,
)
from report_io import save_report


class UsageError(ValueError):
    """An invalid value for a command-line argument."""


def add_experiment_options(parser):
    parser.add_argument("corpus", metavar="CORPUS", type=Path, help="Local training text")
    parser.add_argument("--proxy-width", type=int, default=WIDTHS["proxy"],
                        help="Proxy model width")
    parser.add_argument("--target-width", type=int, default=WIDTHS["target"],
                        help="Target model width")
    parser.add_argument("--target-steps", type=int, default=DEFAULT_TARGET_STEPS,
                        help="Training steps per target trial")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="Distinct repeat seeds")
    add_device_option(parser)


def add_device_option(parser):
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Training device")


def make_parser():
    parser = argparse.ArgumentParser(
        prog="python -m harness", description=__doc__, epilog="Example: python -m harness doctor",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    def command(name, description, example, handler):
        sub = commands.add_parser(
            name, help=description, description=description, epilog=f"Example: python -m harness {example}",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        sub.set_defaults(handler=handler, parser=sub)
        return sub

    doctor = command("doctor", "Check Python, PyTorch, and the selected device.", "doctor", doctor_command)
    add_device_option(doctor)
    plan = command("plan", "Preview trials and estimated compute without training.",
                   "plan data/tinyshakespeare.txt", plan_command)
    add_experiment_options(plan)
    run = command("run", "Train both strategies and save a JSON report.",
                  "run data/tinyshakespeare.txt --output runs/shakespeare.json", run_command)
    add_experiment_options(run)
    run.add_argument("--output", type=Path, default=Path("runs/results.json"), help="JSON report path")
    run.add_argument("--overwrite", action="store_true", help="Replace an existing report after success")
    report = command("report", "Summarize a saved JSON report.",
                     "report results/tinyshakespeare.json", report_command)
    report.add_argument("path", metavar="REPORT", type=Path, help="Saved JSON report")
    return parser


def load_torch():
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"PyTorch could not load. Install it from requirements.txt. {exc}") from exc
    return torch


def check_device(torch, name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Use --device cpu or install a CUDA-enabled PyTorch build.")
    device = torch.device(name)
    torch.empty(1, device=device)
    return device


def doctor_command(args):
    print(f"Python {platform.python_version()}", flush=True)
    torch = load_torch()
    print(f"PyTorch {torch.__version__}")
    print(f"CUDA available: {'yes' if torch.cuda.is_available() else 'no'}", flush=True)
    device = check_device(torch, args.device)
    print(f"{device.type.upper()} ready")


def prepare(args):
    try:
        validate_seeds(args.seeds)
        if min(args.proxy_width, args.target_width) < 1:
            raise ValueError("Model widths must be positive.")
        if args.proxy_width >= args.target_width:
            raise ValueError("--proxy-width must be smaller than --target-width.")
        if args.target_steps < 1:
            raise ValueError("--target-steps must be positive.")
        widths = {"proxy": args.proxy_width, "target": args.target_width}
        budget = experiment_budget(widths, args.target_steps)
    except ValueError as exc:
        raise UsageError(str(exc)) from exc
    try:
        corpus = load_corpus(args.corpus, CONTEXT)
    except ValueError as exc:
        raise ValueError(f"{args.corpus}: {exc}") from exc
    return corpus, widths, budget


def print_plan(args, corpus, widths, budget):
    data = corpus.metadata(args.corpus.name)
    train, validation, test = data["split_bytes"].values()
    trials = len(RECIPES) + PROMOTIONS + DIRECT_TRIALS
    print(f"Corpus        {data['name']}")
    print(f"SHA-256       {data['sha256']}")
    print(f"Split bytes   {train:,} train / {validation:,} validation / {test:,} test")
    print(f"Device        {args.device.upper()}")
    print(f"Model widths  {widths['proxy']} proxy / {widths['target']} target")
    repeats = len(args.seeds)
    print(f"Seeds         {', '.join(map(str, args.seeds))} ({repeats} {'repeat' if repeats == 1 else 'repeats'})")
    print(f"Trials        {trials} per seed ({trials * repeats} total)")
    print(f"Screened      {len(RECIPES)} proxy x {budget.proxy_steps} steps + "
          f"{PROMOTIONS} target x {budget.target_steps} steps, {budget.screening_cost:,} estimated FLOPs/seed")
    print(f"Direct        {DIRECT_TRIALS} target x {budget.target_steps} steps, "
          f"{budget.cap:,} estimated FLOPs/seed")
    print(f"Cap           {budget.cap:,} estimated FLOPs per strategy per seed")
    print(f"Unused        {budget.cap - budget.screening_cost:,} screened / 0 direct per seed", flush=True)


def plan_command(args):
    print_plan(args, *prepare(args))


def run_command(args):
    corpus, widths, budget = prepare(args)
    if args.output.resolve() == args.corpus.resolve() or (
        args.output.exists() and args.output.samefile(args.corpus)
    ):
        raise ValueError("Write the output to a different file than the corpus.")
    if args.output.is_dir():
        raise ValueError(f"{args.output} is a directory. Choose a JSON report file.")
    if args.output.parent.is_file():
        raise ValueError(f"{args.output.parent} is not a directory. Choose a different --output.")
    if os.path.lexists(args.output) and not args.overwrite:
        raise FileExistsError(f"{args.output} already exists. Choose a new --output or use --overwrite.")

    device = check_device(load_torch(), args.device)
    from run import execute, print_seed

    print_plan(args, corpus, widths, budget)
    report = execute(
        corpus, args.corpus.name, widths, budget, args.seeds, device,
        on_seed=lambda index, result: print_seed(result, f"[{index}/{len(args.seeds)}] "),
    )
    save_report(args.output, report, overwrite=args.overwrite)
    print_outcome(report["summary"])
    print(f"Report: {args.output}")


def print_outcome(summary):
    deltas = summary["paired_deltas"]
    wins = sum(delta < 0 for delta in deltas)
    ties = sum(delta == 0 for delta in deltas)
    losses = sum(delta > 0 for delta in deltas)
    print(f"Mean held-out delta (screened - direct): {summary['mean_delta']:+.4f} bits/byte")
    print(f"Screening: {wins} {'win' if wins == 1 else 'wins'}, "
          f"{ties} {'tie' if ties == 1 else 'ties'}, {losses} {'loss' if losses == 1 else 'losses'}")


def report_command(args):
    try:
        report = json.loads(args.path.read_text(encoding="utf-8"))
        repeats = report["repeats"]
        if not isinstance(repeats, list) or not repeats or report["loss_unit"] != LOSS_UNIT:
            raise ValueError("expected one or more repeats with loss in bits per byte")
        lines = []
        for repeat in repeats:
            seed = repeat["seed"]
            if type(seed) is not int:
                raise ValueError("repeat seed must be an integer")
            lines.append(f"Seed {seed}")
            for strategy in ("screened", "direct"):
                result = repeat[strategy]
                winner = result["winner"]
                recipe = winner["recipe"]
                flops, seconds = result["estimated_flops"], result["seconds"]
                validation, held_out = winner["validation_loss"], result["held_out_loss"]
                rate, schedule = recipe["learning_rate"], recipe["schedule"]
                values = (seconds, validation, held_out, rate)
                if (type(flops) is not int or flops < 0 or schedule not in ("constant", "cosine")
                        or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                               for value in values)):
                    raise ValueError(f"invalid {strategy} result for seed {seed}")
                lines.append(
                    f"  {strategy.capitalize():8} {validation:8.4f} validation | {held_out:8.4f} held-out"
                    f" | lr {rate:g}, {schedule} | {flops:,} estimated FLOPs | {seconds:.2f} s"
                )
        summary = summarize(repeats)
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        raise ValueError(f"Invalid experiment report {args.path}: {exc}") from exc
    print("Winning recipes and losses (bits per byte)")
    print("\n".join(lines))
    print_outcome(summary)


def main(argv=None):
    args = make_parser().parse_args(argv)
    try:
        args.handler(args)
    except UsageError as exc:
        args.parser.error(str(exc))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"harness {args.command}: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"harness {args.command}: interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
