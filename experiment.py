"""Compare two ways of spending the same training-compute budget."""

import math
import random
from dataclasses import dataclass
from itertools import combinations
from statistics import mean


@dataclass(frozen=True)
class Recipe:
    learning_rate: float
    schedule: str


RECIPES = tuple(
    Recipe(rate, schedule)
    for rate in (0.0003, 0.001, 0.003, 0.01)
    for schedule in ("constant", "cosine")
)
DIRECT_TRIALS = 4
PROMOTIONS = 3
WIDTHS = {"proxy": 8, "target": 16}
CONTEXT = 12
BATCH_SIZE = 2
EVAL_WINDOWS = 32
DEFAULT_TARGET_STEPS = 400
DEFAULT_SEEDS = (7, 19, 41)
LOSS_UNIT = "bits per byte"


@dataclass(frozen=True)
class Budget:
    proxy_step_cost: int
    target_step_cost: int
    proxy_steps: int
    target_steps: int

    @property
    def cap(self):
        return DIRECT_TRIALS * self.target_step_cost * self.target_steps

    @property
    def screening_cost(self):
        return (
            len(RECIPES) * self.proxy_step_cost * self.proxy_steps
            + PROMOTIONS * self.target_step_cost * self.target_steps
        )


def plan_budget(proxy_step_cost, target_step_cost, target_steps):
    available = (DIRECT_TRIALS - PROMOTIONS) * target_steps * target_step_cost
    proxy_steps = available // (len(RECIPES) * proxy_step_cost)
    if proxy_steps < 1:
        raise ValueError("Training budget gives zero proxy steps. Increase --target-steps.")
    return Budget(proxy_step_cost, target_step_cost, proxy_steps, target_steps)


def estimated_step_flops(width, context, batch_size):
    """Count forward/backward matmul FLOPs, with two FLOPs per multiply-add."""
    projections = context * (12 * width**2 + width * 256)
    causal_pairs = context * (context + 1) // 2
    attention = 2 * causal_pairs * width
    return 6 * batch_size * (projections + attention)


def experiment_budget(widths, target_steps):
    return plan_budget(
        estimated_step_flops(widths["proxy"], CONTEXT, BATCH_SIZE),
        estimated_step_flops(widths["target"], CONTEXT, BATCH_SIZE),
        target_steps,
    )


def validate_seeds(seeds):
    if len(set(seeds)) != len(seeds):
        raise ValueError("Use distinct seeds for each repeat.")
    if any(not 0 <= seed < 2**63 for seed in seeds):
        raise ValueError("Seeds must be between 0 and 2**63 - 1.")


@dataclass(frozen=True)
class Measurement:
    validation_loss: float | None
    seconds: float
    peak_cuda_mib: float | None


def compare(trainer, budget, seed):
    candidates = list(RECIPES)
    random.Random(seed).shuffle(candidates)

    def run(recipe, scale, steps, cost):
        measurement, checkpoint = trainer.train(recipe, scale, steps, seed)
        return {
            "recipe": recipe,
            "validation_loss": measurement.validation_loss,
            "seconds": measurement.seconds,
            "peak_cuda_mib": measurement.peak_cuda_mib,
            "steps": steps,
            "estimated_flops": steps * cost,
        }, checkpoint

    proxy_trials = [
        run(recipe, "proxy", budget.proxy_steps, budget.proxy_step_cost)[0]
        for recipe in candidates
    ]
    viable = [trial for trial in proxy_trials if trial["validation_loss"] is not None]
    if len(viable) < PROMOTIONS:
        raise ValueError(f"Seed {seed}: fewer than {PROMOTIONS} proxy runs had finite loss.")
    viable.sort(key=lambda trial: trial["validation_loss"])

    def target_search(recipes, name, preceding_trials=()):
        trials = []
        winner, checkpoint = None, None
        for recipe in recipes:
            trial, weights = run(recipe, "target", budget.target_steps, budget.target_step_cost)
            trials.append(trial)
            loss = trial["validation_loss"]
            if loss is not None and (winner is None or loss < winner["validation_loss"]):
                winner, checkpoint = trial, weights
        if winner is None:
            raise ValueError(f"Seed {seed}: every {name} target run diverged.")
        charged = [*preceding_trials, *trials]
        return {
            "trials": trials,
            "winner": winner,
            "estimated_flops": sum(t["estimated_flops"] for t in charged),
            "seconds": sum(t["seconds"] for t in charged),
        }, checkpoint

    screened, screened_checkpoint = target_search(
        [trial["recipe"] for trial in viable[:PROMOTIONS]], "screened", proxy_trials
    )
    direct, direct_checkpoint = target_search(candidates[:DIRECT_TRIALS], "direct")

    # Fix both winners before evaluating either on the held-out split.
    for outcome, checkpoint in [(screened, screened_checkpoint), (direct, direct_checkpoint)]:
        loss = trainer.held_out_loss(checkpoint)
        if not math.isfinite(loss):
            raise ValueError(f"Seed {seed}: the selected model had non-finite held-out loss.")
        outcome["held_out_loss"] = loss

    proxy_losses = {trial["recipe"]: trial["validation_loss"] for trial in viable}
    paired = [
        (proxy_losses[trial["recipe"]], trial["validation_loss"])
        for trial in direct["trials"]
        if trial["recipe"] in proxy_losses and trial["validation_loss"] is not None
    ]
    comparisons = [(a - c) * (b - d) for (a, b), (c, d) in combinations(paired, 2)]
    return {
        "seed": seed,
        "proxy_trials": proxy_trials,
        "screened": screened,
        "direct": direct,
        "direct_subset_rank_agreement": {
            "agree": sum(value > 0 for value in comparisons),
            "disagree": sum(value < 0 for value in comparisons),
            "ties": sum(value == 0 for value in comparisons),
        },
    }


def summarize(repeats):
    deltas = [r["screened"]["held_out_loss"] - r["direct"]["held_out_loss"] for r in repeats]
    return {"paired_deltas": deltas, "mean_delta": mean(deltas)}
