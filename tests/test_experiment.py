import unittest

from experiment import RECIPES, Measurement, compare, plan_budget, summarize


class RecordingTrainer:
    def __init__(self):
        self.events = []

    def train(self, recipe, scale, steps, seed):
        self.events.append(("train", scale, recipe))
        index = RECIPES.index(recipe)
        loss = index if scale == "proxy" else len(RECIPES) - index
        return Measurement(float(loss), 0.0, None), (scale, recipe, seed)

    def held_out_loss(self, checkpoint):
        scale, recipe, seed = checkpoint
        assert scale == "target"
        self.events.append(("test", scale, recipe))
        return float(RECIPES.index(recipe))


class ExperimentTest(unittest.TestCase):
    def test_screening_is_charged_and_caps_are_equal(self):
        plan = plan_budget(proxy_step_cost=5, target_step_cost=14, target_steps=20)
        self.assertEqual(plan.cap, 4 * 20 * 14)
        self.assertEqual(plan.proxy_steps, 7)
        self.assertEqual(plan.screening_cost, 4 * 20 * 14)

        rounded = plan_budget(proxy_step_cost=5, target_step_cost=14, target_steps=19)
        self.assertLessEqual(rounded.screening_cost, rounded.cap)
        self.assertLess(rounded.cap - rounded.screening_cost, len(RECIPES) * 5)

    def test_selects_both_winners_before_reading_test_metrics(self):
        trainer = RecordingTrainer()
        plan = plan_budget(5, 14, 20)
        result = compare(trainer, plan, seed=42)
        self.assertEqual(len(result["proxy_trials"]), len(RECIPES))
        self.assertEqual([r["recipe"] for r in result["screened"]["trials"]], list(RECIPES[:3]))
        self.assertEqual(len({r["recipe"] for r in result["direct"]["trials"]}), 4)
        self.assertEqual(result["screened"]["winner"]["recipe"], RECIPES[2])
        self.assertEqual([e[0] for e in trainer.events], ["train"] * 15 + ["test"] * 2)
        self.assertEqual(result["screened"]["held_out_loss"], 2.0)
        self.assertEqual(result["screened"]["estimated_flops"], plan.cap)
        self.assertEqual(result["direct"]["estimated_flops"], plan.cap)

    def test_summary_has_raw_paired_deltas_and_mean(self):
        plan = plan_budget(5, 14, 20)
        repeats = [compare(RecordingTrainer(), plan, seed) for seed in (42, 43)]
        expected = [r["screened"]["held_out_loss"] - r["direct"]["held_out_loss"] for r in repeats]
        self.assertEqual(summarize(repeats)["paired_deltas"], expected)
        self.assertEqual(summarize(repeats)["mean_delta"], sum(expected) / len(expected))

    def test_rejects_budget_that_cannot_screen_every_candidate(self):
        with self.assertRaisesRegex(ValueError, "zero proxy steps"):
            plan_budget(5, 14, 2)
