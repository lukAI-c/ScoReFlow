import ast
import math
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


POLICY_PATH = Path(__file__).parents[1] / "openpi_action_model.LIVE.py"


def load_weights_method():
    tree = ast.parse(POLICY_PATH.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_cr_weights_from_advantages"
    )
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {"math": math, "torch": torch}
    exec(compile(ast.fix_missing_locations(module), str(POLICY_PATH), "exec"), namespace)
    return namespace["_cr_weights_from_advantages"]


class DummyPolicy:
    def __init__(self, **overrides):
        config = {
            "cr_reflow_kl_epsilon": 0.05,
            "cr_reflow_eta_min": 0.01,
            "cr_reflow_eta_max": 10.0,
            "cr_reflow_weight_clip": 10.0,
        }
        config.update(overrides)
        self.config = SimpleNamespace(**config)
        self.weights_from_advantages = types.MethodType(load_weights_method(), self)


def achieved_kl(weights, mask):
    flat = weights.float()[mask > 0]
    probs = flat / flat.sum()
    return float((probs * (torch.log(probs) + math.log(flat.numel()))).sum())


def exact_candidate(advantages, mask, eta, clip_value):
    valid = mask > 0
    flat_adv = advantages.float()[valid]
    flat_weights = torch.softmax(flat_adv / eta, dim=0) * float(flat_adv.numel())
    positive_floor = torch.finfo(flat_weights.dtype).tiny
    if clip_value > 0.0:
        flat_weights = flat_weights.clamp(
            min=positive_floor,
            max=max(clip_value, positive_floor),
        )
    else:
        flat_weights = flat_weights.clamp_min(positive_floor)
    flat_weights = flat_weights / flat_weights.mean().clamp_min(1.0e-12)
    weights = torch.zeros_like(mask, dtype=torch.float32)
    weights[valid] = flat_weights
    return weights


class CRReflowWeightsTest(unittest.TestCase):
    def setUp(self):
        self.advantages = torch.tensor([[-1.0, 0.0, 1.0]])
        self.mask = torch.ones_like(self.advantages)

    def assert_valid_weights(self, weights, mask):
        valid_weights = weights[mask > 0]
        self.assertTrue(torch.isfinite(valid_weights).all())
        self.assertTrue((valid_weights > 0).all())
        self.assertTrue((weights[mask <= 0] == 0).all())

    def test_zero_epsilon_uses_uniform_fallback(self):
        result = DummyPolicy(cr_reflow_kl_epsilon=0.0).weights_from_advantages(
            self.advantages, True, self.mask
        )
        weights, eta, ess, weight_kl, eta_at_bound, uniform_fallback = result
        torch.testing.assert_close(weights, self.mask)
        self.assertEqual(eta, 10.0)
        self.assertEqual(ess, 1.0)
        self.assertEqual(weight_kl, 0.0)
        self.assertEqual(eta_at_bound, 1.0)
        self.assertEqual(uniform_fallback, 1.0)

    def test_infeasible_eta_max_uses_uniform_fallback(self):
        policy = DummyPolicy(cr_reflow_kl_epsilon=0.05, cr_reflow_eta_max=0.1)
        weights, _, ess, weight_kl, eta_at_bound, uniform_fallback = policy.weights_from_advantages(
            self.advantages, True, self.mask
        )
        torch.testing.assert_close(weights, self.mask)
        self.assertEqual(ess, 1.0)
        self.assertEqual(weight_kl, 0.0)
        self.assertEqual(eta_at_bound, 1.0)
        self.assertEqual(uniform_fallback, 1.0)

    def test_normal_bisection_satisfies_bound(self):
        policy = DummyPolicy(cr_reflow_kl_epsilon=0.05)
        weights, eta, _, weight_kl, eta_at_bound, uniform_fallback = policy.weights_from_advantages(
            self.advantages, True, self.mask
        )
        self.assertGreater(eta, policy.config.cr_reflow_eta_min)
        self.assertLess(eta, policy.config.cr_reflow_eta_max)
        self.assertLessEqual(weight_kl, policy.config.cr_reflow_kl_epsilon)
        self.assertFalse(torch.allclose(weights, self.mask))
        self.assertEqual(eta_at_bound, 0.0)
        self.assertEqual(uniform_fallback, 0.0)
        torch.testing.assert_close(
            weights,
            exact_candidate(self.advantages, self.mask, eta, policy.config.cr_reflow_weight_clip),
        )

    def test_constant_advantages_return_uniform_weights(self):
        weights, eta, ess, weight_kl, eta_at_bound, uniform_fallback = DummyPolicy().weights_from_advantages(
            torch.ones_like(self.advantages), True, self.mask
        )
        torch.testing.assert_close(weights, self.mask)
        self.assertEqual((eta, ess, weight_kl, eta_at_bound), (0.0, 1.0, 0.0, 0.0))
        self.assertEqual(uniform_fallback, 0.0)

    def test_masking_keeps_invalid_entries_zero(self):
        mask = torch.tensor([[1.0, 0.0, 1.0]])
        weights, _, _, weight_kl, _, uniform_fallback = DummyPolicy().weights_from_advantages(
            self.advantages, True, mask
        )
        self.assertEqual(float(weights[0, 1]), 0.0)
        self.assert_valid_weights(weights, mask)
        self.assertAlmostEqual(float(weights[mask > 0].mean()), 1.0, places=6)
        self.assertAlmostEqual(weight_kl, achieved_kl(weights, mask), places=6)
        self.assertEqual(uniform_fallback, 0.0)

    def test_reported_kl_uses_post_clipping_weights(self):
        policy = DummyPolicy(
            cr_reflow_kl_epsilon=10.0,
            cr_reflow_eta_min=1.0,
            cr_reflow_weight_clip=1.1,
        )
        weights, eta, _, weight_kl, eta_at_bound, uniform_fallback = policy.weights_from_advantages(
            self.advantages, True, self.mask
        )
        preclip_probs = torch.softmax(self.advantages.flatten() / eta, dim=0)
        preclip_kl = float(
            (
                preclip_probs
                * (torch.log(preclip_probs.clamp_min(1.0e-12)) + math.log(3))
            ).sum()
        )
        self.assertAlmostEqual(weight_kl, achieved_kl(weights, self.mask), places=6)
        self.assertNotAlmostEqual(weight_kl, preclip_kl, places=4)
        self.assertEqual(eta_at_bound, 1.0)
        self.assertEqual(uniform_fallback, 0.0)

    def test_fp16_input_returns_exact_bounded_float32_weights(self):
        advantages = torch.tensor([[-1.0, 0.0, 1.0]], dtype=torch.float16)
        mask = torch.ones_like(advantages)
        weights, _, _, weight_kl, _, uniform_fallback = DummyPolicy().weights_from_advantages(
            advantages, True, mask
        )
        returned_kl = achieved_kl(weights, mask)
        self.assertEqual(weights.dtype, torch.float32)
        self.assert_valid_weights(weights, mask)
        self.assertEqual(weight_kl, returned_kl)
        self.assertLessEqual(returned_kl, 0.05)
        self.assertEqual(uniform_fallback, 0.0)

    def test_bf16_input_returns_exact_bounded_float32_weights(self):
        advantages = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=torch.bfloat16)
        mask = torch.ones_like(advantages)
        weights, _, _, weight_kl, _, uniform_fallback = DummyPolicy().weights_from_advantages(
            advantages, True, mask
        )
        returned_kl = achieved_kl(weights, mask)
        self.assertEqual(weights.dtype, torch.float32)
        self.assert_valid_weights(weights, mask)
        self.assertEqual(weight_kl, returned_kl)
        self.assertLessEqual(returned_kl, 0.05)
        self.assertEqual(uniform_fallback, 0.0)

    def test_reproduced_feasible_seed_zero_case_stays_nonuniform(self):
        torch.manual_seed(0)
        advantages = torch.randn(8, 10)
        mask = torch.ones_like(advantages)
        policy = DummyPolicy()

        weights, eta, _, weight_kl, _, uniform_fallback = policy.weights_from_advantages(
            advantages, True, mask
        )

        self.assertEqual(uniform_fallback, 0.0)
        self.assert_valid_weights(weights, mask)
        self.assertFalse(torch.allclose(weights, mask))
        self.assertLessEqual(weight_kl, policy.config.cr_reflow_kl_epsilon)
        torch.testing.assert_close(
            weights,
            exact_candidate(advantages, mask, eta, policy.config.cr_reflow_weight_clip),
        )

    def test_seeded_feasible_batches_stay_exact_nonuniform_and_bounded(self):
        generator = torch.Generator().manual_seed(7)
        policy = DummyPolicy()
        mask = torch.ones(8, 10)
        tested = 0

        for _ in range(100):
            advantages = torch.randn(8, 10, generator=generator)
            eta_max_weights = exact_candidate(
                advantages,
                mask,
                policy.config.cr_reflow_eta_max,
                policy.config.cr_reflow_weight_clip,
            )
            if achieved_kl(eta_max_weights, mask) > policy.config.cr_reflow_kl_epsilon:
                continue

            weights, eta, _, weight_kl, _, uniform_fallback = policy.weights_from_advantages(
                advantages, True, mask
            )
            tested += 1
            self.assertEqual(uniform_fallback, 0.0)
            self.assert_valid_weights(weights, mask)
            self.assertFalse(torch.allclose(weights, mask))
            self.assertLessEqual(weight_kl, policy.config.cr_reflow_kl_epsilon)
            torch.testing.assert_close(
                weights,
                exact_candidate(advantages, mask, eta, policy.config.cr_reflow_weight_clip),
            )

        self.assertGreater(tested, 0)

    def test_eta_min_underflow_keeps_all_valid_weights_positive(self):
        policy = DummyPolicy(cr_reflow_kl_epsilon=10.0, cr_reflow_eta_min=0.01)
        weights, eta, _, weight_kl, eta_at_bound, uniform_fallback = policy.weights_from_advantages(
            self.advantages, True, self.mask
        )

        self.assertEqual(eta, policy.config.cr_reflow_eta_min)
        self.assertEqual(eta_at_bound, 1.0)
        self.assertEqual(uniform_fallback, 0.0)
        self.assert_valid_weights(weights, self.mask)
        self.assertLessEqual(weight_kl, policy.config.cr_reflow_kl_epsilon)


if __name__ == "__main__":
    unittest.main()
