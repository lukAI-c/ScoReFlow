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
    return float((probs * (torch.log(probs.clamp_min(1.0e-12)) + math.log(flat.numel()))).sum())


class CRReflowWeightsTest(unittest.TestCase):
    def setUp(self):
        self.advantages = torch.tensor([[-1.0, 0.0, 1.0]])
        self.mask = torch.ones_like(self.advantages)

    def test_zero_epsilon_uses_uniform_fallback(self):
        result = DummyPolicy(cr_reflow_kl_epsilon=0.0).weights_from_advantages(
            self.advantages, True, self.mask
        )
        weights, eta, ess, weight_kl, eta_at_bound = result
        torch.testing.assert_close(weights, self.mask)
        self.assertEqual(eta, 10.0)
        self.assertEqual(ess, 1.0)
        self.assertEqual(weight_kl, 0.0)
        self.assertEqual(eta_at_bound, 1.0)

    def test_infeasible_eta_max_uses_uniform_fallback(self):
        policy = DummyPolicy(cr_reflow_kl_epsilon=0.05, cr_reflow_eta_max=0.1)
        weights, _, ess, weight_kl, eta_at_bound = policy.weights_from_advantages(
            self.advantages, True, self.mask
        )
        torch.testing.assert_close(weights, self.mask)
        self.assertEqual(ess, 1.0)
        self.assertEqual(weight_kl, 0.0)
        self.assertEqual(eta_at_bound, 1.0)

    def test_normal_bisection_satisfies_bound(self):
        policy = DummyPolicy(cr_reflow_kl_epsilon=0.05)
        weights, eta, _, weight_kl, eta_at_bound = policy.weights_from_advantages(
            self.advantages, True, self.mask
        )
        self.assertGreater(eta, policy.config.cr_reflow_eta_min)
        self.assertLess(eta, policy.config.cr_reflow_eta_max)
        self.assertLessEqual(weight_kl, policy.config.cr_reflow_kl_epsilon)
        self.assertFalse(torch.allclose(weights, self.mask))
        self.assertEqual(eta_at_bound, 0.0)

    def test_constant_advantages_return_uniform_weights(self):
        weights, eta, ess, weight_kl, eta_at_bound = DummyPolicy().weights_from_advantages(
            torch.ones_like(self.advantages), True, self.mask
        )
        torch.testing.assert_close(weights, self.mask)
        self.assertEqual((eta, ess, weight_kl, eta_at_bound), (0.0, 1.0, 0.0, 0.0))

    def test_masking_keeps_invalid_entries_zero(self):
        mask = torch.tensor([[1.0, 0.0, 1.0]])
        weights, _, _, weight_kl, _ = DummyPolicy().weights_from_advantages(
            self.advantages, True, mask
        )
        self.assertEqual(float(weights[0, 1]), 0.0)
        self.assertAlmostEqual(float(weights[mask > 0].mean()), 1.0, places=6)
        self.assertAlmostEqual(weight_kl, achieved_kl(weights, mask), places=6)

    def test_reported_kl_uses_post_clipping_weights(self):
        policy = DummyPolicy(
            cr_reflow_kl_epsilon=10.0,
            cr_reflow_eta_min=1.0,
            cr_reflow_weight_clip=1.1,
        )
        weights, eta, _, weight_kl, eta_at_bound = policy.weights_from_advantages(
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

    def test_fp16_input_returns_exact_bounded_float32_weights(self):
        advantages = torch.tensor([[-1.0, 0.0, 1.0]], dtype=torch.float16)
        mask = torch.ones_like(advantages)
        weights, _, _, weight_kl, _ = DummyPolicy().weights_from_advantages(
            advantages, True, mask
        )
        returned_kl = achieved_kl(weights, mask)
        self.assertEqual(weights.dtype, torch.float32)
        self.assertEqual(weight_kl, returned_kl)
        self.assertLessEqual(returned_kl, 0.05)

    def test_bf16_input_returns_exact_bounded_float32_weights(self):
        advantages = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=torch.bfloat16)
        mask = torch.ones_like(advantages)
        weights, _, _, weight_kl, _ = DummyPolicy().weights_from_advantages(
            advantages, True, mask
        )
        returned_kl = achieved_kl(weights, mask)
        self.assertEqual(weights.dtype, torch.float32)
        self.assertEqual(weight_kl, returned_kl)
        self.assertLessEqual(returned_kl, 0.05)


if __name__ == "__main__":
    unittest.main()
