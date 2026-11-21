import unittest

import torch
from torch.nn import functional as F

from model import ByteTransformer, estimated_step_flops


class TransformerTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.model = ByteTransformer(width=4, context=3, vocab_size=8)
        self.tokens = torch.tensor([[1, 2, 3]])
        self.targets = torch.tensor([2, 3, 4])

    def test_attention_cannot_see_future_tokens(self):
        left = self.model(self.tokens)
        right = self.model(torch.tensor([[1, 2, 5]]))
        torch.testing.assert_close(left[:, :2], right[:, :2])
        self.assertFalse(torch.allclose(left[:, 2], right[:, 2]))

    def test_every_parameter_receives_nonzero_finite_gradients(self):
        F.cross_entropy(self.model(self.tokens).flatten(0, 1), self.targets).backward()
        for name, parameter in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_adam_learns_repeated_example(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.03)
        baseline = F.cross_entropy(self.model(self.tokens).flatten(0, 1), self.targets).item()
        for _ in range(30):
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(self.model(self.tokens).flatten(0, 1), self.targets).backward()
            optimizer.step()
        self.assertLess(F.cross_entropy(self.model(self.tokens).flatten(0, 1), self.targets).item(),
                        baseline / 2)

    def test_step_estimate_counts_causal_attention_and_output_projection(self):
        # At T=2, causal attention covers three query/key pairs.
        self.assertEqual(estimated_step_flops(4, 2, 1), 6 * (2 * (12 * 4**2 + 256 * 4) + 2 * 3 * 4))
