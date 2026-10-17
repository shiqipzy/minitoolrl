"""单次 GRPO 更新的 CPU 回归检查；不由正式实验入口调用。"""
import copy
import unittest
from types import SimpleNamespace

import torch

from minitoolrl.train.ds_correctness import build_tiny_model, synth_grpo_batch
from minitoolrl.train.grpo_loss import grpo_loss
from minitoolrl.train.policy_loss import pad_batch, per_token_logps, shift_mask
from minitoolrl.train.train_grpo import GRPOTrainer, make_grpo_loss


class SingleUpdate(unittest.TestCase):
    def test_reused_forward_preserves_gradient_update_and_diagnostics(self):
        model = build_tiny_model(vocab=71)
        reference = copy.deepcopy(model)
        samples = synth_grpo_batch(71, 3, seq_len=16)
        old_lps = []
        with torch.no_grad():
            for sample in samples:
                ids, attn, mask = pad_batch([sample['ids']], [sample['mask']], 0, 'cpu')
                old = per_token_logps(reference, ids, attn, policy_mask=mask)
                old_lps.append(old)
                # 用已知偏差核对更新后的诊断仍读取更新前快照。
                sample['sampler_logprobs'] = (old[shift_mask(mask).bool()] + 0.125).tolist()
        for sample, old in zip(samples, old_lps):
            ids, attn, mask = pad_batch([sample['ids']], [sample['mask']], 0, 'cpu')
            lp = per_token_logps(reference, ids, attn, policy_mask=mask)
            loss = grpo_loss(lp, old, torch.tensor([sample['adv']]), shift_mask(mask))
            (loss / len(samples)).backward()

        calls = []
        hook = model.model.register_forward_hook(lambda *args: calls.append(1))
        loss_fn = make_grpo_loss(len(samples), 0, 0.2, 'cpu')
        try:
            for sample in samples:
                loss, _ = loss_fn(model, sample)
                loss.backward()
        finally:
            hook.remove()
        self.assertEqual(len(calls), len(samples))
        self.assertTrue(any(p.grad.abs().sum() > 0 for p in model.parameters()))
        for (name, got), (_, want) in zip(model.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(got.grad, want.grad, rtol=1e-6, atol=1e-7, msg=name)
        for candidate in (model, reference):
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
            torch.optim.AdamW(candidate.parameters(), lr=1e-3, weight_decay=0.0).step()
        for (name, got), (_, want) in zip(model.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-7, msg=name)
        for sample, old in zip(samples, old_lps):
            self.assertFalse(sample['old_lp'].requires_grad)
            torch.testing.assert_close(sample['old_lp'], old)
        trainer = GRPOTrainer.__new__(GRPOTrainer)
        trainer.tok = SimpleNamespace(pad_token_id=0)
        check = trainer._logp_divergence(samples, 'cpu')
        self.assertEqual(check['n_tokens'], 24)
        self.assertEqual(check['mean_abs_diff'], 0.125)
        self.assertEqual(check['max_abs_diff'], 0.125)


if __name__ == '__main__':
    unittest.main()
