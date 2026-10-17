"""vLLM 采样协议的纯序列化与一致性检查测试（不安装 vllm/zmq）。

覆盖：GenRequest/Generation 的 dict 往返、采样概率提取、策略版本断言，
以及 VLLMRollout 权重同步的版本单调约束（用假传输通道，不启动任何进程）。
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from minitoolrl.train import vllm_backend as V


class Serialization(unittest.TestCase):
    def test_request_roundtrip_preserves_all_fields(self):
        req = V.GenRequest(request_id='r1', prompt_token_ids=[1, 2, 3],
                           max_new_tokens=64, temperature=0.8, top_p=0.95,
                           top_k=0, seed=12345)
        back = V.request_from_dict(V.request_to_dict(req))
        self.assertEqual(back.request_id, 'r1')
        self.assertEqual(back.prompt_token_ids, [1, 2, 3])
        self.assertEqual(back.max_new_tokens, 64)
        self.assertAlmostEqual(back.temperature, 0.8)
        self.assertAlmostEqual(back.top_p, 0.95)
        self.assertEqual(back.top_k, 0)
        self.assertEqual(back.seed, 12345)

    def test_request_defaults_and_null_seed(self):
        row = dict(request_id=0, prompt_token_ids=[5], max_new_tokens=8)
        req = V.request_from_dict(row)
        self.assertEqual(req.temperature, 0.0)
        self.assertEqual(req.top_p, 1.0)
        self.assertIsNone(req.seed)

    def test_generation_roundtrip_with_and_without_logprobs(self):
        gen = V.Generation(request_id=7, token_ids=[9, 8, 7],
                           token_logprobs=[-0.1, -2.0, -2.5], finish_reason='stop',
                           policy_version=3)
        back = V.generation_from_dict(V.generation_to_dict(gen))
        self.assertEqual(back.request_id, 7)
        self.assertEqual(back.token_ids, [9, 8, 7])
        self.assertEqual(back.token_logprobs, [-0.1, -2.0, -2.5])
        self.assertEqual(back.finish_reason, 'stop')
        self.assertEqual(back.policy_version, 3)
        none = V.Generation(request_id=1, token_ids=[2], token_logprobs=None,
                            finish_reason='length', policy_version=0)
        back = V.generation_from_dict(V.generation_to_dict(none))
        self.assertIsNone(back.token_logprobs)


class LogprobExtraction(unittest.TestCase):
    def test_sampled_logprobs_picks_the_sampled_token(self):
        def lp(value):
            return SimpleNamespace(logprob=value)
        got = V._sampled_logprobs([4, 6], [{4: lp(-0.5), 5: lp(-1.0)}, {6: lp(-0.25)}])
        self.assertEqual(got, [-0.5, -0.25])

    def test_missing_entry_returns_none(self):
        got = V._sampled_logprobs([4, 9], [{4: SimpleNamespace(logprob=-0.5)}, {}])
        self.assertIsNone(got)
        self.assertIsNone(V._sampled_logprobs([4], None))

    def test_outputs_to_generations_keeps_request_order_and_version(self):
        requests = [V.GenRequest('a', [1], 4), V.GenRequest('b', [2], 4)]
        outputs = [
            SimpleNamespace(outputs=[SimpleNamespace(
                token_ids=[1, 2], logprobs=[{1: SimpleNamespace(logprob=-0.1)},
                                          {2: SimpleNamespace(logprob=-0.2)}],
                finish_reason='stop')]),
            SimpleNamespace(outputs=[SimpleNamespace(
                token_ids=[3], logprobs=None, finish_reason='length')]),
        ]
        gens = V.outputs_to_generations(requests, outputs, policy_version=5)
        self.assertEqual([g.request_id for g in gens], ['a', 'b'])
        self.assertEqual(gens[0].token_ids, [1, 2])
        self.assertEqual(gens[0].token_logprobs, [-0.1, -0.2])
        self.assertEqual(gens[0].finish_reason, 'stop')
        self.assertEqual(gens[1].token_logprobs, None)
        self.assertEqual([g.policy_version for g in gens], [5, 5])


class PolicyVersion(unittest.TestCase):
    def test_consistent_versions_pass(self):
        gens = [V.Generation(i, [1], None, 'stop', 4) for i in range(3)]
        V.assert_policy_version(gens, 4)  # 不抛错

    def test_stale_sampler_is_rejected(self):
        gens = [V.Generation(0, [1], None, 'stop', 4),
                V.Generation(1, [1], None, 'stop', 3)]
        with self.assertRaisesRegex(RuntimeError, 'policy version mismatch'):
            V.assert_policy_version(gens, 4)


class _FakeTransport:
    """记录调用并按需返回应答的假命令通道。"""

    def __init__(self, replies=None):
        self.calls = []
        self.replies = list(replies or [])

    def call(self, message):
        self.calls.append(message)
        if self.replies:
            return self.replies.pop(0)
        return {'ok': True}


class WeightSyncVersioning(unittest.TestCase):
    def _rollout(self, transport):
        rollout = V.VLLMRollout.__new__(V.VLLMRollout)  # 不启动子进程
        rollout.version = 2
        rollout._engine = None
        rollout._closed = False
        rollout.transport = transport
        rollout.process = None
        rollout._engine = SimpleNamespace(
            send_weights=lambda: None)  # 跳过真实 NCCL 引擎
        return rollout

    def test_sync_requires_increasing_version_and_confirms(self):
        transport = _FakeTransport([{'version': 3}])
        rollout = self._rollout(transport)
        rollout.sync_weights(model=None, version=3)
        self.assertEqual(rollout.version, 3)
        confirm = [c for c in transport.calls if c['op'] == 'confirm_version']
        self.assertEqual(confirm, [dict(op='confirm_version', version=3)])

    def test_same_or_older_version_is_rejected(self):
        rollout = self._rollout(_FakeTransport())
        with self.assertRaisesRegex(RuntimeError, 'must increase'):
            rollout.sync_weights(model=None, version=2)
        with self.assertRaisesRegex(RuntimeError, 'must increase'):
            rollout.sync_weights(model=None, version=1)

    def test_unconfirmed_version_is_rejected(self):
        transport = _FakeTransport([{'version': 99}])  # 采样器没有确认 3
        rollout = self._rollout(transport)
        with self.assertRaisesRegex(RuntimeError, 'did not confirm'):
            rollout.sync_weights(model=None, version=3)
        self.assertEqual(rollout.version, 2)


if __name__ == '__main__':
    unittest.main()
