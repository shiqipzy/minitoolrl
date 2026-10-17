"""SFT 更新前后检查；只读模型，保留模式和随机数状态。"""
from __future__ import annotations

import hashlib
import math
import random
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_steps(text: str) -> list[int]:
    if not text:
        return []
    try:
        steps = [int(x) for x in text.split(',')]
    except ValueError as exc:
        raise ValueError('diagnostic steps must be comma-separated integers') from exc
    if any(x <= 0 for x in steps) or steps != sorted(set(steps)):
        raise ValueError('diagnostic steps must be positive, unique and increasing')
    return steps


def file_sha(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def batch_sha(group) -> str:
    """同时标识输入和 labels，比较单／双 rank 时核对实际更新组相同。"""
    digest = hashlib.sha256()
    for ids, labels in group:
        for tensor in (ids, labels):
            value = tensor.detach().to(device='cpu', dtype=torch.int64).contiguous()
            digest.update(str(tuple(value.shape)).encode('ascii'))
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def read_only_model(model):
    modes = [(module, module.training) for module in model.modules()]
    python_state = random.getstate()
    device = next(model.parameters()).device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            model.eval()
            yield
    finally:
        random.setstate(python_state)
        for module, training in modes:
            module.training = training


def score_update_batch(model, group, device, project_logits):
    """在同一真实更新组上比较正确 token 概率；不把该指标当任务成功率。"""
    loss_sum = 0.0
    correct = tokens = 0
    examples = []
    with read_only_model(model):
        for ids, labels in group:
            logits, target, keep = project_logits(model, ids.to(device), labels.to(device))
            losses = F.cross_entropy(logits, target, reduction='none')
            predicted = logits.argmax(-1)
            hits = predicted == target
            loss_sum += float(losses.sum())
            correct += int(hits.sum())
            tokens += target.numel()
            offset = 0
            for row_index, n in enumerate(keep.sum(1).tolist()):
                if n <= 0:
                    raise ValueError('diagnostic example has no supervised target')
                positions = keep[row_index].nonzero().flatten() + 1
                bad = (~hits[offset:offset + n]).nonzero().flatten()[:8].tolist()
                examples.append(dict(
                    supervised_tokens=n, loss=float(losses[offset:offset + n].mean()),
                    exact=bool(hits[offset:offset + n].all()),
                    mistakes=[dict(position=int(positions[j]), target_id=int(target[offset + j]),
                                   predicted_id=int(predicted[offset + j]),
                                   target_probability=math.exp(-float(losses[offset + j]))) for j in bad]))
                offset += n
    if not tokens or not math.isfinite(loss_sum):
        raise ValueError('empty or non-finite diagnostic loss')
    return dict(batch_sha256=batch_sha(group), loss=loss_sum / tokens,
                n_tokens=tokens, token_acc=correct / tokens, examples=examples)
