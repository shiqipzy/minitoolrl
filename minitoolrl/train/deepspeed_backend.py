"""训练后端：以“一次全局更新”为边界。

DeepSpeedTrainer 是生产路径唯一的训练后端：ZeRO-2 分片优化器状态和梯度，
模型参数每卡保留完整副本；bf16 计算、fp32 优化器主权重；不使用 offload。

归一化约定：loss_fn(module, microbatch) 返回按全局口径归一化的损失张量
（例如 CE_sum / n_total，n_total 为整个更新组的全局监督 token 数），使得
单进程直接累加所有 microbatch 的梯度就是期望的一次全局更新。DeepSpeed 内部
会把损失除以每个 rank 的累积步数（gas_local），并在 ZeRO-2 梯度规约时跨
rank 求平均，因此 DeepSpeedTrainer 把每个真实 microbatch 的损失乘以
gas_local * world，抵消这两处缩放，保证单/双卡与参照实现的梯度一致。

累积计数：DeepSpeed 在托管累积模式（默认）下，micro-step 计数由 step() 推进，
累积边界为 ``(micro_steps + 1) % gas_local == 0``（见 DeepSpeed 0.19.7
runtime/engine.py 的 is_gradient_accumulation_boundary），backward() 本身不推进它。
因此每次全局更新必须交替执行 backward(); step() 恰好 gas_local 次，边界落在最后一次
——每次全局更新只发生一次优化器更新与一次梯度裁剪（裁剪在 _take_model_step 内，仅在
边界执行），且微步计数始终保持对齐，不会随更新次数漂移。真实 microbatch 按
rank r, r+N, ... 分配，不足 gas_local 的部分用 zero_loss 产生的零损失补齐；零损失只
用于对齐累积边界，不复制真实样本、不产生梯度，同时保证各 rank 通信次数一致（覆盖
U 组 9 个 microbatch 无法均分双卡的情形）。用 was_step_applied 核对真实更新，
同时核对 micro-step 和边界计数；溢出跳步或非有限损失、梯度直接报错。
"""
from __future__ import annotations

import datetime
import math
import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Sequence

import torch

# loss_fn(module, microbatch) -> (loss_tensor, aux_dict)；aux 中的数值键按求和汇总。
LossFn = Callable[[torch.nn.Module, object], tuple]
# zero_loss(module) -> 带计算图的零损失张量，只用于对齐 DeepSpeed 累积边界。
ZeroLossFn = Callable[[torch.nn.Module], torch.Tensor]


@dataclass
class UpdateMetrics:
    """一次全局更新的结果：全局损失和、全局梯度范数、耗时与计数。"""
    loss: float
    grad_norm: float
    seconds: float
    n_microbatches: int
    applied_updates: int
    aux: Dict[str, float] = field(default_factory=dict)


def local_slice(items: Sequence, rank: int, world: int) -> list:
    """rank r 取全局 microbatch 列表的第 r、r+N、… 个，与单进程顺序一致。"""
    return list(items)[rank::world]


def local_gas(gas_global: int, world: int) -> int:
    """每个 rank 的累积步数：全局 microbatch 数按 rank 数向上取整。"""
    return -(-int(gas_global) // int(world))


def ds_loss_scale(gas_global: int, world: int) -> float:
    """抵消 DeepSpeed 的 1/gas_local 损失缩放和跨 rank 梯度平均。"""
    return float(local_gas(gas_global, world) * world)


def merge_aux(into: Dict[str, float], extra: Dict[str, float]) -> None:
    for key, value in (extra or {}).items():
        into[key] = into.get(key, 0.0) + float(value)


def init_process_group(device: str = 'cuda') -> torch.device:
    """DeepSpeed 统一建立训练进程组；训练脚本不再另建一套初始化入口。"""
    import deepspeed
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available')
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        selected = torch.device('cuda', local_rank)
        torch.cuda.set_device(selected)
    elif device == 'cpu':
        selected = torch.device('cpu')
    else:
        raise ValueError('device must be cuda or cpu, got ' + repr(device))
    if not torch.distributed.is_initialized():
        if platform.system() == 'Windows':
            os.environ.setdefault('USE_LIBUV', '0')
        deepspeed.init_distributed(
            dist_backend='nccl' if selected.type == 'cuda' else 'gloo',
            timeout=datetime.timedelta(hours=2))
    return selected


class DeepSpeedTrainer:
    """DeepSpeed ZeRO-2 后端：单卡或多卡同一种更新语义。

    gas_global 是一次完整更新（非尾批）的全局 microbatch 数；尾批真实
    microbatch 更少时，差额由零损失补齐，不复制真实样本。
    """

    backend_name = 'deepspeed_zero2'

    def __init__(self, model, *, lr: float, weight_decay: float, grad_clip: float,
                 gas_global: int, zero_loss: ZeroLossFn, device: str = 'cuda'):
        import deepspeed  # 懒加载：无机/无 deepspeed 环境下模块仍可导入。
        if int(gas_global) <= 0:
            raise ValueError('gas_global must be positive')
        self.device = init_process_group(device)
        self.gas_global = int(gas_global)
        self.zero_loss = zero_loss
        self.world = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()
        self.gas_local = local_gas(self.gas_global, self.world)
        self.scale = ds_loss_scale(self.gas_global, self.world)
        if self.device.type == 'cuda':
            model = model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                      weight_decay=weight_decay,
                                      fused=self.device.type == 'cuda')
        config = {
            'train_micro_batch_size_per_gpu': 1,
            'gradient_accumulation_steps': self.gas_local,
            'bf16': {'enabled': True, 'check_grad_overflow': True},
            'zero_optimization': {'stage': 2},
            'gradient_clipping': float(grad_clip),
            'steps_per_print': 10 ** 9,
            'wall_clock_breakdown': False,
        }
        self.engine, _, _, _ = deepspeed.initialize(
            model=model, optimizer=optimizer, config=config)

    @property
    def module(self):
        return self.engine.module

    @property
    def micro_steps(self) -> int:
        """DeepSpeed 已推进的 micro-step 数：每校验一次 step() 加一。"""
        return int(self.engine.micro_steps)

    @property
    def optimizer_updates(self) -> int:
        """真实发生的优化器更新次数（只在累积边界递增）。"""
        return int(self.engine.global_steps - self.engine.skipped_steps)

    def set_lr(self, lr: float) -> None:
        for group in self.engine.optimizer.param_groups:
            group['lr'] = lr

    def update(self, global_batch: Sequence, loss_fn: LossFn) -> UpdateMetrics:
        if not global_batch:
            raise RuntimeError('empty update group')
        local_microbatches = local_slice(global_batch, self.rank, self.world)
        started = time.monotonic()
        loss_local = 0.0
        aux: Dict[str, float] = {}
        updates_before = int(self.engine.global_steps)
        micro_before = self.micro_steps
        if micro_before % self.gas_local:
            raise RuntimeError('global update started inside an accumulation boundary')
        applied = 0
        # backward() 不推进 micro-step，step() 才推进；二者必须交替，且本组恰好
        # gas_local 次，边界才落在最后一次，一次全局更新只更新一次参数。
        for index in range(self.gas_local):
            if index < len(local_microbatches):
                loss, extra = loss_fn(self.engine.module, local_microbatches[index])
                self.engine.backward(loss * self.scale)
                loss_local += float(loss.detach())
                merge_aux(aux, extra)
            else:
                # 零损失只对齐累积边界；不产生梯度，不复制真实样本。
                self.engine.backward(self.zero_loss(self.engine.module))
            self.engine.step()
            applied += int(self.engine.was_step_applied())
        norm = self.engine.get_global_grad_norm()
        norm = float(norm) if norm is not None else float('nan')
        invalid = (applied != 1 or int(self.engine.global_steps) - updates_before != 1
                   or self.micro_steps - micro_before != self.gas_local
                   or not math.isfinite(loss_local) or not math.isfinite(norm) or norm < 0)
        if self._all_reduce_sum(float(invalid)):
            raise RuntimeError(
                'invalid global update: applied=%d, gas_local=%d, micro_steps=%s, loss=%s, grad_norm=%s'
                % (applied, self.gas_local, self.micro_steps, loss_local, norm))
        loss_total = self._all_reduce_sum(loss_local)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        return UpdateMetrics(loss=loss_total,
                             grad_norm=norm,
                             seconds=time.monotonic() - started,
                             n_microbatches=len(global_batch), applied_updates=applied,
                             aux=aux)

    def _all_reduce_sum(self, value: float) -> float:
        if self.world == 1:
            return float(value)
        tensor = torch.tensor([float(value)], dtype=torch.float64, device=self.device)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        return float(tensor.item())

    def export_model(self, destination, tokenizer=None) -> None:
        """只有 rank 0 写盘；本方法不做任何集合通信，同步由调用方负责。

        调用方必须保证所有 rank 以同样的次数调用同一段同步代码：把 barrier 放在
        "只有 rank 0 执行" 的分支里会让各 rank 的通信次数不一致，直接错位卡死。
        """
        dest = Path(destination)
        if self.rank == 0:
            state = {k: v.detach().to(device='cpu', dtype=torch.bfloat16)
                     for k, v in self.engine.module.state_dict().items()}
            self.engine.module.save_pretrained(
                dest, state_dict=state, safe_serialization=True)
            if tokenizer is not None:
                tokenizer.save_pretrained(dest)

    def describe(self) -> dict:
        return dict(backend=self.backend_name, zero_stage=2, bf16=True,
                    optimizer_states='fp32', offload=None,
                    gas_global=self.gas_global, gas_local=self.gas_local,
                    loss_scale=self.scale, world_size=self.world, rank=self.rank,
                    micro_steps=self.micro_steps,
                    optimizer_updates=self.optimizer_updates)
