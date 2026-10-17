"""Process-group context, collectives and rank-0 bookkeeping shared by the trainers.

Terms used below:
- rank / world_size: this process's index and the number of processes (one per GPU).
- Production training uses the DeepSpeed backend (train/deepspeed_backend.py); this
  module only keeps the necessary process communication and statistics helpers:
  torchrun environment discovery, small collectives for flags and metrics, rank-0
  file writing, and throughput records.

Without torchrun (no RANK/WORLD_SIZE in the environment) everything degrades to
single-process behaviour: world_size 1, no process group.

The module never imports the training scripts, so it can be imported by all of them.
"""
from __future__ import annotations

import datetime
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistCtx:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    backend: Optional[str]

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def cuda(self) -> bool:
        return self.device.type == 'cuda'

    def describe(self) -> dict:
        return dict(rank=self.rank, world_size=self.world_size, local_rank=self.local_rank,
                    device=str(self.device), backend=self.backend, device_name=device_name(self))


def env_world_size() -> int:
    return int(os.environ.get('WORLD_SIZE', '1'))


def init(device: str = 'cuda', timeout_hours: float = 2.0) -> DistCtx:
    """Read torchrun's environment; fall back to a plain single process."""
    world = env_world_size()
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('--device cuda requested but CUDA is not available')
        dev = torch.device('cuda', local_rank)
        torch.cuda.set_device(dev)
    elif device == 'cpu':
        dev = torch.device('cpu')
    else:
        raise ValueError('device must be cuda or cpu, got ' + repr(device))
    backend = None
    if world > 1:
        backend = 'nccl' if dev.type == 'cuda' else 'gloo'
        if platform.system() == 'Windows':
            # torch's TCPStore on Windows needs the legacy (non-libuv) backend.
            os.environ.setdefault('USE_LIBUV', '0')
        dist.init_process_group(backend=backend, timeout=datetime.timedelta(hours=timeout_hours))
    return DistCtx(rank=rank, world_size=world, local_rank=local_rank, device=dev, backend=backend)


def threads_per_rank(ctx: DistCtx, default: int = 8) -> int:
    """Use 8 CPU threads for one process; divide host threads between ranks otherwise."""
    if not ctx.distributed:
        return default
    cpus = os.cpu_count() or default * ctx.world_size
    return max(2, min(default, cpus // ctx.world_size))


def cleanup(ctx: DistCtx) -> None:
    if ctx.distributed and dist.is_initialized():
        dist.destroy_process_group()


# ------------------------------------------------------------------ collectives
def _coll_device(ctx: DistCtx) -> torch.device:
    return ctx.device if ctx.backend == 'nccl' else torch.device('cpu')


def all_reduce_flag(flag: bool, ctx: DistCtx) -> bool:
    """True on every rank if it is True on any rank (MAX)."""
    if not ctx.distributed:
        return bool(flag)
    t = torch.tensor([1 if flag else 0], dtype=torch.int64, device=_coll_device(ctx))
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item())


def all_gather_object(obj, ctx: DistCtx) -> list:
    if not ctx.distributed:
        return [obj]
    out = [None] * ctx.world_size
    dist.all_gather_object(out, obj)
    return out


def barrier(ctx: DistCtx) -> None:
    if not ctx.distributed:
        return
    if ctx.backend == 'nccl':
        dist.barrier(device_ids=[ctx.local_rank])
    else:
        dist.barrier()


# ------------------------------------------------------------------ device helpers
def synchronize(ctx: DistCtx) -> None:
    if ctx.cuda:
        torch.cuda.synchronize(ctx.device)


def peak_gb(ctx: DistCtx) -> float:
    if not ctx.cuda:
        return 0.0
    return round(torch.cuda.max_memory_allocated(ctx.device) / 1e9, 2)


def device_name(ctx: DistCtx) -> str:
    return torch.cuda.get_device_name(ctx.device) if ctx.cuda else 'cpu'


# ------------------------------------------------------------------ rank-0 bookkeeping
class MainFile:
    """A line-buffered text file that only rank 0 actually writes."""

    def __init__(self, path: Path, ctx: DistCtx, mode: str = 'a'):
        self.handle = open(path, mode, buffering=1, encoding='utf-8') if ctx.is_main else None

    def write(self, text: str) -> None:
        if self.handle is not None:
            self.handle.write(text)

    def flush(self) -> None:
        if self.handle is not None:
            self.handle.flush()

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def enrich_row(row: dict, ctx: DistCtx) -> dict:
    row['world_size'] = ctx.world_size
    row['rank'] = ctx.rank
    return row


def write_throughput(path: Path, ctx: DistCtx, *, stage: str, updates: int, train_seconds: float,
                     supervised_tokens: int, input_tokens: int, extra: Optional[dict] = None) -> dict:
    """Gather per-rank peak memory and write one throughput.json (rank 0). Returns the record."""
    peaks = all_gather_object(peak_gb(ctx), ctx)
    names = all_gather_object(device_name(ctx), ctx)
    secs = max(1e-6, float(train_seconds))
    record = dict(stage=stage, world_size=ctx.world_size, backend=ctx.backend, updates=int(updates),
                  train_seconds=round(secs, 2), gpu_seconds=round(secs * ctx.world_size, 2),
                  updates_per_second=round(updates / secs, 4),
                  seconds_per_update=round(secs / max(1, updates), 4),
                  supervised_tokens=int(supervised_tokens), input_tokens=int(input_tokens),
                  supervised_tokens_per_second=round(supervised_tokens / secs, 1),
                  input_tokens_per_second=round(input_tokens / secs, 1),
                  peak_gb_per_rank=peaks, peak_gb_max=max(peaks) if peaks else 0.0,
                  device_names=names, torch=torch.__version__)
    if extra:
        record.update(extra)
    if ctx.is_main:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
    return record


def launcher_note(ctx: DistCtx) -> str:
    return ('torchrun world_size=%d backend=%s' % (ctx.world_size, ctx.backend)
            if ctx.distributed else 'single process')
