"""vLLM 采样后端：模型生成与参数版本，不处理业务。

三种使用方式：

- LocalVLLM：评测与同卡基准使用的进程内采样器。
- VLLMRollout：GRPO 训推分离的客户端。采样进程常驻独立显卡，训练进程通过
  命令通道（ZMQ）请求生成；权重同步使用 vLLM 官方跨 GPU NCCL 权重传输
  （vllm.distributed.weight_transfer，backend="nccl"）。
- 采样子进程入口：python -m minitoolrl.train.vllm_backend --serve ...

权重同步协议（一次同步只有四件事，失败直接抛错结束本阶段）：
1. 本批采样全部结束（命令通道顺序执行，天然保证）；
2. 训练器完成参数更新；
3. send_weights() 将新参数传给 vLLM，推理侧 finish_weight_update 后
   confirm_version 清除旧参数对应的前缀缓存；
4. vLLM 确认版本号后接受下一批请求；客户端之后断言每批生成的
   policy_version 与本地版本一致。

同步过程不落盘 checkpoint，也不保存参数副本文件。vllm 与 pyzmq 均懒加载，
仅在创建采样器或启动服务时加载这些依赖。
"""

# 一个是训练的卡，一个是采样的卡，训练的卡把采样的任务外包给采样的卡。此外要注意rollout是批量生成，要想数据的形状。
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

# ------------------------------------------------------------------ 协议结构


@dataclass
class GenRequest:
    """一条生成请求：输入 token IDs 与采样参数；request_id 由调用方保证唯一。"""
    request_id: Any
    prompt_token_ids: List[int]
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: Optional[int] = None


@dataclass
class Generation:
    """一条生成结果：只保留当前消费者实际使用的字段。"""
    request_id: Any
    token_ids: List[int]
    token_logprobs: Optional[List[float]]
    finish_reason: str
    policy_version: int


def request_to_dict(req: GenRequest) -> dict:
    return asdict(req)


def request_from_dict(row: dict) -> GenRequest:
    return GenRequest(
        request_id=row['request_id'],
        prompt_token_ids=[int(t) for t in row['prompt_token_ids']],
        max_new_tokens=int(row['max_new_tokens']),
        temperature=float(row.get('temperature', 0.0)),
        top_p=float(row.get('top_p', 1.0)),
        top_k=int(row.get('top_k', 0)),
        seed=None if row.get('seed') is None else int(row['seed']),
    )


def generation_to_dict(gen: Generation) -> dict:
    return asdict(gen)


def generation_from_dict(row: dict) -> Generation:
    logprobs = row.get('token_logprobs')
    return Generation(
        request_id=row['request_id'],
        token_ids=[int(t) for t in row['token_ids']],
        token_logprobs=None if logprobs is None else [float(x) for x in logprobs],
        finish_reason=str(row['finish_reason']),
        policy_version=int(row['policy_version']),
    )


def assert_policy_version(generations: Sequence[Generation], expected: int) -> None:
    """采样→更新→同步协议的一致性检查：本批轨迹必须来自同一策略版本。"""
    bad = [g.request_id for g in generations if g.policy_version != expected]
    if bad:
        raise RuntimeError(
            'policy version mismatch: expected %d, requests %s used a stale sampler'
            % (expected, bad[:5]))


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


# ------------------------------------------------------------------ vLLM 结果提取


def _sampled_logprobs(token_ids: Sequence[int], logprobs_per_token) -> Optional[List[float]]:
    """vLLM 每个位置返回 {token_id: Logprob}；取出实际采样 token 的对数概率。"""
    if logprobs_per_token is None:
        return None
    if len(token_ids) != len(logprobs_per_token):
        raise RuntimeError('sampler logprob count differs from generated tokens')
    out = []
    for tid, entry in zip(token_ids, logprobs_per_token):
        if entry is None or tid not in entry:
            return None
        value = float(entry[tid].logprob)
        if not math.isfinite(value):
            raise RuntimeError('sampler returned a non-finite logprob')
        out.append(value)
    return out


def outputs_to_generations(requests: Sequence[GenRequest], outputs,
                           policy_version: int) -> List[Generation]:
    generations = []
    if len(requests) != len(outputs):
        raise RuntimeError('sampler output count differs from request count')
    for req, out in zip(requests, outputs):
        completion = out.outputs[0]
        token_ids = [int(t) for t in completion.token_ids]
        generations.append(Generation(
            request_id=req.request_id,
            token_ids=token_ids,
            token_logprobs=_sampled_logprobs(token_ids, completion.logprobs),
            finish_reason=str(completion.finish_reason),
            policy_version=policy_version,
        ))
    return generations


def _sampling_params(req: GenRequest, logprobs: bool):
    from vllm import SamplingParams  # 懒加载
    return SamplingParams(
        temperature=req.temperature, top_p=req.top_p, top_k=req.top_k,
        max_tokens=req.max_new_tokens, seed=req.seed, detokenize=False,
        logprobs=1 if logprobs else None)


# ------------------------------------------------------------------ 进程内采样器


class LocalVLLM:
    """评测与同卡基准使用的进程内 vLLM 采样器（不含权重同步）。"""

    def __init__(self, model_path: str, *, max_model_len: int = 4096,
                 gpu_memory_utilization: float = 0.85, dtype: str = 'bfloat16'):
        from vllm import LLM  # 懒加载
        self.llm = LLM(model=model_path, dtype=dtype, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization,
                       enforce_eager=False)
        self.version = 0

    def generate(self, requests: Sequence[GenRequest], *, logprobs: bool = False
                 ) -> List[Generation]:
        from vllm import TokensPrompt  # 懒加载
        prompts = [TokensPrompt(prompt_token_ids=list(r.prompt_token_ids)) for r in requests]
        params = [_sampling_params(r, logprobs) for r in requests]
        outputs = self.llm.generate(prompts, params)
        return outputs_to_generations(requests, outputs, self.version)

    def close(self) -> None:
        del self.llm
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ------------------------------------------------------------------ 训推分离客户端


class _ZMQTransport:
    """REQ/REP JSON 命令通道；生成 token 走 JSON（量小），权重走 NCCL 不经过这里。"""

    def __init__(self, host: str, port: int, timeout_seconds: float):
        import zmq  # 懒加载
        self._zmq = zmq
        self.context = zmq.Context()
        self.endpoint = 'tcp://%s:%d' % (host, port)
        self.timeout_ms = max(1, int(timeout_seconds * 1000))

    def call(self, message: dict, *, timeout_seconds=None) -> dict:
        # vLLM 的 NCCL 握手在工作线程调用 RPC。Context 可跨线程共享，REQ socket
        # 必须在调用线程创建；每次请求关闭后也不会留下超时后的 REQ 等待状态。
        socket = self.context.socket(self._zmq.REQ)
        timeout = self.timeout_ms if timeout_seconds is None else max(1, int(timeout_seconds * 1000))
        socket.setsockopt(self._zmq.RCVTIMEO, timeout)
        socket.setsockopt(self._zmq.SNDTIMEO, timeout)
        socket.setsockopt(self._zmq.LINGER, 0)
        socket.connect(self.endpoint)
        try:
            socket.send_string(json.dumps(message))
            reply = json.loads(socket.recv_string())
        except self._zmq.Again as exc:
            raise RuntimeError('vLLM sampler did not answer in time: '
                               + str(message.get('op'))) from exc
        finally:
            socket.close(0)
        if reply.get('error'):
            raise RuntimeError('vLLM sampler error on %s: %s'
                               % (message.get('op'), reply['error']))
        return reply

    def close(self) -> None:
        self.context.term()


class _WeightSyncClient:
    """实现 vllm VLLMWeightSyncClient 协议：把四个传输阶段调用转发给采样进程。"""

    def __init__(self, transport):
        self._transport = transport

    def init_weight_transfer_engine(self, init_info: Dict[str, Any]) -> None:
        self._transport.call({'op': 'init_weight_transfer_engine', 'init_info': init_info})

    def start_weight_update(self) -> None:
        self._transport.call({'op': 'start_weight_update'})

    def update_weights(self, update_info: Dict[str, Any]) -> None:
        self._transport.call({'op': 'update_weights', 'update_info': update_info})

    def finish_weight_update(self, weight_version: Optional[str] = None) -> None:
        self._transport.call({'op': 'finish_weight_update', 'weight_version': weight_version})


class VLLMRollout:
    """GRPO 训推分离：训练进程一侧的采样与权重同步接口。

    采样进程常驻另一张显卡；generate 只接受 token IDs，返回生成 token、概率和
    结束原因；业务与工具执行仍在训练进程的 Episode 中完成。
    """

    def __init__(self, model_path: str, *, gpu: str, host: str = '127.0.0.1',
                 max_model_len: int = 4096, gpu_memory_utilization: float = 0.85,
                 startup_timeout: float = 900.0, request_timeout: float = 600.0):
        self.host = host
        self.port = _free_port()
        self.nccl_port = _free_port()
        self.version = 0
        self._engine = None
        self._closed = False
        self.process = None
        self.transport = None
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
        for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
            env.pop(key, None)
        argv = [sys.executable, '-u', '-m', 'minitoolrl.train.vllm_backend',
                '--serve', '--model', model_path, '--host', host,
                '--port', str(self.port),
                '--max-model-len', str(max_model_len),
                '--gpu-memory-utilization', str(gpu_memory_utilization)]
        self.process = subprocess.Popen(argv, env=env)
        deadline = time.monotonic() + startup_timeout
        try:
            self.transport = _ZMQTransport(host, self.port, request_timeout)
            while True:
                if self.process.poll() is not None:
                    raise RuntimeError('vLLM sampler exited during startup with code %s'
                                       % self.process.returncode)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError('vLLM sampler did not become ready')
                try:
                    reply = self.transport.call({'op': 'health'}, timeout_seconds=min(2., remaining))
                    if reply.get('ok'):
                        break
                except RuntimeError:
                    continue
        except BaseException:
            self.close(graceful=False)
            raise

    # ------------------------------------------------------------- 采样
    def generate(self, requests: Sequence[GenRequest], *, logprobs: bool = True
                 ) -> List[Generation]:
        reply = self.transport.call({
            'op': 'generate',
            'logprobs': bool(logprobs),
            'requests': [request_to_dict(r) for r in requests],
        })
        generations = [generation_from_dict(row) for row in reply['generations']]
        assert_policy_version(generations, self.version)
        return generations

    # ---------------------------------------------------------- 权重同步
    def sync_weights(self, model, version: int) -> None:
        """传输参数、清除旧缓存、确认新版本可用；任何失败直接抛错。"""
        if version != self.version + 1:
            raise RuntimeError('weight version must increase: %s <= %s'
                               % (version, self.version))
        try:
            self._sync_weights(model, version)
        except BaseException:
            self.close(graceful=False)
            raise

    def _sync_weights(self, model, version):
        if self._engine is None:
            from vllm.distributed.weight_transfer import (  # 懒加载
                ModuleSource, WeightTransferTrainerFactory)
            from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo
            self._engine = WeightTransferTrainerFactory.trainer_init(
                init_info=NCCLTrainerInitInfo(
                    master_address=self.host, master_port=self.nccl_port,
                    world_size=2,  # 训练进程 rank 0 + 一个推理 worker
                    rank=0, packed=True),
                client=_WeightSyncClient(self.transport),
                source=ModuleSource(model),
            )
        self._engine.send_weights()
        reply = self.transport.call({'op': 'confirm_version', 'version': int(version)})
        if int(reply.get('version', -1)) != int(version):
            raise RuntimeError('sampler did not confirm weight version %d' % version)
        self.version = int(version)

    # ------------------------------------------------------------- 结束
    def close(self, *, graceful=True) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if graceful and self.transport is not None:
                self.transport.call({'op': 'shutdown'}, timeout_seconds=5.)
        except Exception:
            pass
        try:
            self.transport.close()
        except Exception:
            pass
        if self.process is not None:
            try:
                self.process.wait(timeout=5 if graceful else 0)
            except Exception:
                self.process.kill()
                self.process.wait(timeout=5)


# ------------------------------------------------------------------ 采样子进程


def _serve(args) -> None:
    """常驻采样进程：持有 LLM，顺序执行命令通道上的请求。"""
    import zmq  # 懒加载
    from vllm import LLM, TokensPrompt  # 懒加载
    from vllm.config import WeightTransferConfig  # 懒加载

    llm = LLM(model=args.model, dtype='bfloat16', max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              weight_transfer_config=WeightTransferConfig(backend='nccl'))
    state = dict(version=0, phase='ready')
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind('tcp://%s:%d' % (args.host, args.port))

    def handle(message: dict) -> dict:
        op = message.get('op')
        if op == 'health':
            return {'ok': True, 'version': state['version']}
        if op == 'generate':
            if state['phase'] != 'ready':
                raise RuntimeError('new weights must be confirmed before sampling')
            requests = [request_from_dict(r) for r in message['requests']]
            prompts = [TokensPrompt(prompt_token_ids=list(r.prompt_token_ids))
                       for r in requests]
            params = [_sampling_params(r, bool(message.get('logprobs'))) for r in requests]
            outputs = llm.generate(prompts, params)
            generations = outputs_to_generations(requests, outputs, state['version'])
            return {'generations': [generation_to_dict(g) for g in generations]}
        if op == 'init_weight_transfer_engine':
            llm.init_weight_transfer_engine({'init_info': message['init_info']})
            return {'ok': True}
        if op == 'start_weight_update':
            if state['phase'] != 'ready':
                raise RuntimeError('weight update already in progress')
            state['phase'] = 'receiving'
            llm.start_weight_update()
            return {'ok': True}
        if op == 'update_weights':
            if state['phase'] != 'receiving':
                raise RuntimeError('weight update has not started')
            llm.update_weights({'update_info': message['update_info']})
            return {'ok': True}
        if op == 'finish_weight_update':
            if state['phase'] != 'receiving':
                raise RuntimeError('weight update has not started')
            llm.finish_weight_update(message.get('weight_version'))
            state['phase'] = 'loaded'
            return {'ok': True}
        if op == 'confirm_version':
            if state['phase'] != 'loaded' or int(message['version']) != state['version'] + 1:
                raise RuntimeError('weight confirmation requires the next completed update')
            # 新参数已就位：清除旧参数对应的前缀缓存，再确认版本。
            if not llm.reset_prefix_cache():
                raise RuntimeError('sampler prefix cache was not cleared')
            llm.update_weight_version(str(message['version']))
            if llm.get_weight_version() != str(message['version']):
                raise RuntimeError('vLLM engine did not commit the requested weight version')
            state['version'] = int(message['version'])
            state['phase'] = 'ready'
            return {'version': state['version']}
        if op == 'shutdown':
            return {'ok': True}
        raise RuntimeError('unknown op: ' + str(op))

    try:
        while True:
            message = json.loads(socket.recv_string())
            try:
                reply = handle(message)
            except Exception as exc:  # 让客户端拿到可读错误并终止本阶段
                reply = {'error': '%s: %s' % (type(exc).__name__, exc)}
            socket.send_string(json.dumps(reply))
            if message.get('op') == 'shutdown':
                break
    finally:
        socket.close(0)
        context.term()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--model', default=None)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    args = parser.parse_args(argv)
    if not args.serve:
        parser.error('vllm_backend is a worker module; pass --serve')
    if not args.model or not args.port:
        parser.error('--serve requires --model and --port')
    _serve(args)


if __name__ == '__main__':
    main()
