"""阶段执行会话：构造子进程命令、串行或并行启动、记录阶段标记与检查点 pin。

一次 `run_experiment --stage X` 构造一个 Session。它只负责“怎么执行、执行过什么”，
不决定训练内容（见 train_sft.py / train_grpo.py），也不负责输入打包与回传清单
（见 run_artifacts.py）。子进程失败、超时或产物缺失直接抛 LaunchError，调用方
据此不写成功标记。
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import List, Optional

from minitoolrl import files
from minitoolrl.data.input_bundle import MANIFEST
from minitoolrl.paths import PROJECT_ROOT


class LaunchError(RuntimeError):
    """子进程或阶段产物不满足要求；阶段不得记为成功。"""


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def checkpoint_template_sha256(checkpoint: Path):
    template = Path(checkpoint) / 'chat_template.jinja'
    text = (template.read_text(encoding='utf-8') if template.exists() else
            files.load_json(Path(checkpoint) / 'tokenizer_config.json', {}).get('chat_template', ''))
    return hashlib.sha256(text.encode('utf-8')).hexdigest() if text else None


class Session:
    """一个运行目录上的执行会话：命令、日志、pin 与阶段标记。"""

    def __init__(self, args):
        self.args = args
        self.run_id = args.run_id
        self.run = Path(args.run_dir)
        self.python = args.python
        self.world = int(args.world_size)
        self.gpus = [g.strip() for g in args.gpus.split(',') if g.strip()] if args.gpus else [
            str(i) for i in range(self.world)]
        self.repo = Path(args.repo) if args.repo else PROJECT_ROOT
        self.pipeline_version = args.pipeline_version
        self.plan = files.load_json(self.run / 'protocol.json', {})
        self.dry = bool(args.dry_run)
        self.pins_path = self.run / 'pins.json'
        self.pins = files.load_json(self.pins_path, {})
        self._verified_pins = set()
        self.ledger_path = self.run / 'budget.json'
        self.ledger = files.load_json(self.ledger_path, dict(run_id=self.run_id, events=[]))

    # ------------------------------------------------------------- bookkeeping
    def path(self, rel: str) -> Path:
        return self.run / rel

    def pin(self, name: str, path: Path, stage: str, **meta) -> dict:
        """记录一个模型终点；后续阶段从这里取路径和预期权重。"""
        path = Path(path)
        weight = path / 'model.safetensors'
        if self.dry:
            return dict(path=str(path), sha256='dry-run')
        self._verified_pins.discard(name)
        if not weight.is_file():
            raise LaunchError('cannot pin absent checkpoint: ' + str(path))
        record = dict(path=str(path), sha256=files.sha256_file(weight), stage=stage,
                      input_version=self.args.input_version,
                      inputs_manifest_sha256=files.sha256_file(self.run / MANIFEST),
                      pipeline_version=self.pipeline_version,
                      chat_template_sha256=checkpoint_template_sha256(path),
                      recorded_local=files.now(), **meta)
        self.pins[name] = record
        files.dump_json(self.pins_path, self.pins)
        print('PIN %s %s' % (name, json.dumps(record, ensure_ascii=False)), flush=True)
        return record

    def need_pin(self, name: str) -> dict:
        """下一阶段只使用本次运行已登记的检查点。"""
        pin = self.pins.get(name)
        if pin is None:
            if self.dry:
                return dict(path=str(self.run / ('<%s>' % name)), sha256='dry-run')
            raise LaunchError('pin %s missing; run the earlier stage first' % name)
        if self.dry or name in self._verified_pins:
            return pin
        if (pin.get('input_version') != self.args.input_version
                or pin.get('inputs_manifest_sha256') != files.sha256_file(self.run / MANIFEST)
                or pin.get('pipeline_version') != self.pipeline_version):
            raise LaunchError('pin %s belongs to another run; rebuild the chain' % name)
        path = Path(pin['path'])
        if files.sha256_file(path / 'model.safetensors') != pin['sha256']:
            raise LaunchError('checkpoint missing or changed: ' + name)
        if checkpoint_template_sha256(path) != pin['chat_template_sha256']:
            raise LaunchError('checkpoint template changed: ' + name)
        self._verified_pins.add(name)
        return pin

    def event(self, stage: str, note: str, **extra) -> None:
        self.ledger['events'].append(dict(stage=stage, note=note, local=files.now(), **extra))
        if not self.dry:
            files.dump_json(self.ledger_path, self.ledger)

    def record_stage(self, stage: str, value: dict) -> None:
        value = dict(value, stage=stage, finished_local=files.now())
        if not self.dry:
            files.dump_json(self.run / (stage + '_status.json'), value)
        print(json.dumps(dict(stage=stage, ok=True), ensure_ascii=False), flush=True)

    def base_env(self, extra: Optional[dict] = None) -> dict:
        env = dict(os.environ)
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONPATH'] = str(self.repo) + (
            os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        env.setdefault('TOKENIZERS_PARALLELISM', 'false')
        for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT'):
            env.pop(key, None)
        if extra:
            env.update(extra)
        return env

    def py(self, module: str, *argv: str) -> List[str]:
        return [self.python, '-u', '-m', module, *map(str, argv)]

    def torchrun(self, nproc: int, module: str, *argv: str) -> List[str]:
        """所有可分布式训练都经 torch.distributed.run 启动，nproc=1 也一样：
        DeepSpeed 的 init_distributed 需要启动器提供的 RANK/WORLD_SIZE/MASTER_*。"""
        return [self.python, '-u', '-m', 'torch.distributed.run', '--standalone', '--nnodes', '1',
                '--nproc_per_node', str(nproc), '--master_port', str(free_port()),
                '-m', module, *map(str, argv)]

    def gpu_env(self, gpu_ids: List[str]) -> dict:
        return {'CUDA_VISIBLE_DEVICES': ','.join(gpu_ids)}

    def launch(self, name: str, argv: List[str], timeout: float, log_rel: str,
               env_extra: Optional[dict] = None) -> dict:
        """运行一个子进程；失败和超时不会返回成功。"""
        job = dict(name=name, argv=argv, timeout=timeout, log=log_rel, env=env_extra)
        return self.parallel([job], max_parallel=1)[0]

    def parallel(self, jobs: List[dict], max_parallel: int) -> List[dict]:
        """同时最多运行 max_parallel 个任务，同一 GPU 不并行。"""
        for job in jobs:
            print('QUEUE %s (gpu %s): %s' % (job['name'], job.get('gpu'), ' '.join(job['argv'])),
                  flush=True)
        if self.dry:
            return [dict(stage=job['name'], command=job['argv'], dry_run=True) for job in jobs]
        pending = list(jobs)
        running = []
        results = []
        failures = []
        while pending or running:
            while pending and len(running) < max_parallel:
                busy = {job.get('gpu') for job, _, _, _ in running}
                index = next((i for i, job in enumerate(pending) if job.get('gpu') not in busy), None)
                if index is None:
                    break
                job = pending.pop(index)
                log = self.run / 'logs' / job['log']
                log.parent.mkdir(parents=True, exist_ok=True)
                handle = log.open('ab')
                handle.write(('\n===== %s %s =====\n' % (files.now(),
                                                         ' '.join(job['argv']))).encode('utf-8'))
                env = job.get('env')
                if env is None and job.get('gpu') is not None:
                    env = self.gpu_env([job['gpu']])
                options = dict(cwd=str(self.repo), stdout=handle, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, env=self.base_env(env))
                if platform.system() != 'Windows':
                    options['start_new_session'] = True
                running.append((job, subprocess.Popen(job['argv'], **options), handle,
                                time.monotonic()))
            time.sleep(0.1)
            still = []
            for job, proc, handle, started in running:
                code = proc.poll()
                seconds = time.monotonic() - started
                timed_out = code is None and job.get('timeout') and seconds > job['timeout']
                if timed_out:
                    code = _kill(proc)
                if code is None:
                    still.append((job, proc, handle, started))
                    continue
                handle.close()
                self.event(job['name'], 'launch', seconds=round(seconds, 1), exit_code=code,
                           timed_out=timed_out, gpu=job.get('gpu'))
                results.append(dict(stage=job['name'], exit_code=code, seconds=round(seconds, 1),
                                    gpu=job.get('gpu')))
                if timed_out:
                    failures.append('%s timed out after %.1fs' % (job['name'], seconds))
                elif code != 0:
                    failures.append('%s exited with %s' % (job['name'], code))
            running = still
        if failures:
            raise LaunchError('; '.join(failures))
        return results


def _kill(proc: subprocess.Popen) -> None:
    """阶段已超时，结束整个进程组并回收启动进程。"""
    if platform.system() == 'Windows':
        proc.kill()
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return proc.wait()
