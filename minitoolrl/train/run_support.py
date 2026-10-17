"""Process launching, checkpoint pins and input/output helpers for the current pipeline."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Optional
from importlib.metadata import PackageNotFoundError, version

from minitoolrl.paths import PROJECT_ROOT
from minitoolrl.data.input_bundle import DEFAULT_ROOT, REQUIRED_FILES, MARKER, MANIFEST, verify_bundle
from minitoolrl.train import experiment_config as C

BASE_REPO = 'Qwen/Qwen3-0.6B'


BASE_REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
BASE_TEMPLATE_SHA256 = 'a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8'


BASE_SHA256 = 'f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'


ARMS = {
    'C_mixed': dict(train='inputs/c_recipe/train_C.jsonl', plan='inputs/c_recipe/schedules/C_mixed.json', accum=8),
    'U_mixed': dict(train='inputs/c_recipe/train_U.jsonl', plan='inputs/c_recipe/schedules/U_mixed.json', accum=9),
    'C_blocked': dict(train='inputs/control/train_control.jsonl', plan=None, accum=8),
}


def now() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


def installed_torch_version():
    # Planning, packaging and input validation do not load a training runtime.
    try:
        return version('torch')
    except PackageNotFoundError:
        return None


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding='utf-8'))


def dump_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


class LaunchError(RuntimeError):
    pass


class Session:
    def __init__(self, args):
        self.args = args
        self.run_id = args.run_id
        self.run = Path(args.run_dir)
        self.python = args.python
        self.world = int(args.world_size)
        self.gpus = [g.strip() for g in args.gpus.split(',') if g.strip()] if args.gpus else [str(i) for i in range(self.world)]
        self.repo = Path(args.repo) if args.repo else PROJECT_ROOT
        self.dry = bool(args.dry_run)
        self.pins_path = self.run / 'pins.json'
        self.pins = load_json(self.pins_path, {})
        self.ledger_path = self.run / 'budget.json'
        self.ledger = load_json(self.ledger_path, dict(run_id=self.run_id, events=[]))

    # ------------------------------------------------------------- bookkeeping
    def path(self, rel: str) -> Path:
        return self.run / rel

    def pin(self, name: str, path: Path, stage: str, **meta) -> dict:
        weight = Path(path) / 'model.safetensors'
        record = dict(path=str(path), sha256=(sha256_file(weight) if weight.exists() and not self.dry else None),
                      bytes=(weight.stat().st_size if weight.exists() else None), stage=stage,
                      recorded_local=now(), **meta)
        if self.args.input_version:
            record['input_version'] = self.args.input_version
            record['inputs_manifest_sha256'] = sha256_file(self.run / MANIFEST)
        self.pins[name] = record
        if not self.dry:
            dump_json(self.pins_path, self.pins)
        print('PIN %s %s' % (name, json.dumps(record, ensure_ascii=False)), flush=True)
        return record

    def need_pin(self, name: str) -> dict:
        if name not in self.pins or not self.pins[name].get('sha256'):
            if self.dry:
                placeholder = dict(path=str(self.run / ('<%s>' % name)), sha256='<%s-sha>' % name, dry_run=True)
                if self.args.input_version:
                    placeholder.update(input_version=self.args.input_version,
                        inputs_manifest_sha256=sha256_file(self.run / MANIFEST))
                self.pins.setdefault(name, placeholder)
                return placeholder
            raise LaunchError('pin %s missing; run the earlier stage first' % name)
        if self.args.input_version and (
                self.pins[name].get('input_version') != self.args.input_version or
                self.pins[name].get('inputs_manifest_sha256') != sha256_file(
                    self.run / MANIFEST)):
            raise LaunchError('pin %s belongs to another input version; rebuild the chain' % name)
        return self.pins[name]

    def event(self, stage: str, note: str, **extra) -> None:
        self.ledger['events'].append(dict(stage=stage, note=note, local=now(), **extra))
        if not self.dry:
            dump_json(self.ledger_path, self.ledger)

    def record_stage(self, stage: str, value: dict) -> None:
        value = dict(value, stage=stage, finished_local=now())
        if not self.dry:
            dump_json(self.run / (stage + '_status.json'), value)
        print(json.dumps(dict(stage=stage, ok=True), ensure_ascii=False), flush=True)

    def stop(self, reason: str, **extra) -> None:
        dump_json(self.run / 'stopped.json', dict(reason=reason, local=now(), **extra))
        raise LaunchError(reason)

    # ---------------------------------------------------------------- commands
    def base_env(self, extra: Optional[dict] = None) -> dict:
        env = dict(os.environ)
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONPATH'] = str(self.repo) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        env.setdefault('TOKENIZERS_PARALLELISM', 'false')
        for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT'):
            env.pop(key, None)
        if extra:
            env.update(extra)
        return env

    def py(self, module: str, *argv: str) -> List[str]:
        return [self.python, '-u', '-m', module, *map(str, argv)]

    def torchrun(self, nproc: int, module: str, *argv: str) -> List[str]:
        """Every distributed-capable training launches through torch.distributed.run,
        including nproc=1: DeepSpeed's init_distributed needs RANK/WORLD_SIZE/MASTER_*
        from the launcher environment even for a single process."""
        return [self.python, '-u', '-m', 'torch.distributed.run', '--standalone', '--nnodes', '1',
                '--nproc_per_node', str(nproc), '--master_port', str(free_port()),
                '-m', module, *map(str, argv)]

    def gpu_env(self, gpu_ids: List[str]) -> dict:
        return {'CUDA_VISIBLE_DEVICES': ','.join(gpu_ids)}

    def launch(self, name: str, argv: List[str], timeout: float, log_rel: str,
               env_extra: Optional[dict] = None) -> dict:
        """Run one child to completion (own process group on POSIX, hard timeout, appended log)."""
        log = self.run / 'logs' / log_rel
        print('LAUNCH %s: %s' % (name, ' '.join(argv)), flush=True)
        if self.dry:
            return dict(stage=name, command=argv, dry_run=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        env = self.base_env(env_extra)
        started = time.monotonic()
        with log.open('ab') as handle:
            handle.write(('\n===== %s %s =====\n' % (now(), ' '.join(argv))).encode('utf-8'))
            handle.flush()
            popen_kw = dict(cwd=str(self.repo), stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
            if platform.system() != 'Windows':
                popen_kw['start_new_session'] = True
            proc = subprocess.Popen(argv, **popen_kw)
            timed_out = False
            while proc.poll() is None:
                if timeout and time.monotonic() - started > timeout:
                    timed_out = True
                    _kill(proc)
                    break
                time.sleep(2)
        seconds = time.monotonic() - started
        self.event(name, 'launch', seconds=round(seconds, 1), exit_code=proc.returncode, timed_out=timed_out)
        if timed_out:
            raise LaunchError('%s exceeded its %ds cap after %.1fs (see %s)' % (name, timeout, seconds, log))
        if proc.returncode != 0:
            raise LaunchError('%s exited with code %s after %.1fs (see %s)' % (name, proc.returncode, seconds, log))
        return dict(stage=name, command=argv, exit_code=proc.returncode, seconds=round(seconds, 1), log=str(log))

    def parallel(self, jobs: List[dict], max_parallel: int) -> List[dict]:
        """jobs: dict(name, argv, timeout, log, gpu). Runs up to max_parallel at once, one GPU each."""
        results = []
        pending = list(jobs)
        running: List[tuple] = []
        for job in pending:
            print('QUEUE %s (gpu %s): %s' % (job['name'], job.get('gpu'), ' '.join(job['argv'])), flush=True)
        if self.dry:
            return [dict(stage=j['name'], command=j['argv'], dry_run=True) for j in jobs]
        failures = []
        while pending or running:
            while pending and len(running) < max_parallel:
                busy_gpus = {job.get('gpu') for job, _, _, _ in running}
                index = next((i for i, job in enumerate(pending)
                              if job.get('gpu') not in busy_gpus), None)
                if index is None:
                    break
                job = pending.pop(index)
                log = self.run / 'logs' / job['log']
                log.parent.mkdir(parents=True, exist_ok=True)
                handle = log.open('ab')
                handle.write(('\n===== %s %s =====\n' % (now(), ' '.join(job['argv']))).encode('utf-8'))
                handle.flush()
                env = self.base_env(self.gpu_env([job['gpu']]) if job.get('gpu') is not None else None)
                popen_kw = dict(cwd=str(self.repo), stdout=handle, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=env)
                if platform.system() != 'Windows':
                    popen_kw['start_new_session'] = True
                proc = subprocess.Popen(job['argv'], **popen_kw)
                running.append((job, proc, handle, time.monotonic()))
            time.sleep(2)
            still = []
            for job, proc, handle, started in running:
                code = proc.poll()
                elapsed = time.monotonic() - started
                if code is None and job.get('timeout') and elapsed > job['timeout']:
                    _kill(proc)
                    code = proc.wait()
                    failures.append('%s timed out after %.0fs' % (job['name'], elapsed))
                if code is None:
                    still.append((job, proc, handle, started))
                    continue
                handle.close()
                self.event(job['name'], 'parallel', seconds=round(elapsed, 1), exit_code=code, gpu=job.get('gpu'))
                results.append(dict(stage=job['name'], exit_code=code, seconds=round(elapsed, 1), gpu=job.get('gpu')))
                if code != 0:
                    failures.append('%s exited with %s' % (job['name'], code))
            running = still
        if failures:
            raise LaunchError('parallel jobs failed: ' + '; '.join(failures))
        return results


def _kill(proc: subprocess.Popen) -> None:
    try:
        if platform.system() != 'Windows':
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(os.getpgid(proc.pid), sig)
                except ProcessLookupError:
                    return
                time.sleep(5)
                if proc.poll() is not None:
                    return
        else:
            proc.kill()
    except Exception:
        proc.kill()


def source_files(package: Path):
    """只收集当前模块，不把数据、旧模型或残留缓存当成源码。"""
    return sorted([*package.glob('*.py'), *(p for name in ('data', 'env', 'train', 'eval', 'tests')
                   for p in (package / name).glob('*.py'))])


def checkpoint_template_sha256(checkpoint: Path):
    """按 Transformers 的优先级读取模板：独立 Jinja 优先于 tokenizer 配置。"""
    template = checkpoint / 'chat_template.jinja'
    text = (template.read_text(encoding='utf-8') if template.exists() else
            load_json(checkpoint / 'tokenizer_config.json', {}).get('chat_template', ''))
    return hashlib.sha256(text.encode('utf-8')).hexdigest() if text else None


def verify_base_template(checkpoint: Path):
    """底座权重与 tokenizer 模板必须来自同一个固定 Qwen revision。"""
    if checkpoint_template_sha256(checkpoint) != BASE_TEMPLATE_SHA256:
        raise LaunchError('base tokenizer must use the pinned official Qwen3 chat template')


def stage_pack(S: Session):
    """Local helper: bundle every frozen input into one tarball for the GPU host."""
    research = Path(S.args.inputs_root) if S.args.inputs_root else DEFAULT_ROOT
    out = Path(S.args.pack_out or (S.repo / 'dist' / S.run_id / 'full_chain_inputs.tar.gz'))
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    verify_bundle(research)
    files = load_json(research / MANIFEST)['files']
    extras = [MARKER, MANIFEST, 'validation.json']
    with tarfile.open(out, 'w:gz') as tar:
        for rel in sorted(REQUIRED_FILES):
            src = research / rel
            tar.add(str(src), arcname=rel)
            manifest[rel] = dict(source=str(src), bytes=src.stat().st_size, sha256=files[rel]['sha256'])
        for rel in extras:
            tar.add(str(research / rel), arcname=rel)
    dump_json(out.with_suffix('.manifest.json'), dict(run_id=S.run_id, files=manifest, created_local=now()))
    source_archive = out.parent / 'minitoolrl_source.tar.gz'
    package = S.repo / 'minitoolrl'
    sources = [*source_files(package), *package.joinpath('docs').glob('*.md'), package / 'README.md',
               *(S.repo / name for name in ('README.md', 'AGENTS.md', 'pyproject.toml', 'LICENSE', 'NOTICE',
                                            'requirements.txt', 'requirements-gpu.txt'))]
    with tarfile.open(source_archive, 'w:gz') as tar:
        for path in sorted(sources):
            tar.add(path, arcname=str(path.relative_to(S.repo)))
    print(json.dumps(dict(tarball=str(out), source_archive=str(source_archive),
                          files=len(manifest), bytes=out.stat().st_size), ensure_ascii=False))
    return dict(tarball=str(out), source_archive=str(source_archive), files=manifest)


def stage_fetch_base(S: Session):
    dest = S.run / 'models' / 'Qwen3-0.6B'
    weight = dest / 'model.safetensors'
    if S.args.base_dir:
        src = Path(S.args.base_dir)
        if src.resolve() != dest.resolve():
            shutil.copytree(src, dest, dirs_exist_ok=True)
    elif not weight.exists() or sha256_file(weight) != BASE_SHA256:
        attempts = []
        # 使用公开下载 API，在 GPU 的 huggingface_hub 1.x 上同样按固定 revision 下载。
        code = ('from huggingface_hub import snapshot_download; import sys; '
                'snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])')
        hf = [S.python, '-c', code, BASE_REPO, BASE_REVISION, str(dest)]
        try:
            S.launch('fetch_base_hf', hf, S.args.timeout or 1800, 'fetch_base.log',
                     env_extra={'HF_ENDPOINT': S.args.hf_endpoint})
            attempts.append(dict(source='huggingface', ok=weight.exists()))
        except LaunchError as exc:
            attempts.append(dict(source='huggingface', ok=False, error=str(exc)[:500]))
        dump_json(S.run / 'models' / 'fetch_attempts.json', attempts)
    if S.dry:
        S.pin('base_qwen', dest, 'fetch_base', repo=BASE_REPO, revision=BASE_REVISION)
        return dict(dry_run=True)
    if not weight.exists():
        S.stop('base_weights_not_downloaded')
    got = sha256_file(weight)
    if got != BASE_SHA256:
        S.stop('base_sha_mismatch', got=got, expected=BASE_SHA256)
    verify_base_template(dest)
    record = S.pin('base_qwen', dest, 'fetch_base', repo=BASE_REPO, revision=BASE_REVISION)
    return dict(base=record)


def _arm_cmd(S: Session, arm: str, n: int, out: Path) -> List[str]:
    g0 = S.need_pin('G0')
    spec = ARMS[arm]
    argv = ['--model', g0['path'], '--output', out, '--train-data', S.path(spec['train']),
            '--val-data', S.path('inputs/c_recipe/val_common.jsonl'), '--expected-sha', g0['sha256'],
            '--batch-size', 2, '--grad-accum', spec['accum'], '--epochs', 2, '--lr', S.args.arm_lr,
            '--weight-decay', 0.01, '--grad-clip', 1.0, '--seed', 17, '--max-length', 4096,
            '--max-seconds', 2400, '--keep-file-order']
    if spec['plan']:
        argv += ['--update-plan', S.path(spec['plan'])]
    return S.torchrun(n, 'minitoolrl.train.train_sft', *argv)


def _eval_cmd(S: Session, model: str, pool: Path, out: Path, tag: str) -> List[str]:
    argv = ['--model', model, '--pool', pool, '--out-dir', out, '--tag', tag, '--mode', 'greedy',
            '--batch-size', 8, '--max-seconds', 1800, '--top-p', 0.95, '--top-k', 0]
    argv += ['--temperature', 0, '--tool-order', S.args.eval_tool_order]
    return S.py('minitoolrl.eval.evaluate', *argv)


def write_comp_suite_protocol(S: Session, models: Dict[str, dict]) -> Path:
    from minitoolrl.data.composition_suite import TARGET_FAMILIES
    root = S.run / 'eval' / 'comp_suite'
    root.mkdir(parents=True, exist_ok=True)
    pool_src = S.path('inputs/comp_suite/test_comp_suite.jsonl')
    pool = root / 'test_comp_suite.jsonl'
    if not pool.exists():
        shutil.copy2(pool_src, pool)
    template = load_json(S.path('inputs/comp_suite/protocol_template.json'), {})
    protocol = dict(
        run_id=S.run_id, input_version=S.args.input_version, status='prepared', parent_protocol=template.get('run_id'),
        objective='SFT baseline, GRPO baseline and three continuation arms comparing decomposition and basic tasks.',
        families=list(TARGET_FAMILIES), explicitly_excluded=['bulk_refund'], n_families=len(TARGET_FAMILIES),
        n_worlds_per_family=template.get('n_worlds_per_family', 60), variants_per_world=['A', 'B'],
        n_worlds=template.get('n_worlds', 180), n_records=template.get('n_records', 360),
        models={tag: dict(recorded_path=pin['path'], sha256=pin['sha256'], source_run=S.run_id)
                for tag, pin in models.items()},
        episode_budget=template.get('episode_budget', dict(C.EPISODE_BUDGET)),
        decode=template.get('decode', dict(mode='greedy', temperature=0, top_p=0.95, top_k=0, batch_size=8)),
        policy_version=template.get('policy_version'), metrics=template.get('metrics'),
        limitations=template.get('limitations'),
        artifacts=dict(pool=dict(path=pool.name, sha256=sha256_file(pool))),
    )
    dump_json(root / 'protocol.json', protocol)
    return root


def stage_package(S: Session):
    entries = []
    weights = []
    for path in sorted(S.run.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(S.run).as_posix()
        if path.suffix in ('.safetensors', '.pt', '.pth') or '.part.' in path.name or rel.startswith('models/'):
            if path.name == 'model.safetensors':
                weights.append(dict(path=rel, bytes=path.stat().st_size))
            continue
        entries.append(dict(path=rel, bytes=path.stat().st_size, sha256=sha256_file(path)))
    manifest = dict(run_id=S.run_id, files=entries, weight_artifacts=weights, pins=S.pins,
                    total_bytes=sum(e['bytes'] for e in entries), local=now())
    dump_json(S.run / 'transfer_manifest.json', manifest)
    return dict(files=len(entries), weights=len(weights))
