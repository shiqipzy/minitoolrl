"""One SFT, one GRPO, three arms and paired performance measurements.

Planning and input preparation require only Python. GPU jobs run explicitly,
one stage per invocation; completed stages are never silently retrained.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tarfile
import time
from pathlib import Path

from minitoolrl.data.input_bundle import VERSION, DEFAULT_ROOT, REQUIRED_FILES, MARKER, MANIFEST, verify_bundle
from minitoolrl.paths import RUNS_ROOT, PACKAGE_ROOT, CURRENT_RUN_ID
# 评测元数据里的 policy_version 来自 env/policy.py，而本模块的 VERSION 是输入包版本；
# 两者含义不同，校验评测身份必须用 policy 版本。
from minitoolrl.env.policy import VERSION as POLICY_VERSION
from minitoolrl.train import run_support as IO
from minitoolrl.train import experiment_config as C

PIPELINE_VERSION = 'deepspeed-vllm-engineering-qwen3-v5'
RUN_ID = CURRENT_RUN_ID
STAGES = ('plan', 'pack', 'init', 'fetch_base', 'preflight', 'sft', 'grpo',
          'arms', 'engineering', 'evals', 'analysis', 'package', 'status')
MODELS = ('SFT', 'G0', 'C_blocked', 'C_mixed', 'U_mixed')
COMPARISONS = ('G0-SFT', 'C_mixed-U_mixed', 'C_mixed-G0', 'C_blocked-G0',
               'C_mixed-C_blocked', 'U_mixed-G0')
TRAINER = 'minitoolrl.train.train_sft'
DEFAULTS = dict(world_size=2, gpus='', sft_epochs=2, sft_lr=2e-5, sft_seconds=3600,
                grpo_passes=2, grpo_seconds=3600, arm_seconds=2400, arm_lr=1e-5,
                eval_tool_order='registry',
                probe_updates=40, probe_iterations=20, probe_seconds=1500,
                probe_seq_len=4096, max_gb=23.)
CONFIG_KEYS = tuple(DEFAULTS)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def make_plan(args, root):
    n_sft = len(read_rows(root / 'data/sft_train_v2.jsonl'))
    n_rl = len(read_rows(root / 'data/rl_train.jsonl'))
    iterations = math.ceil(n_rl / C.RL['prompts_per_iteration']) * args.grpo_passes
    sizes = sorted({1, args.world_size})
    return dict(pipeline_version=PIPELINE_VERSION, input_version=VERSION,
        run_id=args.run_id, config={k: getattr(args, k) for k in CONFIG_KEYS},
        objective='Train tool use and compare held-out composition; measure SFT/GRPO performance.',
        base=dict(repo=IO.BASE_REPO, revision=IO.BASE_REVISION, sha256=IO.BASE_SHA256),
        chat_format=dict(source='checkpoint tokenizer', enable_thinking=False,
                         template_sha256=IO.BASE_TEMPLATE_SHA256),
        sft=dict(records=n_sft, epochs=args.sft_epochs,
                 updates=math.ceil(n_sft / 16) * args.sft_epochs, lr=args.sft_lr,
                 global_batch=16, microbatch=2, grad_accum=8, seed=1,
                 backend='DeepSpeed ZeRO-2 (bf16 compute, fp32 optimizer states, no offload)',
                 selection='fixed complete-budget endpoint; dev results are diagnostics',
                 optimizer='one AdamW and one 3%-warmup cosine schedule, final LR ratio 0.1'),
        grpo=dict(pool_tasks=n_rl, pool_passes=args.grpo_passes, iterations=iterations,
                  episodes=iterations * 8,
                  trainer='DeepSpeed ZeRO-2 single training GPU',
                  sampler='resident vLLM process on the sampling GPU; NCCL weight '
                          'transfer after every accepted update',
                  rl=dict(C.RL),
                  endpoint='fixed iteration budget; record actual updates and zero-signal groups'),
        arms={name: dict(spec, epochs=2, updates=144, world_size=args.world_size, lr=args.arm_lr)
              for name, spec in IO.ARMS.items()},
        engineering=dict(included=True, world_sizes=sizes, sft_updates=args.probe_updates,
            grpo_iterations=args.probe_iterations, exclude_first=2,
            sft='same official base, data, order, global batch and LR schedule; '
                'single-rank vs dual-rank strong scaling',
            grpo='DeepSpeed training card + resident vLLM sampling card; verify the '
                 'rollout-update-sync path from the SFT endpoint for a fixed iteration budget',
            metrics='wall-clock seconds and GPU-seconds',
            scope='prefix update drift and performance, not full-training endpoint equivalence'),
        evaluations=dict(models=list(MODELS), greedy=dict(comp_suite=360, c_recipe=372, long=60),
            tool_order=args.eval_tool_order,
            comparisons=list(COMPARISONS), selection='no test-based checkpoint selection'),
        execution=dict(main_training_jobs=5, sft_probe_jobs=len(sizes), grpo_probe_jobs=1,
                       greedy_jobs=len(MODELS)*3,
                       note='time caps are failure limits, not intended budgets or duration estimates'),
        episode_budget=dict(C.EPISODE_BUDGET))


class Session(IO.Session):
    def need_pin(self, name):
        pin = super().need_pin(name)
        if not self.dry:
            if pin.get('pipeline_version') != PIPELINE_VERSION:
                raise IO.LaunchError('checkpoint belongs to a different pipeline: ' + name)
            weight = Path(pin['path']) / 'model.safetensors'
            if not weight.exists() or IO.sha256_file(weight) != pin['sha256']:
                raise IO.LaunchError('checkpoint missing or changed: ' + name)
            if pin.get('chat_template_sha256') and (
                    IO.checkpoint_template_sha256(Path(pin['path'])) != pin['chat_template_sha256']):
                raise IO.LaunchError('checkpoint template changed: ' + name)
        return pin

    def pin(self, name, path, stage, **meta):
        if not self.dry and not (Path(path) / 'model.safetensors').exists():
            raise IO.LaunchError('cannot pin absent checkpoint: ' + str(path))
        return super().pin(name, path, stage, pipeline_version=PIPELINE_VERSION,
                           chat_template_sha256=IO.checkpoint_template_sha256(Path(path)), **meta)


def stage_init(S):
    if (S.run / 'protocol.json').exists() or S.pins:
        raise IO.LaunchError('init requires a fresh run directory; existing protocol is preserved')
    S.run.mkdir(parents=True, exist_ok=True)
    if S.args.inputs_tar:
        with tarfile.open(S.args.inputs_tar, 'r:gz') as tar:
            root = S.run.resolve()
            for member in tar.getmembers():
                target = (root / member.name).resolve()
                if not target.is_relative_to(root) or member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                    raise IO.LaunchError('unsafe input archive member: ' + member.name)
            tar.extractall(root)
    else:
        source = Path(S.args.inputs_root)
        verify_bundle(source)
        for rel in [*sorted(REQUIRED_FILES), MARKER, MANIFEST, 'validation.json']:
            dst = S.run / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / rel, dst)
    verify_bundle(S.run)
    plan = make_plan(S.args, S.run)
    sources = IO.source_files(PACKAGE_ROOT)
    plan.update(created_local=IO.now(), torch=IO.installed_torch_version(),
                inputs=dict(mode='read_only_reference',
                            note='copied into the run directory once at init and verified '
                                 'against the bundle manifest; no stage rewrites them',
                            manifest_sha256=IO.sha256_file(S.run / MANIFEST)),
                source_sha256={str(p.relative_to(PACKAGE_ROOT)): IO.sha256_file(p) for p in sources},
                inputs_manifest_sha256=IO.sha256_file(S.run / MANIFEST))
    IO.dump_json(S.run / 'protocol.json', plan)
    IO.dump_json(S.run / 'inputs_manifest.json', IO.load_json(S.run / MANIFEST))
    IO.dump_json(S.run / 'pins.json', {})
    return dict(protocol=plan)


def require_preflight(S):
    if not S.dry and not IO.load_json(S.run / 'preflight/gates.json', {}).get('all_ok'):
        raise IO.LaunchError('preflight must pass before training')


def stage_preflight(S):
    base = S.need_pin('base_qwen')
    out = S.run / 'preflight'
    for mode in ('smoke', 'memory'):
        argv = S.torchrun(S.world, 'minitoolrl.train.preflight_deepspeed', '--mode', mode, '--out', out)
        if mode == 'memory':
            argv += list(map(str, ['--model', base['path'], '--seq-len', S.args.probe_seq_len,
                                  '--batch', 2, '--accum', max(8, S.world), '--steps', 3,
                                  '--max-gb', S.args.max_gb]))
        S.launch('preflight_' + mode, argv, 900, 'preflight.log', S.gpu_env(S.gpus[:S.world]))
    checks = {mode: IO.load_json(out / ('ds_' + mode + '.json'), {}).get('ok') for mode in ('smoke', 'memory')}
    if not S.dry:
        IO.dump_json(out / 'gates.json', dict(checks=checks, all_ok=all(checks.values())))
        require_preflight(S)
    return dict(checks=checks)


def sft_command(S, n, out, probe=False):
    base = S.need_pin('base_qwen')
    argv = S.torchrun(n, TRAINER, '--model', base['path'], '--expected-sha', base['sha256'],
        '--output', out, '--train-data', S.path('data/sft_train_v2.jsonl'),
        '--val-data', S.path('data/sft_val_v2.jsonl'),
        '--batch-size', 2, '--grad-accum', 8, '--epochs', S.args.sft_epochs,
        '--lr', S.args.sft_lr, '--weight-decay', 0.01, '--grad-clip', 1,
        '--max-length', 4096, '--seed', 1,
        '--max-seconds', S.args.probe_seconds if probe else S.args.sft_seconds)
    return argv + ['--probe', '--max-updates', str(S.args.probe_updates)] if probe else argv


def complete_sft(out, expected, parent_sha=None):
    summary = IO.load_json(out / 'summary.json', {})
    if (summary.get('steps') != expected or summary.get('total_steps_planned') != expected
            or summary.get('optimizer_updates') != expected or summary.get('complete_budget') is not True
            or summary.get('probe') is not False):
        raise IO.LaunchError('SFT budget incomplete; do not use partial endpoint: ' + str(out))
    if parent_sha and IO.load_json(out / 'config.json', {}).get('warm_start_sha') != parent_sha:
        raise IO.LaunchError('saved SFT belongs to a different parent: ' + str(out))
    return summary


def fresh_output(S, out):
    if not S.dry and out.exists() and any(out.iterdir()):
        raise IO.LaunchError('unfinished output already exists; inspect it and use a new run directory: ' + str(out))


def stage_sft(S):
    require_preflight(S)
    base = S.need_pin('base_qwen')
    out = S.run / 'sft'
    expected = make_plan(S.args, S.run)['sft']['updates']
    if not S.dry and (out / 'summary.json').exists():
        summary = complete_sft(out, expected, base['sha256'])
    else:
        fresh_output(S, out)
        S.launch('sft', sft_command(S, S.world, out), S.args.sft_seconds + 1200,
                 'sft.log', S.gpu_env(S.gpus[:S.world]))
        summary = {} if S.dry else complete_sft(out, expected, base['sha256'])
    checkpoint = out / 'endpoint'
    S.launch('sft_dev', S.py('minitoolrl.eval.rollout_eval', '--model', checkpoint,
        '--pool', S.path('data/dev.jsonl'), '--out-dir', out / 'dev_eval', '--tag', 'SFT_dev'),
        900, 'sft_dev.log', S.gpu_env(S.gpus[:1]))
    if not S.dry:
        dev = IO.load_json(out / 'dev_eval/SFT_dev_run.json', {})
        if (dev.get('n_tasks') != len(read_rows(S.path('data/dev.jsonl')))
                or dev.get('weight_sha256') != IO.sha256_file(checkpoint / 'model.safetensors')):
            raise IO.LaunchError('SFT task-success diagnostics missing or incomplete')
    S.pin('SFT', checkpoint, 'sft', world_size=S.world, selection='fixed_budget', updates=expected)
    if not S.dry:
        IO.dump_json(S.run / 'frozen_start.json', dict(path=str(checkpoint),
            weight_sha256=S.pins['SFT']['sha256'],
            chat_template_sha256=IO.sha256_file(checkpoint / 'chat_template.jinja'),
            pipeline_version=PIPELINE_VERSION, selection='fixed_budget', updates=expected))
    return summary


def grpo_command(S, out, probe=False):
    """GRPO uses one DeepSpeed training process and a resident vLLM sampler.

    训练进程占 S.gpus[0]，采样进程用 S.gpus[1] 覆盖自己的 CUDA_VISIBLE_DEVICES。
    """
    start = S.need_pin('SFT')
    iterations = S.args.probe_iterations if probe else make_plan(S.args, S.run)['grpo']['iterations']
    if len(S.gpus) < 2:
        raise IO.LaunchError(
            'the vLLM sampler needs its own GPU: --gpus must list two distinct devices '
            '(trainer uses %s)' % S.gpus[0])
    sampler_gpu = S.gpus[1]
    argv = S.torchrun(1, 'minitoolrl.train.train_grpo', '--model', start['path'],
        '--run-dir', S.run, '--train-pool', S.path('data/rl_train.jsonl'),
        '--dev-subset', S.path('data/dev_curve.jsonl'), '--max-iterations', iterations,
        '--seconds', S.args.probe_seconds if probe else S.args.grpo_seconds,
        '--out', out, '--run-id', S.run_id,
        '--sampler-gpu', sampler_gpu)
    return argv + ['--probe', '--save-reserve', '0'] if probe else argv


def stage_grpo(S):
    require_preflight(S)
    if not S.dry and S.world < 2:
        raise IO.LaunchError('train/inference-separated GRPO needs two GPUs: '
                             'one training card and one vLLM sampling card')
    out = S.run / 'grpo' / 'main'
    fresh_output(S, out)
    S.launch('grpo', grpo_command(S, out), S.args.grpo_seconds + 1200, 'grpo.log', S.gpu_env(S.gpus))
    expected = make_plan(S.args, S.run)['grpo']['iterations']
    final = IO.load_json(out / 'checkpoint_final.json', {})
    if not S.dry:
        cost = IO.load_json(out / 'grpo_cost.json', {})
        if cost.get('iterations') != expected or not final:
            raise IO.LaunchError('GRPO iteration budget incomplete; G0 is not accepted')
    S.pin('G0', Path(final.get('path', str(out / 'models/final'))), 'grpo',
          world_size=1, iterations=expected, sampler='vllm')
    return dict(iterations=expected)


def stage_arms(S):
    require_preflight(S)
    if not S.dry:
        from minitoolrl.eval.runtime import tools_for_order
        expected_tools = list(tools_for_order(S.args.eval_tool_order))
        paths = {spec['train'] for spec in IO.ARMS.values()} | {'inputs/c_recipe/val_common.jsonl'}
        for rel in sorted(paths):
            for row in read_rows(S.path(rel)):
                if row.get('exposed_tools') != expected_tools:
                    raise IO.LaunchError('continuation/evaluation tool order mismatch: ' + rel)
    g0 = S.need_pin('G0')
    results = {}
    for arm in IO.ARMS:
        out = S.run / 'arms' / arm / ('n%d' % S.world) / 'train'
        if not S.dry and (out / 'summary.json').exists():
            results[arm] = complete_sft(out, 144, g0['sha256'])
        else:
            fresh_output(S, out)
            argv = IO._arm_cmd(S, arm, S.world, out)
            argv[argv.index('--max-seconds') + 1] = str(S.args.arm_seconds)
            S.launch('arm_' + arm, argv, S.args.arm_seconds + 900, 'arms.log', S.gpu_env(S.gpus[:S.world]))
            results[arm] = {} if S.dry else complete_sft(out, 144, g0['sha256'])
        S.pin(arm, out / 'endpoint', 'arms', world_size=S.world, updates=144)
    return results


def stage_engineering(S):
    require_preflight(S)
    if not S.dry and S.world != 2:
        raise IO.LaunchError('paired engineering requires two physical GPUs')
    sizes = sorted({1, S.world})
    # SFT 成对性能测量：同一 DeepSpeed 后端单 rank vs 双 rank。
    for n in sizes:
        out = S.run / 'engineering' / 'sft' / ('n%d' % n)
        if not S.dry:
            summary = IO.load_json(out / 'summary.json', {})
            if summary.get('steps') == S.args.probe_updates:
                continue
        fresh_output(S, out)
        S.launch('engineering_sft_n%d' % n, sft_command(S, n, out, True),
                 S.args.probe_seconds + 900, 'engineering.log', S.gpu_env(S.gpus[:n]))
    # GRPO 训推分离性能测量：使用主链相同的 DeepSpeed + vLLM 路径。
    out = S.run / 'engineering' / 'grpo' / 'vllm_separated'
    cost = IO.load_json(out / 'grpo_cost.json', {}) if not S.dry else {}
    if cost.get('iterations') != S.args.probe_iterations:
        fresh_output(S, out)
        S.launch('engineering_grpo_vllm_separated', grpo_command(S, out, True),
                 S.args.probe_seconds + 900, 'engineering.log', S.gpu_env(S.gpus))
    S.launch('engineering_report', S.py('minitoolrl.eval.engineering_report', '--run-dir', S.run,
                                      '--require-complete'), 120, 'engineering.log')
    return dict(world_sizes=sizes, grpo_deployments=['vllm_separated'])


def evaluation_jobs(S):
    models = {tag: S.need_pin(tag) for tag in MODELS}
    jobs = []
    for suite, pool in [('comp_suite', 'inputs/comp_suite/test_comp_suite.jsonl'),
                        ('c_recipe', 'inputs/c_recipe/eval_variants.jsonl'), ('long', 'inputs/long/test_long.jsonl')]:
        for tag, pin in models.items():
            out = S.run / 'eval' / suite / 'run' / tag / 'greedy'
            jobs.append(dict(name=suite+'_'+tag, log='evals.log', timeout=2400,
                argv=IO._eval_cmd(S, pin['path'], S.path(pool), out, tag),
                summary=out / (tag + '_summary.json'), quota=len(read_rows(S.path(pool)))))
    for k, job in enumerate(jobs):
        job['gpu'] = S.gpus[k % len(S.gpus)]
    return jobs


def verify_evaluations(jobs):
    summaries = {}
    for job in jobs:
        row = IO.load_json(job['summary'], {})
        if row.get('filled') != job['quota'] or row.get('quota') != job['quota'] or row.get('incomplete', True):
            raise IO.LaunchError('evaluation missing or incomplete: ' + job['name'])
        from minitoolrl.eval.evaluate import load_done
        cmd = job['argv']
        pool = read_rows(Path(cmd[cmd.index('--pool') + 1]))
        tag = cmd[cmd.index('--tag') + 1]
        expected = {(r['task_id'], None) for r in pool}
        folder = Path(job['summary']).parent
        meta = IO.load_json(folder / (tag + '_run.json'), {})
        if (not meta or meta.get('weight_sha256') != IO.sha256_file(Path(cmd[cmd.index('--model') + 1]) / 'model.safetensors')
                or meta.get('pool_sha256') != IO.sha256_file(Path(cmd[cmd.index('--pool') + 1]))
                or meta.get('policy_version') != POLICY_VERSION or meta.get('backend') != 'vllm'):
            raise IO.LaunchError('evaluation checkpoint or input identity missing: ' + job['name'])
        done = load_done(folder / (tag + '_rollouts.jsonl'))
        analyses = IO.load_json(folder / (tag + '_analysis.json'), [])
        keys = [(r['task_id'], r.get('sample_index')) for r in analyses]
        if (len(expected) != job['quota'] or set(done) != expected
                or len(keys) != len(expected) or set(keys) != expected
                or any(r.get('success') != done[k].get('success') for r, k in zip(analyses, keys))
                or row.get('success') != sum(bool(r.get('success')) for r in done.values())):
            raise IO.LaunchError('evaluation artifacts do not match the quota: ' + job['name'])
        summaries[job['name']] = row
    return summaries


def protect_evaluation_outputs(S, jobs):
    """A resumed rollout must have the same weights, input pool and decode command."""
    for job in jobs:
        cmd = job['argv']
        model = Path(cmd[cmd.index('--model')+1])
        pool = Path(cmd[cmd.index('--pool')+1])
        tag = cmd[cmd.index('--tag')+1]
        folder = job['summary'].parent
        identity_path = folder / 'evaluation_identity.json'
        # Model/pool locations may change when transferring a run, hashes may not.
        mode = cmd[cmd.index('--mode')+1]
        identity = dict(weight_sha256=IO.sha256_file(model / 'model.safetensors'),
            template_sha256=IO.sha256_file(model / 'chat_template.jinja'), pool_sha256=IO.sha256_file(pool),
            mode=mode, temperature=cmd[cmd.index('--temperature')+1], top_p='0.95', top_k='0',
            tool_order=cmd[cmd.index('--tool-order')+1],
            backend='vllm',  # 终点评测固定用 vLLM 采样；换后端必须视为新的评测身份
            episode_budget=dict(C.EPISODE_BUDGET), pipeline_version=PIPELINE_VERSION)
        old = IO.load_json(identity_path)
        if old is not None and old != identity:
            raise IO.LaunchError('evaluation identity changed: ' + job['name'])
        if old is None and (folder / (tag + '_rollouts.jsonl')).exists():
            raise IO.LaunchError('cannot resume trajectories without checkpoint identity: ' + job['name'])
        if old is None:
            IO.dump_json(identity_path, identity)


def stage_evals(S):
    models = {tag: S.need_pin(tag) for tag in MODELS}
    IO.write_comp_suite_protocol(S, models)
    jobs = evaluation_jobs(S)
    if not S.dry:
        protect_evaluation_outputs(S, jobs)
    S.parallel(jobs, max_parallel=len(S.gpus))
    summaries = {} if S.dry else verify_evaluations(jobs)
    return dict(jobs=len(jobs), summaries=summaries)


def stage_analysis(S):
    jobs = evaluation_jobs(S)
    if not S.dry:
        protect_evaluation_outputs(S, jobs)
    summaries = {} if S.dry else verify_evaluations(jobs)
    S.launch('composition_analysis', S.py('minitoolrl.eval.analyze_comp_suite',
        '--root', S.run / 'eval/comp_suite', '--models', ','.join(MODELS),
        '--comparisons', ','.join(COMPARISONS)), 1200, 'analysis.log')
    S.launch('engineering_report', S.py('minitoolrl.eval.engineering_report', '--run-dir', S.run,
                                      '--require-complete'), 120, 'analysis.log')
    if not S.dry:
        from minitoolrl.eval.current_report import write_report
        write_report(S.run, MODELS, summaries)
    return dict(evaluations=len(summaries))


def stage_status(S):
    return dict(pins=S.pins, pipeline_version=PIPELINE_VERSION,
                stages={stage: (S.run / (stage + '_status.json')).exists()
                        for stage in STAGES if stage not in ('plan', 'status', 'pack')})


def verify_completed_stage(S, stage):
    """跳过成功阶段前只读核对实际产物，缺失或改变不能由旧标记代替。"""
    if stage == 'fetch_base':
        S.need_pin('base_qwen')
    elif stage == 'preflight':
        require_preflight(S)
        if any(IO.load_json(S.run / ('preflight/ds_' + mode + '.json'), {}).get('ok') is not True
               for mode in ('smoke', 'memory')):
            raise IO.LaunchError('completed preflight evidence missing or failed')
    elif stage == 'sft':
        S.need_pin('SFT')
        complete_sft(S.run / 'sft', make_plan(S.args, S.run)['sft']['updates'], S.need_pin('base_qwen')['sha256'])
        if not IO.load_json(S.run / 'sft/dev_eval/SFT_dev_run.json'):
            raise IO.LaunchError('completed SFT diagnostics missing')
    elif stage == 'grpo':
        pin = S.need_pin('G0')
        cost = IO.load_json(S.run / 'grpo/main/grpo_cost.json', {})
        final = IO.load_json(S.run / 'grpo/main/checkpoint_final.json', {})
        if (cost.get('iterations') != make_plan(S.args, S.run)['grpo']['iterations']
                or final.get('weight_sha256') != pin['sha256']):
            raise IO.LaunchError('completed GRPO evidence missing or changed')
    elif stage == 'arms':
        g0 = S.need_pin('G0')
        for arm in IO.ARMS:
            S.need_pin(arm)
            complete_sft(S.run / 'arms' / arm / ('n%d' % S.world) / 'train', 144, g0['sha256'])
    elif stage in ('engineering', 'analysis'):
        from minitoolrl.eval.engineering_report import collect
        report = collect(S.run)
        if not report['complete']:
            raise IO.LaunchError('completed engineering evidence incomplete: ' + '; '.join(report['issues']))
    if stage in ('evals', 'analysis'):
        verify_evaluations(evaluation_jobs(S))
    if stage == 'analysis':
        if (IO.load_json(S.run / 'eval/comp_suite/final_report.json', {}).get('status') != 'complete'
                or not IO.load_json(S.run / 'eval/eval_summary.json')
                or not (S.run / 'eval/report.md').exists()):
            raise IO.LaunchError('completed analysis artifacts missing')


HANDLERS = dict(pack=IO.stage_pack, init=stage_init, fetch_base=IO.stage_fetch_base,
                preflight=stage_preflight, sft=stage_sft, grpo=stage_grpo, arms=stage_arms,
                engineering=stage_engineering, evals=stage_evals, analysis=stage_analysis,
                package=IO.stage_package, status=stage_status)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', required=True, choices=STAGES)
    ap.add_argument('--run-id', default=None)
    ap.add_argument('--run-dir', default=None)
    ap.add_argument('--inputs-root', default=str(DEFAULT_ROOT))
    ap.add_argument('--inputs-tar')
    ap.add_argument('--pack-out')
    ap.add_argument('--base-dir')
    ap.add_argument('--hf-endpoint', default='https://hf-mirror.com')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--repo')
    ap.add_argument('--dry-run', action='store_true')
    for key, value in DEFAULTS.items():
        ap.add_argument('--' + key.replace('_', '-'), type=type(value), default=None)
    args = ap.parse_args(argv)
    args.run_dir = args.run_dir or str(RUNS_ROOT / (args.run_id or RUN_ID))
    saved = IO.load_json(Path(args.run_dir) / 'protocol.json')
    args.run_id = args.run_id or (saved or {}).get('run_id') or RUN_ID
    if saved and args.stage not in ('pack', 'status'):
        if saved.get('pipeline_version') != PIPELINE_VERSION or saved.get('run_id') != args.run_id:
            ap.error('this directory belongs to another pipeline or run; use a new run directory')
        for key in CONFIG_KEYS:
            supplied = getattr(args, key)
            if key not in saved['config']:
                ap.error('saved configuration lacks ' + key + '; use a new run directory')
            frozen = saved['config'][key]
            if supplied is not None and supplied != frozen:
                ap.error('configuration fixed by init: ' + key)
            setattr(args, key, frozen)
    for key, value in DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.world_size not in (1, 2):
        ap.error('this paired experiment supports one or two training ranks')
    for key in CONFIG_KEYS:
        value = getattr(args, key)
        if key not in ('gpus', 'eval_tool_order') and value <= 0:
            ap.error(key + ' must be positive')
    if args.eval_tool_order not in ('registry', 'alphabetical'):
        ap.error('--eval-tool-order must be registry or alphabetical')
    if args.probe_updates <= 2 or args.probe_iterations <= 2:
        ap.error('probes need more than two warmup measurements')
    ids = [x.strip() for x in args.gpus.split(',')] if args.gpus else list(map(str, range(args.world_size)))
    if len(ids) != args.world_size or len(set(ids)) != len(ids) or any(not x.strip() for x in ids):
        ap.error('--gpus must list exactly world-size distinct devices')
    args.input_version = VERSION
    args.timeout = 0
    return args


def main(argv=None):
    args = parse_args(argv)
    S = Session(args)
    if args.stage == 'plan':
        verify_bundle(Path(args.inputs_root))
        print(json.dumps(make_plan(args, Path(args.inputs_root)), ensure_ascii=False, indent=2))
        return
    if args.stage not in ('pack', 'init', 'status'):
        if not (S.run / 'protocol.json').exists():
            raise IO.LaunchError('run init first')
        verify_bundle(S.run)
        protocol = IO.load_json(S.run / 'protocol.json')
        if IO.sha256_file(S.run / MANIFEST) != protocol.get('inputs_manifest_sha256'):
            raise IO.LaunchError('input manifest changed after init')
    if args.stage == 'status':
        print(json.dumps(stage_status(S), ensure_ascii=False, indent=2))
        return
    marker = S.run / (args.stage + '_status.json')
    if marker.exists() and args.stage not in ('pack', 'init') and not S.dry:
        verify_completed_stage(S, args.stage)
        print(json.dumps(dict(stage=args.stage, already_complete=True), ensure_ascii=False))
        return
    if S.dry and args.stage in ('init', 'pack', 'fetch_base', 'package'):
        raise IO.LaunchError('use --stage plan for preparation; --dry-run is for GPU and analysis stages')
    started = time.monotonic()
    try:
        result = HANDLERS[args.stage](S)
    except Exception as exc:
        if not S.dry and args.stage != 'pack':
            IO.dump_json(S.run / (args.stage + '_failure.json'), dict(error=str(exc), stage=args.stage))
        raise
    if args.stage != 'pack':
        S.record_stage(args.stage, dict(result=result, seconds=round(time.monotonic() - started, 1)))


if __name__ == '__main__':
    main()
