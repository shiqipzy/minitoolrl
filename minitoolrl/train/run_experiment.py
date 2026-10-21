"""唯一实验入口：一次 SFT、一次 GRPO、三组续训与成对性能测量。

阶段按 `--stage` 逐个执行：plan / pack / init / fetch_base 只准备输入与身份，
preflight 检查 GPU 环境，sft / grpo / arms 训练，engineering 测性能，
evals 完成评测与报告，package 生成回传清单。已成功的阶段只做产物核验后跳过，
不会静默重训。
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

from minitoolrl import files
from minitoolrl.data.input_bundle import (CURRENT_MANIFEST_SHA256, DEFAULT_ROOT, MANIFEST,
                                          MARKER, REQUIRED_FILES, VERSION, verify_bundle)
from minitoolrl.paths import CURRENT_RUN_ID, PACKAGE_ROOT, RUNS_ROOT
from minitoolrl.train import experiment_config as C
from minitoolrl.train import run_artifacts as artifacts
from minitoolrl.train.run_session import LaunchError, Session

PIPELINE_VERSION = 'deepspeed-vllm-engineering-qwen3-v8'
RUN_ID = CURRENT_RUN_ID
STAGES = ('plan', 'pack', 'init', 'fetch_base', 'preflight', 'sft', 'grpo',
          'arms', 'engineering', 'evals', 'package', 'status')
MODELS = ('SFT', 'G0', 'C_blocked', 'C_mixed', 'U_mixed')
COMPARISONS = ('G0-SFT', 'C_mixed-U_mixed', 'C_mixed-G0', 'C_blocked-G0',
               'C_mixed-C_blocked', 'U_mixed-G0')
TRAINER = 'minitoolrl.train.train_sft'
DEFAULTS = dict(world_size=2, gpus='', sft_epochs=2, sft_lr=2e-5, sft_seconds=3600,
                grpo_passes=2, grpo_seconds=3600, arm_seconds=2400, arm_lr=2e-6,
                eval_tool_order='registry',
                probe_updates=40, probe_iterations=20, probe_seconds=1500,
                probe_seq_len=4096, max_gb=23.)
CONFIG_KEYS = tuple(DEFAULTS)


def make_plan(args, root):
    n_sft = len(files.read_jsonl(root / 'data/sft_train_v2.jsonl'))
    n_rl = len(files.read_jsonl(root / 'data/rl_train.jsonl'))
    iterations = math.ceil(n_rl / C.RL['prompts_per_iteration']) * args.grpo_passes
    sizes = sorted({1, args.world_size})
    return dict(pipeline_version=PIPELINE_VERSION, input_version=VERSION,
        run_id=args.run_id, inputs_manifest_sha256=files.sha256_file(root / MANIFEST),
        config={k: getattr(args, k) for k in CONFIG_KEYS},
        objective='Train tool use and compare held-out composition; measure SFT/GRPO performance.',
        base=dict(repo=C.BASE_REPO, revision=C.BASE_REVISION, sha256=C.BASE_SHA256),
        chat_format=dict(source='checkpoint tokenizer', enable_thinking=False,
                         template_sha256=C.BASE_TEMPLATE_SHA256),
        sft=dict(records=n_sft, epochs=args.sft_epochs,
                 updates=math.ceil(n_sft / 16) * args.sft_epochs, lr=args.sft_lr,
                 global_batch=16, microbatch=2, grad_accum=8, seed=1,
                 backend='DeepSpeed ZeRO-2 (bf16 compute, fp32 optimizer states, no offload)',
                 selection='fixed complete-budget endpoint; data/gate_spec.json dev acceptance required',
                 optimizer='one AdamW and one 3%-warmup cosine schedule, final LR ratio 0.1'),
        grpo=dict(pool_tasks=n_rl, pool_passes=args.grpo_passes, iterations=iterations,
                  episodes=iterations * 8,
                  trainer='DeepSpeed ZeRO-2 single training GPU',
                  sampler='resident vLLM process on the sampling GPU; NCCL weight '
                          'transfer after every accepted update',
                  rl=dict(C.RL),
                  endpoint='fixed iteration budget; record actual updates and zero-signal groups'),
        arms={name: dict(spec, epochs=2, updates=144, world_size=args.world_size, lr=args.arm_lr)
              for name, spec in C.ARMS.items()},
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


def stage_init(S):
    if (S.run / 'protocol.json').exists() or S.pins:
        raise LaunchError('init requires a fresh run directory; existing protocol is preserved')
    S.run.mkdir(parents=True, exist_ok=True)
    if S.args.inputs_tar:
        with tarfile.open(S.args.inputs_tar, 'r:gz') as tar:
            root = S.run.resolve()
            for member in tar.getmembers():
                target = (root / member.name).resolve()
                if not target.is_relative_to(root) or member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                    raise LaunchError('unsafe input archive member: ' + member.name)
            tar.extractall(root)
    else:
        source = Path(S.args.inputs_root)
        verify_bundle(source, expected_manifest_sha256=CURRENT_MANIFEST_SHA256)
        for rel in [*sorted(REQUIRED_FILES), MARKER, MANIFEST, 'validation.json']:
            dst = S.run / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / rel, dst)
    verify_bundle(S.run, expected_manifest_sha256=CURRENT_MANIFEST_SHA256)
    plan = make_plan(S.args, S.run)
    sources = artifacts.source_files(PACKAGE_ROOT)
    plan.update(created_local=files.now(), torch=artifacts.installed_torch_version(),
                inputs=dict(mode='read_only_reference',
                            note='copied into the run directory once at init and verified '
                                 'against the bundle manifest; no stage rewrites them',
                            manifest_sha256=files.sha256_file(S.run / MANIFEST)),
                source_sha256={str(p.relative_to(PACKAGE_ROOT)): files.sha256_file(p) for p in sources})
    files.dump_json(S.run / 'protocol.json', plan)
    files.dump_json(S.run / 'inputs_manifest.json', files.load_json(S.run / MANIFEST))
    files.dump_json(S.run / 'pins.json', {})
    return dict(protocol=plan)


def require_preflight(S):
    if not S.dry and not files.load_json(S.run / 'preflight/gates.json', {}).get('all_ok'):
        raise LaunchError('preflight must pass before training')


def stage_preflight(S):
    base = S.need_pin('base_qwen')
    out = S.run / 'preflight'
    for mode in ('smoke', 'memory'):
        argv = S.torchrun(S.world, 'minitoolrl.preflight', '--mode', mode, '--out', out)
        if mode == 'memory':
            argv += list(map(str, ['--model', base['path'], '--seq-len', S.args.probe_seq_len,
                                  '--batch', 2, '--accum', max(8, S.world), '--steps', 3,
                                  '--max-gb', S.args.max_gb]))
        S.launch('preflight_' + mode, argv, 900, 'preflight.log', S.gpu_env(S.gpus[:S.world]))
    checks = {mode: files.load_json(out / ('ds_' + mode + '.json'), {}).get('ok') for mode in ('smoke', 'memory')}
    if not S.dry:
        files.dump_json(out / 'gates.json', dict(checks=checks, all_ok=all(checks.values())))
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


def arm_command(S, arm, n, out, seconds):
    """续训命令：从同一 G0 出发，按该组的数据、更新日程和累积步数训练。"""
    g0 = S.need_pin('G0')
    spec = C.ARMS[arm]
    argv = ['--model', g0['path'], '--output', out, '--train-data', S.path(spec['train']),
            '--val-data', S.path('inputs/c_recipe/val_common.jsonl'), '--expected-sha', g0['sha256'],
            '--batch-size', 2, '--grad-accum', spec['accum'], '--epochs', 2, '--lr', S.args.arm_lr,
            '--weight-decay', 0.01, '--grad-clip', 1.0, '--seed', 17, '--max-length', 4096,
            '--max-seconds', seconds]
    if spec['plan']:
        argv += ['--update-plan', S.path(spec['plan'])]
    return S.torchrun(n, TRAINER, *argv)


def complete_sft(out, expected, parent_sha=None):
    summary = files.load_json(out / 'summary.json', {})
    if (summary.get('steps') != expected or summary.get('total_steps_planned') != expected
            or summary.get('optimizer_updates') != expected or summary.get('complete_budget') is not True
            or summary.get('probe') is not False):
        raise LaunchError('SFT budget incomplete; do not use partial endpoint: ' + str(out))
    if parent_sha and files.load_json(out / 'config.json', {}).get('initial_model_sha256') != parent_sha:
        raise LaunchError('saved SFT belongs to a different parent: ' + str(out))
    return summary


def complete_sft_dev(S, checkpoint):
    """冻结开发集门槛验收 SFT 终点；不达标不能登记 pin。"""
    out = S.run / 'sft/dev_eval'
    dev = files.load_json(out / 'SFT_dev_run.json', {})
    metrics = files.load_json(out / 'SFT_dev_metrics.json', {})
    pool = S.path('data/dev.jsonl')
    if (dev.get('n_tasks') != len(files.read_jsonl(pool))
            or dev.get('pool_sha256') != files.sha256_file(pool)
            or dev.get('weight_sha256') != files.sha256_file(checkpoint / 'model.safetensors')):
        raise LaunchError('SFT dev task-success evaluation missing, incomplete or from another checkpoint/pool')
    # 整包 manifest 固定当前开发池；规格附属的旧池哈希不参与验收。
    spec = files.load_json(S.path('data/gate_spec.json'))
    thresholds = spec.get('thresholds', {})
    gates = (
        ('ordinary_min', '>=', ('ordinary',), 'denominator_ordinary'),
        ('policy_query_min', '>=', ('compensation_flow', 'policy_query'), 'denominator_compensation'),
        ('apply_credit_min', '>=', ('compensation_flow', 'apply_credit_called'), 'denominator_compensation'),
        ('repeat_loop_max', '<=', ('compensation_flow', 'repeat_query_loop'), 'denominator_compensation'),
    )

    def count(value, label):
        if type(value) is not int or value < 0:
            raise LaunchError('SFT dev gate requires a non-negative integer at ' + label)
        return value

    failures = []
    for threshold_name, operator, path, denominator_name in gates:
        metric = metrics
        for key in path:
            metric = metric.get(key, {}) if isinstance(metric, dict) else {}
        observed = count(metric.get('correct'), '.'.join(path) + '.correct')
        denominator = count(metric.get('n'), '.'.join(path) + '.n')
        required_denominator = count(thresholds.get(denominator_name), 'thresholds.' + denominator_name)
        if denominator != required_denominator:
            raise LaunchError('SFT dev gate %s denominator %d != frozen %d' %
                              (denominator_name, denominator, required_denominator))
        threshold = count(thresholds.get(threshold_name), 'thresholds.' + threshold_name)
        passed = observed >= threshold if operator == '>=' else observed <= threshold
        if not passed:
            failures.append('%s: %d/%d does not satisfy %s %d' %
                            (threshold_name, observed, denominator, operator, threshold))
    if failures:
        raise LaunchError('SFT dev gate failed; SFT pin and GRPO are blocked: ' + '; '.join(failures))
    return dev


def fresh_output(S, out):
    if not S.dry and out.exists() and any(out.iterdir()):
        raise LaunchError('unfinished output already exists; inspect it and use a new run directory: ' + str(out))


def stage_sft(S):
    require_preflight(S)
    base = S.need_pin('base_qwen')
    out = S.run / 'sft'
    expected = S.plan['sft']['updates']
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
        complete_sft_dev(S, checkpoint)
    S.pin('SFT', checkpoint, 'sft', world_size=S.world,
          selection='fixed_budget_dev_gate_passed', updates=expected)
    return summary


def grpo_command(S, out, iterations, probe=False):
    """训练进程占第一张卡，采样进程使用第二张物理卡。"""
    if len(S.gpus) < 2:
        raise LaunchError('GRPO needs separate training and sampling GPUs')
    start = S.need_pin('SFT')
    if not S.dry:
        complete_sft_dev(S, Path(start['path']))
    argv = S.torchrun(1, 'minitoolrl.train.train_grpo', '--model', start['path'],
        '--expected-sha', start['sha256'],
        '--run-dir', S.run, '--train-pool', S.path('data/rl_train.jsonl'),
        '--max-iterations', iterations,
        '--seconds', S.args.probe_seconds if probe else S.args.grpo_seconds,
        '--out', out, '--run-id', S.run_id, '--sampler-gpu', S.gpus[1])
    return argv + ['--probe', '--save-reserve', '0'] if probe else argv


def complete_grpo(S, expected):
    """只验收训练器已导出的模型，不重新制定计划或加载权重。"""
    out = S.run / 'grpo' / 'main'
    final = files.load_json(out / 'checkpoint_final.json', {})
    path = Path(final.get('path', str(out / 'models/final')))
    if not S.dry:
        cost = files.load_json(out / 'grpo_cost.json', {})
        if (cost.get('iterations') != expected or not final
                or not (path / 'model.safetensors').is_file()):
            raise LaunchError('GRPO iteration budget or endpoint is incomplete')
    return path


def stage_grpo(S):
    require_preflight(S)
    expected = S.plan['grpo']['iterations']
    out = S.run / 'grpo' / 'main'
    fresh_output(S, out)
    S.launch('grpo', grpo_command(S, out, expected), S.args.grpo_seconds + 1200,
             'grpo.log', S.gpu_env(S.gpus))
    checkpoint = complete_grpo(S, expected)
    S.pin('G0', checkpoint, 'grpo', world_size=1, iterations=expected, sampler='vllm')
    return dict(iterations=expected)


def stage_arms(S):
    require_preflight(S)
    g0 = S.need_pin('G0')
    results = {}
    for arm in C.ARMS:
        expected = S.plan['arms'][arm]['updates']
        out = S.run / 'arms' / arm / ('n%d' % S.world) / 'train'
        if not S.dry and (out / 'summary.json').exists():
            results[arm] = complete_sft(out, expected, g0['sha256'])
        else:
            fresh_output(S, out)
            argv = arm_command(S, arm, S.world, out, S.args.arm_seconds)
            S.launch('arm_' + arm, argv, S.args.arm_seconds + 900, 'arms.log',
                     S.gpu_env(S.gpus[:S.world]))
            results[arm] = {} if S.dry else complete_sft(out, expected, g0['sha256'])
        S.pin(arm, out / 'endpoint', 'arms', world_size=S.world, updates=expected)
    return results


def stage_engineering(S):
    require_preflight(S)
    if not S.dry and S.world != 2:
        raise LaunchError('paired engineering requires two physical GPUs')
    sizes = sorted({1, S.world})
    # SFT 成对性能测量：同一 DeepSpeed 后端单 rank vs 双 rank。
    for n in sizes:
        out = S.run / 'engineering' / 'sft' / ('n%d' % n)
        if not S.dry:
            summary = files.load_json(out / 'summary.json', {})
            if (summary.get('steps') == S.args.probe_updates
                    and summary.get('optimizer_updates') == S.args.probe_updates
                    and summary.get('probe') is True
                    and summary.get('stop_reason') == 'max_updates'):
                continue
        fresh_output(S, out)
        S.launch('engineering_sft_n%d' % n, sft_command(S, n, out, True),
                 S.args.probe_seconds + 900, 'engineering.log', S.gpu_env(S.gpus[:n]))
    # GRPO 训推分离性能测量：使用主链相同的 DeepSpeed + vLLM 路径。
    out = S.run / 'engineering' / 'grpo' / 'vllm_separated'
    cost = files.load_json(out / 'grpo_cost.json', {}) if not S.dry else {}
    if cost.get('iterations') != S.args.probe_iterations:
        fresh_output(S, out)
        S.launch('engineering_grpo_vllm_separated',
                 grpo_command(S, out, S.args.probe_iterations, probe=True),
                 S.args.probe_seconds + 900, 'engineering.log', S.gpu_env(S.gpus))
    S.launch('engineering_report', S.py('minitoolrl.eval.engineering_report', '--run-dir', S.run,
                                      '--require-complete'), 120, 'engineering.log')
    return dict(world_sizes=sizes, grpo_deployments=['vllm_separated'])


def evaluation_jobs(S):
    models = {tag: S.need_pin(tag) for tag in MODELS}
    jobs = []
    for suite, rel in [('comp_suite', 'inputs/comp_suite/test_comp_suite.jsonl'),
                       ('c_recipe', 'inputs/c_recipe/eval_variants.jsonl'),
                       ('long', 'inputs/long/test_long.jsonl')]:
        for tag, pin in models.items():
            pool, out = S.path(rel), S.run / 'eval' / suite / 'run' / tag / 'greedy'
            argv = S.py('minitoolrl.eval.evaluate', '--model', pin['path'], '--pool', pool,
                        '--out-dir', out, '--tag', tag, '--mode', 'greedy', '--batch-size', 8,
                        '--max-seconds', 1800, '--top-p', 0.95, '--top-k', 0,
                        '--temperature', 0, '--tool-order', S.args.eval_tool_order)
            jobs.append(dict(name=suite + '_' + tag, log='evals.log', timeout=2400, argv=argv,
                             pool=pool, out=out, tag=tag))
    for index, job in enumerate(jobs):
        job['gpu'] = S.gpus[index % len(S.gpus)]
        S.need_pin(job['tag'])
    return jobs


def write_comp_suite_protocol(S, models):
    """固定本次组合评测的题目、模型身份与解码条件；已有协议不重写。"""
    from minitoolrl.data.composition_suite import TARGET_FAMILIES
    root = S.run / 'eval' / 'comp_suite'
    root.mkdir(parents=True, exist_ok=True)
    pool_src = S.path('inputs/comp_suite/test_comp_suite.jsonl')
    pool = root / 'test_comp_suite.jsonl'
    if not pool.exists():
        shutil.copy2(pool_src, pool)
    template = files.load_json(S.path('inputs/comp_suite/protocol_template.json'), {})
    protocol = dict(
        run_id=S.run_id, input_version=S.args.input_version, status='prepared',
        parent_protocol=template.get('run_id'),
        objective='SFT baseline, GRPO baseline and three continuation arms comparing decomposition and basic tasks.',
        families=list(TARGET_FAMILIES), explicitly_excluded=['bulk_refund'],
        n_families=len(TARGET_FAMILIES),
        n_worlds_per_family=template.get('n_worlds_per_family', 60), variants_per_world=['A', 'B'],
        n_worlds=template.get('n_worlds', 180), n_records=template.get('n_records', 360),
        models={tag: dict(recorded_path=pin['path'], sha256=pin['sha256'], source_run=S.run_id)
                for tag, pin in models.items()},
        episode_budget=template.get('episode_budget', dict(C.EPISODE_BUDGET)),
        decode=template.get('decode', dict(mode='greedy', temperature=0, top_p=0.95, top_k=0,
                                           batch_size=8)),
        policy_version=template.get('policy_version'), metrics=template.get('metrics'),
        limitations=template.get('limitations'),
        artifacts=dict(pool=dict(path=pool.name, sha256=files.sha256_file(pool))),
    )
    files.dump_json(root / 'protocol.json', protocol)
    return root


def stage_evals(S):
    from minitoolrl.eval.evaluate import verify_evaluations, write_reports

    models = {tag: S.need_pin(tag) for tag in MODELS}
    write_comp_suite_protocol(S, models)
    jobs = evaluation_jobs(S)
    S.parallel(jobs, max_parallel=len(S.gpus))
    summaries = {} if S.dry else verify_evaluations(jobs)
    S.launch('engineering_report', S.py('minitoolrl.eval.engineering_report', '--run-dir', S.run,
                                      '--require-complete'), 120, 'evals.log')
    if not S.dry:
        write_reports(S.run, MODELS, summaries, COMPARISONS)
    return dict(jobs=len(jobs), summaries=summaries)


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
        if any(files.load_json(S.run / ('preflight/ds_' + mode + '.json'), {}).get('ok') is not True
               for mode in ('smoke', 'memory')):
            raise LaunchError('completed preflight evidence missing or failed')
    elif stage == 'sft':
        S.need_pin('SFT')
        complete_sft(S.run / 'sft', S.plan['sft']['updates'], S.need_pin('base_qwen')['sha256'])
        complete_sft_dev(S, S.run / 'sft/endpoint')
    elif stage == 'grpo':
        S.need_pin('G0')
        complete_grpo(S, S.plan['grpo']['iterations'])
    elif stage == 'arms':
        g0 = S.need_pin('G0')
        for arm in C.ARMS:
            S.need_pin(arm)
            complete_sft(S.run / 'arms' / arm / ('n%d' % S.world) / 'train',
                         S.plan['arms'][arm]['updates'], g0['sha256'])
    elif stage in ('engineering', 'evals'):
        from minitoolrl.eval.engineering_report import collect
        report = collect(S.run)
        if not report['complete']:
            raise LaunchError('completed engineering evidence incomplete: ' + '; '.join(report['issues']))
    if stage == 'evals':
        from minitoolrl.eval.evaluate import verify_evaluations
        verify_evaluations(evaluation_jobs(S))
        if (files.load_json(S.run / 'eval/comp_suite/final_report.json', {}).get('status') != 'complete'
                or not files.load_json(S.run / 'eval/eval_summary.json')
                or not files.load_json(S.run / 'eval/train_curves.json')
                or not (S.run / 'eval/report.md').exists()):
            raise LaunchError('completed evaluation reports missing')


HANDLERS = dict(pack=artifacts.stage_pack, init=stage_init, fetch_base=artifacts.stage_fetch_base,
                preflight=stage_preflight, sft=stage_sft, grpo=stage_grpo, arms=stage_arms,
                engineering=stage_engineering, evals=stage_evals,
                package=artifacts.stage_package, status=stage_status)


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
    saved = files.load_json(Path(args.run_dir) / 'protocol.json')
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
    if args.eval_tool_order != 'registry':
        ap.error('--eval-tool-order must be registry for the current input bundle')
    if args.probe_updates <= 2 or args.probe_iterations <= 2:
        ap.error('probes need more than two warmup measurements')
    ids = [x.strip() for x in args.gpus.split(',')] if args.gpus else list(map(str, range(args.world_size)))
    if len(ids) != args.world_size or len(set(ids)) != len(ids) or any(not x.strip() for x in ids):
        ap.error('--gpus must list exactly world-size distinct devices')
    args.input_version = VERSION
    args.pipeline_version = PIPELINE_VERSION
    args.timeout = 0
    return args


def main(argv=None):
    args = parse_args(argv)
    S = Session(args)
    if args.stage == 'plan':
        verify_bundle(Path(args.inputs_root), expected_manifest_sha256=CURRENT_MANIFEST_SHA256)
        print(json.dumps(make_plan(args, Path(args.inputs_root)), ensure_ascii=False, indent=2))
        return
    if args.stage not in ('pack', 'init', 'status'):
        if not (S.run / 'protocol.json').exists():
            raise LaunchError('run init first')
        verify_bundle(S.run, expected_manifest_sha256=CURRENT_MANIFEST_SHA256)
    if args.stage == 'status':
        print(json.dumps(stage_status(S), ensure_ascii=False, indent=2))
        return
    marker = S.run / (args.stage + '_status.json')
    if marker.exists() and args.stage not in ('pack', 'init') and not S.dry:
        verify_completed_stage(S, args.stage)
        print(json.dumps(dict(stage=args.stage, already_complete=True), ensure_ascii=False))
        return
    if S.dry and args.stage in ('init', 'pack', 'fetch_base', 'package'):
        raise LaunchError('use --stage plan for preparation; --dry-run is for GPU stages')
    started = time.monotonic()
    try:
        result = HANDLERS[args.stage](S)
    except Exception as exc:
        if not S.dry and args.stage != 'pack':
            files.dump_json(S.run / (args.stage + '_failure.json'), dict(error=str(exc), stage=args.stage))
        raise
    if args.stage != 'pack':
        S.record_stage(args.stage, dict(result=result, seconds=round(time.monotonic() - started, 1)))


if __name__ == '__main__':
    main()
