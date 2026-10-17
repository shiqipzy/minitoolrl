"""供评测入口共用的 vLLM 采样器构建、模型自主执行和轨迹保存。"""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from minitoolrl.env.session import EpisodeBudget
from minitoolrl.env import policy as P, tools as T
from minitoolrl.train import experiment_config as C
from minitoolrl.train.rollout_align import rollout_pool

BUDGET = EpisodeBudget(**C.EPISODE_BUDGET)
# 基础 SFT 未指定 exposed_tools，GRPO 采样也使用工具注册顺序。
VISIBLE_TOOLS = tuple(T.ALL_TOOL_NAMES)
EVAL_BATCH_SIZE = 8


def tools_for_order(order):
    """按训练记录选择验证顺序；三个已有续训分支显式使用名称排序。"""
    if order == 'registry':
        return tuple(T.ALL_TOOL_NAMES)
    if order == 'alphabetical':
        return tuple(sorted(T.ALL_TOOL_NAMES))
    raise ValueError('unknown tool order: ' + str(order))


def resolved_tools(visible_tools=None):
    names = tuple(VISIBLE_TOOLS if visible_tools is None else visible_tools)
    if len(names) != len(T.ALL_TOOL_NAMES) or set(names) != set(T.ALL_TOOL_NAMES):
        raise ValueError('evaluation must expose every tool exactly once')
    return names


def content_sha(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def evaluation_context(tok, budget=BUDGET, *, top_p=0.95, top_k=0, batch_size=EVAL_BATCH_SIZE,
                       visible_tools=None):
    """记录实际使用的工具顺序、说明、模板、规则和采样条件。"""
    if asdict(budget) != C.EPISODE_BUDGET:
        raise ValueError('evaluation budget differs from the shared episode budget')
    names = resolved_tools(visible_tools)
    return dict(
        visible_tools=list(names),
        tool_schema_sha256=content_sha(T.tool_schemas(names)),
        chat_template_sha256=hashlib.sha256(tok.chat_template.encode('utf-8')).hexdigest(),
        system_prompt_sha256=content_sha(P.OPERATING_RULES),
        policy_docs_sha256=content_sha(P.POLICY_DOCS),
        enable_thinking=False, episode_budget=asdict(budget),
        temperature=0.0, top_p=top_p, top_k=top_k, seed=17, batch_size=batch_size,
    )


def bind_evaluation_context(rollouts_path, tok, instances, budget=BUDGET, *,
                            top_p=0.95, top_k=0, batch_size=EVAL_BATCH_SIZE, identity=None,
                            visible_tools=None):
    """续跑前核对输入条件；缺少旧上下文记录时拒绝混入已有轨迹。"""
    rollouts_path = Path(rollouts_path)
    context = evaluation_context(tok, budget, top_p=top_p, top_k=top_k, batch_size=batch_size,
                                 visible_tools=visible_tools)
    record = dict(context=context, context_sha256=content_sha(context),
                  task_pool_sha256=content_sha([inst.to_json() for inst in instances]),
                  identity=identity)
    path = rollouts_path.with_suffix('.context.json')
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != record:
            raise ValueError('evaluation context or inputs changed; use a new output directory')
    else:
        if rollouts_path.exists():
            raise ValueError('saved rollouts have no evaluation context; use a new output directory')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return record


def greedy_rollouts(sampler, tok, instances, budget=BUDGET, *, top_p=0.95, top_k=0,
                    seed_base=17, visible_tools=None):
    """正式评测与开发诊断共用同一工具列表、顺序和 greedy 调度。"""
    if asdict(budget) != C.EPISODE_BUDGET:
        raise ValueError('evaluation budget differs from the shared episode budget')
    return rollout_pool(sampler, tok, instances, budget, temperature=0.0,
                        visible_tools=resolved_tools(visible_tools), top_p=top_p, top_k=top_k,
                        seed_base=seed_base)


def load_done(path):
    """只读已有任务轨迹；重复执行记录不能当成已完成配额。"""
    path = Path(path)
    done = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row['task_id'], row.get('sample_index'))
            if key in done:
                raise ValueError('duplicate saved rollout: ' + str(key))
            done[key] = row
    return done


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()


def make_sampler(model_path, max_model_len=4096,
                 gpu_memory_utilization=0.85):
    """评测统一使用 vLLM 进程内实例。"""
    from minitoolrl.train.vllm_backend import LocalVLLM  # 懒加载
    return LocalVLLM(model_path, max_model_len=max_model_len,
                     gpu_memory_utilization=gpu_memory_utilization)


def close_sampler(sampler) -> None:
    """关闭 vLLM 实例并释放显存。"""
    sampler.close()
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def free_rollouts(sampler, tok, insts, path, *, identity=None, visible_tools=None):
    bind_evaluation_context(path, tok, insts, identity=identity, visible_tools=visible_tools)
    done = load_done(path)
    expected = {(inst.task_id, None) for inst in insts}
    if len(expected) != len(insts) or not set(done).issubset(expected):
        raise ValueError('duplicate input tasks or saved rollouts outside the greedy quota')
    pending = [inst for inst in insts if (inst.task_id, None) not in done]
    rows = list(done.values())
    for start in range(0,len(pending),EVAL_BATCH_SIZE):
        group=greedy_rollouts(sampler, tok, pending[start:start+EVAL_BATCH_SIZE],
                             seed_base=17+start, visible_tools=visible_tools)
        batch=[e.transcript() for e in group.episodes]
        with Path(path).open('a',encoding='utf-8') as f:
            for row in batch: f.write(json.dumps(row,ensure_ascii=False)+'\n')
        rows.extend(batch)
        print(json.dumps(dict(event='rollout_progress',file=Path(path).name,done=len(rows),
            total=len(insts),success=sum(r['success'] for r in rows))),flush=True)
    return rows
