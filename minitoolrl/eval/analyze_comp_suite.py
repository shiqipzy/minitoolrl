"""Analyze the frozen three-family composition suite; CPU only and deterministic."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from minitoolrl.data.composition_suite import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    MODEL_TAGS,
    TARGET_FAMILIES,
    VARIANTS,
    sha256_file,
)


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot take percentile of an empty sequence")
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    fraction = index - lo
    return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction


def paired_bootstrap(
    left: Mapping[Tuple[str, str], float],
    right: Mapping[Tuple[str, str], float],
    worlds_by_family: Mapping[str, Sequence[str]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    families = list(TARGET_FAMILIES)
    keys = {(family, world) for family in families for world in worlds_by_family[family]}
    if set(left) != keys or set(right) != keys:
        raise RuntimeError("Bootstrap inputs do not match the frozen family/world keys")

    point_by_family = {
        family: sum(
            left[(family, world)] - right[(family, world)]
            for world in worlds_by_family[family]
        )
        / len(worlds_by_family[family])
        for family in families
    }
    point = sum(point_by_family.values()) / len(families)
    rng = random.Random(seed)
    samples: List[float] = []
    for _ in range(draws):
        family_deltas = []
        for family in families:
            worlds = worlds_by_family[family]
            delta = 0.0
            for _ in worlds:
                world = worlds[rng.randrange(len(worlds))]
                delta += left[(family, world)] - right[(family, world)]
            family_deltas.append(delta / len(worlds))
        samples.append(sum(family_deltas) / len(families))
    return {
        "point": point,
        "ci_low": percentile(samples, 0.025),
        "ci_high": percentile(samples, 0.975),
        "draws": draws,
        "seed": seed,
        "point_by_family": point_by_family,
    }


def validate_and_index(
    analyses: Sequence[Mapping[str, Any]],
    expected: Mapping[str, Mapping[str, Any]],
    model: str,
) -> Dict[Tuple[str, str, str], Mapping[str, Any]]:
    by_task: Dict[str, Mapping[str, Any]] = {}
    for row in analyses:
        task_id = row.get("task_id")
        if task_id in by_task:
            raise RuntimeError(f"{model}: duplicate analysis row {task_id}")
        by_task[task_id] = row
    missing = sorted(set(expected) - set(by_task))
    extra = sorted(set(by_task) - set(expected))
    if missing or extra:
        raise RuntimeError(
            f"{model}: task mismatch missing={missing[:3]} extra={extra[:3]} "
            f"counts={len(missing)}/{len(extra)}"
        )

    indexed: Dict[Tuple[str, str, str], Mapping[str, Any]] = {}
    for task_id, meta in expected.items():
        row = by_task[task_id]
        family = meta["family"]
        world_id = meta["world_id"]
        variant = meta["variant"]
        for key, want in (("family", family), ("world_id", world_id), ("variant", variant)):
            if row.get(key) != want:
                raise RuntimeError(
                    f"{model}: {task_id} {key}={row.get(key)!r}, expected {want!r}"
                )
        indexed[(family, world_id, variant)] = row
    return indexed


def summarize_model(
    indexed: Mapping[Tuple[str, str, str], Mapping[str, Any]],
    worlds_by_family: Mapping[str, Sequence[str]],
) -> Dict[str, Any]:
    per_family: Dict[str, Any] = {}
    avg_scores: Dict[Tuple[str, str], float] = {}
    both_scores: Dict[Tuple[str, str], float] = {}
    failure_tags: Dict[str, Any] = {}

    for family in TARGET_FAMILIES:
        worlds = worlds_by_family[family]
        by_variant = {}
        for variant in VARIANTS:
            rows = [indexed[(family, world, variant)] for world in worlds]
            by_variant[variant] = {
                "success": sum(bool(row.get("success")) for row in rows),
                "n": len(rows),
            }
        both = 0
        either = 0
        for world in worlds:
            a = float(bool(indexed[(family, world, "A")].get("success")))
            b = float(bool(indexed[(family, world, "B")].get("success")))
            avg_scores[(family, world)] = (a + b) / 2.0
            both_scores[(family, world)] = min(a, b)
            both += int(a == 1.0 and b == 1.0)
            either += int(a == 1.0 or b == 1.0)
        avg_rate = sum(avg_scores[(family, world)] for world in worlds) / len(worlds)
        per_family[family] = {
            "worlds": len(worlds),
            "by_variant": by_variant,
            "mean_single_prompt_success": avg_rate,
            "both_wordings_success": both,
            "both_wordings_rate": both / len(worlds),
            "either_wording_success": either,
            "either_wording_rate": either / len(worlds),
        }
        failure_tags[family] = {
            variant: dict(
                Counter(
                    tag
                    for world in worlds
                    for tag in indexed[(family, world, variant)].get("failure_tags", [])
                )
            )
            for variant in VARIANTS
        }

    return {
        "per_family": per_family,
        "macro_mean_single_prompt_success": sum(
            row["mean_single_prompt_success"] for row in per_family.values()
        )
        / len(per_family),
        "macro_both_wordings_rate": sum(
            row["both_wordings_rate"] for row in per_family.values()
        )
        / len(per_family),
        "avg_world_scores": avg_scores,
        "both_world_scores": both_scores,
        "failure_tags": failure_tags,
    }


def serializable_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in summary.items()
        if key not in ("avg_world_scores", "both_world_scores")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", default=",".join(MODEL_TAGS),
                        help="comma-separated model tags to analyse (default: SFT, G0 and the three continuation groups)")
    parser.add_argument("--comparisons", default="C_mixed-U_mixed,C_mixed-G0",
                        help="comma-separated LEFT-RIGHT pairs for the paired bootstrap")
    args = parser.parse_args()
    models = tuple(tag.strip() for tag in args.models.split(",") if tag.strip())
    pairs = []
    for item in args.comparisons.split(","):
        item = item.strip()
        if not item:
            continue
        left, right = item.split("-", 1)
        if left not in models or right not in models:
            raise RuntimeError(f"comparison {item} names a model outside --models")
        pairs.append((left, right))
    root = args.root.resolve()
    protocol = load_json(root / "protocol.json")
    if tuple(protocol.get("families", [])) != TARGET_FAMILIES:
        raise RuntimeError(f"Unexpected family freeze: {protocol.get('families')}")
    if "bulk_refund" in protocol.get("families", []):
        raise RuntimeError("bulk_refund is outside this analysis protocol")

    pool_path = root / protocol["artifacts"]["pool"]["path"]
    got_pool_sha = sha256_file(pool_path)
    if got_pool_sha != protocol["artifacts"]["pool"]["sha256"]:
        raise RuntimeError("Frozen pool SHA256 mismatch")
    pool = load_jsonl(pool_path)
    expected = {row["task_id"]: row for row in pool}
    if len(expected) != protocol["n_records"]:
        raise RuntimeError(f"Pool count mismatch: {len(expected)}")

    worlds_by_family: Dict[str, List[str]] = {}
    for family in TARGET_FAMILIES:
        worlds = sorted({row["world_id"] for row in pool if row["family"] == family})
        if len(worlds) != protocol["n_worlds_per_family"]:
            raise RuntimeError(f"{family}: expected 60 worlds, got {len(worlds)}")
        worlds_by_family[family] = worlds

    summaries: Dict[str, Dict[str, Any]] = {}
    run_checks: Dict[str, Any] = {}
    for model in models:
        run_dir = root / "run" / model / "greedy"
        analysis_path = run_dir / f"{model}_analysis.json"
        summary_path = run_dir / f"{model}_summary.json"
        run_path = run_dir / f"{model}_run.json"
        analyses = load_json(analysis_path)
        run_summary = load_json(summary_path)
        run_meta = load_json(run_path)
        if run_summary.get("incomplete") or run_summary.get("filled") != len(pool):
            raise RuntimeError(f"{model}: incomplete run {run_summary}")
        expected_sha = protocol["models"][model]["sha256"]
        if run_meta.get("weight_sha256") != expected_sha:
            raise RuntimeError(
                f"{model}: weight SHA mismatch {run_meta.get('weight_sha256')} != {expected_sha}"
            )
        if run_meta.get("pool_sha256") != got_pool_sha:
            raise RuntimeError(f"{model}: pool SHA mismatch")
        indexed = validate_and_index(analyses, expected, model)
        summaries[model] = summarize_model(indexed, worlds_by_family)
        run_checks[model] = {
            "analysis_sha256": sha256_file(analysis_path),
            "summary_sha256": sha256_file(summary_path),
            "run_sha256": sha256_file(run_path),
            "filled": run_summary["filled"],
            "quota": run_summary["quota"],
            "incomplete": run_summary["incomplete"],
            "weight_sha256": run_meta["weight_sha256"],
        }

    comparisons = {}
    for left, right in pairs:
        label = f"{left}-{right}"
        comparisons[label] = {
            "mean_single_prompt_success": paired_bootstrap(
                summaries[left]["avg_world_scores"],
                summaries[right]["avg_world_scores"],
                worlds_by_family,
            ),
            "both_wordings_rate": paired_bootstrap(
                summaries[left]["both_world_scores"],
                summaries[right]["both_world_scores"],
                worlds_by_family,
                seed=BOOTSTRAP_SEED + 1,
            ),
        }

    report = {
        "status": "complete",
        "run_id": protocol["run_id"],
        "pool_sha256": got_pool_sha,
        "models": {model: serializable_summary(summary) for model, summary in summaries.items()},
        "comparisons": comparisons,
        "run_checks": run_checks,
        "scope": {
            "families": list(TARGET_FAMILIES),
            "worlds_per_family": protocol["n_worlds_per_family"],
            "variants": list(VARIANTS),
            "bulk_refund_included": False,
            "training_seed_uncertainty_measured": False,
        },
    }
    out_path = root / "final_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    compact = {
        model: {
            family: {
                "A": row["by_variant"]["A"]["success"],
                "B": row["by_variant"]["B"]["success"],
                "both": row["both_wordings_success"],
            }
            for family, row in summaries[model]["per_family"].items()
        }
        for model in models
    }
    print(
        json.dumps(
            {
                "status": "complete",
                "results": compact,
                "comparisons": comparisons,
                "report": str(out_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
