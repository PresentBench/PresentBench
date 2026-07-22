#!/usr/bin/env python3
"""
Aggregate *repeated* judge evaluations and report mean / std / range.

Motivation
----------
The judge model is stochastic, so a single evaluation pass only gives a point
estimate of each score. To quantify how stable those scores are, ``judge_all.py
--repeats N`` re-evaluates every test case ``N`` times, writing each repeat into
its own directory (namespaced per judge model so different judges never
collide)::

    results/<agent>/<...>/generation_task/results/repeats/<judge_model>/rep1/{judge_model}_{ts}_score.yaml
    results/<agent>/<...>/generation_task/results/repeats/<judge_model>/rep2/...
    ...

This module reads those per-repeat score files and reports, for each metric
(overall score and the two section aggregates):

* **per-case** statistics across the ``N`` repeats
  (mean, sample std, min, max, range);
* **dataset-level** statistics:
    - ``reported_mean``          : mean over cases of the per-case mean
                                   (the number you would normally report);
    - ``within_case_std_mean``   : average per-case std (how much a *single*
                                   case wobbles between repeats);
    - ``within_case_range_max``  : worst-case per-case range;
    - ``per_repeat_dataset_mean``: for each repeat index, the dataset mean over
                                   cases; and the mean / std / min / max / range
                                   of those per-repeat dataset means
                                   (``across_run_*``). This last group is the
                                   run-to-run variability of the final reported
                                   benchmark number -- the most useful
                                   "fluctuation interval".

Layouts supported
-----------------
1. Repeats (preferred): ``results/.../repeats/<judge_model>/repK/``.
2. Legacy single-pass (pre-repeats) format: one or more timestamped
   ``{judge_model}_{ts}_score.yaml`` files written directly under a case's
   ``results`` dir. Used automatically when no per-model ``repeats/`` dir is
   present, so results produced before the repeats feature are still readable
   (each timestamp is treated as one repeat, oldest -> newest = 1..K; a single
   legacy result yields n=1).

Usage
-----
    python -m utils.statistics.aggregate_repeats \
        --result_root results/NotebookLM \
        --judge_model gemini-3-flash-preview

    # dump the full (per-case) report to a custom path
    python -m utils.statistics.aggregate_repeats \
        --result_root results/NotebookLM --output my_report.yaml
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# Allow ``python utils/statistics/aggregate_repeats.py`` (script style) in
# addition to ``python -m utils.statistics.aggregate_repeats``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.score_utils import (  # noqa: E402
    find_single_score_yaml,
    find_all_score_yamls,
    extract_yaml_metric,
)

# Metrics reported by default: overall score + the two section aggregates.
DEFAULT_METRICS: Dict[str, Tuple[str, ...]] = {
    "overall": ("total", "weighted_arithmetic_mean_percent"),
    "material_independent": (
        "total", "material_independent", "weighted_arithmetic_mean_percent",
    ),
    "material_dependent": (
        "total", "material_dependent", "weighted_arithmetic_mean_percent",
    ),
}

_REP_DIR_PATTERN = re.compile(r"^rep(?P<idx>\d+)$")


# ---------------------------------------------------------------------------
# Small stats helpers
# ---------------------------------------------------------------------------

def _std(values: List[float]) -> float:
    """Sample standard deviation (ddof=1); 0.0 for fewer than 2 values."""
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def _summ(values: List[float]) -> Dict[str, float]:
    """mean / std / min / max / range / n for a list of numbers."""
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None,
                "range": None, "n": 0}
    lo, hi = min(values), max(values)
    return {
        "mean": statistics.mean(values),
        "std": _std(values),
        "min": lo,
        "max": hi,
        "range": hi - lo,
        "n": len(values),
    }


# ---------------------------------------------------------------------------
# Repeat discovery
# ---------------------------------------------------------------------------

def _collect_repeat_score_files(
    results_dir: Path,
    judge_model: str,
) -> Dict[int, Path]:
    """Return ``{repeat_index: newest score yaml}`` for one case's results dir.

    Preferred layout: ``results_dir/repeats/<judge_model>/rep{k}/`` (repeats are
    namespaced per judge model, so different models never collide).

    Backward-compatible layout: when no per-model ``repeats/`` dir exists, fall
    back to the pre-repeats single-pass format -- one or more timestamped
    ``{judge_model}_{ts}_score.yaml`` files directly in ``results_dir`` (each
    timestamp treated as one repeat, ordered oldest -> newest = 1..K).
    """
    found: Dict[int, Path] = {}

    # Preferred layout: per-model namespaced repeats -> repeats/<judge_model>/repK.
    model_repeats_dir = results_dir / "repeats" / judge_model
    if model_repeats_dir.is_dir():
        for rep_dir in model_repeats_dir.iterdir():
            if not rep_dir.is_dir():
                continue
            m = _REP_DIR_PATTERN.match(rep_dir.name)
            if not m:
                continue
            score = find_single_score_yaml(rep_dir, judge_model=judge_model)
            if score is not None:
                found[int(m.group("idx"))] = score
        if found:
            return found

    # Backward compatibility with the pre-repeats (single-pass) format: one or
    # more timestamped ``{judge_model}_{ts}_score.yaml`` files written directly
    # in results_dir. Each timestamp is treated as one repeat (sorted oldest ->
    # newest = 1..K); a single legacy result therefore yields n=1 (std/range=0).
    score_files = find_all_score_yamls(results_dir, judge_model=judge_model)
    for i, p in enumerate(score_files, start=1):
        found[i] = p
    return found


def _read_metrics(score_path: Path, metrics: Dict[str, Tuple[str, ...]]) -> Dict[str, Optional[float]]:
    """Load a score yaml and extract each requested metric (or None)."""
    try:
        with open(score_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        return {name: None for name in metrics}
    if not isinstance(data, dict):
        return {name: None for name in metrics}
    return {name: extract_yaml_metric(data, keys) for name, keys in metrics.items()}


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

def aggregate_repeats(
    result_root: Path | str,
    judge_model: str = "gemini-3-flash-preview",
    metrics: Optional[Dict[str, Tuple[str, ...]]] = None,
    output_path: Path | str | None = None,
    print_summary: bool = True,
) -> dict:
    """Aggregate repeated evaluations under ``result_root``.

    Returns the full report dict and (unless ``output_path=""``) writes it to a
    YAML file.
    """
    result_root = Path(result_root).resolve()
    metrics = metrics or DEFAULT_METRICS

    # case_rel -> {repeat_index -> {metric -> value}}
    per_case_values: Dict[str, Dict[int, Dict[str, Optional[float]]]] = {}

    for results_dir in sorted(result_root.glob("**/generation_task/results")):
        if not results_dir.is_dir():
            continue
        rep_files = _collect_repeat_score_files(results_dir, judge_model)
        if not rep_files:
            continue
        try:
            case_rel = str(results_dir.parents[1].relative_to(result_root))
        except ValueError:
            case_rel = str(results_dir.parents[1])
        per_case_values[case_rel] = {
            k: _read_metrics(p, metrics) for k, p in rep_files.items()
        }

    # ---- Build the report ----
    rep_counts = {rel: len(v) for rel, v in per_case_values.items()}
    rep_distribution: Dict[int, int] = {}
    for n in rep_counts.values():
        rep_distribution[n] = rep_distribution.get(n, 0) + 1

    report: dict = {
        "judge_model": judge_model,
        "result_root": str(result_root),
        "num_cases": len(per_case_values),
        "repeats_per_case": {
            "min": min(rep_counts.values()) if rep_counts else 0,
            "max": max(rep_counts.values()) if rep_counts else 0,
            "distribution": dict(sorted(rep_distribution.items())),
        },
        "metrics": {},
    }

    for metric_name in metrics:
        per_case_stats: Dict[str, dict] = {}
        per_case_means: List[float] = []
        per_case_stds: List[float] = []
        per_case_ranges: List[float] = []
        # repeat_index -> list of per-case values (for across-run dataset means)
        by_repeat: Dict[int, List[float]] = {}

        for case_rel, rep_map in per_case_values.items():
            vals: List[float] = []
            values_by_idx: Dict[int, float] = {}
            for idx, mvals in sorted(rep_map.items()):
                v = mvals.get(metric_name)
                if v is None:
                    continue
                vals.append(v)
                values_by_idx[idx] = v
                by_repeat.setdefault(idx, []).append(v)
            if not vals:
                continue
            s = _summ(vals)
            s["values"] = {int(k): values_by_idx[k] for k in sorted(values_by_idx)}
            per_case_stats[case_rel] = s
            per_case_means.append(s["mean"])
            per_case_stds.append(s["std"])
            per_case_ranges.append(s["range"])

        # Per-repeat dataset mean (mean over cases for each repeat index).
        per_repeat_dataset_mean = {
            int(idx): statistics.mean(vs) for idx, vs in sorted(by_repeat.items()) if vs
        }
        across_run_vals = list(per_repeat_dataset_mean.values())
        across = _summ(across_run_vals)

        report["metrics"][metric_name] = {
            "dataset": {
                "reported_mean": statistics.mean(per_case_means) if per_case_means else None,
                "within_case_std_mean": statistics.mean(per_case_stds) if per_case_stds else None,
                "within_case_range_mean": statistics.mean(per_case_ranges) if per_case_ranges else None,
                "within_case_range_max": max(per_case_ranges) if per_case_ranges else None,
                "per_repeat_dataset_mean": per_repeat_dataset_mean,
                "across_run_mean": across["mean"],
                "across_run_std": across["std"],
                "across_run_min": across["min"],
                "across_run_max": across["max"],
                "across_run_range": across["range"],
            },
            "per_case": per_case_stats,
        }

    # ---- Write report ----
    if output_path != "":
        out = (
            Path(output_path).resolve()
            if output_path
            else result_root / f"repeats_summary__{judge_model}.yaml"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            yaml.safe_dump(report, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        report["_output_path"] = str(out)

    if print_summary:
        _print_summary(report, metrics)

    return report


def _fmt(x: Optional[float]) -> str:
    return "  n/a" if x is None else f"{x:6.2f}"


def _print_summary(report: dict, metrics: Dict[str, Tuple[str, ...]]) -> None:
    print(f"\n{'=' * 92}")
    print(f"Repeated-evaluation summary  (judge_model = {report['judge_model']})")
    print(f"  cases = {report['num_cases']}, "
          f"repeats/case = {report['repeats_per_case']['min']}"
          f"..{report['repeats_per_case']['max']} "
          f"(distribution: {report['repeats_per_case']['distribution']})")
    print(f"{'=' * 92}")
    header = (
        f"{'metric':<22} {'reported':>9} {'±within-case std':>17} "
        f"{'across-run std':>15} {'across-run range':>17}"
    )
    print(header)
    print("-" * len(header))
    for name in metrics:
        d = report["metrics"].get(name, {}).get("dataset", {})
        print(
            f"{name:<22} {_fmt(d.get('reported_mean')):>9} "
            f"{_fmt(d.get('within_case_std_mean')):>17} "
            f"{_fmt(d.get('across_run_std')):>15} "
            f"{_fmt(d.get('across_run_range')):>17}"
        )

    # Show the per-repeat dataset means for the overall metric.
    overall = report["metrics"].get("overall", {}).get("dataset", {})
    prm = overall.get("per_repeat_dataset_mean") or {}
    if prm:
        pretty = ", ".join(f"rep{k}={v:.2f}" for k, v in sorted(prm.items()))
        print(f"\noverall per-repeat dataset means: {pretty}")

    # Top-5 most variable cases by overall range.
    per_case = report["metrics"].get("overall", {}).get("per_case", {})
    ranked = sorted(
        ((c, s.get("range", 0.0) or 0.0, s) for c, s in per_case.items()),
        key=lambda x: x[1],
        reverse=True,
    )[:5]
    if ranked and ranked[0][1] > 0:
        print("\nMost variable cases (overall score, by range across repeats):")
        for case_rel, rng, s in ranked:
            print(f"  {rng:5.2f}  (mean={s['mean']:.2f}, n={s['n']})  {case_rel}")

    if report.get("_output_path"):
        print(f"\nFull report written to: {report['_output_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate repeated judge evaluations (mean/std/range)."
    )
    parser.add_argument("--result_root", type=str, required=True,
                        help="Agent result root, e.g. results/NotebookLM.")
    parser.add_argument("--judge_model", type=str, default="gemini-3-flash-preview",
                        help="Judge model filename prefix (default: gemini-3-flash-preview).")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to write the YAML report "
                             "(default: <result_root>/repeats_summary__<judge_model>.yaml).")
    args = parser.parse_args()

    result_root = Path(args.result_root).expanduser().resolve()
    if not result_root.is_dir():
        raise SystemExit(f"result_root not found or not a directory: {result_root}")

    aggregate_repeats(
        result_root=result_root,
        judge_model=args.judge_model,
        output_path=args.output,
        print_summary=True,
    )


if __name__ == "__main__":
    main()
