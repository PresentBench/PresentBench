"""
Shared utilities and constants for handling score files in the user_study subset.

This module provides:
- Score file pattern definitions (regular expressions)
- Metric definitions (YAML_METRIC_SPECS)
- Score file discovery helpers
- Score data extraction helpers
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Score file pattern.
SCORE_FILE_PATTERN = re.compile(
    r"^(?P<judge_model>.+)_(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_score\.yaml$"
)
# JSON score file pattern (judge_model required): {judge_model}_{yyyymmdd}_{hhmmss}_score (optional .json extension)
JSON_SCORE_FILE_PATTERN = re.compile(
    r"^(?P<judge_model>.+)_(?P<ts>\d{8}_\d{6})_score(?:\.json)?$"
)
_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def model_filename_prefix(model: str, thinking_level: str | None = None) -> str:
    """将模型名转换为跨平台安全、与评测输出一致的文件名前缀。"""
    base = model.replace('\\', '/').rsplit('/', 1)[-1].strip()
    base = _INVALID_FILENAME_CHARS.sub('-', base).strip(' .-') or 'model'
    if thinking_level:
        safe_level = _INVALID_FILENAME_CHARS.sub('-', thinking_level).strip(' .-')
        if safe_level:
            base += f'_{safe_level}'
    return base


# Define all YAML-based scoring metrics:
# - AM: original total.* metric
# - SRC_*: aggregate metrics under total.material_independent / total.material_dependent (2 metrics)
# - LOO_*: for 5 leave-one-out configurations under total.leave_one_out_analysis,
#          take weighted arithmetic mean respectively
#
# Note: metric_name is a "flat" string identifier; downstream code does not depend on its internal structure.
YAML_METRIC_SPECS: Dict[str, Tuple[str, ...]] = {
    # Original overall metric.
    "AM": ("total", "weighted_arithmetic_mean_percent"),
    # Aggregate metrics under total.material_independent / total.material_dependent.
    "SRC_INDEP_AM": (
        "total",
        "material_independent",
        "weighted_arithmetic_mean_percent",
    ),
    "SRC_DEP_AM": (
        "total",
        "material_dependent",
        "weighted_arithmetic_mean_percent",
    ),
    # Leave-one-out: material_independent_*
    "LOO_material_independent_1_AM": (
        "total",
        "leave_one_out_analysis",
        "material_independent_1",
        "weighted_arithmetic_mean_percent",
    ),
    "LOO_material_independent_2_AM": (
        "total",
        "leave_one_out_analysis",
        "material_independent_2",
        "weighted_arithmetic_mean_percent",
    ),
    # Leave-one-out: material_dependent_*
    "LOO_material_dependent_1_AM": (
        "total",
        "leave_one_out_analysis",
        "material_dependent_1",
        "weighted_arithmetic_mean_percent",
    ),
    "LOO_material_dependent_2_AM": (
        "total",
        "leave_one_out_analysis",
        "material_dependent_2",
        "weighted_arithmetic_mean_percent",
    ),
    "LOO_material_dependent_3_AM": (
        "total",
        "leave_one_out_analysis",
        "material_dependent_3",
        "weighted_arithmetic_mean_percent",
    ),
}

# All supported metric names (including JSON-based PPTEval).
METRICS: Tuple[str, ...] = tuple(YAML_METRIC_SPECS.keys()) + ("PPTEval", "LLMRanking")

def discover_subset_item_rel_paths(base_dir: Path) -> List[Path]:
    """
    Discover the list of items under a user_study subset directory.

    Notes:
    - Do not call get_all_data_item_dirs(base_dir=base_dir) directly, because a subset may not include
      all data source directories listed in utils/paths.py, which can cause iterdir() errors.
    - Each subset item should contain generation_task/results, so discovering based on that structure is more robust.
    """
    item_dirs: set[Path] = set()
    for results_dir in base_dir.glob("**/generation_task/results"):
        if not results_dir.is_dir():
            continue
        # base_dir/<rel_path>/generation_task/results
        # Note: Path.parents[0] is the parent, so <rel_path> corresponds to parents[1].
        item_dir = results_dir.parents[1]
        try:
            item_dirs.add(item_dir.relative_to(base_dir))
        except ValueError:
            # Not under base_dir; ignore (should not happen in practice).
            continue
    return sorted(item_dirs, key=lambda p: str(p).replace("\\", "/"))


def find_all_score_yamls(results_dir: Path, judge_model: str | None = None) -> List[Path]:
    """
    Find all files in results_dir matching {judge_model}_{timestamp}_score.yaml.
    
    Args:
        results_dir: Results directory.
        judge_model: Optional judge model name; if provided, only return matching files.
    
    Returns:
        Matching files, sorted by file name.
    """
    expected_model = model_filename_prefix(judge_model) if judge_model is not None else None
    matches: List[Path] = []
    for p in results_dir.glob("*_score.yaml"):
        if m := SCORE_FILE_PATTERN.match(p.name):
            # If judge_model is specified, it must match.
            if expected_model is not None:
                if m.group("judge_model") == expected_model:
                    matches.append(p)
            else:
                # If judge_model is not specified, keep all matches.
                matches.append(p)
    return sorted(matches)


def find_single_score_yaml(results_dir: Path, prefer_newest: bool = True, judge_model: str | None = None) -> Path | None:
    """
    Find a single {judge_model}_{timestamp}_score.yaml file in results_dir.
    
    Args:
        results_dir: Results directory.
        prefer_newest: If multiple files exist, True selects newest, False selects oldest. Defaults to True.
        judge_model: Optional judge model name; if provided, only consider matching files.
    
    Returns:
        File path if found, otherwise None. If multiple files exist, choose newest/oldest based on prefer_newest.
    """
    matches = find_all_score_yamls(results_dir, judge_model=judge_model)
    if len(matches) == 0:
        return None
    
    if len(matches) == 1:
        return matches[0]
    
    # If multiple files exist, pick newest/oldest by timestamp.
    # Extract timestamp from: {judge_model}_{timestamp}_score.yaml
    def extract_timestamp(path: Path) -> str:
        m = SCORE_FILE_PATTERN.match(path.name)
        if m:
            return m.group("ts")
        return ""
    
    # Sort by timestamp (format: YYYY-MM-DD_HH-MM-SS).
    matches_with_ts = [(extract_timestamp(p), p) for p in matches]
    matches_with_ts.sort(key=lambda x: x[0])
    
    if prefer_newest:
        return matches_with_ts[-1][1]
    else:
        return matches_with_ts[0][1]


def find_all_score_jsons(results_dir: Path, judge_model: str | None = None) -> List[Path]:
    """
    Find all JSON files in results_dir matching JSON_SCORE_FILE_PATTERN.
    
    Args:
        results_dir: Results directory.
        judge_model: Optional judge model name; if provided, only return matching files.
    
    Returns:
        Matching files, sorted by file name.
    """
    expected_model = model_filename_prefix(judge_model) if judge_model is not None else None
    matches: List[Path] = []
    for p in results_dir.iterdir():
        if not p.is_file():
            continue
        if m := JSON_SCORE_FILE_PATTERN.match(p.name):
            # If judge_model is specified, it must match.
            if expected_model is not None:
                if m.group("judge_model") == expected_model:
                    matches.append(p)
            else:
                # If judge_model is not specified, keep all matches.
                matches.append(p)
    return sorted(matches)


def find_single_score_json(results_dir: Path, prefer_newest: bool, judge_model: str | None = None) -> Path | None:
    """
    Find a JSON score file whose name ends with yyyymmdd_hhmmss_score (JSON format).

    Notes:
    - Matching rule (as requested): the file name **ends with `yyyymmdd_hhmmss_score`**, independent of extension.
    - These files coexist with `*_score.yaml` in the same directory without conflict (yaml ends with `_score.yaml`).
    - Return None if no matching file exists.
    - If multiple matches exist, pick newest/oldest based on prefer_newest.
    
    Args:
        results_dir: Results directory.
        prefer_newest: If multiple files exist, True selects newest, False selects oldest.
        judge_model: Optional judge model name; if provided, only consider matching files.
    
    Returns:
        File path if found, otherwise None.
    """
    matches = find_all_score_jsons(results_dir, judge_model=judge_model)
    if not matches:
        return None
    
    if len(matches) == 1:
        return matches[0]
    
    # If multiple files exist, pick newest/oldest by timestamp.
    # Extract timestamp from file name.
    def extract_timestamp(path: Path) -> str:
        m = JSON_SCORE_FILE_PATTERN.match(path.name)
        if m:
            return m.group("ts")
        return ""
    
    # Sort by timestamp (format: yyyymmdd_hhmmss; zero-padded digits, so lexicographic sort works).
    matches_with_ts = [(extract_timestamp(p), p) for p in matches]
    matches_with_ts.sort(key=lambda x: x[0])
    
    if prefer_newest:
        return matches_with_ts[-1][1]
    else:
        return matches_with_ts[0][1]


def extract_yaml_metric(data: dict, path_keys: Tuple[str, ...]) -> float | None:
    """
    Extract a metric value from YAML data by a given key path.

    Args:
        data: YAML-parsed dict.
        path_keys: Key path in YAML, e.g. ("total", "weighted_arithmetic_mean_percent").

    Returns:
        Metric value, or None if the path does not exist or value is invalid.
    """
    try:
        node = data
        for key in path_keys:
            node = node[key]
        return float(node)
    except (KeyError, TypeError, ValueError):
        return None


def extract_ppteval_score(json_path: Path) -> float | None:
    """
    Extract PPTEval overall score from a JSON file.

    Args:
        json_path: JSON score file path.

    Returns:
        Overall score, or None if the file does not exist or value is invalid.
    """
    try:
        with json_path.open("r", encoding="utf-8") as f:
            j = json.load(f)
        return float(j["scores"]["overall"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
