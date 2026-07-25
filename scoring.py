"""
Usage example:

python scoring.py \
    --weights_path /path/to/judge_weights.yaml \
    --result_path    /path/to/result.yaml
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import yaml


# ============================================================================
# File I/O utilities
# ============================================================================

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_yaml(path: str) -> Dict[str, Any]: 
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_config(path: str) -> Dict[str, Any]:
    """
    Load configuration file (JSON or YAML) based on file extension.
    """
    if Path(path).suffix.lower() in {".yaml", ".yml"}:
        return load_yaml(path)
    else:
        return load_json(path)


# ============================================================================
# Statistics computation helpers
# ============================================================================

def _sum_stats(class_info: Dict[str, Dict[str, Any]], key: str) -> int:
    """Sum a statistic across all classes."""
    return sum(info.get(key, 0) for info in class_info.values())


def _compute_leave_one_out(class_info: Dict[str, Dict[str, Any]], class_weights: Dict[str, Any]) -> Dict[str, Any]:
    results = {}
    # Iterate through each class to "leave it out"
    for remove_id in class_info.keys():
        # 1. Create subset excluding the current class
        subset_info = {k: v for k, v in class_info.items() if k != remove_id}
        subset_weights = {k: v for k, v in class_weights.items() if k != remove_id}
        # 2. Compute Arithmetic Mean for the subset
        subset_total_score = sum(info["score"] for info in subset_info.values())
        subset_max_total = sum(float(subset_weights.get(k, 0.0)) for k in subset_info.keys())
        arithmetic_mean = (subset_total_score / subset_max_total * 100.0) if subset_max_total > 0 else 0.0
        results[remove_id] = {
            "weighted_arithmetic_mean_percent": arithmetic_mean,
        }
        
    return results

# ============================================================================
# Core scoring functions
# ============================================================================

def score_section(
    answers_section: Dict[str, Any],
    class_weights: Dict[str, Any],
) -> Tuple[float, Dict[str, Dict[str, Any]]]:
    """
    Compute scores for one top-level group, e.g. "material_independent" or
    "material_dependent".

    Returns:
        total_score: sum over all classes in this section
        class_info: mapping "class_id" -> {"score": float, "max_score": float, "yes_count": int, "valid_count": int, "not_applicable_count": int}
    """
    class_info: Dict[str, Dict[str, Any]] = {}
    total_score = 0.0

    for cls_id, cls_answers in answers_section.items():
        if cls_id == "total":
            continue

        # class_weights has shape:
        # { "total": 20.0, "1": 5.0, "2": 5.0, ... }
        cls_total = float(class_weights.get(cls_id, 0.0))

        # Flatten items: keys like "1.1", "1.2", ...
        yes_count = 0
        valid_count = 0
        not_applicable_count = 0

        for item_id, item in cls_answers.items():
            if not isinstance(item, dict):
                continue
            raw_ans = item.get("answer")
            if raw_ans is None:
                # API 调用失败：无 answer 视为不计
                continue
            ans = str(raw_ans).strip().lower()
            if ans in ["not applicable", r"not\ applicable"]:
                not_applicable_count += 1
                continue
            if ans in ["yes", "no"]:
                valid_count += 1
                if ans == "yes":
                    yes_count += 1

        if valid_count == 0 or cls_total == 0:
            cls_score = 0.0
        else:
            # Evenly distribute the class score to all applicable items
            cls_score = cls_total * yes_count / valid_count

        class_info[cls_id] = {
            "score": cls_score,
            "max_score": cls_total,
            "yes_count": yes_count,
            "valid_count": valid_count,
            "not_applicable_count": not_applicable_count,
        }
        total_score += cls_score

    return total_score, class_info


def _compute_section_stats(total: float, max_total: float, class_info: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compute statistics for a section."""
    return {
        "weighted_arithmetic_mean_percent": (total / max_total * 100.0) if max_total > 0 else 0.0,
        "yes_count": _sum_stats(class_info, "yes_count"),
        "valid_count": _sum_stats(class_info, "valid_count"),
        "not_applicable_count": _sum_stats(class_info, "not_applicable_count"),
    }


def compute_scores(result_path: str, weights_path: str) -> Dict[str, Any]:
    combined_result = load_config(result_path)
    weights_all = load_config(weights_path)
    
    material_independent_weights = weights_all.get("material_independent", {})
    material_dependent_weights = weights_all.get("material_dependent", {})
    
    # Result structure fields: renamed to material_independent / material_dependent.
    mi_section = combined_result.get("material_independent")
    md_section = combined_result.get("material_dependent")

    mi_total, mi_class_info = score_section(
        mi_section,
        material_independent_weights,
    )
    md_total, md_class_info = score_section(
        md_section,
        material_dependent_weights,
    )
    
    mi_max_total = sum(float(material_independent_weights.get(cls_id, 0.0)) for cls_id in material_independent_weights.keys())
    md_max_total = sum(float(material_dependent_weights.get(cls_id, 0.0)) for cls_id in material_dependent_weights.keys())
    overall_max_total = float(weights_all.get("total", mi_max_total + md_max_total))
    
    mi_stats = _compute_section_stats(mi_total, mi_max_total, mi_class_info)
    md_stats = _compute_section_stats(md_total, md_max_total, md_class_info)
    
    all_class_info = {f"material_independent_{k}": v for k, v in mi_class_info.items()}
    all_class_info.update({f"material_dependent_{k}": v for k, v in md_class_info.items()})
    all_class_weights = {
        f"material_independent_{k}": float(material_independent_weights.get(k, 0.0))
        for k in mi_class_info.keys()
    }
    all_class_weights.update(
        {
            f"material_dependent_{k}": float(material_dependent_weights.get(k, 0.0))
            for k in md_class_info.keys()
        }
    )
    leave_one_out_stats = _compute_leave_one_out(all_class_info, all_class_weights)

    overall_stats = {
        "weighted_arithmetic_mean_percent": ((mi_total + md_total) / overall_max_total * 100.0) if overall_max_total > 0 else 0.0,
        "yes_count": mi_stats["yes_count"] + md_stats["yes_count"],
        "valid_count": mi_stats["valid_count"] + md_stats["valid_count"],
        "not_applicable_count": mi_stats["not_applicable_count"] + md_stats["not_applicable_count"],
        "leave_one_out_analysis": leave_one_out_stats,
    }
    
    mi_stats["weight_percent"] = (mi_max_total / overall_max_total * 100.0) if overall_max_total > 0 else 0.0
    md_stats["weight_percent"] = (md_max_total / overall_max_total * 100.0) if overall_max_total > 0 else 0.0
    
    return {
        "total": overall_stats,
        "material_independent_total": mi_stats,
        "material_dependent_total": md_stats,
        "material_independent_class_info": mi_class_info,
        "material_dependent_class_info": md_class_info,
        "material_independent_weights": material_independent_weights,
        "material_dependent_weights": material_dependent_weights,
        "overall_max_total": overall_max_total,
    }


# ============================================================================
# Output formatting functions
# ============================================================================

def _build_tree_data(scores: Dict[str, Any]) -> Dict[str, Any]:
    """Build tree-structured data for YAML output."""
    overall_max_total = scores["overall_max_total"]
    
    def _build_section(section_key_prefix: str) -> Dict[str, Any]:
        section_total = scores[f"{section_key_prefix}_total"]
        class_info = scores[f"{section_key_prefix}_class_info"]
        weights = scores[f"{section_key_prefix}_weights"]
        return {
            **section_total,
            "classes": {
                f"class_{cls_id}": {
                    "weight_percent": (float(weights.get(cls_id, 0.0)) / overall_max_total * 100.0) if overall_max_total > 0 else 0.0,
                    "score_percent": (info["score"] / info["max_score"] * 100.0) if info["max_score"] > 0 else 0.0,
                    "yes_count": info["yes_count"],
                    "valid_count": info["valid_count"],
                    "not_applicable_count": info["not_applicable_count"],
                }
                for cls_id, info in class_info.items()
            },
        }
    
    return {
        "total": scores["total"],
        "material_independent": _build_section("material_independent"),
        "material_dependent": _build_section("material_dependent"),
    }


def _print_summary(scores: Dict[str, Any], output_path: str):
    """Print score summary in tree format."""
    def _print_section(section_name: str, section_label: str, tree_prefix: str, class_prefix: str, has_next_section: bool):
        section_info = scores[f"{section_name}_total"]
        class_info = scores[f"{section_name}_class_info"]
        weights = scores[f"{section_name}_weights"]
        class_ids = sorted(class_info.keys(), key=lambda x: int(x))
        
        print(f"\n{tree_prefix} {section_label} ({section_info['weight_percent']:.0f}%):")
        print(f"{class_prefix}  ├─ Statistics: {section_info['yes_count']} / {section_info['valid_count']} yes, {section_info['not_applicable_count']} NA")
        print(f"{class_prefix}  └─ Weighted arithmetic mean: {section_info['weighted_arithmetic_mean_percent']:.1f}%")
        
        overall_max_total = scores["overall_max_total"]
        for i, cls_id in enumerate(class_ids):
            info = class_info[cls_id]
            class_weight_percent = (float(weights.get(cls_id, 0.0)) / overall_max_total * 100.0) if overall_max_total > 0 else 0.0
            class_score_percent = (info["score"] / info["max_score"] * 100.0) if info["max_score"] > 0 else 0.0
            is_last = (i == len(class_ids) - 1)
            
            if section_name == "material_independent":
                if is_last and has_next_section:
                    prefix = "│   └──"
                elif is_last:
                    prefix = "    └──"
                else:
                    prefix = "│   ├──"
            else:  # material_dependent
                prefix = "   └──" if is_last else "   ├──"
            
            print(f"{prefix} Class {cls_id} ({class_weight_percent:.0f}%): {class_score_percent:.1f}% ({info['yes_count']} / {info['valid_count']} yes, {info['not_applicable_count']} NA)")
    
    total_info = scores['total']
    print(f"Total")
    print(f"  ├─ Statistics: {total_info['yes_count']} / {total_info['valid_count']} yes, {total_info['not_applicable_count']} NA")
    print(f"  └─ Weighted arithmetic mean: {total_info['weighted_arithmetic_mean_percent']:.1f}%")
    
    has_md = len(scores["material_dependent_class_info"]) > 0
    _print_section("material_independent", "Material Independent", "├─", "│", has_md)
    if has_md:
        _print_section("material_dependent", "Material Dependent", "└─", "   ", False)
    
    print(f"\nScores saved to: {output_path}")


# ============================================================================
# Public API
# ============================================================================

def compute_and_save_scores(
    result_path: str = None,
    weights_path: str = None,
    output_path: str = None,
    print_summary: bool = True,
) -> str:
    """
    Compute scores from result file(s) and save to output file.
    
    Args:
        result_path: Path to combined result YAML/JSON file containing both material_independent and material_dependent results.
        weights_path: Path to weights YAML/JSON file
        output_path: Path to save computed scores. If None, defaults to 'score.yaml' in the same directory as result_path
        print_summary: Whether to print score summary to console
    
    Returns:
        Path to the saved score file
    """
    if output_path is None:
        if result_path:
            name, ext = os.path.splitext(result_path)
            output_path = name + '_score' + ext
        else:
            output_path = "score.yaml"
    
    scores = compute_scores(result_path=result_path, weights_path=weights_path)
    tree_data = _build_tree_data(scores)
    
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(tree_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    
    if print_summary:
        _print_summary(scores, output_path)
    
    return output_path


# ============================================================================
# Command-line interface
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute weighted scores from judge results and weight configs."
    )
    parser.add_argument(
        "--result_path",
        help=(
            "Path to combined result file containing both material_independent and material_dependent results. "
            "Format: {'material_dependent': ..., 'material_independent': ...}."
        ),
    )
    parser.add_argument(
        "--weights_path",
        help="Path to combined judge_weights.yaml (or .json)",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        default=None,
        help=(
            "Path to save computed scores as YAML. "
            "If omitted, defaults to 'score.yaml' in the same directory as --material_dependent_result_path or --result_path."
        ),
    )
    return parser.parse_args()




if __name__ == "__main__":
    args = parse_args()

    compute_and_save_scores(
        result_path=args.result_path,
        weights_path=args.weights_path,
        output_path=args.output_path,
        print_summary=True,
    )


