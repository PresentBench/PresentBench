#!/usr/bin/env python3
"""
Compute average scores across all data.

Traverse all data sources and items, read score files, and compute per-level averages in a tree structure.
Supports two scoring methods:
- ours: read generation_task/results/{judge_model}_{timestamp}_score.yaml
- ppteval: read generation_task/results/{judge_model}_{yyyymmdd}_{hhmmss}_score.json

Usage:
    python -m utils.statistics.calculate_average_scores --result_root_dir ../data_results/Doubao/ --judge_model gemini-3-flash-preview --prefer_newest --scoring_methods ours ppteval
     
    
Args:
    --result_root_dir: Results root directory.
    --output: Output file name (default: average_scores.yaml; only for the "ours" method).
    --prefer_newest: If multiple scoring files exist, choose the newest one (default: choose oldest).
    --judge_model: Optional judge model name; if provided, only select files matching this judge model.
    --scoring_methods: Scoring methods to include: ours and/or ppteval. Default: ours only.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import yaml

# Reuse data source directory definitions from paths.py.
from utils.paths import DATA_SOURCE_DIRS_REL
# Reuse helpers from score_utils.py.
from utils.score_utils import (
    find_single_score_yaml, 
    find_single_score_json,
    extract_yaml_metric, 
    extract_ppteval_score,
    model_filename_prefix,
    YAML_METRIC_SPECS
)


def find_score_files(data_item_dir: Path, scoring_method: str, prefer_newest: bool, judge_model: str | None = None) -> list[Path]:
    """
    Find score files under a data item directory.
    Only considers files directly under generation_task*/results matching *_score.yaml or *_score.json (no subdirectories).
    If multiple files exist in the same directory, choose newest/oldest based on prefer_newest.
    
    Args:
        data_item_dir: Data item directory.
        scoring_method: 'ours' for YAML files, 'ppteval' for JSON files.
        prefer_newest: If multiple files exist, True selects newest, False selects oldest.
        judge_model: Optional judge model name; if provided, only select matching files.
    
    Returns:
        List of found score file paths (at most one per results directory).
    """
    score_files = []
    
    # Find results directories under generation_task*.
    for results_dir in data_item_dir.glob("generation_task*/results"):
        if not results_dir.is_dir():
            continue
        # Choose helper based on scoring method.
        if scoring_method == 'ours':
            score_file = find_single_score_yaml(results_dir, prefer_newest=prefer_newest, judge_model=judge_model)
        elif scoring_method == 'ppteval':
            score_file = find_single_score_json(results_dir, prefer_newest=prefer_newest, judge_model=judge_model)
        else:
            score_file = None
        
        if score_file:
            score_files.append(score_file)
    
    return score_files


def read_scores(score_file: Path, scoring_method: str) -> dict | None:
    """
    Read scores from a score file.
    
    Args:
        score_file: Score file path.
        scoring_method: 'ours' reads YAML; 'ppteval' reads JSON.
    
    Returns:
        A dict containing scores, or None if reading fails.
    """
    try:
        if scoring_method == 'ours':
            # Read YAML score file.
            with open(score_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if not data or 'total' not in data:
                return None
            
            # Extract metrics using score_utils helpers.
            arithmetic_mean = extract_yaml_metric(data, YAML_METRIC_SPECS['AM'])
            
            result = {
                'weighted_arithmetic_mean_percent': arithmetic_mean,
                'file': str(score_file),
                'dimensions': {
                    'material_independent': {},
                    'material_dependent': {},
                }
            }

            # Try to read per-dimension scores (class_1, class_2, ...).
            total = data['total']
            for category in ['material_independent', 'material_dependent']:
                node = total.get(category)
                if node and isinstance(node, dict) and 'classes' in node:
                    for class_name, class_data in node['classes'].items():
                        score = class_data.get('score_percent')
                        if score is not None:
                            result['dimensions'][category][class_name] = score

            return result
        
        elif scoring_method == 'ppteval':
            # Read PPTEval JSON score file (scores.overall).
            overall_score = extract_ppteval_score(score_file)
            if overall_score is None:
                return None
            
            # PPTEval has a single overall score.
            result = {
                'overall_score': overall_score,  # raw score in [1, 5]
                'overall_score_scaled_to_0_100': (overall_score - 1) * 25,  # linearly mapped to [0, 100]
                'file': str(score_file),
                'scoring_method': 'ppteval',
            }
            return result
        
        else:
            return None
            
    except Exception as e:
        print(f"Warning: Failed to read {score_file}: {e}")
    return None

def collect_all_scores(result_root_dir: Path, scoring_method: str, prefer_newest: bool, judge_model: str | None = None) -> dict[str, list[dict]]:
    """
    Collect scores for all data items.
    
    Args:
        result_root_dir: Results root directory.
        scoring_method: 'ours' or 'ppteval'.
        prefer_newest: If multiple files exist, True selects newest, False selects oldest.
        judge_model: Optional judge model name; if provided, only select matching files.
    
    Returns:
        Dict keyed by relative data source path; values are lists of score dicts for that source.
    """
    all_scores = defaultdict(list)
    
    for rel_source_path in DATA_SOURCE_DIRS_REL:
        source_dir = result_root_dir / rel_source_path
        
        if not source_dir.exists():
            print(f"Warning: Source directory not found: {source_dir}")
            continue
        
        # Iterate over each data item (subdirectory) under the source directory.
        for item_dir in source_dir.iterdir():
            if not item_dir.is_dir():
                continue
            
            # Skip hidden and special directories.
            if item_dir.name.startswith('.') or item_dir.name.startswith('__'):
                continue
            
            # Confirm this is a data item by checking for generation_task*.
            has_gen_task = any(item_dir.glob("generation_task*"))
            if not has_gen_task:
                continue
            
            # Find score files.
            score_files = find_score_files(item_dir, scoring_method, prefer_newest=prefer_newest, judge_model=judge_model)
            
            for score_file in score_files:
                scores = read_scores(score_file, scoring_method)
                if scores:
                    scores['data_item'] = item_dir.name
                    scores['data_source'] = str(rel_source_path)
                    all_scores[str(rel_source_path)].append(scores)
    
    return all_scores


def calculate_average(scores_list: list[dict], scoring_method: str = 'ours') -> dict:
    """
    Compute the average of a list of scores.
    
    Args:
        scores_list: List of score dicts.
        scoring_method: 'ours' or 'ppteval'.
    
    Returns:
        Dict containing averages and count.
    """
    if not scores_list:
        if scoring_method == 'ppteval':
            return {
                'count': 0,
                'overall_score': None,
                'overall_score_scaled_to_0_100': None,
            }
        else:
            return {
                'count': 0,
                'arithmetic_mean': None,
            }
    
    if scoring_method == 'ppteval':
        # PPTEval: average overall_score and overall_score_scaled_to_0_100.
        overall_values = [s['overall_score'] for s in scores_list 
                          if s.get('overall_score') is not None]
        overall_scaled_values = [s['overall_score_scaled_to_0_100'] for s in scores_list 
                                 if s.get('overall_score_scaled_to_0_100') is not None]
        result = {
            'count': len(scores_list),
            'overall_score': sum(overall_values) / len(overall_values) if overall_values else None,
            'overall_score_scaled_to_0_100': sum(overall_scaled_values) / len(overall_scaled_values) if overall_scaled_values else None,
        }
        return result
    else:
        # Ours: compute arithmetic mean.
        arithmetic_values = [s['weighted_arithmetic_mean_percent'] for s in scores_list 
                             if s.get('weighted_arithmetic_mean_percent') is not None]
        
        result = {
            'count': len(scores_list),
            'arithmetic_mean': sum(arithmetic_values) / len(arithmetic_values) if arithmetic_values else None,
            'dimensions': {'material_independent': {}, 'material_dependent': {}}
        }

        temp_dim_values = {
            'material_independent': defaultdict(list),
            'material_dependent': defaultdict(list),
        }
        for s in scores_list:
            dims = s.get('dimensions', {})
            # Dimensions use the new material_* naming; iterate over existing keys directly.
            for category in ['material_independent', 'material_dependent']:
                if category in dims:
                    for cls_name, val in dims[category].items():
                        if val is not None:
                            temp_dim_values[category][cls_name].append(val)

        for category, classes_dict in temp_dim_values.items():
            for cls_name, val_list in classes_dict.items():
                if val_list:
                    result['dimensions'][category][cls_name] = sum(val_list) / len(val_list)
        return result


def build_result_tree(
    all_scores: dict[str, list[dict]],
    scoring_method: str = 'ours',
    include_leaf_nodes: bool = False,
) -> dict:
    """
    Build a tree structure with score statistics, down to each data item.
    
    Args:
        all_scores: Scores grouped by data source path.
        scoring_method: 'ours' or 'ppteval'.
        include_leaf_nodes: Whether to keep leaf nodes (each concrete data item) in the result tree.
            - False (default): keep only directory-level nodes; leaf entries are only used for aggregation.
            - True: keep per-item leaf nodes in the output.
    
    Returns:
        Tree-structured result dict.
    """
    
    def create_node() -> dict:
        """Create a tree node."""
        return {
            'average': None,
            'subtrees': {},
        }
    
    # Step 1: build the tree structure.
    root = create_node()
    
    for source_path_str, scores_list in all_scores.items():
        parts = Path(source_path_str).parts
        
        # 1) Descend into the data source path.
        current = root
        for part in parts:
            if part not in current['subtrees']:
                current['subtrees'][part] = create_node()
            current = current['subtrees'][part]
        
        # 2) Attach each data item's scores to the tree.
        for item_score in scores_list:
            item_name = item_score['data_item']

            if include_leaf_nodes:
                # Under the current directory node, create a leaf node per data_item.
                item_node = create_node()
                # Leaf node average is the item's own score (re-aggregated later if multiple files exist).
                item_node['average'] = calculate_average([item_score], scoring_method)

                # If multiple entries share the same item name (e.g. under different task dirs),
                # accumulate them in a temporary list for aggregation.
                if item_name not in current['subtrees']:
                    current['subtrees'][item_name] = item_node
                    current['subtrees'][item_name]['_leaf_scores'] = [item_score]
                else:
                    current['subtrees'][item_name]['_leaf_scores'].append(item_score)
            else:
                # No leaf nodes: accumulate raw scores at the current directory node for later aggregation.
                if '_leaf_scores' not in current:
                    current['_leaf_scores'] = []
                current['_leaf_scores'].append(item_score)

    # Step 2: recursively compute averages bottom-up.
    def calculate_node_averages(node: dict) -> list[dict]:
        """Recursively aggregate scores."""
        all_node_scores = []
        
        # Leaf nodes: take their raw scores.
        if '_leaf_scores' in node:
            all_node_scores.extend(node['_leaf_scores'])
            # Recompute exact average (handles multiple score files for the same item).
            node['average'] = calculate_average(node['_leaf_scores'], scoring_method)
        
        # Recurse into children and aggregate.
        for child in node['subtrees'].values():
            all_node_scores.extend(calculate_node_averages(child))
        
        # Compute directory-level average at this node.
        if all_node_scores:
            node['average'] = calculate_average(all_node_scores, scoring_method)
        
        return all_node_scores
    
    calculate_node_averages(root)
    
    # Step 3: clean temporary data.
    def clean_temp_data(node: dict):
        if '_leaf_scores' in node:
            del node['_leaf_scores']
        for child in node['subtrees'].values():
            clean_temp_data(child)
    
    clean_temp_data(root)
    
    return root


def format_output(result_tree: dict, scoring_method: str = 'ours') -> dict:
    """
    Format output for readability.
    
    Args:
        result_tree: Result tree.
        scoring_method: 'ours' or 'ppteval'.
    """
    def format_float(val):
        return round(val, 4) if val is not None else None

    def format_node(node: dict) -> dict:
        """Format a single node."""
        formatted = {}
        
        if node.get('average'):
            avg = node['average']
            if scoring_method == 'ppteval':
                # PPTEval outputs overall_score and overall_score_scaled_to_0_100.
                formatted_avg = {
                    'count': avg['count'],
                    'overall_score': format_float(avg.get('overall_score')),  # raw score in [1, 5]
                    'overall_score_scaled_to_0_100': format_float(avg.get('overall_score_scaled_to_0_100')),  # scaled to [0, 100]
                }
            else:
                # Ours outputs arithmetic_mean.
                formatted_avg = {
                    'count': avg['count'],
                    'arithmetic_mean': format_float(avg.get('arithmetic_mean')),
                }
                if 'dimensions' in avg:
                    dim_out = {}
                    has_dim_data = False
                    for cat, classes in avg['dimensions'].items():
                        if classes:
                            dim_out[cat] = {k: format_float(v) for k, v in classes.items()}
                            has_dim_data = True
                    if has_dim_data: formatted_avg['dimensions'] = dim_out
            formatted['average'] = formatted_avg
        if node.get('subtrees'):
            formatted['subtrees'] = {key: format_node(child) for key, child in sorted(node['subtrees'].items())}
        return formatted
    return format_node(result_tree)


def process_scoring_method(
    result_root_dir: Path,
    scoring_method: str,
    prefer_newest: bool,
    judge_model: str | None,
    output_file: str | None = None,
    include_leaf_nodes: bool = False,
):
    """
    Process aggregation for one scoring method.
    
    Args:
        result_root_dir: Results root directory.
        scoring_method: 'ours' or 'ppteval'.
        prefer_newest: If multiple files exist, True selects newest, False selects oldest.
        judge_model: Optional judge model name; if provided, only select matching files.
        output_file: Output file name (relative path or file name); if None, use defaults.
        include_leaf_nodes: Whether to keep leaf nodes (each concrete data item) in the output.
    """
    method_name = 'Ours' if scoring_method == 'ours' else 'PPTEval'
    
    # Collect all scores.
    print(f"\nCollecting {method_name} scores...")
    if judge_model:
        print(f"Filtering judge_model: {judge_model}")
    print(f"File selection policy: {'newest' if prefer_newest else 'oldest'}")
    all_scores = collect_all_scores(result_root_dir, scoring_method, prefer_newest=prefer_newest, judge_model=judge_model)
    
    total_count = sum(len(scores) for scores in all_scores.values())
    print(f"Found {total_count} score record(s) from {len(all_scores)} data source(s)")
    
    if total_count == 0:
        print(f"No {method_name} score records found. Please check the data directory.")
        return
    
    # Build result tree.
    print(f"\nBuilding {method_name} aggregate results...")
    result_tree = build_result_tree(all_scores, scoring_method, include_leaf_nodes=include_leaf_nodes)
    
    # Format output.
    formatted_result = format_output(result_tree, scoring_method)
    
    # Determine output file name.
    if output_file is None:
        base_name = "average_scores" if scoring_method == 'ours' else "average_scores_ppteval"
        if judge_model:
            # 将模型名转换为跨平台安全的文件名片段，避免不同 judge 互相覆盖。
            safe_judge_model = model_filename_prefix(judge_model)
            file_name = f"{base_name}__{safe_judge_model}.yaml" if safe_judge_model else f"{base_name}.yaml"
        else:
            file_name = f"{base_name}.yaml"
        output_file_path = result_root_dir / file_name
    else:
        output_file_path = result_root_dir / output_file
    
    # Save results.
    with open(output_file_path, 'w', encoding='utf-8') as f:
        yaml.dump(formatted_result, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"{method_name} results saved to: {output_file_path}")
    
    # Print overall stats.
    if formatted_result.get('average'):
        avg = formatted_result['average']
        print(f"\n=== {method_name} Overall Summary ===")
        print(f"Count: {avg['count']}")
        
        if scoring_method == 'ppteval':
            # PPTEval: print overall_score and overall_score_scaled_to_0_100.
            print(f"Overall score (raw [1,5]): {avg.get('overall_score', 'N/A')}")
            print(f"Overall score (scaled to [0,100]): {avg.get('overall_score_scaled_to_0_100', 'N/A')}")
            
            # Print per-category stats.
            categories = ['academia', 'advertising', 'education', 'economics', 'talk']
            col_width = max(max(len(key) for key in categories), 12)
            first_col_width = 10
            
            # Print header.
            header = "Type".ljust(first_col_width)
            header += f"{'Total':>{col_width}}"
            for key in categories:
                header += f"{key:>{col_width}}"
            print(f"\n{header}")
            print("-" * len(header))
            
            # Row 1: raw score [1,5]
            overall_row = "Raw".ljust(first_col_width)
            overall_val = avg.get('overall_score')
            if overall_val is not None:
                overall_row += f"{overall_val:>{col_width}.4f}"
            else:
                overall_row += f"{'N/A':>{col_width}}"
            for key in categories:
                if key in formatted_result.get('subtrees', {}):
                    val = formatted_result['subtrees'][key]['average'].get('overall_score')
                    if val is not None:
                        overall_row += f"{val:>{col_width}.4f}"
                    else:
                        overall_row += f"{'N/A':>{col_width}}"
                else:
                    overall_row += f"{'N/A':>{col_width}}"
            print(overall_row)
            
            # Row 2: scaled score [0,100]
            scaled_row = "Scaled".ljust(first_col_width)
            scaled_val = avg.get('overall_score_scaled_to_0_100')
            if scaled_val is not None:
                scaled_row += f"{scaled_val:>{col_width}.4f}"
            else:
                scaled_row += f"{'N/A':>{col_width}}"
            for key in categories:
                if key in formatted_result.get('subtrees', {}):
                    val = formatted_result['subtrees'][key]['average'].get('overall_score_scaled_to_0_100')
                    if val is not None:
                        scaled_row += f"{val:>{col_width}.4f}"
                    else:
                        scaled_row += f"{'N/A':>{col_width}}"
                else:
                    scaled_row += f"{'N/A':>{col_width}}"
            print(scaled_row)
        else:
            # Ours: print arithmetic means.
            # Print per-category stats as a table (first column is total).
            categories = ['academia', 'advertising', 'education', 'economics', 'talk']
            
            # Compute column width (align numbers and headers).
            col_width = max(max(len(key) for key in categories), 12)
            first_col_width = 10
            
            # Print header.
            header = "Type".ljust(first_col_width)
            header += f"{'Total':>{col_width}}"
            for key in categories:
                header += f"{key:>{col_width}}"
            print(f"\n{header}")
            print("-" * len(header))
            
            arithmetic_row = "Arithmetic".ljust(first_col_width)
            # First column: total arithmetic mean.
            arithmetic_row += f"{avg.get('arithmetic_mean', 'N/A'):>{col_width}.4f}" if avg.get('arithmetic_mean') is not None else f"{'N/A':>{col_width}}"
            # Remaining columns: per-category arithmetic means.
            for key in categories:
                if key in formatted_result.get('subtrees', {}):
                    val = formatted_result['subtrees'][key]['average'].get('arithmetic_mean')
                    if val is not None:
                        arithmetic_row += f"{val:>{col_width}.4f}"
                    else:
                        arithmetic_row += f"{'N/A':>{col_width}}"
                else:
                    arithmetic_row += f"{'N/A':>{col_width}}"
            print(arithmetic_row)


def main():
    parser = argparse.ArgumentParser(description='Compute average scores across all data')
    parser.add_argument('--result_root_dir', type=str, help='Results root directory')
    parser.add_argument(
        '--prefer_newest',
        action='store_true',
        help='If multiple scoring files exist, prefer the newest one (default: prefer oldest).',
    )
    parser.add_argument(
        '--judge_model',
        type=str,
        default=None,
        help='Optional judge model name. If provided, only select scoring files matching this judge model.',
    )
    parser.add_argument(
        '--scoring_methods',
        type=str,
        nargs='+',
        choices=['ours', 'ppteval'],
        default=['ours'],
        help='Scoring methods to include: ours and/or ppteval. "ours" reads YAML; "ppteval" reads PPTEval JSON.',
    )
    parser.add_argument(
        '--include_leaf_nodes',
        action='store_true',
        help='Whether to keep leaf nodes (each concrete data item) in the output. '
             'By default leaf nodes are omitted and only directory-level aggregates are output.',
    )
    args = parser.parse_args()
    
    result_root_dir = Path(args.result_root_dir)
    
    # Process each scoring method.
    if 'ours' in args.scoring_methods:
        process_scoring_method(
            result_root_dir, 
            'ours', 
            args.prefer_newest, 
            args.judge_model,
            output_file=None,
            include_leaf_nodes=args.include_leaf_nodes,
        )
    
    if 'ppteval' in args.scoring_methods:
        process_scoring_method(
            result_root_dir, 
            'ppteval', 
            args.prefer_newest, 
            args.judge_model,
            output_file=None,
            include_leaf_nodes=args.include_leaf_nodes,
        )


if __name__ == '__main__':
    main()
