#!/usr/bin/env python3
"""
Batch scoring script: traverse all result files under result_root, check completeness, and compute scores.

Usage:
    python scoring_all.py --result_root ./results/Doubao/  --data_root .
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import yaml

from scoring import compute_and_save_scores
from utils.judge_utils import load_judge_prompt, is_result_complete
from utils.paths import DATA_SOURCE_DIRS_REL, get_data_source_dirs, get_valid_item_dirs, SKIP_NAMES


def _category_from_rel(rel: str | Path) -> str:
    """Extract category (first-level directory name) from a relative path."""
    if isinstance(rel, Path):
        return rel.parts[0]
    return str(rel).split("/", 1)[0]


def _ensure_data_root_ready(data_root: Path) -> None:
    if not data_root.exists() or not data_root.is_dir():
        raise FileNotFoundError(f"data_root not found or not a directory: {data_root}")

    # Best-effort check: domain-level weights and prompts exist for at least one domain.
    expected_domains = sorted({Path(rel).parts[0] for rel in DATA_SOURCE_DIRS_REL})
    existing = [d for d in expected_domains if (data_root / d).is_dir()]
    if not existing:
        raise FileNotFoundError(
            f"No expected domain directories found under data_root={data_root}. "
            f"Expected one of: {', '.join(expected_domains)}"
        )

    missing = []
    for d in existing:
        w = data_root / d / "judge_weights.yaml"
        c = data_root / d / "common_judge_prompt.json"
        if not w.exists():
            missing.append(str(w))
        if not c.exists():
            missing.append(str(c))
    if missing:
        raise FileNotFoundError("Missing required domain-level files:\n" + "\n".join(missing))


def find_result_files(result_root: Path) -> tuple[list[tuple[Path, Path, str]], list[tuple[Path, str]]]:
    """
    Find all result files.
    
    Args:
        result_root: Results root directory.
        
    Returns:
        (result_files, errors)
        - result_files: List of tuples: (result_file_path, item_dir, category)
          where result_file_path is the path to a result YAML file.
        - errors: List of tuples: (item_dir, error_message)
          directories that do not have a result yaml file and their error messages.
    """
    result_files = []
    errors = []
    
    # Get all data source directories.
    data_source_dirs = get_data_source_dirs(base_dir=result_root)
    
    for rel, src_dir in zip(DATA_SOURCE_DIRS_REL, data_source_dirs, strict=False):
        if not src_dir.is_dir():
            continue
        
        category = _category_from_rel(rel)
        
        try:
            result_item_dirs = get_valid_item_dirs(src_dir, skip_names=SKIP_NAMES)
        except OSError as e:
            logging.warning(f"Cannot list directory {src_dir}: {e}")
            continue
        
        for result_item_dir in sorted(result_item_dirs, key=lambda p: p.name):
            results_dir = result_item_dir / "generation_task" / "results"
            if not results_dir.exists():
                errors.append((result_item_dir, f"results directory does not exist: {results_dir}"))
                continue
            
            if not results_dir.is_dir():
                errors.append((result_item_dir, f"results path is not a directory: {results_dir}"))
                continue
            
            # Find all result files matching: {judge_model}_{timestamp}.yaml
            # Timestamp format: 2026-01-27_12-34-56
            pattern = re.compile(r'^(.+)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.yaml$')
            
            try:
                # If multiple matches exist, keep the one with the newest timestamp.
                latest: tuple[str, Path] | None = None  # (timestamp_str, file_path)
                for file_path in results_dir.iterdir():
                    if not file_path.is_file():
                        continue

                    # Skip _score.yaml files.
                    if file_path.name.endswith('_score.yaml'):
                        continue

                    match = pattern.match(file_path.name)
                    if not match:
                        continue

                    ts = match.group(2)  # "YYYY-MM-DD_HH-MM-SS" (fixed width; string compare works)
                    if latest is None or ts > latest[0]:
                        latest = (ts, file_path)

                if latest is not None:
                    result_files.append((latest[1], result_item_dir, category))
                else:
                    errors.append((result_item_dir, f"no matching result yaml file found in {results_dir}"))
            except OSError as e:
                error_msg = f"cannot list results directory {results_dir}: {e}"
                errors.append((result_item_dir, error_msg))
                logging.warning(error_msg)
                continue
    
    return result_files, errors


def check_result_completeness(
    result_file: Path,
    result_item_dir: Path,
    result_root: Path,
    data_root: Path,
) -> tuple[bool, Optional[str]]:
    """
    Check whether a result file fully contains all checklist items.
    
    Args:
        result_file: Result YAML file path.
        item_dir: Data item directory.
        category: Category name.
        base_dir: Base directory (used to load judge_prompt.json).
        
    Returns:
        (is_complete, error_message)
    """
    # judge_prompt.json / common_judge_prompt.json are taken from data_root.
    try:
        rel_item_dir = result_item_dir.relative_to(result_root)
    except Exception:
        return False, f"Result item dir is not under result_root: {result_item_dir} (root={result_root})"

    weights_item_dir = data_root / rel_item_dir

    # Load judge_prompt.json (note: do NOT use the copy under result_root).
    judge_prompt_path = weights_item_dir / "generation_task" / "judge_prompt.json"
    if not judge_prompt_path.exists():
        return False, f"judge_prompt.json not found: {judge_prompt_path}"
    
    try:
        # Domain-level common_judge_prompt.json: pass in the path directly.
        common_judge_prompt_path = data_root / rel_item_dir.parts[0] / "common_judge_prompt.json"
        if not common_judge_prompt_path.exists():
            return False, f"common_judge_prompt.json not found: {common_judge_prompt_path}"
        
        # Slides are required to generate the checklist.
        # Find slides file.
        slides_path = None
        results_dir = result_item_dir / "generation_task" / "results"
        for slides_file in ["slides.pdf", "slides.pptx"]:
            candidate = results_dir / slides_file
            if candidate.exists():
                slides_path = str(candidate)
                break
        
        if not slides_path:
            return False, f"Slides file not found in {results_dir}"
        
        # Generate checklist (the JSON loader handles prefixing + page-count truncation internally).
        loaded = load_judge_prompt(
            prompt_path=str(judge_prompt_path),
            common_judge_prompt_path=str(common_judge_prompt_path),
            slides_path=str(slides_path),
        )
        if not loaded:
            return False, f"Failed to load judge prompt from {judge_prompt_path}"
        material_independent_checklist, material_dependent_checklist = loaded
        
        # material_independent_checklist[0][0] may be partial(check_slide_count, max_count=...).
        # Truncation is handled by judge.py at runtime; completeness checks do not modify the checklist further.
        
        # Load result file.
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                result_data = yaml.safe_load(f)
        except Exception as e:
            return False, f"Failed to load result file {result_file}: {e}"
        
        if not isinstance(result_data, dict):
            return False, f"Result file {result_file} is not a dictionary"
        
        # Result structure now uses material_independent / material_dependent.
        mi_result = result_data.get("material_independent")
        md_result = result_data.get("material_dependent")

        # Check material_independent results.
        si_result = mi_result
        if not is_result_complete(material_independent_checklist, si_result):
            missing_items = []
            for i, sublist in enumerate(material_independent_checklist):
                outer_key = f"{i+1}"
                if outer_key not in si_result:
                    missing_items.append(f"material_independent.{outer_key} (missing entire class)")
                    continue
                inner_dict = si_result.get(outer_key, {})
                for j, _ in enumerate(sublist):
                    inner_key = f"{i+1}.{j+1}"
                    if inner_key not in inner_dict:
                        missing_items.append(f"material_independent.{inner_key}")
                    elif inner_dict[inner_key].get("answer") is None:
                        missing_items.append(f"material_independent.{inner_key} (answer is None)")
            return False, f"[{result_file}] Incomplete material_independent results. Missing items: {', '.join(missing_items)}"
        
        # Check material_dependent results.
        sd_result = md_result
        if not is_result_complete(material_dependent_checklist, sd_result):
            missing_items = []
            for i, sublist in enumerate(material_dependent_checklist):
                outer_key = f"{i+1}"
                if outer_key not in sd_result:
                    missing_items.append(f"material_dependent.{outer_key} (missing entire class)")
                    continue
                inner_dict = sd_result.get(outer_key, {})
                for j, _ in enumerate(sublist):
                    inner_key = f"{i+1}.{j+1}"
                    if inner_key not in inner_dict:
                        missing_items.append(f"material_dependent.{inner_key}")
                    elif inner_dict[inner_key].get("answer") is None:
                        missing_items.append(f"material_dependent.{inner_key} (answer is None)")
            return False, f"[{result_file}] Incomplete material_dependent results. Missing items: {', '.join(missing_items)}"
        
        return True, None

    except Exception as e:
        return False, f"[{result_file}] Error checking completeness: {e}"



def process_result_file(
    result_file: Path,
    result_item_dir: Path,
    category: str,
    data_root: Path,
    result_root: Path,
) -> tuple[bool, Optional[str]]:
    """
    Process a single result file: check completeness and compute scores.
    
    Args:
        result_file: Result YAML file path.
        item_dir: Data item directory.
        category: Category name.
        result_root: Results root (used to derive judge_weights.yaml path).
        base_dir: Base directory (used to load judge_prompt.json and locate judge_weights.yaml).
        
    Returns:
        (success, error_message)
    """
    # Locate judge_weights.yaml.
    weights_path = data_root / category / "judge_weights.yaml"
    if not weights_path.exists():
        return False, f"judge_weights.yaml not found: {weights_path}"
    
    # Completeness check: judge_prompt.json from data_root; slides/result.yaml from result_root.
    is_complete, error_msg = check_result_completeness(
        result_file,
        result_item_dir,
        result_root=result_root,
        data_root=data_root,
    )
    if not is_complete:
        return False, error_msg
    
    # Compute and save scores.
    try:
        score_output_path = result_file.parent / f"{result_file.stem}_score.yaml"
        compute_and_save_scores(
            result_path=str(result_file),
            weights_path=str(weights_path),
            output_path=str(score_output_path),
            print_summary=False  # Don't print summary during batch processing.
        )
        return True, None
    except Exception as e:
        return False, f"Error computing scores: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Batch scoring script: traverse all result files under result_root, check completeness, and compute scores"
    )
    parser.add_argument(
        "--result_root",
        type=str,
        required=True,
        help="Results root directory (e.g. data_results/AgentName)"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory used to locate judge_weights.yaml. If not provided, it will be inferred from result_root."
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output"
    )
    
    args = parser.parse_args()
    
    # Configure logging.
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    result_root = Path(args.result_root).expanduser().resolve()
    if not result_root.exists():
        logging.error(f"Result root directory does not exist: {result_root}")
        sys.exit(1)
    if not result_root.is_dir():
        logging.error(f"Result root is not a directory: {result_root}")
        sys.exit(1)
    
    data_root = Path(args.data_root).expanduser().resolve()
    if data_root is None:
        logging.error(f"Judge weights root is not specified")
        sys.exit(1)

    try:
        _ensure_data_root_ready(data_root)
    except Exception as e:
        logging.error(str(e))
        logging.error("If you haven't downloaded the benchmark dataset yet, run:")
        logging.error(f"  {sys.executable} scripts/download_data.py --repo_id PresentBench/PresentBench")
        sys.exit(1)
    
    # Find all result files.
    logging.info(f"Scanning result files in {result_root}...")
    result_files, find_errors = find_result_files(result_root)
    logging.info(f"Found {len(result_files)} result files to process")
    
    # Process each result file.
    success_count = 0
    process_error_count = 0
    process_errors = []  # Errors encountered during processing.
    
    for result_file, result_item_dir, category in result_files:
        success, error_msg = process_result_file(
            result_file=result_file,
            result_item_dir=result_item_dir,
            category=category,
            data_root=data_root,
            result_root=result_root,
        )
        
        if success:
            success_count += 1
        else:
            process_error_count += 1
            process_errors.append((result_item_dir, error_msg))
            logging.error(f"  ✗ Failed: {error_msg}")
    
    # Print summary.
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total directories scanned: {len(result_files) + len(find_errors)}")
    print(f"  Directories with result files found: {len(result_files)}")
    print(f"  Directories without result yaml: {len(find_errors)}")
    print(f"  Directories successfully processed: {success_count}")
    print(f"  Directories failed during processing: {process_error_count}")
    print(f"{'='*60}")
    
    # Print directories without a result yaml file.
    if find_errors:
        print(f"\nDirectories without result yaml ({len(find_errors)}):")
        for item_dir, error_msg in find_errors:
            print(f"  ✗ {item_dir}")
            print(f"    Error: {error_msg}")
    
    # Print directories that failed processing (e.g. incomplete result yaml).
    if process_errors:
        print(f"\nDirectories failed during processing ({process_error_count}):")
        for item_dir, error_msg in process_errors:
            print(f"  ✗ {item_dir}")
            print(f"    Error: {error_msg}")
    
    total_errors = len(find_errors) + process_error_count
    if total_errors > 0:
        print(f"\nTotal errors: {total_errors}")
        sys.exit(1)


if __name__ == "__main__":
    main()
