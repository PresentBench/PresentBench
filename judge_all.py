#!/usr/bin/env python3
"""
Batch runner for `judge.py`, executing all evaluations for a specified agent under test.

Usage:
    python judge_all.py --agent_name <agent_name> [--api_type <api_type>] [--model <model>] [--thinking_level <level>]

Examples:
    python judge_all.py --agent_name NotebookLM
    python judge_all.py --agent_name PPTAgent --api_type gemini --model gemini-3-pro-preview
"""
from __future__ import annotations

import io
import os
import sys
import logging
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from tqdm import tqdm  # type: ignore[import-not-found]
import argparse

import judge

from utils.paths import get_data_source_dirs, DATA_SOURCE_DIRS_REL
from utils.material_utils import find_material_files


def _is_data_root_ready(data_root: Path) -> tuple[bool, str]:
    """
    Best-effort check to provide actionable errors for open-source users.
    """
    if not data_root.exists():
        return False, f"data_root does not exist: {data_root}"
    if not data_root.is_dir():
        return False, f"data_root is not a directory: {data_root}"

    # Expect domain directories, e.g. academia/, advertising/, ...
    expected_domains = sorted({Path(rel).parts[0] for rel in DATA_SOURCE_DIRS_REL})

    # Domain-level files required by this runner/judge.
    missing = []
    for d in expected_domains:
        w = data_root / d / "judge_weights.yaml"
        c = data_root / d / "common_judge_prompt.json"
        if not w.exists():
            missing.append(str(w))
        if not c.exists():
            missing.append(str(c))
    if missing:
        return False, "Missing required domain-level files:\n" + "\n".join(missing)

    return True, "OK"


def _print_data_download_help(data_root: Path) -> None:
    repo_root = Path(__file__).resolve().parent
    script = repo_root / "scripts" / "download_data.py"
    print("\nBenchmark data not found or incomplete.", file=sys.stderr)
    print(f"- data_root: {data_root}", file=sys.stderr)
    print("\nTo download the dataset into ./data, run:", file=sys.stderr)
    print(
        f"  {sys.executable} {script} --repo_id PresentBench/PresentBench",
        file=sys.stderr,
    )
    print("\nAfter download, re-run your command.", file=sys.stderr)


def get_type_name(data_source_dir: Path, data_root: Path) -> str:
    """
    Extract `type_name` (the first-level category) from a data source directory path.
    Example: data/academic/CVPR_2023 -> academic
    """
    # Path relative to the data root.
    # Use real paths so that symlinks like `data -> PresentBench_dataset`
    # are treated as the same tree and do not trigger ValueError.
    try:
        rel_path = data_source_dir.resolve().relative_to(data_root.resolve())
    except ValueError:
        # Fallback to the original behavior (helps surface real mismatches)
        rel_path = data_source_dir.relative_to(data_root)
    return rel_path.parts[0]
 

def collect_all_test_cases(data_root: Path) -> list[tuple[str, Path]]:
    """
    Collect all test cases that should be evaluated.
    Returns: list of (type_name, base_dir) tuples.
    """
    test_cases = []
    data_source_dirs = get_data_source_dirs(data_root)
    
    for data_source_dir in data_source_dirs:
        if not data_source_dir.exists():
            continue
        
        type_name = get_type_name(data_source_dir, data_root)
        
        # Iterate over all subdirectories under the data source directory.
        for subdir in data_source_dir.iterdir():
            if not subdir.is_dir():
                continue
            
            # Check whether required files exist.
            material_files = find_material_files(subdir)
            if not material_files:
                continue  # No material files -> skip this case.
            
            judge_prompt = subdir / "generation_task" / "judge_prompt.json"
            if not judge_prompt.exists():
                continue
            
            # This is a valid test case.
            test_cases.append((type_name, subdir))
    
    return test_cases


def run_once(
    api_type: str,
    model: str,
    thinking_level: str | None,
    type_name: str,
    data_item_dir: Path,
    result_root: Path,
    data_root: Path,
    debug: bool = False,
    min_timestamp: str | None = None,
) -> tuple[str, Path, str | None]:
    """
    Run one test case evaluation. Calls `judge.main()`.
    
    Args:
        api_type: API type.
        model: Model name.
        thinking_level: Thinking level.
        type_name: Type/category name (used to locate weights_path).
        data_item_dir: Subdirectory under the data root (e.g. advertisement/Apple_iPhone/iPhone_17_Pro).
        result_root: Root directory for the agent-under-test results.
        data_root: Data root directory.
    
    Returns:
        Tuple of (type_name, data_item_dir, error_message).
        On success, error_message is None; on failure, it contains the error details.
    """
    # Ensure Path objects are absolute (needed for multiprocessing).
    data_item_dir = Path(data_item_dir).resolve()
    result_root = Path(result_root).resolve()
    data_root = Path(data_root).resolve()
    
    # Switch to data_root to ensure relative paths resolve correctly.
    original_cwd = os.getcwd()
    try:
        os.chdir(str(data_root))

        def _run_judge_with_logging(judge_args: argparse.Namespace, log_dir: Path, error_context: str) -> str | None:
            """
            Shared wrapper: configure logging and call judge.main(); return an error message (None if no error).
            """
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_file = log_dir / f"{timestamp}.log"

            log_handler = logging.FileHandler(str(log_file), encoding="utf-8")
            log_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            )

            root_logger = logging.getLogger()
            root_logger.setLevel(logging.INFO)
            root_logger.handlers.clear()
            root_logger.addHandler(log_handler)

            import sys

            # Ensure data_root is on sys.path so packages like `academia`, `advertising`, etc. are importable.
            data_root_str = str(data_root)
            added_to_syspath = False

            try:
                if data_root_str not in sys.path:
                    sys.path.insert(0, data_root_str)
                    added_to_syspath = True

                # Call the already-imported judge.main to avoid re-loading the module each time.
                judge.main(args=judge_args)
                return None
            except Exception as e:
                logging.exception(f"Error when {error_context}: {e}")
                return str(e)
            finally:
                if added_to_syspath and data_root_str in sys.path:
                    sys.path.remove(data_root_str)
                root_logger.removeHandler(log_handler)
                log_handler.close()
        
        # material and judge_prompt are taken from the data item directory.
        material_files = find_material_files(data_item_dir)
        if not material_files:
            return (type_name, data_item_dir, f"No material files found in {data_item_dir}")
        
        judge_prompt = data_item_dir / "generation_task" / "judge_prompt.json"
        # Domain-level common_judge_prompt.json is loaded from data_root by type_name.
        common_judge_prompt = data_root / type_name / "common_judge_prompt.json"
        
        # Slides are taken from the agent-under-test directory (same relative layout).
        # Compute the path relative to data_root.
        rel_path = data_item_dir.relative_to(data_root)
        agent_slides_dir = result_root / rel_path / "generation_task" / "results"
        
        # Prefer slides.pdf; fall back to slides.pptx.
        slides = agent_slides_dir / "slides.pdf"
        if not slides.exists():
            slides = agent_slides_dir / "slides.pptx"
        
        # If slides were not generated but slides_generation_failed.txt exists, write an all-zero score directly.
        if not slides.exists():
            failed_flag = agent_slides_dir / "slides_generation_failed.txt"
            if failed_flag.exists():
                # Generate an all-zero score file without running evaluation.
                weights_path = data_root / type_name / "judge_weights.yaml"

                judge_args = argparse.Namespace(
                    api_type=api_type,
                    model=model,
                    thinking_level=thinking_level,
                    slides=str(slides),
                    judge_prompt=str(judge_prompt),
                    weights_path=str(weights_path),
                    material=[str(material_file) for material_file in material_files],
                    output=None,
                    retry=5,
                    debug=debug,
                    zero_score=True,
                    min_timestamp=min_timestamp,
                )

                error_msg = _run_judge_with_logging(
                    judge_args=judge_args,
                    log_dir=agent_slides_dir / "log",
                    error_context="writing zero score",
                )
                return (type_name, data_item_dir, error_msg)

            # No slides and no failure marker -> keep the original error behavior.
            return (type_name, data_item_dir, f"Slides not found: {slides}")
        
        weights_path = data_root / type_name / "judge_weights.yaml"
        
        # Log directory lives next to the agent slides output.
        log_dir = slides.parent / "log"

        # Build argparse.Namespace to pass into judge.main().
        judge_args = argparse.Namespace(
            api_type=api_type,
            model=model,
            thinking_level=thinking_level,
            slides=str(slides),
            judge_prompt=str(judge_prompt),
            common_judge_prompt=str(common_judge_prompt),
            weights_path=str(weights_path),
            material=[str(material_file) for material_file in material_files],
            output=None,  # Use default output path.
            retry=5,
            debug=debug,  # Allow debug in single-process mode.
            min_timestamp=min_timestamp,
        )

        error_msg = _run_judge_with_logging(
            judge_args=judge_args,
            log_dir=log_dir,
            error_context="running judge.main",
        )
        return (type_name, data_item_dir, error_msg)
    finally:
        # Restore original working directory.
        os.chdir(original_cwd)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch evaluation script")
    parser.add_argument("--agent_name", type=str, required=True, help="Name of the agent under test")
    parser.add_argument("--api_type", type=str, default="gemini", help="API type")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview", help="Model name")
    parser.add_argument("--thinking_level", type=str, default=None, help="Thinking level")
    parser.add_argument("--data_root", type=str, default=None, help="Data root directory (defaults to the repo's data/ directory)")
    parser.add_argument("--result_root", type=str, default=None, help="Root directory of slides to evaluate (default: [repo_root]/results/agent_name)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode: call judge.py directly (no subprocess), supports ipdb.set_trace(), etc.")
    parser.add_argument("--max_workers", type=int, default=None, help="Maximum worker processes (default: CPU count; forced to 1 in debug mode)")
    parser.add_argument("--min_timestamp", type=str, default=None, help="Minimum timestamp for resume (format: YYYY-MM-DD_HH-MM-SS). Result files with timestamps older than this time will be skipped.")
    
    args = parser.parse_args()
    
    # Evaluation data root.
    if args.data_root:
        data_root = Path(args.data_root).expanduser().resolve()
    else:
        repo_root = Path(__file__).resolve().parent
        data_root = repo_root / "data"
    
    ok, msg = _is_data_root_ready(data_root)
    if not ok:
        print(msg, file=sys.stderr)
        _print_data_download_help(data_root)
        sys.exit(1)

    # Collect all test cases.
    print("Collecting test cases...")
    test_cases = collect_all_test_cases(data_root)
    total = len(test_cases)

    if total != 238:
        print(f"The number of test cases should be 238, but found {total}.")
        _print_data_download_help(data_root)
        sys.exit(1)
        
    print(f"Found {total} test cases")
    
    # Agent-under-test results root.
    if args.result_root:
        result_root = Path(args.result_root).expanduser().resolve()
    else:
        result_root = Path("results").resolve() / args.agent_name
    result_root.mkdir(parents=True, exist_ok=True)
    
    # Determine concurrency.
    if args.debug:
        max_workers = 1  # Force single process in debug mode.
        print("Debug mode: using single process")
    else:
        max_workers = args.max_workers
        if max_workers is None:
            import multiprocessing
            max_workers = multiprocessing.cpu_count()
        print(f"Using {max_workers} worker process(es)")
    
    # Use tqdm for a progress bar.
    failed_cases = []
    
    if max_workers == 1:
        # Single-process mode (serial execution, supports debugging).
        with tqdm(total=total, desc="Judging", unit="case") as pbar:
            for type_name, base_dir in test_cases:
                pbar.set_postfix({"current": base_dir.name})
                try:
                    error_msg = run_once(
                        api_type=args.api_type,
                        model=args.model,
                        thinking_level=args.thinking_level,
                        type_name=type_name,
                        data_item_dir=base_dir,
                        result_root=result_root,
                        data_root=data_root,
                        debug=args.debug,  # Pass debug flag in single-process mode.
                        min_timestamp=args.min_timestamp,
                    )[2]  # Get error message.
                    if error_msg:
                        failed_cases.append((type_name, base_dir, error_msg))
                        print(f"\nError processing {base_dir}: {error_msg}", file=sys.stderr)
                except Exception as e:
                    failed_cases.append((type_name, base_dir, str(e)))
                    print(f"\nError processing {base_dir}: {e}", file=sys.stderr)
                finally:
                    pbar.update(1)
    else:
        # Multi-process mode (parallel execution).
        with tqdm(total=total, desc="Judging", unit="case") as pbar:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks.
                future_to_case = {
                    executor.submit(
                        run_once,
                        args.api_type,
                        args.model,
                        args.thinking_level,
                        type_name,
                        base_dir,
                        result_root,
                        data_root,
                        False,  # Do not use debug in multi-process mode.
                        args.min_timestamp,
                    ): (type_name, base_dir)
                    for type_name, base_dir in test_cases
                }
                
                # Collect results.
                for future in as_completed(future_to_case):
                    type_name, base_dir = future_to_case[future]
                    pbar.set_postfix({"current": base_dir.name})
                    try:
                        result_type_name, result_base_dir, error_msg = future.result()
                        if error_msg:
                            failed_cases.append((result_type_name, result_base_dir, error_msg))
                            print(f"\nError processing {result_base_dir}: {error_msg}", file=sys.stderr)
                    except Exception as e:
                        failed_cases.append((type_name, base_dir, str(e)))
                        print(f"\nError processing {base_dir}: {e}", file=sys.stderr)
                    finally:
                        pbar.update(1)
    if len(failed_cases) == 0:
        # If all cases succeeded, compute average scores.
        print("\nAll cases succeeded, calculating average scores...")
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "utils.statistics.calculate_average_scores",
                    "--result_root_dir",
                    str(result_root),
                    "--prefer_newest",
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Failed to calculate average scores: {e}", file=sys.stderr)
    
    # Print summary.
    print(f"\n{'='*60}")
    print(f"Total: {total}")
    print(f"Success: {total - len(failed_cases)}")
    print(f"Failed: {len(failed_cases)}")
    
    if failed_cases:
        print("\nFailed cases:")
        for type_name, base_dir, error in failed_cases:
            print(f"  - {base_dir}: {error}")
        sys.exit(1)
