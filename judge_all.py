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
from collections.abc import Iterator
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from tqdm import tqdm  # type: ignore[import-not-found]
import argparse

import judge

from utils.paths import get_data_source_dirs, DATA_SOURCE_DIRS_REL
from utils.material_utils import find_material_files
from utils.statistics.calculate_average_scores import process_scoring_method


logger = logging.getLogger(__name__)


@contextmanager
def _per_case_file_log(log_dir: Path) -> Iterator[Path]:
    """
    Attach a per-case FileHandler to the root logger for the duration of the
    `with` block. Any pre-existing handlers (e.g. the stdout StreamHandler
    configured at program startup) are preserved, so logs are also visible on
    the console in single-process/debug mode.

    In multi-process mode, worker processes inherit the root logger state
    from the parent (under `fork`), or start fresh (under `spawn`); either
    way, the per-case FileHandler added here is only active inside the
    `with` block in the current process.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"{timestamp}.log"

    handler = logging.FileHandler(str(log_file), encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        yield log_file
    finally:
        root_logger.removeHandler(handler)
        handler.close()


@contextmanager
def _sys_path_prepended(path: str) -> Iterator[None]:
    """Temporarily prepend `path` to sys.path (no-op if already present)."""
    added = False
    if path not in sys.path:
        sys.path.insert(0, path)
        added = True
    try:
        yield
    finally:
        if added:
            try:
                sys.path.remove(path)
            except ValueError:
                pass



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


def _largest_remainder_allocate(
    # Items to allocate a quota over, each with an integer "size" (capacity).
    # List of (key, size) pairs; keys are used only for deterministic tie-breaking.
    items: list[tuple[str, int]],
    quota: int,
) -> dict[str, int]:
    """
    Allocate ``quota`` units across ``items`` proportionally to each item's
    size, using the Hamilton / Largest Remainder Method, with the constraint
    that no item receives more than its own size (capacity).

    Guarantees:
    - sum of returned values == quota (as long as 0 <= quota <= sum(sizes))
    - |alloc[i]/quota - size[i]/total| is minimized subject to capacity
    - deterministic under ties: earlier ``items`` (lexicographic key order
      passed by caller) win ties.
    """
    total_size = sum(s for _, s in items)
    if quota <= 0 or total_size <= 0:
        return {k: 0 for k, _ in items}
    if quota >= total_size:
        return {k: s for k, s in items}

    # Initial floor allocation.
    alloc: dict[str, int] = {}
    remainders: list[tuple[float, int, str]] = []  # (-remainder, tiebreak_idx, key)
    allocated = 0
    for idx, (key, size) in enumerate(items):
        exact = size * quota / total_size
        base = int(exact)  # floor
        # Cap at capacity (shouldn't exceed since quota < total_size, but be safe).
        base = min(base, size)
        alloc[key] = base
        allocated += base
        remainders.append((-(exact - base), idx, key))

    # Distribute the remainder one by one to items with the largest fractional
    # part, skipping any that are already at capacity.
    remaining = quota - allocated
    # Sort by (-remainder, idx) -> largest remainder first, stable by input order.
    remainders.sort()
    i = 0
    while remaining > 0 and i < len(remainders):
        _, _, key = remainders[i]
        i += 1
        size = dict(items)[key]
        if alloc[key] < size:
            alloc[key] += 1
            remaining -= 1

    # If we still have leftover quota (all picks were capped), loop again
    # giving one more unit to any item with spare capacity, in input order.
    if remaining > 0:
        for key, size in items:
            while remaining > 0 and alloc[key] < size:
                alloc[key] += 1
                remaining -= 1
            if remaining == 0:
                break

    return alloc


def _stratified_sample_by_subcategory(
    test_cases: list[tuple[str, Path]],
    limit: int,
    data_root: Path,
) -> list[tuple[str, Path]]:
    """
    Two-level stratified sampling that returns EXACTLY ``limit`` cases.

    Directory layout: data_root / <domain> / <subcategory> / <test case>
    - ``type_name`` in each tuple is the domain (first-level category).
    - The subcategory is the immediate parent directory of the test case dir.

    Priorities (in order):
    1. Balance across domains:     allocate the ``limit`` quota across the 5
       domains via the Largest Remainder Method, proportional to each
       domain's case count.
    2. Balance across subcategories within each domain: apply the Largest
       Remainder Method a second time inside each domain, proportional to
       each subcategory's case count.
    3. Within each subcategory, pick the first ``k`` cases in dictionary
       order of path.

    This guarantees ``len(result) == min(limit, total)`` exactly, while
    keeping inter-domain and intra-domain distributions as balanced as the
    integer constraints allow. Results are returned sorted by
    ``(type_name, path)`` for deterministic execution order.
    """
    total = len(test_cases)
    if total == 0 or limit <= 0:
        return []
    if limit >= total:
        return sorted(test_cases, key=lambda tc: (tc[0], str(tc[1])))

    # ---- Group by domain, then by subcategory ----
    # domain_name -> subcat_key -> list of (type_name, base_dir)
    domains: dict[str, dict[str, list[tuple[str, Path]]]] = {}

    def _subcat_key(p: Path) -> str:
        """Path relative to data_root (for stable, human-readable keys)."""
        try:
            return str(p.resolve().relative_to(data_root.resolve()))
        except ValueError:
            return str(p)

    for type_name, base_dir in test_cases:
        subcat_dir = base_dir.parent
        skey = _subcat_key(subcat_dir)
        domains.setdefault(type_name, {}).setdefault(skey, []).append(
            (type_name, base_dir)
        )

    # ---- Level 1: allocate quota across domains ----
    domain_items: list[tuple[str, int]] = sorted(
        ((d, sum(len(v) for v in subs.values())) for d, subs in domains.items()),
        key=lambda x: x[0],
    )
    domain_alloc = _largest_remainder_allocate(domain_items, limit)

    # ---- Level 2: within each domain, allocate across subcategories ----
    sampled: list[tuple[str, Path]] = []
    for domain, _domain_size in domain_items:
        d_quota = domain_alloc[domain]
        if d_quota <= 0:
            continue
        subs = domains[domain]
        subcat_items: list[tuple[str, int]] = sorted(
            ((skey, len(cases)) for skey, cases in subs.items()),
            key=lambda x: x[0],
        )
        subcat_alloc = _largest_remainder_allocate(subcat_items, d_quota)

        for skey, _subcat_size in subcat_items:
            k = subcat_alloc[skey]
            if k <= 0:
                continue
            cases = sorted(subs[skey], key=lambda tc: str(tc[1]))
            sampled.extend(cases[:k])

    return sorted(sampled, key=lambda tc: (tc[0], str(tc[1])))


def _invoke_judge(
    judge_args: argparse.Namespace,
    log_dir: Path,
    data_root: Path,
    error_context: str,
) -> str | None:
    """
    Call `judge.main(judge_args)` with:
      - a per-case FileHandler attached to the root logger (log_dir/<timestamp>.log),
      - `data_root` temporarily on sys.path so package-style imports work.
    Returns None on success, or the stringified error on failure.
    """
    with _per_case_file_log(log_dir), _sys_path_prepended(str(data_root)):
        try:
            judge.main(args=judge_args)
            return None
        except Exception as e:
            logger.exception("Error when %s: %s", error_context, e)
            return str(e)


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
    retry: int = 5,
    temperature: float | None = None,
    seed: int | None = None,
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

    # material and judge_prompt are taken from the data item directory.
    material_files = find_material_files(data_item_dir)
    if not material_files:
        return (type_name, data_item_dir, f"No material files found in {data_item_dir}")

    judge_prompt = data_item_dir / "generation_task" / "judge_prompt.json"
    # Domain-level common_judge_prompt.json is loaded from data_root by type_name.
    common_judge_prompt = data_root / type_name / "common_judge_prompt.json"

    # Slides are taken from the agent-under-test directory (same relative layout).
    rel_path = data_item_dir.relative_to(data_root)
    agent_slides_dir = result_root / rel_path / "generation_task" / "results"

    # Prefer slides.pdf; fall back to slides.pptx.
    slides = agent_slides_dir / "slides.pdf"
    if not slides.exists():
        slides = agent_slides_dir / "slides.pptx"

    weights_path = data_root / type_name / "judge_weights.yaml"

    # If slides were not generated but slides_generation_failed.txt exists,
    # write an all-zero score directly (no evaluation).
    if not slides.exists():
        failed_flag = agent_slides_dir / "slides_generation_failed.txt"
        if not failed_flag.exists():
            return (type_name, data_item_dir, f"Slides not found: {slides}")

        judge_args = argparse.Namespace(
            api_type=api_type,
            model=model,
            thinking_level=thinking_level,
            slides=str(slides),
            judge_prompt=str(judge_prompt),
            weights_path=str(weights_path),
            material=[str(f) for f in material_files],
            output=None,
            output_dir=None,
            retry=retry,
            debug=debug,
            zero_score=True,
            min_timestamp=min_timestamp,
            temperature=temperature,
            seed=seed,
        )
        err = _invoke_judge(
            judge_args=judge_args,
            log_dir=agent_slides_dir / "log",
            data_root=data_root,
            error_context="writing zero score",
        )
        return (type_name, data_item_dir, err)

    judge_args = argparse.Namespace(
        api_type=api_type,
        model=model,
        thinking_level=thinking_level,
        slides=str(slides),
        judge_prompt=str(judge_prompt),
        common_judge_prompt=str(common_judge_prompt),
        weights_path=str(weights_path),
        material=[str(f) for f in material_files],
        output=None,
        output_dir=None,
        retry=retry,
        debug=debug,
        min_timestamp=min_timestamp,
        temperature=temperature,
        seed=seed,
    )
    err = _invoke_judge(
        judge_args=judge_args,
        log_dir=slides.parent / "log",
        data_root=data_root,
        error_context="running judge.main",
    )
    return (type_name, data_item_dir, err)


if __name__ == "__main__":
    import argparse

    # Configure root logger once for the main process. Per-case file logs are
    # additionally attached inside `_run_judge_with_logging` without touching
    # this stdout handler, so in single-process/debug mode logs are visible
    # on the console AND written to each case's log file. In multi-process
    # mode, worker processes start with a fresh (handler-less) root logger,
    # so only the per-case FileHandler is active there.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Batch evaluation script")
    parser.add_argument("--agent_name", type=str, required=True, help="Name of the agent under test")
    parser.add_argument("--api_type", type=str, default="gemini", help="API type (gemini, gemini_inline)")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview", help="Model name")
    parser.add_argument("--thinking_level", type=str, default=None, help="Thinking level")
    parser.add_argument("--data_root", type=str, default=None, help="Data root directory (defaults to the repo's data/ directory)")
    parser.add_argument("--result_root", type=str, default=None, help="Root directory of slides to evaluate (default: [repo_root]/results/agent_name)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode: call judge.py directly (no subprocess), supports ipdb.set_trace(), etc.")
    parser.add_argument("--max_workers", type=int, default=None, help="Maximum worker processes (default: CPU count; forced to 1 in debug mode)")
    parser.add_argument("--min_timestamp", type=str, default=None, help="Minimum timestamp for resume (format: YYYY-MM-DD_HH-MM-SS). Result files with timestamps older than this time will be skipped.")
    parser.add_argument("--retry", type=int, default=5, help="Number of retries for each judge item (default: 5)")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "Judge sampling temperature. Default: not sent (the API server "
            "default decoding is used, i.e. the original behaviour). Set to 0 "
            "for the most reproducible verdicts."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Base judge random seed, forwarded to backends that support it. "
            "Default: not sent."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "If set to a positive integer N, run EXACTLY N test cases selected "
            "by two-level stratified sampling (dataset layout: domain/subcategory/case): "
            "(1) the N-quota is first allocated across the 5 domains proportional "
            "to each domain's size (Largest Remainder Method), so domains are as "
            "balanced as integer constraints allow; (2) each domain's sub-quota is "
            "then allocated across its subcategories the same way; (3) within each "
            "subcategory the first k cases in dictionary order are taken. "
            "Useful for smoke tests. Default: run all cases."
        ),
    )
    
    args = parser.parse_args()

    # Resolve effective fixed-decoding settings for the judge. By default
    # neither temperature nor seed is sent (the API server default decoding is
    # used, i.e. the original behaviour); pass --temperature/--seed to fix them.
    eff_temperature = args.temperature
    eff_seed = args.seed

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
    total_found = len(test_cases)

    if total_found != 238:
        print(f"The number of test cases should be 238, but found {total_found}.")
        _print_data_download_help(data_root)
        sys.exit(1)

    print(f"Found {total_found} test cases")

    # Optionally restrict to exactly N cases via two-level stratified sampling.
    if args.limit is not None:
        if args.limit <= 0:
            print(f"--limit must be a positive integer, got {args.limit}", file=sys.stderr)
            sys.exit(2)
        if args.limit < total_found:
            test_cases = _stratified_sample_by_subcategory(
                test_cases, args.limit, data_root
            )
            # Report per-domain allocation for transparency.
            from collections import Counter
            per_domain = Counter(tc[0] for tc in test_cases)
            dist = ", ".join(f"{d}={per_domain[d]}" for d in sorted(per_domain))
            print(
                f"--limit={args.limit}: two-level stratified sampling selected "
                f"exactly {len(test_cases)} case(s) out of {total_found} "
                f"[per-domain: {dist}]"
            )
        else:
            # Still sort for a deterministic running order.
            test_cases = sorted(test_cases, key=lambda tc: (tc[0], str(tc[1])))
            print(f"--limit={args.limit} >= total ({total_found}); running all cases")

    total = len(test_cases)

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
                        retry=args.retry,
                        temperature=eff_temperature,
                        seed=eff_seed,
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
                        args.retry,
                        eff_temperature,
                        eff_seed,
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
        judge_model_filter = judge._model_filename_prefix(args.model, args.thinking_level)
        print(
            f"\nAll cases succeeded, calculating average scores "
            f"(judge_model={judge_model_filter})..."
        )
        try:
            process_scoring_method(
                result_root_dir=result_root,
                scoring_method="ours",
                prefer_newest=True,
                judge_model=judge_model_filter,
            )
        except Exception as e:
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
