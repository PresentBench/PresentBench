#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Iterable


DEFAULT_DATASET_REPO_ID = "PresentBench/PresentBench"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download benchmark dataset from a Hugging Face dataset repo into ./data"
    )
    p.add_argument(
        "--repo_id",
        type=str,
        default=DEFAULT_DATASET_REPO_ID,
        help=f"Hugging Face dataset repo id (default: {DEFAULT_DATASET_REPO_ID})",
    )
    p.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional git revision/tag for reproducibility (recommended)",
    )
    p.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Local data root directory (default: <repo_root>/data)",
    )
    p.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="Optional Hugging Face Hub endpoint. Use this to override HF_ENDPOINT (e.g. https://huggingface.co).",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=6,
        help="Number of retries on transient network errors (default: 6).",
    )
    p.add_argument(
        "--retry_backoff",
        type=float,
        default=1.5,
        help="Exponential backoff base in seconds between retries (default: 1.5).",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Per-request download timeout in seconds (sets HF_HUB_DOWNLOAD_TIMEOUT).",
    )
    p.add_argument(
        "--etag_timeout",
        type=int,
        default=None,
        help="Timeout in seconds for ETag/metadata requests (sets HF_HUB_ETAG_TIMEOUT).",
    )
    p.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Max parallel download workers for snapshot_download (default: huggingface_hub default).",
    )
    return p.parse_args()


def _default_data_root() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "data"


def _basic_integrity_check(data_root: Path) -> tuple[bool, str]:
    """
    Best-effort sanity check. We only verify that domain directories exist and contain
    domain-level config files that the evaluator expects.
    """
    data_root = Path(data_root).expanduser().resolve()
    try:
        from utils.material_utils import find_material_files
        from utils.paths import DATA_SOURCE_DIRS_REL, get_data_source_dirs, get_valid_item_dirs
    except Exception as e:
        return False, f"Failed to import utils.paths for integrity check: {e}"

    expected_domains = sorted({Path(rel).parts[0] for rel in DATA_SOURCE_DIRS_REL})
    missing_domains = [d for d in expected_domains if not (data_root / d).is_dir()]
    if missing_domains:
        return (
            False,
            "Missing expected domain directories under data_root: "
            + ", ".join(missing_domains)
            + f" (data_root={data_root})",
        )

    missing_files: list[str] = []
    for domain in expected_domains:
        w = data_root / domain / "judge_weights.yaml"
        c = data_root / domain / "common_judge_prompt.json"
        if not w.exists():
            missing_files.append(str(w))
        if not c.exists():
            missing_files.append(str(c))

    if missing_files:
        return False, "Missing expected domain-level files:\n" + "\n".join(missing_files)

    # Item-level checks: each data item must contain:
    # - generation prompt: generation_task/instructions.md
    # - judge prompt: generation_task/judge_prompt.json
    # - at least one material file (material.md/pdf or material_1.md/pdf, ... at item root)
    item_missing: list[str] = []
    checked_items = 0
    material_missing_count = 0
    expected_item_counts = {
        "academia": 91,
        "advertising": 16,
        "economics": 41,
        "education": 60,
        "talk": 30,
    }
    item_counts_by_domain: dict[str, int] = {k: 0 for k in expected_item_counts}

    try:
        for src_dir in get_data_source_dirs(base_dir=data_root):
            domain = src_dir.relative_to(data_root).parts[0]
            for item_dir in get_valid_item_dirs(src_dir):
                checked_items += 1
                if domain in item_counts_by_domain:
                    item_counts_by_domain[domain] += 1

                gen_prompt = item_dir / "generation_task" / "instructions.md"
                if not gen_prompt.exists():
                    item_missing.append(str(gen_prompt))

                judge_prompt = item_dir / "generation_task" / "judge_prompt.json"
                if not judge_prompt.exists():
                    item_missing.append(str(judge_prompt))

                if not find_material_files(item_dir):
                    material_missing_count += 1
                    item_missing.append(str(item_dir / "material(.md|.pdf|_N.md|_N.pdf)"))
    except Exception as e:
        return False, f"Failed during item-level integrity check: {e}"

    if item_missing:
        max_show = 80
        shown = item_missing[:max_show]
        more = len(item_missing) - len(shown)
        extra = f"\n... and {more} more" if more > 0 else ""
        return (
            False,
            "Missing expected item-level files.\n"
            f"- checked_items: {checked_items}\n"
            f"- missing_entries: {len(item_missing)}\n"
            f"- items_missing_material: {material_missing_count}\n"
            "Examples:\n"
            + "\n".join(shown)
            + extra,
        )

    count_mismatches = {
        k: (item_counts_by_domain.get(k, 0), v)
        for k, v in expected_item_counts.items()
        if item_counts_by_domain.get(k, 0) != v
    }
    if count_mismatches:
        lines = ["Evaluation instance count mismatch by domain:"]
        for k in sorted(expected_item_counts):
            got = item_counts_by_domain.get(k, 0)
            exp = expected_item_counts[k]
            delta = got - exp
            lines.append(f"- {k}: got {got}, expected {exp} (delta {delta:+d})")
        return False, "\n".join(lines)

    return True, f"OK (checked_items={checked_items})"
def _set_hf_timeouts(download_timeout_s: int | None, etag_timeout_s: int | None) -> None:
    if download_timeout_s is not None:
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(download_timeout_s)
    if etag_timeout_s is not None:
        os.environ["HF_HUB_ETAG_TIMEOUT"] = str(etag_timeout_s)


def _iter_transient_error_signatures() -> Iterable[str]:
    # Keep this conservative: only match common transient network symptoms.
    return (
        "handshake operation timed out",
        "timed out",
        "Temporary failure in name resolution",
        "Name or service not known",
        "Connection reset by peer",
        "Connection aborted",
        "Connection refused",
        "RemoteDisconnected",
        "TLSV1_ALERT",
        "EOF occurred in violation of protocol",
        "Network is unreachable",
        "ConnectionError",
        "ConnectError",
        "ReadTimeout",
    )


def _is_transient_download_error(err: BaseException) -> bool:
    msg = str(err)
    lower = msg.lower()
    for sig in _iter_transient_error_signatures():
        if sig.lower() in lower:
            return True
    return False


def _snapshot_download_with_retries(*, snapshot_download, repo_id: str, revision: str | None, local_dir: str, endpoint: str | None, max_workers: int | None, retries: int, retry_backoff: float) -> None:
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                local_dir=local_dir,
                endpoint=endpoint,
                max_workers=max_workers,
            )
            return
        except Exception as e:
            last_err = e
            if attempt >= retries or not _is_transient_download_error(e):
                raise
            sleep_s = min(60.0, retry_backoff * (2**attempt))
            print(
                f"Transient download error (attempt {attempt + 1}/{retries + 1}). "
                f"Sleeping {sleep_s:.1f}s then retrying...\n"
                f"Error: {e}",
                file=sys.stderr,
            )
            time.sleep(sleep_s)

    if last_err is not None:
        raise last_err


def main() -> int:
    args = _parse_args()

    # Ensure repository root is importable (so `import utils.*` works reliably).
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else _default_data_root()
    data_root.mkdir(parents=True, exist_ok=True)
    
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print(
            "Failed to import huggingface_hub. Did you install dependencies?\n"
            "Try: pip install -r requirements.txt\n"
            f"Import error: {e}",
            file=sys.stderr,
        )
        return 2

    endpoint = args.endpoint or os.environ.get("HF_ENDPOINT")
    print(f"Downloading dataset repo: {args.repo_id}")
    if args.revision:
        print(f"  revision: {args.revision}")
    if endpoint:
        print(f"  endpoint: {endpoint}")
    print(f"  -> local data_root: {data_root}")

    try:
        _set_hf_timeouts(args.timeout, args.etag_timeout)
        _snapshot_download_with_retries(
            snapshot_download=snapshot_download,
            repo_id=args.repo_id,
            revision=args.revision,
            local_dir=str(data_root),
            endpoint=endpoint,
            max_workers=args.max_workers,
            retries=max(0, int(args.retries)),
            retry_backoff=max(0.1, float(args.retry_backoff)),
        )
    except Exception as e:
        print(
            "Download failed.\n"
            "- If this is a gated dataset, run `huggingface-cli login` first.\n"
            "- If you are using a mirror endpoint, try: --endpoint https://huggingface.co\n"
            "- If you are behind a proxy, configure HF_HOME / HTTPS_PROXY accordingly.\n"
            "- If you see TLS/SSL handshake timeouts, try lowering parallelism: --max_workers 1\n"
            "- You can also raise timeouts: --timeout 120 --etag_timeout 60\n"
            f"Error: {e}",
            file=sys.stderr,
        )
        return 1

    ok, msg = _basic_integrity_check(data_root)
    if not ok:
        print("Downloaded, but integrity check failed:", file=sys.stderr)
        print(msg, file=sys.stderr)
        return 1

    print("Done. Dataset is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

