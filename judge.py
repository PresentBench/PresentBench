"""
Uasge example:

python judge.py \
    --api_type gemini \
    --model gemini-3-flash-preview \
    --slides /path/to/slides.pdf \
    --material /path/to/material.md \
    --judge_prompt /path/to/judge_prompt.json \
    --common_judge_prompt /path/to/common_judge_prompt.json \
    --weights_path /path/to/judge_weights.yaml
"""

import os
import argparse
import importlib.util
import re
from pathlib import Path
import yaml
import json
import logging
import time
import threading
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from scoring import compute_and_save_scores, load_config
from dataclasses import dataclass
from typing import Optional, List, Any, TYPE_CHECKING, Callable
from enum import Enum

from utils.api.base import GeminiAPI, GeminiInlineAPI, OpenAIChatAPI
from utils.api.judge_api import JudgeAPI
from utils.pptx_to_pdf import convert_pptx_to_pdf
from utils.count_pages import count_pages
from utils.truncate_pages import truncate_slides
from utils.judge_utils import find_resume_yaml, load_judge_prompt, is_result_complete, load_existing_results
from utils.score_utils import find_single_score_yaml, model_filename_prefix
from functools import partial

if TYPE_CHECKING:
    from typing import Protocol


# Module-level logger. Logging configuration (handlers/level/format) is set up
# only at the application entry point (see `if __name__ == '__main__':` below)
# or by the caller when `judge.main()` is invoked as a library (e.g. from
# `judge_all.py`). Do NOT call `logging.basicConfig(...)` from within library
# code, otherwise the caller's logging setup can be silently shadowed.
logger = logging.getLogger(__name__)



class JudgeType(str, Enum):
    """Judge type enum (split by whether material is required)."""
    MATERIAL_INDEPENDENT = "material_independent"
    MATERIAL_DEPENDENT = "material_dependent"



@dataclass
class JudgeContext:
    """Holds the context required to run judging."""
    judge_api: JudgeAPI  # Use JudgeAPI rather than BaseAPI directly.
    model: str
    thinking_level: Optional[str]
    slides_obj: Optional[Any]  # File object (Gemini API)
    material_objs: Optional[List[Any]]  # List of file objects (Gemini API)
    slides_path: Optional[str]  # File path (OpenAI-like API)
    material_paths: Optional[List[str]]  # List of file paths (OpenAI-like API)
    judge_type: JudgeType = JudgeType.MATERIAL_INDEPENDENT
    max_retries: int = 5
    debug: bool = False
    save_interval: int = 20  # Save after finishing this many items.
    # Fixed-decoding settings for the judge model. ``None`` means "let the API
    # server decide" (original behaviour); ``temperature=0`` (optionally with a
    # ``seed``) makes verdicts as deterministic / reproducible as the backend allows.
    temperature: Optional[float] = None
    seed: Optional[int] = None


def create_judge_api(
    api_type: str,
) -> JudgeAPI:
    """Create an API client based on api_type."""
    if api_type == 'gemini_inline':
        api_key = os.getenv("GENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "GENAI_API_KEY environment variable is not set. "
                "Please configure it in .env file."
            )
        api_client = GeminiInlineAPI(
            api_key=api_key,
            **(
                dict(http_options={"": os.getenv("GENAI_BASE_URL")})
                if os.getenv("GENAI_BASE_URL")
                else dict()
            ),
        )
    elif api_type == 'gemini':
        api_key = os.getenv("GENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "GENAI_API_KEY environment variable is not set. "
                "Please configure it in .env file."
            )
        api_client = GeminiAPI(
            api_key=api_key,
            **(
                dict(http_options={"base_url": os.getenv("GENAI_BASE_URL")})
                if os.getenv("GENAI_BASE_URL")
                else dict()
            ),
        )
    elif api_type == 'minimax':
        api_key = os.getenv("MINIMAX_API_KEY")
        if not api_key:
            raise ValueError(
                "MINIMAX_API_KEY environment variable is not set. "
                "Please configure it in .env file."
            )
        api_client = OpenAIChatAPI(
            api_key=api_key,
            base_url=os.getenv(
                "MINIMAX_BASE_URL",
                "https://api.minimaxi.com/v1/chat/completions",
            ),
            reasoning_split=True,
        )
    else:
        raise ValueError(f"Unsupported API type: {api_type}")
    
    return JudgeAPI(api_client)



def _model_filename_prefix(model: str, thinking_level: str | None = None) -> str:
    """兼容旧调用方，并统一使用跨平台安全的模型文件名前缀。"""
    return model_filename_prefix(model, thinking_level)


# ------------------------------------------------------------------------------
# Resume aliases
# ------------------------------------------------------------------------------
# Some model names refer to the same underlying judge model but are routed
# through different API providers. When the current model is one
# of these aliases, we also allow resuming from existing result/score files
# produced by any of the sibling aliases.
_RESUME_ALIAS_GROUPS: tuple[frozenset[str], ...] = ()


def _resume_alias_prefixes(model_prefix: str) -> list[str]:
    """Return extra filename prefixes that ``model_prefix`` is allowed to resume from.

    The returned list never contains ``model_prefix`` itself; it only lists the
    *additional* aliases. Returns an empty list if no alias group matches.
    """
    for group in _RESUME_ALIAS_GROUPS:
        if model_prefix in group:
            return sorted(m for m in group if m != model_prefix)
    return []


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate content using Gemini API with slides and material files'
    )
    parser.add_argument(
        '--api_type', type=str, required=True,
        choices=['gemini', 'gemini_inline', 'minimax'],
        help='API type to use: gemini, gemini_inline, or minimax'
    )
    parser.add_argument(
        '--model', type=str, default='gemini-3-flash-preview',
        help='Model name to use (default: gemini-3-flash-preview)'
    )
    parser.add_argument(
        '--thinking_level', type=str,
        help='Thinking level to use (default: None)'
    )
    parser.add_argument('--slides', type=str, help='Path to slides file')
    parser.add_argument('--material', type=str, nargs='+', help='Path(s) to material file(s). Can specify multiple files.')
    parser.add_argument(
        '--judge_prompt', type=str, default='',
        help='Path to judge prompt file'
    )
    parser.add_argument(
        '--common_judge_prompt', type=str, default='',
        help='Path to common judge prompt file (domain-level)'
    )
    parser.add_argument(
        '--weights_path', type=str,
        help='Path to weights YAML file for scoring'
    )
    parser.add_argument(
        '--output', type=str,
        help='Path to output YAML file for saving judge results'
    )
    parser.add_argument(
        '--output_dir', type=str,
        help='Directory to write result/score files into, using the default '
             'timestamped filenames. Overrides the default location (next to '
             '--slides). Handy for isolating repeated evaluations (see '
             'judge_all.py --repeats). Ignored when --output is given.'
    )
    parser.add_argument(
        '--temperature', type=float, default=None,
        help='Sampling temperature for the judge model. Default: not sent '
             '(the API server default decoding is used, i.e. the original '
             'behaviour). Set to 0 for the most deterministic / reproducible '
             'verdicts.'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for the judge model, forwarded to backends that '
             'support it. Default: not sent. Note: thinking models may not '
             'honour the seed strictly.'
    )
    parser.add_argument(
        '--retry', type=int, default=5,
        help='Number of retries for each judge item (default: 5)'
    )
    parser.add_argument(
        '--debug', action='store_true', 
        help='Enable debug mode: directly call main() function instead of using subprocess',
    )
    parser.add_argument(
        '--zero_score',
        action='store_true',
        help='If set, skip all judging and file reading, and directly save an all-zero score YAML file to --output',
    )
    parser.add_argument(
        '--min_timestamp', type=str,
        help='Minimum timestamp for resume (format: YYYY-MM-DD_HH-MM-SS). Result files with timestamps older than this time will be skipped.',
    )
    return parser.parse_args()


def validate_args(args):
    """Validate command-line arguments."""
    if not args.judge_prompt:
        raise ValueError("--judge_prompt is required")
    if not os.path.exists(args.judge_prompt):
        raise FileNotFoundError(f"Judge prompt file not found: {args.judge_prompt}")
    
    if not args.common_judge_prompt:
        raise ValueError("--common_judge_prompt is required")
    if not os.path.exists(args.common_judge_prompt):
        raise FileNotFoundError(f"Common judge prompt file not found: {args.common_judge_prompt}")
    
    if not args.slides:
        raise ValueError("--slides is required for source-dependent judge mode")
    if not os.path.exists(args.slides):
        raise FileNotFoundError(f"Slides file not found: {args.slides}")
    
    if not args.material:
        raise ValueError("--material is required for source-dependent judge mode")
    for material_file in args.material:
        if not os.path.exists(material_file):
            raise FileNotFoundError(f"Material file not found: {material_file}")


# Common LaTeX text/format wrappers the judge occasionally emits *inside*
# \boxed{...}, e.g. \boxed{\text{no}}, \boxed{\textbf{Yes}}, \boxed{\mathrm{no}}.
# The naive capture group ([^}]+) stops at the first '}', so such answers are
# captured as e.g. '\text{no' -- neither None nor yes/no -- and used to be
# accepted as "answered" then silently dropped at scoring time.
_LATEX_WRAPPER_RE = re.compile(r'\\[a-zA-Z]+\s*')


def _normalize_boxed_token(raw: str) -> str:
    r"""Normalize the raw text captured inside ``\boxed{...}`` to a bare token.

    Strips common LaTeX formatting commands (``\text``, ``\textbf``,
    ``\mathrm``, ...) and any stray braces / surrounding punctuation, so
    wrapped verdicts such as ``\text{no`` or ``\textbf{Yes}`` normalize back to
    ``no`` / ``Yes``. Plain tokens (``yes``, ``no``) are returned unchanged
    apart from whitespace/period trimming.
    """
    s = raw.strip()
    s = _LATEX_WRAPPER_RE.sub('', s)          # drop \cmd wrappers (e.g. '\text')
    s = s.replace('{', '').replace('}', '')   # drop any leftover braces
    s = s.strip().strip('.').strip()          # drop surrounding spaces / trailing period
    return s


def process_single_item(item_info, context: JudgeContext):
    """
    Process a single checklist item.

    item_info: tuple (i, j, item) where i is the sublist index, j is the index within the sublist,
               and item is the checklist entry to process.
    context: JudgeContext with all required information.

    Returns: tuple (outer_key, inner_key, result_dict), where inner_key is "{i+1}.{j+1}".
    """
    i, j, item = item_info
    key = f"{i+1}.{j+1}"
    
    # Check whether the item is callable (a function).
    if callable(item):
        # If callable, invoke it directly.
        logger.info(f'\n\nCalling function for {context.judge_type} item {key}:')

        if context.slides_path:
            try:
                function_result = item(context.slides_path)
            except Exception as e:
                function_result = {
                    'answer': None,
                    'explanation': None,
                    'log': f"Error: {str(e)}"
                }
        else:
            # If no file path is provided, return an error.
            function_result = {
                'answer': None,
                'explanation': None,
                'log': "Error: No slides file path provided."
            }

        logger.info(f"Answer: {function_result['answer']}")
        logger.info(f"Explanation: {function_result['explanation']}")
        logger.info('====================================\n\n')
        
        return (f"{i+1}", key, function_result)

    else:
        # Otherwise, call the LLM.
        answer = None
        response_text = None
        error_msg = None

        # Retry up to max_retries times until \boxed{...} is matched.
        for attempt in range(context.max_retries):
            try:
                # Use JudgeAPI's unified interface.
                response_text, response_json = context.judge_api.generate_content(
                    model=context.model,
                    prompt=item,
                    slides_obj=context.slides_obj,
                    material_objs=context.material_objs,
                    slides_path=context.slides_path,
                    material_paths=context.material_paths,
                    thinking_level=context.thinking_level,
                    temperature=context.temperature,
                    seed=context.seed,
                )

                # If the underlying API returns an empty response (e.g. network error returning (None, None)),
                # treat it as a failure and retry with backoff rather than as a valid response.
                if not response_text:
                    error_msg = f"Empty response on attempt {attempt + 1}/{context.max_retries} for item {key} of {context.slides_path}"
                    logger.error(f'Error: {error_msg}')
                    if attempt < context.max_retries - 1:
                        time.sleep(min(2**attempt, 60))
                    continue

                logger.info(f'\n\nAttempt {attempt + 1}/{context.max_retries} for {context.judge_type} item {key} of {context.slides_path}:')
                logger.info(f"{response_text=}")
                logger.info('====================================\n\n')

                # Extract the content inside \boxed{...} from the response text.
                # Each captured token is normalized to tolerate LaTeX wrappers
                # (e.g. \boxed{\text{no}} / \boxed{\textbf{Yes}}); then, as before,
                # if the first \boxed{} is yes/no use it, otherwise use the last.
                if all_matches := re.findall(r'\\boxed\{([^}]+)\}', response_text):
                    normalized = [_normalize_boxed_token(m) for m in all_matches]
                    first = normalized[0]
                    if first.lower() in ('yes', 'no'):
                        answer = first
                    else:
                        answer = normalized[-1]
                    break  # Match succeeded -> exit retry loop.
            except Exception as e:
                # Other unexpected exceptions.
                error_msg = f"Unexpected error on attempt {attempt + 1}/{context.max_retries}: {str(e)}"
                logger.error(f'Error: {error_msg}')
                # sleep 2**attempt seconds, but not more than 60 seconds
                if attempt < context.max_retries - 1:  # No sleep on the last attempt.
                    time.sleep(min(2**attempt, 60))
        
        # Build the final result.
        if answer is not None:
            # Extraction succeeded.
            return (f"{i+1}", key, {
                "answer": answer,
                "explanation": response_text if response_text else None
            })
        else:
            # Extraction failed or the API call failed.
            final_error = error_msg or f'Failed to extract answer after {context.max_retries} attempts for item {key}'
            logger.error(f'Error: {final_error}')
            return (f"{i+1}", key, {
                "answer": None,
                "explanation": response_text if response_text else None,
                "log": final_error
            })



def run_judge(
    checklist: List[List[Any]],
    context: JudgeContext,
    existing_result: Optional[dict] = None,
    save_callback: Optional[Callable[[dict, JudgeType], None]] = None,
):
    """
    Run judging logic and return an ordered result dictionary.
    
    Args:
        checklist: Checklist items.
        context: JudgeContext with all required information.
        existing_result: Existing results used for resuming.
        save_callback: Periodic save callback, called with (ordered_result, judge_type).
    """

    # Flatten the checklist while keeping index information.
    flat_items = [(i, j, item) for i, sublist in enumerate(checklist) for j, item in enumerate(sublist)]
    
    # Temporary storage for results (unordered).
    result = defaultdict(dict)

    # Resume: pre-fill existing results and skip items that already exist.
    # Note: only items whose answer is not None are considered complete; answer=None will be re-judged.
    existing_result = existing_result or {}
    existing_keys: set[str] = set()
    if isinstance(existing_result, dict):
        for outer_key, inner in existing_result.items():
            if not isinstance(inner, dict):
                continue
            for inner_key, inner_val in inner.items():
                if isinstance(inner_key, str):
                    # Only resume when answer is not None.
                    if isinstance(inner_val, dict) and inner_val.get("answer") is not None:
                        existing_keys.add(inner_key)
                        result[str(outer_key)][inner_key] = inner_val
                    # If answer is None, do not add to existing_keys; it will be re-judged.
    
    def _build_ordered_result(flat_items, result):
        """Build an ordered result following flat_items order.
        
        NOTE: Callers must ensure appropriate locking when used from concurrent code.
        """
        ordered = {}
        for i, j, _ in flat_items:
            outer_key = f"{i+1}"
            inner_key = f"{i+1}.{j+1}"
            if outer_key not in ordered:
                ordered[outer_key] = {}
            if inner_key in result.get(outer_key, {}):
                ordered[outer_key][inner_key] = result[outer_key][inner_key]
        return ordered

    # Run the first 5 items in a single thread (warm up / build cache).
    warmup_count = min(5, len(flat_items))
    
    # Single-thread execution for the first 5 items.
    for item_info in flat_items[:warmup_count]:
        i, j, _ = item_info
        inner_key = f"{i+1}.{j+1}"
        if inner_key in existing_keys:
            continue
        i_key, key, item_result = process_single_item(item_info, context)
        result[i_key][key] = item_result
    
    # Multi-thread execution for the remaining items.
    if len(flat_items) > warmup_count:
        remaining_items = flat_items[warmup_count:]

        # Resume: filter out items that already exist.
        remaining_items = [
            item_info for item_info in remaining_items
            if f"{item_info[0]+1}.{item_info[1]+1}" not in existing_keys
        ]
        
        # Execute remaining items via a thread pool.
        if remaining_items:
            if context.debug:
                max_workers = 1  # Force single worker in debug mode.
            else:
                max_workers = None
            
            # Lock and counter for periodic saving.
            result_lock = threading.Lock()
            completed_count = 0
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks.
                future_to_item = {
                    executor.submit(process_single_item, item_info, context): item_info
                    for item_info in remaining_items
                }
                
                # Collect results.
                for future in as_completed(future_to_item):
                    try:
                        i_key, key, item_result = future.result()
                        should_save = False
                        ordered_result = None
                        with result_lock:
                            result[i_key][key] = item_result
                            completed_count += 1
                            
                            # Periodic save: save once every save_interval items.
                            if save_callback and completed_count % context.save_interval == 0:
                                ordered_result = _build_ordered_result(flat_items, result)
                                should_save = True
                        
                        # Call save_callback outside the lock to avoid holding it for too long.
                        if should_save and save_callback:
                            save_callback(ordered_result, context.judge_type)
                    except Exception as e:
                        item_info = future_to_item[future]
                        i, j, _ = item_info
                        key = f"{i+1}.{j+1}"
                        logger.error(f'Error processing item {key}: {str(e)}')
                        should_save = False
                        ordered_result = None
                        with result_lock:
                            result[f"{i+1}"][key] = {
                                "answer": None,
                                "explanation": None,
                                "log": f"Thread execution error: {str(e)}"
                            }
                            completed_count += 1
                            
                            # Save even on errors.
                            if save_callback and completed_count % context.save_interval == 0:
                                ordered_result = _build_ordered_result(flat_items, result)
                                should_save = True
                        
                        # Call save_callback outside the lock to avoid holding it for too long.
                        if should_save and save_callback:
                            save_callback(ordered_result, context.judge_type)
    
    # Build an ordered result following flat_items order (Python 3.7+ dict preserves insertion order).
    ordered_result = _build_ordered_result(flat_items, result)
    
    return ordered_result


def main(args=None):
    """
    Main entrypoint; can be called by other modules.
    
    Args:
        args: Optional argparse.Namespace. If None, parse from command line.

    Note on logging:
        This function does NOT configure the logging system (no basicConfig,
        no handlers). Logging setup is the responsibility of the caller:
        - When run as a script (``python judge.py ...``), the ``__main__``
          block below configures stdout logging.
        - When called from another module (e.g. ``judge_all.py``), the caller
          is expected to have set up the desired handlers/levels beforehand.
    """
    # Load environment variables from .env.
    load_dotenv()
    if args is None:
        args = parse_args()

    # zero_score mode: do not run evaluation and do not read slides/material/judge_prompt/result files;
    # directly output an all-zero score YAML file.
    if getattr(args, "zero_score", False):
        if args.output:
            # If an output path is provided, use it as the base path.
            name, ext = os.path.splitext(args.output)
            score_output_path = name + '_score' + ext
        else:
            # Otherwise, use the default output path (respecting --output_dir).
            output_dir = getattr(args, 'output_dir', None) or os.path.dirname(args.slides) or '.'
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = _model_filename_prefix(args.model, args.thinking_level) + f'_{timestamp}_score.yaml'
            score_output_path = os.path.join(output_dir, filename)

        # Load weights to construct the class structure for zero-score output.
        mi_weights = {}
        md_weights = {}
        overall_max_total = 0.0
        if args.weights_path and os.path.exists(args.weights_path):
            try:
                weights_all = load_config(args.weights_path)
                if isinstance(weights_all, dict):
                    # New weight naming: material_independent / material_dependent.
                    mi_weights = weights_all.get("material_independent", {}) or {}
                    md_weights = weights_all.get("material_dependent", {}) or {}
                    mi_total = sum(float(v) for v in mi_weights.values())
                    md_total = sum(float(v) for v in md_weights.values())
                    overall_max_total = float(weights_all.get("total", mi_total + md_total))
            except Exception as e:
                logger.warning(f"Failed to load weights from {args.weights_path} in zero_score mode: {e}")

        def _build_zero_classes(weights_dict):
            classes = {}
            for cls_id, w in (weights_dict or {}).items():
                try:
                    w_float = float(w)
                except Exception:
                    w_float = 0.0
                weight_percent = (w_float / overall_max_total * 100.0) if overall_max_total > 0 else 0.0
                cls_id_str = str(cls_id)
                classes[f"class_{cls_id_str}"] = {
                    "weight_percent": weight_percent,
                    "score_percent": 0.0,
                    "yes_count": 0,
                    "valid_count": 0,
                }
            return classes

        zero_score_data = {
            "total": {
                "force_zero": True,
                "yes_count": 0,
                "valid_count": 0,
                "weighted_arithmetic_mean_percent": 0.0,
                "material_independent": {
                    "yes_count": 0,
                    "valid_count": 0,
                    "weighted_arithmetic_mean_percent": 0.0,
                    "classes": _build_zero_classes(mi_weights),
                },
                "material_dependent": {
                    "yes_count": 0,
                    "valid_count": 0,
                    "weighted_arithmetic_mean_percent": 0.0,
                    "classes": _build_zero_classes(md_weights),
                },
            }
        }

        output_dir = os.path.dirname(score_output_path) or "."
        os.makedirs(output_dir, exist_ok=True)
        with open(score_output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(zero_score_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        logger.info(f"Zero score results saved to: {score_output_path}")
        return

    validate_args(args)

    # Effective fixed-decoding settings for the judge. By default neither
    # temperature nor seed is sent (the API server default decoding is used,
    # i.e. the original behaviour); pass --temperature/--seed to fix them for
    # reproducible verdicts.
    _judge_temperature = getattr(args, "temperature", None)
    _judge_seed = getattr(args, "seed", None)
    logger.info(
        f"Judge decoding: temperature={_judge_temperature}, seed={_judge_seed}"
    )

    material_independent_checklist, material_dependent_checklist = load_judge_prompt(
        args.judge_prompt,
        args.common_judge_prompt,
        args.slides
    )

    # --- Resume / skip logic ---
    # Priority:
    #   1. If a _score.yaml already exists (newest, filtered by min_timestamp), reuse it directly and return.
    #   2. If a result .yaml (non-score) exists and is complete, skip judge and only recompute scores.
    #   3. Otherwise, run judge (possibly resuming from an incomplete result .yaml).

    if args.output:
        _default_output_dir = os.path.dirname(args.output) or '.'
    elif getattr(args, 'output_dir', None):
        _default_output_dir = args.output_dir
    else:
        _default_output_dir = os.path.dirname(args.slides) or '.'
    _filename_prefix = _model_filename_prefix(args.model, args.thinking_level) + '_'

    # Parse min_timestamp (if provided).
    min_timestamp = None
    if args.min_timestamp:
        try:
            min_timestamp = datetime.strptime(args.min_timestamp, "%Y-%m-%d_%H-%M-%S")
        except ValueError as e:
            logger.error(f"Invalid min_timestamp format: {args.min_timestamp}. Expected format: YYYY-MM-DD_HH-MM-SS")
            raise ValueError(f"Invalid min_timestamp format: {args.min_timestamp}. Expected format: YYYY-MM-DD_HH-MM-SS") from e

    # --- Priority 1: check for existing _score.yaml ---
    _judge_model = _model_filename_prefix(args.model, args.thinking_level)
    _extra_judge_models = _resume_alias_prefixes(_judge_model)
    if _extra_judge_models:
        logger.info(
            f"Resume alias enabled for judge_model={_judge_model!r}: "
            f"also accepting {_extra_judge_models} as resume sources."
        )
    existing_score_yaml_path = find_single_score_yaml(
        Path(_default_output_dir),
        judge_model=_judge_model,
        min_timestamp=min_timestamp,
        extra_judge_models=_extra_judge_models,
    )
    if existing_score_yaml_path:
        logger.info(f"Found existing score YAML: {existing_score_yaml_path}. Skipping judge and scoring.")
        return

    # --- Priority 2 & 3: check for existing result .yaml ---
    _extra_filename_prefixes = [p + '_' for p in _extra_judge_models]
    resume_yaml_path = find_resume_yaml(
        _default_output_dir,
        _filename_prefix,
        min_timestamp=min_timestamp,
        extra_prefixes=_extra_filename_prefixes,
    )
    existing_output_data = load_existing_results(resume_yaml_path) if resume_yaml_path else None

    # Did the resume source come from an alias (i.e. a different model's file)?
    resume_is_alias = False
    resume_source_prefix: Optional[str] = None
    if resume_yaml_path:
        _resume_basename = os.path.basename(resume_yaml_path)
        if _resume_basename.startswith(_filename_prefix):
            resume_source_prefix = _judge_model
        else:
            for _alias_prefix, _alias_model in zip(_extra_filename_prefixes, _extra_judge_models):
                if _resume_basename.startswith(_alias_prefix):
                    resume_is_alias = True
                    resume_source_prefix = _alias_model
                    break
        if resume_is_alias:
            logger.info(
                f"Resume source is an alias: {resume_yaml_path} "
                f"(source judge_model={resume_source_prefix!r}, "
                f"current judge_model={_judge_model!r})."
            )

    def _build_resume_info(target_yaml_path: str) -> Optional[dict]:
        """Metadata describing where resumed data came from.

        The ``resumed_from`` field is stored as a path *relative to the target
        YAML file's directory* (i.e. the YAML we're about to write), so that
        the record stays valid when the whole output directory is moved.

        Args:
            target_yaml_path: Path of the YAML file into which this metadata
                will be embedded. Used as the anchor for computing the
                relative path to ``resume_yaml_path``.

        Returns:
            The metadata dict, or ``None`` if there is nothing to resume from,
            so that the key is only added to the output YAML when it's
            actually meaningful.
        """
        if not resume_yaml_path:
            return None
        try:
            target_dir = os.path.dirname(os.path.abspath(target_yaml_path)) or "."
            resume_rel = os.path.relpath(os.path.abspath(resume_yaml_path), target_dir)
        except Exception:
            resume_rel = resume_yaml_path
        return {
            "resumed_from": resume_rel,
            "source_judge_model": resume_source_prefix,
            "current_judge_model": _judge_model,
            "is_alias": resume_is_alias,
        }

    # Check whether existing results are complete.
    existing_complete_results = False
    if resume_yaml_path and existing_output_data:
        logger.info(f"Resume enabled: found existing result YAML: {resume_yaml_path}")

        material_independent_complete = is_result_complete(
            material_independent_checklist,
            existing_output_data.get(JudgeType.MATERIAL_INDEPENDENT.value) if isinstance(existing_output_data, dict) else None
        )
        material_dependent_complete = is_result_complete(
            material_dependent_checklist,
            existing_output_data.get(JudgeType.MATERIAL_DEPENDENT.value) if isinstance(existing_output_data, dict) else None
        )

        # If both judge types are complete, skip judge execution.
        if material_independent_complete and material_dependent_complete:
            existing_complete_results = True
            logger.info(f"Results are already complete. Using existing result YAML: {resume_yaml_path}")
            output_path = resume_yaml_path

            # When we resumed from an alias (a different model's result YAML),
            # materialise a new result file under the *current* model's prefix
            # so that subsequent lookups / statistics can find it directly
            # without relying on alias matching. The new file embeds a
            # `_resume_info` block that records where the data came from.
            if resume_is_alias:
                _alias_copy_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                _alias_copy_name = (
                    _model_filename_prefix(args.model, args.thinking_level)
                    + f"_{_alias_copy_ts}.yaml"
                )
                _alias_copy_path = os.path.join(_default_output_dir, _alias_copy_name)
                os.makedirs(os.path.dirname(_alias_copy_path) or ".", exist_ok=True)

                _resume_info = _build_resume_info(_alias_copy_path)
                _alias_copy_data = dict(existing_output_data) if isinstance(existing_output_data, dict) else {}
                if _resume_info is not None:
                    _alias_copy_data["_resume_info"] = _resume_info
                with open(_alias_copy_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(
                        _alias_copy_data,
                        f,
                        default_flow_style=False,
                        allow_unicode=True,
                        sort_keys=False,
                    )
                logger.info(
                    f"Materialised alias-resumed results to current-model file: "
                    f"{_alias_copy_path} (source: {resume_yaml_path})"
                )
                # Subsequent scoring should read from the new file so that the
                # generated *_score.yaml also uses the current-model prefix.
                output_path = _alias_copy_path
        else:
            # Results are incomplete; continue running judge.
            existing_complete_results = False
    else:
        # No resume data; continue running judge.
        existing_complete_results = False

    # Only run judge if results are not complete.
    if not existing_complete_results:

        # Create API client.
        judge_api = create_judge_api(
            args.api_type,
        )

        # Handle slides file.
        if args.slides.endswith('.pptx'):
            slides_pdf_path = convert_pptx_to_pdf(args.slides)
            if slides_pdf_path is None:
                raise ValueError(f"Failed to convert {args.slides} to PDF")
            args.slides = str(slides_pdf_path)
        # If slide page count exceeds the maximum, truncate to the limit.
        try:
            # Extract max_count from material_independent_checklist[0][0].
            if (material_independent_checklist and
                len(material_independent_checklist) > 0 and
                len(material_independent_checklist[0]) > 0):
                first_item = material_independent_checklist[0][0]
                if isinstance(first_item, partial):
                    # Extract max_count from the partial object.
                    max_count = first_item.keywords.get('max_count')
                    if max_count is not None:
                        # Compute current slide page count.
                        num_pages = count_pages(args.slides)
                        logger.info(f'Current slides page count: {num_pages}, max_count: {max_count}')

                        # If page count exceeds the limit, truncate the file.
                        if num_pages > max_count:
                            logger.info(f'Truncating slides from {num_pages} pages to {max_count} pages')
                            truncated_path = truncate_slides(args.slides, max_count)
                            args.slides = truncated_path
                    else:
                        logger.error('Could not extract max_count from partial object')
                else:
                    logger.error('First checklist item is not a partial object, skipping truncation check')
            else:
                logger.error('material_independent_checklist is empty or invalid, skipping truncation check')
        except Exception as e:
            logger.error(f'Error during slides truncation check: {e}. Continuing without truncation.')

        slides_obj = judge_api.upload_file(args.slides)
        logger.debug(f'{slides_obj=}')

        # Handle material files (multiple files supported).
        material_objs = None
        if args.material:
            material_objs = [judge_api.upload_file(material_file) for material_file in args.material]
            logger.debug(f'{material_objs=}')

        # Determine base output path and generate a timestamped file name (used for both checkpoints and final save).
        if args.output:
            # Use the provided output path as the base path.
            output_path = args.output
        else:
            # Otherwise, use the default output path (respecting --output_dir).
            output_dir = getattr(args, 'output_dir', None) or os.path.dirname(args.slides) or '.'
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = _model_filename_prefix(args.model, args.thinking_level) + f'_{timestamp}.yaml'
            output_path = os.path.join(output_dir, filename)

        # Ensure output directory exists.
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Store current results (used for periodic saving).
        current_output_data = {
            JudgeType.MATERIAL_INDEPENDENT.value: {},
            JudgeType.MATERIAL_DEPENDENT.value: {},
        }
        # If existing results are present, load them into current_output_data first.
        if isinstance(existing_output_data, dict):
            for judge_type_value in [JudgeType.MATERIAL_INDEPENDENT.value, JudgeType.MATERIAL_DEPENDENT.value]:
                if judge_type_value in existing_output_data:
                    current_output_data[judge_type_value] = existing_output_data[judge_type_value]

        # Record where we resumed from (if any), so that checkpoints and the
        # final YAML carry this provenance information. The path in
        # `_resume_info` is stored relative to `output_path` (which is the
        # YAML file we will actually be writing).
        _resume_info = _build_resume_info(output_path)
        if _resume_info is not None:
            current_output_data["_resume_info"] = _resume_info

        # Lock for periodic saving.
        save_lock = threading.Lock()

        def save_intermediate_result(ordered_result: dict, judge_type: JudgeType):
            """Callback to periodically save intermediate results (uses a fixed timestamped file name)."""
            with save_lock:
                # Update current results.
                current_output_data[judge_type.value] = ordered_result

                # Save to file (same file name as the final output).
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        yaml.safe_dump(current_output_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
                    logger.info(f'Checkpoint saved to: {output_path} (judge_type: {judge_type.value})')
                except Exception as e:
                    logger.error(f'Failed to save checkpoint: {e}')

        def execute_judge(checklist: List[List[Any]], judge_type: JudgeType):
            """Shared logic for running judge."""
            # Include material files depending on judge_type:
            # - Material independent: slides only
            # - Material dependent: slides + material

            context = JudgeContext(
                judge_api=judge_api,
                model=args.model,
                thinking_level=args.thinking_level,
                slides_obj=slides_obj,
                material_objs=material_objs if judge_type == JudgeType.MATERIAL_DEPENDENT else None,
                slides_path=args.slides,
                material_paths=args.material if judge_type == JudgeType.MATERIAL_DEPENDENT else None,
                judge_type=judge_type,
                max_retries=args.retry,
                debug=args.debug,
                save_interval=20,  # Save after every 20 completed items.
                temperature=_judge_temperature,
                seed=_judge_seed,
            )

            existing_for_type = None
            if isinstance(existing_output_data, dict):
                existing_for_type = existing_output_data.get(judge_type.value)
            return run_judge(
                checklist,
                context,
                existing_result=existing_for_type,
                save_callback=save_intermediate_result,
            )

        # Material independent judge
        material_independent_result = execute_judge(material_independent_checklist, JudgeType.MATERIAL_INDEPENDENT)
        # Material dependent judge
        material_dependent_result = execute_judge(material_dependent_checklist, JudgeType.MATERIAL_DEPENDENT)

        # Save final results to YAML (same timestamped file name as checkpoints).
        output_data = {
            JudgeType.MATERIAL_INDEPENDENT.value: material_independent_result,
            JudgeType.MATERIAL_DEPENDENT.value: material_dependent_result,
        }
        if _resume_info is not None:
            output_data["_resume_info"] = _resume_info

        # Ensure output directory exists.
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(output_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        logger.info(f'\nFinal results saved to: {output_path}')

    # Compute scores (skipped only when _score.yaml was found above — which already returned).
    if not args.weights_path:
        raise ValueError("--weights_path is required when using --score")
    if not os.path.exists(args.weights_path):
        raise FileNotFoundError(f"Weights file not found: {args.weights_path}")

    logger.info(f'\nComputing scores...')
    name, ext = os.path.splitext(output_path)
    score_output_path = name + '_score' + ext
    compute_and_save_scores(
        result_path=output_path,
        weights_path=args.weights_path,
        output_path=score_output_path,
        print_summary=True
    )


if __name__ == '__main__':
    # Configure logging only when running as a script. When imported as a
    # library (e.g. by judge_all.py), the caller controls logging setup.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    main()
