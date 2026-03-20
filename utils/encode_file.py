import os
import base64
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=128)
def _encode_file_base64_cached(file_path: str, file_mtime: float) -> str:
    """
    Read a file and encode it as base64 (cached).
    Uses the file's modification time as part of the cache key to ensure re-encoding after updates.
    
    Args:
        file_path: File path.
        file_mtime: File modification time (for cache invalidation).
    
    Returns:
        Base64-encoded string.
    """
    with open(file_path, "rb") as file_file:
        return base64.b64encode(file_file.read()).decode('utf-8')


def encode_file_base64(file_path: str) -> str:
    """
    Read a file and encode it as base64 (with caching).
    
    Args:
        file_path: File path.
    
    Returns:
        Base64-encoded string.
    """
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Use file mtime as part of the cache key.
    mtime = os.path.getmtime(file_path)
    return _encode_file_base64_cached(file_path, mtime)


def read_file_bytes(file_path: os.PathLike) -> bytes:
    """
    Read file bytes.
    
    Args:
        file_path: File path.
    
    Returns:
        File bytes.
    """
    with open(file_path, "rb") as file_file:
        return file_file.read()
