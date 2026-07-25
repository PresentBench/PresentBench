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


_EXT_TO_MIME = {
    'pdf': 'application/pdf',
    'png': 'image/png',
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'webp': 'image/webp',
    'heic': 'image/heic',
    'heif': 'image/heif',
    'md': 'text/markdown',
    'markdown': 'text/markdown',
    'txt': 'text/plain',
}


def get_mime_type(file_path: str, charset: str = 'utf-8') -> str:
    """根据扩展名推断 MIME 类型，并为文本文件声明 UTF-8 编码。"""
    ext = os.path.splitext(file_path.lower())[1].lstrip('.')
    mime_type = _EXT_TO_MIME.get(ext)
    if mime_type is None:
        raise ValueError(f"Unsupported file type: {file_path}")
    if charset and mime_type.startswith('text/'):
        return f'{mime_type};charset={charset}'
    return mime_type


def build_file_data_url(file_path: str) -> str:
    """将文件编码为保留正确 MIME 类型的 data URL。"""
    return f"data:{get_mime_type(file_path)};base64,{encode_file_base64(file_path)}"
