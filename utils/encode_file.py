import os
import base64
from functools import lru_cache


# ---------------------------------------------------------------------------
# Low-level helpers: read bytes / base64-encode bytes
# ---------------------------------------------------------------------------

@lru_cache(maxsize=128)
def _encode_file_base64_cached(file_path: str, file_mtime: float) -> str:
    """
    Read a file as raw bytes and encode it as base64 (cached).
    Uses the file's modification time as part of the cache key to ensure
    re-encoding after updates.

    Args:
        file_path: File path.
        file_mtime: File modification time (for cache invalidation).

    Returns:
        Base64-encoded string of the raw file bytes.
    """
    with open(file_path, "rb") as file_file:
        return base64.b64encode(file_file.read()).decode('utf-8')


def encode_file_base64(file_path: str) -> str:
    """
    Read a file as raw bytes and encode it as base64 (with caching).

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


# ---------------------------------------------------------------------------
# MIME type inference
# ---------------------------------------------------------------------------

# Single source of truth: file extension (lowercase, no leading dot) -> MIME media type.
_EXT_TO_MIME = {
    # Binary (document)
    'pdf': 'application/pdf',
    # Images
    'png': 'image/png',
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'webp': 'image/webp',
    'heic': 'image/heic',
    'heif': 'image/heif',
    # Text
    'md': 'text/markdown',
    'markdown': 'text/markdown',
    'txt': 'text/plain',
}


def get_mime_type(file_path: str, charset: str = 'utf-8') -> str:
    """
    Infer the MIME type from a file path's extension.

    The mapping is driven by :data:`_EXT_TO_MIME`. For ``text/*`` types a
    ``charset`` parameter is appended (defaulting to ``utf-8``) as a
    separate post-processing step.

    Why a ``charset`` parameter is attached for text types:
        We have no reliable way to detect the on-disk encoding of an
        arbitrary text file, so we adopt a project-wide convention:
        **text files are UTF-8**. Declaring the charset explicitly matters
        for ``data:`` URLs in particular — RFC 2397 defaults ``text/*``
        to US-ASCII, so strict receivers would otherwise mangle non-ASCII
        bytes (e.g. Chinese UTF-8) into mojibake. The parameter is also
        harmless (and arguably helpful) for other consumers such as
        Gemini's ``inline_data.mime_type``.

    Args:
        file_path: File path (only the extension is inspected).
        charset: Charset to declare for text types (``text/*``). Defaults
            to ``"utf-8"``. Pass ``None`` or an empty string to omit the
            ``charset`` parameter entirely (useful if the caller knows
            the receiver is strictly RFC-compliant and dislikes parameters).
            Ignored for non-text types.

    Returns:
        The MIME type string (possibly including a ``charset`` parameter).

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = os.path.splitext(file_path.lower())[1].lstrip('.')

    mime = _EXT_TO_MIME.get(ext)
    if mime is None:
        raise ValueError(f"Unsupported file type: {file_path}")

    # Post-process: only ``text/*`` types accept a ``charset`` parameter.
    if charset and mime.startswith('text/'):
        return f'{mime};charset={charset}'
    return mime


# ---------------------------------------------------------------------------
# High-level helper: build a data URL
# ---------------------------------------------------------------------------

def build_file_data_url(file_path: str) -> str:
    """
    Build a ``data:`` URL for ``file_path``.

    For text types the MIME string returned by :func:`get_mime_type`
    already carries ``;charset=utf-8``, so strict receivers decode
    non-ASCII payloads (e.g. Chinese UTF-8 Markdown) correctly.

    Args:
        file_path: File path.

    Returns:
        A full ``data:<mediatype>;base64,<payload>`` URL.
    """
    mime_type = get_mime_type(file_path)
    payload = encode_file_base64(file_path)
    return f"data:{mime_type};base64,{payload}"
