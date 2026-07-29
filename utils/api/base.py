import logging
import os
import tempfile
from abc import ABC, abstractmethod

import requests
from google import genai

from utils.encode_file import (
    build_file_data_url,
    encode_file_base64,
    get_mime_type,
    read_file_bytes,
)


logger = logging.getLogger(__name__)


def _describe_request_error(error: Exception) -> str:
    """Append the response body, which is usually what explains an HTTP error."""
    response = getattr(error, "response", None)
    if response is not None and response.text:
        return f"{error}: {response.text[:200]}"
    return str(error)


def _build_gemini_config(thinking_level=None, temperature=None, seed=None):
    """Build a ``GenerateContentConfig`` for the google-genai SDK.

    Only fields that are explicitly provided are included, so the default
    behaviour (``thinking_level=temperature=seed=None``) reproduces the
    previous "server-default decoding" and returns ``None`` (no config).

    Fixing ``temperature=0`` (optionally with a ``seed``) makes the judge's
    per-item verdicts as deterministic / reproducible as the backend allows.
    """
    config_kwargs = {}
    if thinking_level:
        config_kwargs["thinking_config"] = genai.types.ThinkingConfig(
            thinking_level=thinking_level
        )
    if temperature is not None:
        config_kwargs["temperature"] = temperature
    if seed is not None:
        config_kwargs["seed"] = seed
    return genai.types.GenerateContentConfig(**config_kwargs) if config_kwargs else None


class BaseAPI(ABC):
    """Base API class that defines a unified interface."""
        
    @abstractmethod
    def upload_file(self, file_path: str):
        """Upload a file and return a file object."""
        pass
    
    @abstractmethod
    def generate_content(self, model: str, prompt: str, **kwargs):
        """
        Generate content.
        
        Args:
            model: Model name.
            prompt: Prompt text.
            **kwargs: Other optional parameters.
            
        Returns:
            (response_text, response_json_dict)
            - response_text: Text content from the API response.
            - response_json_dict: Raw JSON dict (used for billing/usage tracking).
        """
        pass


class GeminiAPI(BaseAPI):
    """Gemini API wrapper."""
    
    def __init__(self, *args, **kwargs):
        self.client = genai.Client(*args, **kwargs)
        # Track files uploaded via upload_file (best-effort cleanup on destruction).
        self._uploaded_files_names = []
    
    def upload_file(self, file_path: str):
        """Upload a file to Gemini."""
        myfile = self.client.files.upload(file=file_path)
        self._uploaded_files_names.append(myfile.name)
        return myfile
    
    def delete_file(self, file_name: str):
        """Delete a file."""
        self.client.files.delete(name=file_name)

    def __del__(self):
        """Delete all files uploaded via upload_file on destruction (best-effort)."""
        try:
            for name in set(self._uploaded_files_names):
                self.delete_file(name)
        except Exception:
            # Avoid exceptions in __del__ interfering with interpreter teardown.
            pass
    
    def generate_content(self, model: str, prompt: str, contents: list = None,
                         thinking_level: str = None, temperature: float | None = None,
                         seed: int | None = None, **kwargs):
        """Generate content using the Gemini API.
        
        Args:
            model: Model name.
            prompt: Prompt text.
            contents: Optional list of contents.
            thinking_level: Optional thinking level.
            temperature: Optional temperature.
            seed: Optional seed.
            **kwargs: Other optional parameters.
        """
        contents = contents or []
        
        try:
            config = _build_gemini_config(thinking_level, temperature, seed)
            response = self.client.models.generate_content(
                model=model,
                contents=contents + [prompt],
                config=config,
            )
            return response.text, dict(response)
        except Exception as e:
            logger.error(f"Gemini API call failed: {str(e)}")
            return None, None
    

class GeminiInlineAPI(BaseAPI):
    
    def __init__(self, *args, **kwargs):
        self.client = genai.Client(*args, **kwargs)

    def upload_file(self, file_path: str):
        """Return a Gemini file part object (inline bytes)."""
        mime_type = get_mime_type(file_path)

        return genai.types.Blob(
            mime_type=mime_type,
            data=read_file_bytes(file_path),
        )

    def generate_content(self, model: str, prompt: str, contents: list = None,
                         thinking_level: str = None, temperature: float | None = None,
                         seed: int | None = None, **kwargs):
        """Generate content using the Gemini API.
        
        Args:
            model: Model name.
            prompt: Prompt text.
            contents: Optional list of contents.
            thinking_level: Optional thinking level.
            temperature: Optional temperature.
            seed: Optional seed.
        """
        parts = [
            genai.types.Part(inline_data=item)
            for item in (contents or [])
        ]
        parts.append(genai.types.Part(text=prompt))

        config = _build_gemini_config(thinking_level, temperature, seed)

        try:
            response = self.client.models.generate_content(
                model=model,
                contents=[genai.types.Content(parts=parts)],
                config=config,
            )
            return response.text, dict(response)
        except Exception as e:
            logger.error(f"Gemini API call failed: {str(e)}")
            return None, None



class OpenAIChatAPI(BaseAPI):
    """Multimodal client for the OpenAI Chat Completions protocol."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        reasoning_split: bool = False,
        pdf_dpi: int = 150,
        **kwargs,
    ):
        self.api_key = api_key
        # ``base_url`` is the API root (e.g. https://api.minimaxi.com/v1), the
        # same value OpenAI-compatible SDKs take; the route is appended here.
        self.base_url = base_url.rstrip('/')
        self.url = f"{self.base_url}/chat/completions"
        self.reasoning_split = reasoning_split
        self.pdf_dpi = pdf_dpi
        self.session = requests.Session()

    @property
    def headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def upload_file(self, file_path: str):
        """Chat Completions takes inline content, so keep the local path."""
        return file_path

    def _build_file_content(self, file_path: str) -> list[dict]:
        """Convert a local file into Chat Completions multimodal content."""
        suffix = os.path.splitext(file_path)[1].lower()
        if suffix == '.pdf':
            return self._pdf_to_image_contents(file_path, dpi=self.pdf_dpi)
        if suffix in ('.md', '.txt'):
            with open(file_path, 'r', encoding='utf-8') as file_obj:
                return [{"type": "text", "text": file_obj.read()}]

        payload = encode_file_base64(file_path)
        return [{
            "type": "image_url",
            "image_url": {
                "url": f"data:application/octet-stream;base64,{payload}",
            },
        }]

    @staticmethod
    def _pdf_to_image_contents(pdf_path: str, dpi: int = 150) -> list[dict]:
        """Render every PDF page to a PNG data URL, cached on disk.

        No page limit is applied: how many slides the judge may see is decided
        by the benchmark (``judge.py`` truncates slides to the checklist's
        ``max_count``) and materials are never truncated, so dropping pages
        here would silently score a different deck than the Gemini backends do.
        The default DPI matches ``utils.pdf_to_images``, keeping body text
        legible enough to compare against backends that read the PDF natively.
        """
        try:
            import fitz
        except ImportError as error:
            raise ImportError(
                "pymupdf is required to render PDF pages for OpenAIChatAPI. "
                "Install with: pip install pymupdf"
            ) from error

        import base64
        import binascii
        import hashlib

        try:
            mtime = os.path.getmtime(pdf_path)
        except OSError:
            mtime = 0
        cache_key = hashlib.sha1(
            f"{pdf_path}|{mtime}|{dpi}".encode()
        ).hexdigest()[:16]
        # v3: earlier caches may hold a page-limited render, so never reuse them.
        cache_name = f"openai_chat_pdf_v3_{cache_key}.tsv"
        cache_dir = tempfile.gettempdir()
        cache_path = os.path.join(cache_dir, cache_name)

        # Pages are kept base64-encoded end to end: that is both the cache format
        # and the wire format, so a cache hit needs no re-encoding.
        encoded_pages: list[str] = []
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as cache_file:
                    for line in cache_file:
                        encoded = line.rstrip("\n")
                        if encoded:
                            base64.b64decode(encoded, validate=True)  # detect corruption
                            encoded_pages.append(encoded)
                if not encoded_pages:
                    raise ValueError("PDF cache is empty")
            except (OSError, ValueError, binascii.Error):
                encoded_pages = []
                try:
                    os.unlink(cache_path)
                except FileNotFoundError:
                    pass

        if not encoded_pages:
            matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
            with fitz.open(pdf_path) as document:
                for page_index in range(len(document)):
                    page = document.load_page(page_index)
                    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                    encoded_pages.append(
                        base64.b64encode(pixmap.tobytes("png")).decode("utf-8")
                    )

            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=cache_dir,
                    prefix=f".{cache_name}.",
                    suffix=".tmp",
                    delete=False,
                ) as cache_file:
                    temporary_path = cache_file.name
                    for encoded in encoded_pages:
                        cache_file.write(encoded + "\n")
                os.replace(temporary_path, cache_path)
                temporary_path = None
            finally:
                if temporary_path is not None:
                    try:
                        os.unlink(temporary_path)
                    except FileNotFoundError:
                        pass

        return [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + encoded},
            }
            for encoded in encoded_pages
        ]

    def generate_content(
        self,
        model: str,
        prompt: str,
        file_paths: list | None = None,
        thinking_level: str | None = None,
        **kwargs,
    ):
        """Call a Chat Completions compatible endpoint.

        A failure returns ``(None, None)``; retrying is owned by the caller
        (``judge.process_single_item``), as it is for the Gemini clients.
        """
        content_list: list[dict] = []
        for file_path in file_paths or []:
            content_list.extend(self._build_file_content(file_path))
        content_list.append({"type": "text", "text": prompt})

        data = {
            "model": model,
            "messages": [{"role": "user", "content": content_list}],
            "stream": False,
        }
        if self.reasoning_split:
            data["reasoning_split"] = True
        if thinking_level:
            # The protocol only defines "disabled" and "adaptive"; levels borrowed
            # from other backends (e.g. Gemini's "high") cannot be honoured.
            thinking_type = thinking_level
            if thinking_type not in ("disabled", "adaptive"):
                logger.warning(
                    "thinking_level=%r is not supported by the Chat Completions "
                    "protocol; falling back to 'adaptive'",
                    thinking_level,
                )
                thinking_type = "adaptive"
            data["thinking"] = {"type": thinking_type}
        for option in ("temperature", "seed"):
            if kwargs.get(option) is not None:
                data[option] = kwargs[option]

        try:
            response = self.session.post(
                self.url,
                headers=self.headers,
                json=data,
                timeout=300,
            )
            response.raise_for_status()
            result = response.json()
            choices = result.get('choices') or []
            if not choices:
                logger.error("Chat API returned no choices: %s", str(result)[:200])
                return None, result
            text = (choices[0].get('message') or {}).get('content', '')
            return text, result
        except requests.exceptions.RequestException as error:
            logger.error("Chat API request failed: %s", _describe_request_error(error))
            return None, None
        except Exception as error:
            logger.error("Chat API call failed: %s", error)
            return None, None


class OpenAIResponsesAPI(BaseAPI):
    """OpenAI Responses API client."""
    
    def __init__(self, api_key: str, **kwargs):
        self.api_key = api_key
        self.url = "https://api.openai.com/v1/responses"
        # Reuse connections via a Session for better performance.
        self.session = requests.Session()
    
    @property
    def headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def upload_file(self, file_path: str):
        """OpenAI API does not require uploading; return the file path."""
        return file_path

    @staticmethod
    def _extract_output_text(result: dict) -> str:
        """Extract all plain-text output from an OpenAI Responses API JSON response.

        This strictly follows the official documentation at
        https://developers.openai.com/api/reference/resources/responses/methods/create
        and handles every possible item type in the ``output`` array. All fields
        containing plain text are extracted and joined into a single string using
        readable section delimiters:

        - ``message``: ``content[].output_text.text`` / ``content[].refusal.refusal``
        - ``reasoning``: ``summary[].text`` / ``content[].text``

        Non-plain-text content such as ``compaction`` (encrypted) and
        ``image_generation_call`` (base64-encoded images) is skipped.
        """
        if not isinstance(result, dict):
            return ''

        segments: list[str] = []

        def _section(header: str, body: str) -> None:
            body = body.strip('\n')
            if body:
                segments.append(f"--- {header} ---\n{body}")

        def _as_text_from_content_list(content) -> str:
            """Join an ``output_text`` array into plain text."""
            if isinstance(content, str):
                return content
            if not isinstance(content, list):
                return ''
            parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get('type') in ('output_text', 'text'):
                    t = part.get('text')
                    if isinstance(t, str) and t:
                        parts.append(t)
                elif part.get('type') == 'refusal':
                    r = part.get('refusal')
                    if isinstance(r, str) and r:
                        parts.append(f"[refusal] {r}")
            return '\n'.join(parts)

        output = result.get('output') or []
        for item in output:
            if not isinstance(item, dict):
                continue
            itype = item.get('type')

            if itype == 'message':
                text = _as_text_from_content_list(item.get('content'))
                _section('message', text)

            elif itype == 'reasoning':
                parts: list[str] = []
                for s in (item.get('summary') or []):
                    if isinstance(s, dict) and s.get('type') == 'summary_text':
                        t = s.get('text')
                        if isinstance(t, str) and t:
                            parts.append(f"[summary] {t}")
                for c in (item.get('content') or []):
                    if isinstance(c, dict) and c.get('type') == 'reasoning_text':
                        t = c.get('text')
                        if isinstance(t, str) and t:
                            parts.append(t)
                _section('reasoning', '\n'.join(parts))

        if segments:
            return '\n\n'.join(segments)

        # Fallback: support the top-level ``output_text`` field returned by some proxies
        if isinstance(top_level := result.get('output_text'), str):
            return top_level

        return ''

    def generate_content(self, model: str, prompt: str, file_paths: list = None, **kwargs):
        """
        Call the OpenAI Response API.
        https://platform.openai.com/docs/api-reference/responses/create
        
        Args:
            model: Model name.
            prompt: Prompt text.
            file_paths: Optional list of file paths.
            **kwargs: Other optional parameters.
        """
        file_paths = file_paths or []
        
        content_list = []
        for file_path in file_paths:
            content_list.append({
                "type": "input_file",
                "filename": file_path,
                "file_data": build_file_data_url(file_path),
            })
        
        # Add the text prompt.
        content_list.append({
            "type": "input_text",
            "text": prompt
        })
        
        data = {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": content_list
                }
            ],
            "stream": False
        }
        if kwargs.get("temperature") is not None:
            data["temperature"] = kwargs["temperature"]
        
        try:
            # Reuse connections via session.
            response = self.session.post(self.url, headers=self.headers, json=data, timeout=300)
            response.raise_for_status()

            # Extract text content from the response.
            result = response.json()
            return self._extract_output_text(result), result
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAI Responses API request failed: {_describe_request_error(e)}")
            return None, None
        except Exception as e:
            logger.error(f"OpenAI Responses API call failed: {str(e)}")
            return None, None


# Keep the former name so existing callers do not break.
OpenAIAPI = OpenAIResponsesAPI
