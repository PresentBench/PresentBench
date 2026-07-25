import logging
import os
import time
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

_TRANSIENT_REQUEST_EXCEPTIONS = (
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
)


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
    """兼容 OpenAI Chat Completions 协议的多模态 API 客户端。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        reasoning_split: bool = False,
        **kwargs,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.reasoning_split = reasoning_split
        self.session = requests.Session()

    @property
    def headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def upload_file(self, file_path: str):
        """Chat Completions API 不需要上传文件，直接返回本地路径。"""
        return file_path

    def _build_file_content(self, file_path: str) -> list[dict]:
        """把本地文件转换为 Chat Completions 多模态内容。"""
        suffix = os.path.splitext(file_path)[1].lower()
        if suffix == '.pdf':
            max_pages = 8 if 'material' in file_path.lower() else 30
            return self._pdf_to_image_contents(file_path, max_pages=max_pages, dpi=72)
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
    def _pdf_to_image_contents(
        pdf_path: str,
        max_pages: int = 30,
        dpi: int = 72,
    ) -> list[dict]:
        """将 PDF 页面渲染为 PNG data URL，并用文件缓存避免重复渲染。"""
        try:
            import fitz
        except ImportError as error:
            raise ImportError(
                "pymupdf is required to render PDF pages for OpenAIChatAPI. "
                "Install with: pip install pymupdf"
            ) from error

        import base64
        import hashlib

        try:
            mtime = os.path.getmtime(pdf_path)
        except OSError:
            mtime = 0
        cache_key = hashlib.sha1(
            f"{pdf_path}|{mtime}|{dpi}|{max_pages}".encode()
        ).hexdigest()[:16]
        cache_path = os.path.join("/tmp", f"openai_chat_pdf_{cache_key}.tsv")

        rendered: list[bytes] = []
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as cache_file:
                for line in cache_file:
                    encoded = line.rstrip("\n")
                    if encoded:
                        rendered.append(base64.b64decode(encoded))

        if not rendered:
            matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
            with fitz.open(pdf_path) as document:
                for page_index in range(min(len(document), max_pages)):
                    page = document.load_page(page_index)
                    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                    rendered.append(pixmap.tobytes("png"))
            with open(cache_path, "w", encoding="utf-8") as cache_file:
                for png_bytes in rendered:
                    cache_file.write(base64.b64encode(png_bytes).decode("utf-8") + "\n")

        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        "data:image/png;base64," +
                        base64.b64encode(png_bytes).decode("utf-8")
                    ),
                },
            }
            for png_bytes in rendered
        ]

    def generate_content(
        self,
        model: str,
        prompt: str,
        file_paths: list | None = None,
        thinking_level: str | None = None,
        **kwargs,
    ):
        """调用兼容 Chat Completions 的接口，并对瞬态错误进行重试。"""
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
            thinking_type = (
                thinking_level
                if thinking_level in ("disabled", "adaptive")
                else "adaptive"
            )
            data["thinking"] = {"type": thinking_type}
        for option in ("temperature", "seed"):
            if kwargs.get(option) is not None:
                data[option] = kwargs[option]

        try:
            last_error: Exception | None = None
            for attempt in range(6):
                try:
                    response = self.session.post(
                        self.base_url,
                        headers=self.headers,
                        json=data,
                        timeout=300,
                    )
                except _TRANSIENT_REQUEST_EXCEPTIONS as error:
                    last_error = error
                    if attempt < 5:
                        sleep_seconds = min(60.0, 4.0 * (2 ** attempt))
                        logger.warning(
                            "Chat API transport error on attempt %s: %s; "
                            "sleeping %.1fs then retrying",
                            attempt + 1,
                            error,
                            sleep_seconds,
                        )
                        time.sleep(sleep_seconds)
                    continue

                if response.status_code == 429 or response.status_code >= 500:
                    last_error = requests.exceptions.HTTPError(
                        f"{response.status_code} {response.text[:200]}"
                    )
                    if attempt < 5:
                        sleep_seconds = min(60.0, 4.0 * (2 ** attempt))
                        logger.warning(
                            "Chat API transient %s on attempt %s; "
                            "sleeping %.1fs then retrying",
                            response.status_code,
                            attempt + 1,
                            sleep_seconds,
                        )
                        time.sleep(sleep_seconds)
                    continue

                response.raise_for_status()
                result = response.json()
                text = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                return text, result

            if last_error:
                logger.error("Chat API request failed after retries: %s", last_error)
            return None, None
        except requests.exceptions.RequestException as error:
            logger.error("Chat API request failed: %s", error)
            return None, None
        except Exception as error:
            logger.error("Chat API call failed: %s", error)
            return None, None


class OpenAIResponsesAPI(BaseAPI):
    """OpenAI Responses API 客户端。"""
    
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
            last_error: Exception | None = None
            for attempt in range(6):
                try:
                    response = self.session.post(
                        self.url,
                        headers=self.headers,
                        json=data,
                        timeout=300,
                    )
                except _TRANSIENT_REQUEST_EXCEPTIONS as error:
                    last_error = error
                    if attempt < 5:
                        sleep_seconds = min(60.0, 4.0 * (2 ** attempt))
                        logger.warning(
                            "Responses API transport error on attempt %s: %s; "
                            "sleeping %.1fs then retrying",
                            attempt + 1,
                            error,
                            sleep_seconds,
                        )
                        time.sleep(sleep_seconds)
                    continue

                if response.status_code == 429 or response.status_code >= 500:
                    last_error = requests.exceptions.HTTPError(
                        f"{response.status_code} {response.text[:200]}"
                    )
                    if attempt < 5:
                        sleep_seconds = min(60.0, 4.0 * (2 ** attempt))
                        logger.warning(
                            "Responses API transient %s on attempt %s; "
                            "sleeping %.1fs then retrying",
                            response.status_code,
                            attempt + 1,
                            sleep_seconds,
                        )
                        time.sleep(sleep_seconds)
                    continue

                response.raise_for_status()
                result = response.json()
                return self._extract_output_text(result), result

            if last_error:
                logger.error(
                    "OpenAI Responses API request failed after retries: %s",
                    last_error,
                )
            return None, None
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAI Responses API request failed: {str(e)}")
            return None, None
        except Exception as e:
            logger.error(f"OpenAI Responses API call failed: {str(e)}")
            return None, None


# 保留旧名称，避免外部调用方在升级后中断。
OpenAIAPI = OpenAIResponsesAPI
