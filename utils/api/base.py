from abc import ABC, abstractmethod
import logging
import os
import time
import requests
from google import genai
from utils.encode_file import encode_file_base64, read_file_bytes



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
    
    def generate_content(self, model: str, prompt: str, contents: list = None, thinking_level: str = None, **kwargs):
        """Generate content using the Gemini API.
        
        Args:
            model: Model name.
            prompt: Prompt text.
            contents: Optional list of contents.
            thinking_level: Optional thinking level.
        """
        contents = contents or []
        
        try:
            if not thinking_level:
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents + [prompt],
                )
            else:
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents + [prompt],
                    config=genai.types.GenerateContentConfig(
                        thinking_config=genai.types.ThinkingConfig(thinking_level=thinking_level)
                    ),
                )
            
            return response.text, dict(response)
        except Exception as e:
            logging.error(f"Gemini API call failed: {str(e)}")
            return None, None
    

class GeminiInlineAPI(BaseAPI):
    
    def __init__(self, *args, **kwargs):
        self.client = genai.Client(*args, **kwargs)
    
    def upload_file(self, file_path: str):
        """Return a Gemini file part object (inline bytes)."""
        if file_path.endswith('.pdf'):
            mime_type = 'application/pdf'
        elif file_path.endswith('.md') or file_path.endswith('.txt'):
            mime_type = 'text/plain'
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

        return genai.types.Part.from_bytes(
            data=read_file_bytes(file_path),
            mime_type=mime_type,
        )
    
    def generate_content(self, model: str, prompt: str, contents: list = None, thinking_level: str = None, **kwargs):
        """Generate content using the Gemini API.
        
        Args:
            model: Model name.
            prompt: Prompt text.
            contents: Optional list of contents.
            thinking_level: Optional thinking level.
        """
        contents = contents or []
        
        try:
            if not thinking_level:
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents + [prompt],
                )
            else:
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents + [prompt],
                    config=genai.types.GenerateContentConfig(
                        thinking_config=genai.types.ThinkingConfig(thinking_level=thinking_level)
                    ),
                )
            
            return response.text, dict(response)
        except Exception as e:
            logging.error(f"Gemini API call failed: {str(e)}")
            return None, None
    


class OpenAIChatAPI(BaseAPI):
    """
    OpenAI Chat Completions 兼容 API 基类。
    适用于所有兼容 /v1/chat/completions 格式的 API（MiniMax、DeepSeek 等）。
    """

    def __init__(self, api_key: str, base_url: str, **kwargs):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

    @property
    def headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def upload_file(self, file_path: str):
        """Chat Completions API 不需要上传；直接返回文件路径。"""
        return file_path

    def _build_file_content(self, file_path: str) -> list:
        """
        将文件构建为多模态 content 元素列表。
        - PDF 文件拆成多张 PNG（每页一张），按 data:image/png base64 传入
          （MiniMax M3 / 多数 Chat Completions 实现只接受 image/* 媒体类型）。
          文件路径里包含 'material' 关键词的，限制为 8 页（material 是论文等长文档，
          不需要全量；slides 是给 judge 看的，要尽量全）。
        - 文本文件（.md / .txt）按单条 text 返回。
        - 其他文件按 octet-stream 兜底。

        返回 list[dict]，调用方应展平后再追加 prompt。
        """
        if file_path.endswith('.pdf'):
            # material 文件不需要全文，控制页数避免 token 超限
            max_pages = 8 if 'material' in file_path.lower() else 30
            # 渲染 DPI：固定 72。早期曾支持 MINIMAX_SLIDES_LOW_RES=1 降到 36
            # 绕开 MiniMax "image is sensitive" 422 拦截，但 36 DPI 下文字/细节
            # 太糊，已弃用；遇到 422 改为在调用侧重试或切其他模型。material 文件
            # 仍只取前 8 页，控制 token 消耗。
            dpi = 72
            return self._pdf_to_image_contents(file_path, max_pages=max_pages, dpi=dpi)
        elif file_path.endswith('.md') or file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return [{
                "type": "text",
                "text": text
            }]
        else:
            b64 = encode_file_base64(file_path)
            return [{
                "type": "image_url",
                "image_url": {
                    "url": f"data:application/octet-stream;base64,{b64}"
                }
            }]

    @staticmethod
    def _pdf_to_image_contents(pdf_path: str, max_pages: int = 30, dpi: int = 80) -> list:
        """
        把 PDF 的每一页渲染为 PNG (data URI)，
        返回一组 image_url content 元素。最多 max_pages 页、DPI 较低，
        防止 token 爆炸。MiniMax M3 通过 Chat Completions 不直接吃 PDF，
        必须先转图。

        实现细节：渲染结果缓存到 /tmp 下，键为 (pdf_path, mtime, dpi, max_pages)，
        避免同一 case 的多个 checklist item 重复渲染（每个 case 通常 50+ 次调用）。
        """
        try:
            import fitz  # pymupdf
        except ImportError as e:
            raise ImportError(
                "pymupdf is required to render PDF pages for OpenAIChatAPI. "
                "Install with: pip install pymupdf"
            ) from e

        import base64 as _b64
        import os as _os
        import hashlib as _hl

        try:
            mtime = _os.path.getmtime(pdf_path)
        except OSError:
            mtime = 0
        key = _hl.sha1(f"{pdf_path}|{mtime}|{dpi}|{max_pages}".encode()).hexdigest()[:16]
        cache_path = _os.path.join("/tmp", f"openai_chat_pdf_{key}.tsv")

        rendered: list[bytes] = []
        if _os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    rendered.append(_b64.b64decode(line))

        if not rendered:
            zoom = dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            with fitz.open(pdf_path) as doc:
                n = min(len(doc), max_pages)
                for i in range(n):
                    page = doc.load_page(i)
                    pix = page.get_pixmap(matrix=matrix, alpha=False)
                    rendered.append(pix.tobytes("png"))
            # 写缓存（每行一个 PNG 的 base64）
            with open(cache_path, "w", encoding="utf-8") as fp:
                for png_bytes in rendered:
                    fp.write(_b64.b64encode(png_bytes).decode("utf-8") + "\n")

        contents = []
        for png_bytes in rendered:
            b64 = _b64.b64encode(png_bytes).decode("utf-8")
            contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}"
                }
            })
        return contents

    def generate_content(self, model: str, prompt: str, file_paths: list = None, thinking_level: str = None, **kwargs):
        """
        调用 Chat Completions API。

        Args:
            model: 模型名称。
            prompt: Prompt 文本。
            file_paths: 文件路径列表。
            thinking_level: 思考模式。
                - 传 "disabled" / "adaptive"：映射为 MiniMax 文档的 thinking.type
                - 传 "low" / "medium" / "high"：视作禁用思考之外的强度档位，
                  未被 MiniMax 文档定义，所以这里统一映射为 adaptive。
                  若需精确控制，请直接传 "adaptive" 或 "disabled"。
        """
        file_paths = file_paths or []

        content_list = []
        for file_path in file_paths:
            content_list.extend(self._build_file_content(file_path))

        content_list.append({
            "type": "text",
            "text": prompt
        })

        data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": content_list
                }
            ],
            "stream": False,
            # 让 MiniMax 把思考内容拆到 reasoning_content 字段，
            # 便于我们之后做 prompt / answer 分离的解析。
            "reasoning_split": True,
        }

        if thinking_level:
            if thinking_level in ("disabled", "adaptive"):
                data["thinking"] = {"type": thinking_level}
            else:
                # low/medium/high 在 MiniMax 文档中没有定义；安全起见用 adaptive。
                data["thinking"] = {"type": "adaptive"}

        try:
            last_err: Exception | None = None
            for attempt in range(6):
                response = self.session.post(self.base_url, headers=self.headers, json=data, timeout=300)
                if response.status_code == 429 or response.status_code >= 500:
                    # 限流 / 5xx：长退避，最长 60s
                    sleep_s = min(60.0, 4.0 * (2 ** attempt))
                    logging.warning(
                        f"Chat API transient {response.status_code} on attempt {attempt + 1}; "
                        f"sleeping {sleep_s:.1f}s then retrying"
                    )
                    last_err = requests.exceptions.HTTPError(
                        f"{response.status_code} {response.text[:200]}"
                    )
                    time.sleep(sleep_s)
                    continue
                if response.status_code >= 400:
                    # 4xx 非 429：把 body 打全，便于诊断
                    logging.error(
                        f"Chat API client error {response.status_code}; body={response.text[:1000]}"
                    )
                    response.raise_for_status()
                result = response.json()
                text = result.get('choices', [{}])[0].get('message', {}).get('content', '')

                return text, result

            # 6 次都失败
            if last_err:
                logging.error(f"Chat API request failed after retries: {str(last_err)}")
            return None, None
        except requests.exceptions.RequestException as e:
            logging.error(f"Chat API request failed: {str(e)}")
            return None, None
        except Exception as e:
            logging.error(f"Chat API call failed: {str(e)}")
            return None, None


class OpenAIResponsesAPI(BaseAPI):
    """OpenAI Responses API (v1/responses) wrapper."""

    def __init__(self, api_key: str, **kwargs):
        self.api_key = api_key
        self.url = "https://api.openai.com/v1/responses"
        self.session = requests.Session()

    @property
    def headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def upload_file(self, file_path: str):
        return file_path

    def generate_content(self, model: str, prompt: str, file_paths: list = None, **kwargs):
        file_paths = file_paths or []

        content_list = []
        for file_path in file_paths:
            content_list.append({
                "type": "input_file",
                "filename": file_path,
                "file_data": f"data:application/pdf;base64,{encode_file_base64(file_path)}"
            })

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

        try:
            response = self.session.post(self.url, headers=self.headers, json=data, timeout=300)
            response.raise_for_status()

            result = response.json()
            text = result.get('output', [{}])[0].get('content', [{}])[0].get('text', '')

            return text, result
        except requests.exceptions.RequestException as e:
            logging.error(f"OpenAI Responses API request failed: {str(e)}")
            return None, None
        except Exception as e:
            logging.error(f"OpenAI Responses API call failed: {str(e)}")
            return None, None

