from typing import Optional, Any, List
from .base import BaseAPI, GeminiAPI, OpenAIChatAPI, OpenAIResponsesAPI, GeminiInlineAPI



class JudgeAPI:
    """
    Evaluation-task API adapter.

    Encapsulates evaluation-specific logic and translates the evaluation interface
    (slides_obj, material_objs, etc.) into what the underlying API client expects.
    This keeps the underlying API clients decoupled from evaluation concerns.
    """

    def __init__(self, api_client: BaseAPI):
        """
        Initialize JudgeAPI.

        Args:
            api_client: Underlying API client instance (GeminiAPI, OpenAIChatAPI, etc.).
        """
        self.api_client = api_client
        # Determine API type at init time to avoid runtime isinstance checks.
        self._is_gemini_api = isinstance(api_client, (GeminiAPI, GeminiInlineAPI))
        self._is_file_path_api = isinstance(api_client, (OpenAIChatAPI, OpenAIResponsesAPI))

        if not self._is_gemini_api and not self._is_file_path_api:
            raise ValueError(f"Unsupported API client: {type(api_client)}")
    
    def generate_content(
        self,
        model: str,
        prompt: str,
        slides_obj: Optional[Any] = None,
        slides_path: Optional[str] = None,
        material_objs: Optional[List[Any]] = None,
        material_paths: Optional[List[str]] = None,
        thinking_level: Optional[str] = None,
        **kwargs
    ) -> tuple[Optional[str], Optional[dict]]:
        """
        Generate content (evaluation-task interface).
        
        Automatically handles conversion between file objects and file paths based on the underlying API type.
        
        Args:
            model: Model name.
            prompt: Prompt text.
            slides_obj: Slides file object (for Gemini API).
            slides_path: Slides file path (for OpenAI-like API).
            material_objs: List of material file objects (for Gemini API).
            material_paths: List of material file paths (for OpenAI-like API).
            thinking_level: Thinking level (Gemini API only).
            **kwargs: Other optional parameters.
            
        Returns:
            (response_text, response_json_dict)
            - response_text: Text content from the API response.
            - response_json_dict: Raw JSON dict (used for billing/usage tracking).
        """
        if self._is_gemini_api:
            # Gemini API: use file objects and a contents list.
            contents = []
            if slides_obj:
                contents.append(slides_obj)
            if material_objs:
                contents.extend(material_objs)

            return self.api_client.generate_content(
                model=model,
                prompt=prompt,
                contents=contents,
                thinking_level=thinking_level,
                **kwargs
            )

        elif self._is_file_path_api:
            # OpenAI Chat Completions / Responses API: use file path list.
            file_paths = []
            if slides_path:
                file_paths.append(slides_path)
            if material_paths:
                file_paths.extend(material_paths)

            return self.api_client.generate_content(
                model=model,
                prompt=prompt,
                file_paths=file_paths,
                thinking_level=thinking_level,
                **kwargs
            )

        else:
            raise ValueError(f"Unsupported API client: {type(self.api_client)}")

    def upload_file(self, file_path: str) -> Any:
        """
        Upload a file (evaluation-task interface).
        
        Args:
            file_path: File path.
            
        Returns:
            A file object or file path, depending on the underlying API type.
        """
        return self.api_client.upload_file(file_path)
