from abc import ABC, abstractmethod
import logging
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
    


class OpenAIAPI(BaseAPI):
    """OpenAI Response API wrapper."""
    
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
                "file_data": f"data:application/pdf;base64,{encode_file_base64(file_path)}"
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
        
        try:
            # Reuse connections via session.
            response = self.session.post(self.url, headers=self.headers, json=data, timeout=300)
            response.raise_for_status()

            # Extract text content from the response.
            result = response.json()
            text = result.get('output', [{}])[0].get('content', [{}])[0].get('text', '')
            
            return text, result
        except requests.exceptions.RequestException as e:
            logging.error(f"OpenAI API request failed: {str(e)}")
            return None, None
        except Exception as e:
            logging.error(f"OpenAI API call failed: {str(e)}")
            return None, None

