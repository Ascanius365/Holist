import re
import json
import inspect
import warnings
from copy import deepcopy
from dataclasses import asdict
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from litellm import acompletion, ModelResponse
from dotenv import load_dotenv
from premem_module.src.utils.io import read_yaml
import tiktoken
from transformers import AutoTokenizer

# Load environment variables
load_dotenv()

# List of available models
MODEL_POOL = [
    "gpt-3.5-turbo-0125",
    "gpt-4.1-nano",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
    "qwen/qwen3-coder:free",
    "qwen/qwen-2.5-coder-32b-instruct",
    # "o3-mini",
    # "o1",
    # "claude-3-7-sonnet-20250219",
    # "claude-3-5-haiku-20241022",
    # "gemini/gemini-2.5-pro-preview-03-25",
    # "gemini/gemini-2.0-flash",
    # "gemini/gemini-2.0-flash-lite",
]


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def to_dict(self):
        """
        Convert the LLMResponse object to a dictionary.
        """
        return asdict(self)


@dataclass
class LLMResponse:
    response: str
    model: str
    usage: Usage

    def to_dict(self):
        """
        Convert the LLMResponse object to a dictionary.
        """
        return asdict(self)


class LLMAgent:
    def __init__(self, model_name: str):
        """Initialize LLM client
        model_name: Name of the model to use for completion
        """
        self._model_name = model_name
        self._available_models = MODEL_POOL
        self._valid_parameter = set(inspect.signature(acompletion).parameters.keys()) | {
            "api_base",
            "num_retries",
        }
        # Initialize tokenizer based on model
        self._tokenizer = None  # For Claude models, handled differently

    async def get_completion(
        self,
        prompt_path: Optional[str] = None,
        messages: Dict[str, Any] = None,
        api_base: Optional[str] = None,
        raw_response: bool = False,
        **kwargs,
    ) -> LLMResponse | ModelResponse:
        """Send messages to LLM and get completion response

        Args:
            messages: List of message dictionaries containing 'role' and 'content'
            api_base: Optional API base URL override
            **kwargs: Additional keyword arguments to pass to completion function

        Returns:
            LLMResponse: Object containing:
                - response: Generated text response
                - model: Name of model used
                - usage: Token usage statistics
        """
        if prompt_path is None and messages is None:
            raise ValueError("Either prompt_path or messages must be provided")
        elif prompt_path is not None and messages is not None:
            raise ValueError("prompt_path and messages cannot both be provided")
        elif messages is not None:
            prompt = {"messages": messages}
        elif prompt_path is not None:
            prompt = read_yaml(prompt_path)
        messages = self._make_messages(prompt, **kwargs)

        is_openrouter = self._model_name.startswith("openrouter/")

        messages["model"] = (
            "hosted_vllm/" + self._model_name
            if self._model_name not in self._available_models
               and "hosted_vllm" not in self._model_name
               and not is_openrouter  # Verhindert den AuthenticationError
            else self._model_name
        )
        messages["api_base"] = api_base
        if messages["model"].startswith("hosted_vllm") and api_base is None:
            warnings.warn("api_base is not provided")

        # Generate response through LiteLLM
        response = await acompletion(
            **{k: v for k, v in messages.items() if k in self._valid_parameter},
        )
        if raw_response:
            return response

        # Convert response to LLMResponse object
        return LLMResponse(
            response=response.choices[0].message.content,
            model=response.model,
            usage=Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ),
        )

    def _make_messages(self, prompt, **kwargs):
        messages = self._fill_prompt(prompt, **kwargs)
        for key in list(self._valid_parameter):
            if key in kwargs:
                messages[key] = kwargs[key]
        return messages

    def _fill_prompt(self, prompt, **kwargs):
        def _fill(msg, kwargs):
            all_context_input = []
            context_input = re.findall(r"(\{\{\$.+?\}\})", msg)
            for input_ in context_input:
                str_to_replace = kwargs[input_[3:-2]]
                all_context_input.append(input_[3:-2])
                if isinstance(str_to_replace, int):
                    str_to_replace = str(str_to_replace)
                if isinstance(str_to_replace, list):
                    str_to_replace = "- " + "\n- ".join(str_to_replace)
                if isinstance(str_to_replace, dict):
                    str_to_replace = json.dumps(str_to_replace).replace('", "', '",\n"')
                msg = msg.replace(input_, str_to_replace)
            return msg, all_context_input

        prompt = deepcopy(prompt)
        all_context_input = []

        for msg in prompt["messages"]:
            output = _fill(msg["content"], kwargs)
            msg["content"] = output[0]
            all_context_input.extend(output[1])
        if len(set(kwargs) - (set(all_context_input) | self._valid_parameter)) > 0:
            raise ValueError(
                f"Invalid context input: {set(kwargs) - (set(all_context_input) | self._valid_parameter)}"
            )
        return prompt

    def count_tokens(self, text: Union[str, List[Dict[str, str]]]) -> int:
        """
        Calculate the number of tokens in text or messages.

        Args:
            text: A string or list of messages (list of role-content dictionaries)
            system: System message (for Claude models)

        Returns:
            int: Number of tokens
        """
        model_name = self._model_name

        # Convert list of messages to appropriate format
        messages = None
        if isinstance(text, list) and len(text) > 0 and isinstance(text[0], dict):
            messages = text
            # Combine into single string (for OpenAI/HF tokenizers)
            combined_text = ""
            for msg in messages:
                if "role" in msg and "content" in msg:
                    combined_text += f"{msg['role']}: {msg['content']}\n"
            text = combined_text

        # OpenAI models (GPT)
        if model_name.startswith("gpt-"):
            return len(self._tokenizer.encode(text, allowed_special={"<|endoftext|>"}))

        # All other models (using Transformers)
        else:
            return len(self._tokenizer.encode(text))

    def truncate_text(self, text: str, max_tokens: int) -> str:
        model_name = self._model_name
        if model_name.startswith("gpt-"):
            return self._tokenizer.decode(self._tokenizer.encode(text)[:max_tokens])

        # Claude models
        elif "claude" in model_name:
            raise NotImplementedError("Claude models are not supported")

        # All other models (using Transformers)
        else:
            return self._tokenizer.decode(
                self._tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
            )
