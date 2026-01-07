"""
Data Copilot Lab - LLM Integration
Integration with Large Language Models (OpenAI API)
"""

import json
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Union

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from src.core.config import settings
from src.core.exceptions import InvalidParameterError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"  # Future support
    LOCAL = "local"  # Future support for local models


class LLMModel(str, Enum):
    """OpenAI models"""
    GPT4 = "gpt-4"
    GPT4_TURBO = "gpt-4-turbo-preview"
    GPT35_TURBO = "gpt-3.5-turbo"
    GPT35_TURBO_16K = "gpt-3.5-turbo-16k"


class LLMIntegration:
    """
    Integration with Large Language Models

    Provides unified interface for:
    - Chat completions
    - Code generation
    - Data analysis assistance
    - Token management
    - Rate limiting
    - Error handling and retries
    """

    def __init__(
        self,
        provider: Union[str, LLMProvider] = LLMProvider.OPENAI,
        model: Union[str, LLMModel] = LLMModel.GPT35_TURBO,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: int = 60
    ):
        """
        Initialize LLM Integration

        Args:
            provider: LLM provider
            model: Model name
            api_key: API key (None = use from settings)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
        """
        self.logger = logger

        # Convert to enum
        if isinstance(provider, str):
            provider = LLMProvider(provider)
        if isinstance(model, str) and provider == LLMProvider.OPENAI:
            try:
                model = LLMModel(model)
            except ValueError:
                pass  # Allow custom model names

        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Setup API
        if provider == LLMProvider.OPENAI:
            if not OPENAI_AVAILABLE:
                raise InvalidParameterError("OpenAI package not installed. Install with: pip install openai")

            self.api_key = api_key or settings.openai_api_key
            if not self.api_key:
                raise InvalidParameterError("OpenAI API key not provided")

            openai.api_key = self.api_key
            self.logger.info(f"Initialized OpenAI integration with model: {model}")

        # Token tracking
        self.total_tokens_used = 0
        self.total_requests = 0

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        functions: Optional[List[Dict]] = None,
        function_call: Optional[Union[str, Dict]] = None
    ) -> Dict[str, Any]:
        """
        Generate chat completion

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            stream: Enable streaming
            functions: Function definitions for function calling
            function_call: Control function calling behavior

        Returns:
            Completion response
        """
        if self.provider != LLMProvider.OPENAI:
            raise InvalidParameterError(f"Provider {self.provider} not yet supported")

        self.logger.info(f"Requesting chat completion ({len(messages)} messages)")

        # Prepare parameters
        params = {
            "model": self.model.value if isinstance(self.model, LLMModel) else self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "stream": stream
        }

        if max_tokens or self.max_tokens:
            params["max_tokens"] = max_tokens or self.max_tokens

        if functions:
            params["functions"] = functions

        if function_call:
            params["function_call"] = function_call

        # Make request with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(**params)

                # Track tokens
                if hasattr(response, 'usage'):
                    self.total_tokens_used += response.usage.total_tokens
                    self.total_requests += 1

                    self.logger.debug(
                        f"Tokens used: {response.usage.total_tokens} "
                        f"(prompt: {response.usage.prompt_tokens}, "
                        f"completion: {response.usage.completion_tokens})"
                    )

                return self._format_response(response)

            except openai.error.RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    self.logger.warning(f"Rate limit hit. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

            except openai.error.APIError as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    self.logger.warning(f"API error: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

            except Exception as e:
                self.logger.error(f"LLM request failed: {e}")
                raise

    def generate_text(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate text completion from prompt

        Args:
            prompt: User prompt
            system_message: Optional system message
            **kwargs: Additional parameters for chat_completion

        Returns:
            Generated text
        """
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": prompt})

        response = self.chat_completion(messages, **kwargs)

        return response['content']

    def analyze_data_query(
        self,
        query: str,
        data_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze natural language data query

        Args:
            query: Natural language query
            data_context: Context about available data

        Returns:
            Structured query analysis
        """
        system_message = """You are a data analysis assistant.
Analyze the user's natural language query and extract:
1. Intent (e.g., 'filter', 'aggregate', 'visualize', 'analyze', 'predict')
2. Entities (columns, values mentioned)
3. Operations (what to do with the data)
4. Visualization type (if applicable)

Return a JSON object with these fields."""

        context_info = ""
        if data_context:
            context_info = f"\n\nAvailable data context:\n{json.dumps(data_context, indent=2)}"

        prompt = f"Query: {query}{context_info}"

        response = self.generate_text(
            prompt,
            system_message=system_message,
            temperature=0.3  # Lower temperature for structured output
        )

        # Try to parse JSON response
        try:
            analysis = json.loads(response)
        except json.JSONDecodeError:
            # If not valid JSON, return as plain text
            analysis = {"raw_response": response}

        return analysis

    def generate_code(
        self,
        task_description: str,
        language: str = "python",
        context: Optional[str] = None,
        examples: Optional[List[str]] = None
    ) -> str:
        """
        Generate code from task description

        Args:
            task_description: What the code should do
            language: Programming language
            context: Additional context (imports, available variables, etc.)
            examples: Example code snippets

        Returns:
            Generated code
        """
        system_message = f"""You are an expert {language} programmer specializing in data science.
Generate clean, efficient, well-commented code that follows best practices.
Only return the code without additional explanation unless specifically asked."""

        prompt_parts = [f"Task: {task_description}"]

        if context:
            prompt_parts.append(f"\nContext:\n{context}")

        if examples:
            prompt_parts.append("\nExamples:")
            for i, example in enumerate(examples, 1):
                prompt_parts.append(f"\nExample {i}:\n{example}")

        prompt = "\n".join(prompt_parts)

        code = self.generate_text(
            prompt,
            system_message=system_message,
            temperature=0.2  # Lower temperature for code generation
        )

        return self._clean_code_response(code)

    def explain_code(
        self,
        code: str,
        detail_level: str = "medium"
    ) -> str:
        """
        Explain what code does

        Args:
            code: Code to explain
            detail_level: 'brief', 'medium', or 'detailed'

        Returns:
            Code explanation
        """
        system_message = f"You are a code explainer. Provide {detail_level} explanations of code."

        prompt = f"Explain this code:\n\n```python\n{code}\n```"

        explanation = self.generate_text(prompt, system_message=system_message)

        return explanation

    def suggest_improvements(
        self,
        code: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Suggest code improvements

        Args:
            code: Code to improve
            context: Additional context

        Returns:
            Improvement suggestions
        """
        system_message = """You are a code reviewer. Analyze the code and suggest improvements for:
1. Performance
2. Readability
3. Best practices
4. Potential bugs

Return suggestions as a JSON object with categories."""

        prompt = f"Code to review:\n\n```python\n{code}\n```"
        if context:
            prompt += f"\n\nContext: {context}"

        response = self.generate_text(
            prompt,
            system_message=system_message,
            temperature=0.4
        )

        # Try to parse JSON
        try:
            suggestions = json.loads(response)
        except json.JSONDecodeError:
            suggestions = {"raw_suggestions": response}

        return suggestions

    def answer_question(
        self,
        question: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Answer data science related question

        Args:
            question: Question to answer
            context: Relevant context
            conversation_history: Previous conversation

        Returns:
            Answer
        """
        system_message = """You are a helpful data science assistant.
Answer questions clearly and concisely.
Provide code examples when relevant."""

        messages = [{"role": "system", "content": system_message}]

        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)

        # Add context if provided
        prompt = question
        if context:
            prompt = f"Context:\n{context}\n\nQuestion: {question}"

        messages.append({"role": "user", "content": prompt})

        response = self.chat_completion(messages)

        return response['content']

    def _format_response(self, response: Any) -> Dict[str, Any]:
        """Format OpenAI response"""
        if hasattr(response, 'choices') and len(response.choices) > 0:
            choice = response.choices[0]

            formatted = {
                "content": choice.message.content if hasattr(choice.message, 'content') else "",
                "role": choice.message.role if hasattr(choice.message, 'role') else "assistant",
                "finish_reason": choice.finish_reason if hasattr(choice, 'finish_reason') else None
            }

            # Handle function calls
            if hasattr(choice.message, 'function_call'):
                formatted['function_call'] = {
                    "name": choice.message.function_call.name,
                    "arguments": choice.message.function_call.arguments
                }

            # Add token usage
            if hasattr(response, 'usage'):
                formatted['usage'] = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }

            return formatted

        return {"content": "", "role": "assistant"}

    def _clean_code_response(self, code: str) -> str:
        """Clean code from markdown formatting"""
        # Remove markdown code blocks
        if "```" in code:
            # Extract code from ```python ... ``` or ``` ... ```
            parts = code.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1:  # Code blocks are at odd indices
                    # Remove language identifier if present
                    lines = part.strip().split('\n')
                    if lines[0].strip() in ['python', 'py', 'python3']:
                        code = '\n'.join(lines[1:])
                    else:
                        code = part.strip()
                    break

        return code.strip()

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get token usage statistics"""
        return {
            "total_tokens_used": self.total_tokens_used,
            "total_requests": self.total_requests,
            "average_tokens_per_request": (
                self.total_tokens_used / self.total_requests
                if self.total_requests > 0 else 0
            )
        }

    def estimate_cost(self, pricing_per_1k: Optional[float] = None) -> float:
        """
        Estimate cost based on token usage

        Args:
            pricing_per_1k: Price per 1000 tokens (None = use default)

        Returns:
            Estimated cost in USD
        """
        if pricing_per_1k is None:
            # Default pricing for GPT-3.5-turbo
            pricing_per_1k = 0.002  # $0.002 per 1K tokens

        cost = (self.total_tokens_used / 1000) * pricing_per_1k

        return cost
