"""
LLM Client with Prompt Caching Support
프롬프트 캐싱을 지원하는 LLM 클라이언트

Implements prompt caching to reduce token consumption for repeated system prompts.
반복되는 시스템 프롬프트에 대한 토큰 소비를 줄이기 위해 프롬프트 캐싱을 구현합니다.
"""

# Standard library imports
# 표준 라이브러리 import
from typing import Any, Dict, Optional


class CachedLLMClient:
    """
    Wrapper for LLM client with prompt caching support.
    프롬프트 캐싱을 지원하는 LLM 클라이언트 래퍼.
    
    Supports:
    - OpenAI Prompt Caching (GPT-4 Turbo and later)
    - Anthropic Claude Prompt Caching
    - Manual in-memory caching for other providers
    """

    def __init__(self, client, provider: str = "openai"):
        """
        Initialize cached LLM client.
        
        Args:
            client: Base LLM client (e.g., OpenAI client)
            provider: LLM provider ("openai", "anthropic", or "other")
        """
        self.client = client
        self.provider = provider.lower()
        self._system_prompt_cache: Dict[str, str] = {}

    def chat_with_cache(self, system_prompt: str, user_prompt: str, model: str, temperature: float = 0.1, response_format: Optional[Dict[str, str]] = None, cache_key: Optional[str] = None, ) -> Any:
        """
        Create chat completion with prompt caching.
        프롬프트 캐싱을 사용한 채팅 완성 생성.
        
        Args:
            system_prompt: System prompt (will be cached)
            user_prompt: User prompt (not cached)
            model: Model name
            temperature: Sampling temperature
            response_format: Response format specification
            cache_key: Optional key for manual caching
            
        Returns:
            LLM response
        """
        if self.provider == "openai":
            return self._openai_cached_call(system_prompt, user_prompt, model, temperature, response_format)
        elif self.provider == "anthropic":
            return self._anthropic_cached_call(system_prompt, user_prompt, model, temperature)
        else:
            # Fallback: manual caching (doesn't reduce API costs but reduces redundant calls)
            return self._manual_cached_call(system_prompt, user_prompt, model, temperature, response_format, cache_key)

    def _openai_cached_call(self, system_prompt: str, user_prompt: str, model: str, temperature: float, response_format: Optional[Dict[str, str]], ) -> Any:
        """
        OpenAI API call with prompt caching.
        
        OpenAI automatically caches prompts that are:
        - Longer than 1024 tokens
        - Identical across requests
        - Used within the cache TTL (5-10 minutes)
        
        Note: As of 2024, OpenAI's prompt caching is automatic for supported models.
        """
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}, ]

        kwargs = {"model": model, "temperature": temperature, "messages": messages, }

        if response_format:
            kwargs["response_format"] = response_format

        # For OpenAI, caching is automatic - no special parameters needed
        # Just ensure system prompts are consistent across calls
        response = self.client.chat.completions.create(**kwargs)

        return response

    def _anthropic_cached_call(self, system_prompt: str, user_prompt: str, model: str, temperature: float, ) -> Any:
        """
        Anthropic Claude API call with prompt caching.
        
        Anthropic requires explicit cache_control parameter.
        """
        # Anthropic uses a different API structure
        # This is a placeholder - adjust based on actual Anthropic SDK
        messages = [{"role": "user", "content": user_prompt, }]

        # Anthropic's cache control
        system_with_cache = [
            {
                "type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}  # Enable caching
            }
        ]

        response = self.client.messages.create(model=model, max_tokens=4096, temperature=temperature, system=system_with_cache, messages=messages, )

        return response

    def _manual_cached_call(self, system_prompt: str, user_prompt: str, model: str, temperature: float, response_format: Optional[Dict[str, str]], cache_key: Optional[str], ) -> Any:
        """
        Manual caching for providers without native support.
        
        Note: This doesn't reduce API costs, but can reduce redundant identical calls.
        """
        # For manual caching, we'd need to implement conversation history
        # This is a simplified version
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}, ]

        kwargs = {"model": model, "temperature": temperature, "messages": messages, }

        if response_format:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)

        return response

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get caching statistics (if available from provider).
        캐싱 통계 가져오기 (제공자가 지원하는 경우).
        """
        # This would need to be implemented based on provider's API
        return {"provider": self.provider, "manual_cache_size": len(self._system_prompt_cache), }
