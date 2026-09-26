"""Universal Session Manager with Provider Detection

Automatically detects LLM provider (OpenAI, Ollama, etc.) and applies
appropriate caching strategy.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from config.config import Settings


class UniversalLLMSession:
    """Universal LLM session with provider-specific optimizations.

    Supports:
    - OpenAI: Automatic prompt caching
    - Ollama: KV cache optimization with keep_alive
    - Other providers: Manual conversation history
    """

    def __init__(self, client, system_prompt: str, session_id: str, ttl_minutes: int = 60, provider: str = "auto", base_url: Optional[str] = None, max_history: int = 10):
        """Initialize universal LLM session.

        Args:
            client: LLM client
            system_prompt: System prompt (sent once at session start)
            session_id: Unique session identifier
            ttl_minutes: Session time-to-live in minutes
            provider: LLM provider ("auto", "openai", "ollama", "other")
            base_url: Base URL for API (used for provider detection)
            max_history: Maximum number of conversation turns to keep (0 = no history, -1 = unlimited)

                        Stage 1 should use 0 or 1 (each column is independent)

                        Stage 2 can use higher values if context is needed
        """
        self.client = client
        self.session_id = session_id
        self.created_at = datetime.now()
        self.ttl = timedelta(minutes=ttl_minutes)
        self.last_used = datetime.now()
        self.max_history = max_history

        # Auto-detect provider
        self.provider = self._detect_provider(provider, base_url)

        # Initialize conversation with system prompt
        self.messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

        self.total_tokens_saved = 0
        self.system_prompt_tokens = self._estimate_tokens(system_prompt)

        # Provider-specific settings
        self.ollama_keep_alive = "60m" if self.provider == "ollama" else None

    def _detect_provider(self, provider: str, base_url: Optional[str]) -> str:
        """Auto-detect LLM provider.

        Args:
            provider: Manual provider specification or "auto"
            base_url: Base URL for API

        Returns:
            Detected provider: "openai", "ollama", or "other"
        """
        if provider != "auto":
            return provider.lower()

        # Detect from base_url
        if base_url:
            if "localhost:11434" in base_url or "ollama" in base_url.lower():
                return "ollama"
            elif "openai" in base_url.lower() or "api.openai.com" in base_url:
                return "openai"

        # Detect from client type
        client_type = type(self.client).__name__.lower()
        if "ollama" in client_type:
            return "ollama"
        elif "openai" in client_type:
            return "openai"

        return "other"

    def add_user_message(self, content: str) -> None:
        """Add user message to conversation history."""
        self.messages.append({"role": "user", "content": content})
        # Do not trim here! We need to send this message.
        self.last_used = datetime.now()

    def add_assistant_message(self, content: str) -> None:
        """Add assistant message to conversation history."""
        self.messages.append({"role": "assistant", "content": content})
        self._trim_history()

    def _trim_history(self) -> None:
        """Trim conversation history to max_history limit.

        Keeps system prompt + last N conversation turns.
        """
        if self.max_history < 0:
            # max_history=-1: Unlimited history (no trimming)

            return
        elif self.max_history == 0:
            # max_history=0: Keep only system prompt (no conversation history)

            # Stage 1 should use this mode since each column is independent

            self.messages = [self.messages[0]]  # Keep only system prompt
        elif self.max_history > 0 and len(self.messages) > 1:
            # max_history=N: Keep system prompt + last N*2 messages (N user + N assistant pairs)

            # Calculate how many messages to keep (system + user/assistant pairs)

            max_messages = 1 + (self.max_history * 2)  # 1 system + N pairs

            if len(self.messages) > max_messages:
                # Keep system prompt + last N conversation turns

                system_prompt = self.messages[0]
                recent_messages = self.messages[-(self.max_history * 2):]
                self.messages = [system_prompt] + recent_messages

    def supports_streaming(self) -> bool:
        """Check if current provider supports streaming.

        Returns:
            True if provider supports streaming, False otherwise
        """
        return self.provider == "openai"

    def chat(
            self,
            user_prompt: str,
            model: str,
            temperature: float = 0.0,
            response_format: Optional[Dict[str, str]] = None,
            stream: bool = False,
            stream_callback: Optional[Any] = None,
            **kwargs, ) -> Any:
        """Send a message and get response with provider-specific optimizations.

        Args:
            user_prompt: User's message
            model: Model name
            temperature: Sampling temperature
            response_format: Response format specification
            stream: Enable streaming mode for real-time token delivery
            stream_callback: Optional callback function for streaming chunks
            **kwargs: Additional provider-specific parameters

        Returns:
            LLM response (accumulated if streaming)
        """
        # Stage 1 uses max_history=0 because every column is an independent
        # request. Build an immutable per-call message list in that mode so
        # concurrent requests cannot overwrite one another's prompts.
        if self.max_history == 0:
            request_messages = [
                dict(self.messages[0]),
                {"role": "user", "content": user_prompt},
            ]
            self.last_used = datetime.now()
        else:
            # Ensure history is trimmed before starting new turn

            self._trim_history()

            # Add user message to history
            self.add_user_message(user_prompt)
            request_messages = list(self.messages)

        # Prepare base arguments
        api_kwargs = {
            "model": model,
            "temperature": temperature,
            "messages": request_messages,
        }

        # Add response format if specified
        if response_format:
            api_kwargs["response_format"] = response_format

        # Apply provider-specific optimizations
        if self.provider == "ollama":
            # Ollama with OpenAI-compatible API:
            # Keep response_format. Current Ollama OpenAI-compatible chat
            # completions support JSON mode, which prevents avoidable
            # free-form or syntactically incomplete responses.
            pass

        elif self.provider == "openai":
            # OpenAI: Automatic prompt caching (no special parameters needed)
            # System prompt will be cached automatically if >1024 tokens
            pass

        # Merge additional kwargs
        api_kwargs.update(kwargs)

        # Handle streaming mode
        if stream and self.supports_streaming():
            api_kwargs["stream"] = True
            response = self._handle_streaming_response(api_kwargs, stream_callback)
        else:
            # Non-streaming mode (original behavior)
            response = self.client.chat.completions.create(**api_kwargs)

            # Independent Stage 1 calls must not mutate shared history.
            if self.max_history != 0:
                assistant_message = response.choices[0].message.content
                self.add_assistant_message(assistant_message)

            # Track tokens saved (system prompt not resent)
            self.total_tokens_saved += self.system_prompt_tokens

            self.last_used = datetime.now()

        self._persist_audit_record(
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            response_format=response_format,
            stream=bool(stream and self.supports_streaming()),
            response_content=response.choices[0].message.content,
        )

        return response

    def _persist_audit_record(
            self,
            user_prompt: str,
            model: str,
            temperature: float,
            response_format: Optional[Dict[str, str]],
            stream: bool,
            response_content: str,
    ) -> None:
        """Persist one complete prompt/response exchange for local auditing.

        Each request receives its own atomically renamed JSON file, avoiding
        cross-thread record interleaving. The cache contains no credentials;
        callers are responsible for ensuring prompts contain schema metadata
        rather than raw records.
        """
        if not Settings.LLM_AUDIT_ENABLED:
            return

        try:
            audit_dir = Path(Settings.LLM_AUDIT_DIR)
            audit_dir.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(audit_dir, 0o700)
            except OSError:
                pass

            created_at = datetime.now().astimezone()
            record_id = uuid4().hex
            record = {
                "record_id": record_id,
                "created_at": created_at.isoformat(),
                "session_id": self.session_id,
                "provider": self.provider,
                "model": model,
                "temperature": temperature,
                "stream": stream,
                "response_format": response_format,
                "system_prompt": self.messages[0]["content"],
                "user_prompt": user_prompt,
                "response": response_content,
            }

            filename = (
                f"{created_at.strftime('%Y%m%dT%H%M%S%f')}_{record_id}.json"
            )
            final_path = audit_dir / filename
            temporary_path = audit_dir / f".{filename}.tmp"

            with open(temporary_path, "w", encoding="utf-8") as audit_file:
                json.dump(record, audit_file, ensure_ascii=False, indent=2)
                audit_file.flush()
                os.fsync(audit_file.fileno())

            try:
                os.chmod(temporary_path, 0o600)
            except OSError:
                pass
            os.replace(temporary_path, final_path)
        except Exception as exc:
            # Audit persistence must not turn a successful model decision into
            # a failed pipeline run. Surface the failure in local logs.
            print(f"[WARN] Failed to persist LLM audit record: {exc}")

    def _handle_streaming_response(self, api_kwargs: Dict[str, Any], stream_callback: Optional[Any]) -> Any:
        """Handle streaming response from OpenAI API.

        Args:
            api_kwargs: API call arguments
            stream_callback: Optional callback for streaming chunks

        Returns:
            Mock response object with accumulated content
        """
        # Make streaming API call
        stream = self.client.chat.completions.create(**api_kwargs)

        # Accumulate the full response
        accumulated_content = ""

        # Process streaming chunks
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    content_chunk = delta.content
                    accumulated_content += content_chunk

                    # Call callback if provided
                    if stream_callback:
                        stream_callback(content_chunk)

        # Independent Stage 1 calls must not mutate shared history.
        if self.max_history != 0:
            self.add_assistant_message(accumulated_content)

        # Track tokens saved (system prompt not resent)
        self.total_tokens_saved += self.system_prompt_tokens

        self.last_used = datetime.now()

        # Return a mock response object compatible with existing code
        return self._create_mock_response(accumulated_content)

    def _create_mock_response(self, content: str) -> Any:
        """Create a mock response object for streaming compatibility.

        Args:
            content: Accumulated response content

        Returns:
            Mock response object with structure matching OpenAI response
        """

        class MockMessage:
            def __init__(self, content: str):
                self.content = content

        class MockChoice:
            def __init__(self, message):
                self.message = message

        class MockResponse:
            def __init__(self, content: str):
                self.choices = [MockChoice(MockMessage(content))]

        return MockResponse(content)

    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.now() - self.last_used > self.ttl

    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            "session_id": self.session_id,
            "provider": self.provider,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat(),
            "message_count": len(self.messages),
            "estimated_tokens_saved": self.total_tokens_saved,
            "is_expired": self.is_expired(),
            "optimization": self._get_optimization_info(),
        }

    def _get_optimization_info(self) -> str:
        """Get optimization strategy info."""
        if self.provider == "ollama":
            return f"Ollama KV Cache (keep_alive: {self.ollama_keep_alive})"
        elif self.provider == "openai":
            return "OpenAI Automatic Prompt Caching"
        else:
            return "Manual Conversation History"

    def reset(self, new_system_prompt: Optional[str] = None) -> None:
        """Reset session with optional new system prompt.

        Args:
            new_system_prompt: New system prompt (if None, keeps current)
        """
        if new_system_prompt:
            self.messages = [{"role": "system", "content": new_system_prompt}]
            self.system_prompt_tokens = self._estimate_tokens(new_system_prompt)
        else:
            # Keep only the system prompt
            self.messages = [self.messages[0]]

        self.total_tokens_saved = 0
        self.last_used = datetime.now()

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough estimation of token count.

        Note: For accurate counting, use tiktoken library.
        """
        # Rough estimate: ~4 characters per token for English
        return len(text) // 4


class UniversalSessionManager:
    """Manages multiple universal LLM sessions with provider detection."""

    def __init__(self, client, provider: str = "auto", base_url: Optional[str] = None):
        """Initialize universal session manager.

        Args:
            client: LLM client
            provider: LLM provider ("auto", "openai", "ollama", "other")
            base_url: Base URL for API (used for provider detection)
        """
        self.client = client
        self.provider = provider
        self.base_url = base_url
        self.sessions: Dict[str, UniversalLLMSession] = {}

    def get_or_create_session(self, session_id: str, system_prompt: str, ttl_minutes: int = 60, max_history: int = 10) -> UniversalLLMSession:
        """Get existing session or create new one with provider detection.

        Args:
            session_id: Session identifier
            system_prompt: System prompt for new sessions
            ttl_minutes: Session TTL
            max_history: Maximum conversation history to keep (0 = no history, -1 = unlimited)

        Returns:
            Universal LLM session
        """
        # Clean up expired sessions
        self._cleanup_expired()

        # Get or create session
        if session_id not in self.sessions:
            self.sessions[session_id] = UniversalLLMSession(self.client, system_prompt, session_id, ttl_minutes, provider=self.provider, base_url=self.base_url, max_history=max_history)
        elif self.sessions[session_id].is_expired():
            # Recreate expired session
            self.sessions[session_id] = UniversalLLMSession(self.client, system_prompt, session_id, ttl_minutes, provider=self.provider, base_url=self.base_url, max_history=max_history)

        return self.sessions[session_id]

    def close_session(self, session_id: str) -> None:
        """Close and remove a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def _cleanup_expired(self) -> None:
        """Remove expired sessions."""
        expired = [sid for sid, session in self.sessions.items() if session.is_expired()]
        for sid in expired:
            del self.sessions[sid]

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all sessions."""
        return {"active_sessions": len(self.sessions), "provider": self.provider, "sessions": {sid: session.get_stats() for sid, session in self.sessions.items()}, }
