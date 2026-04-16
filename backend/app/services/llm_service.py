"""LLM service — calls an OpenAI-compatible chat completion API for grounded answer generation.

This service is the final layer in the RAG pipeline.  It receives structured
messages (system + user) from the prompt service and sends them to a chat
completion endpoint.  The current configuration targets Groq's OpenAI-compatible
API with Llama 3.3 70B, but any provider that speaks the OpenAI chat format
(OpenAI, Together, Ollama, vLLM, etc.) works by changing the base URL and model
in ``.env``.
"""

from __future__ import annotations

import logging
import time

from openai import OpenAI

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class LLMService:
    """OpenAI-compatible LLM client for grounded response generation."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        settings = get_settings()
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self._model = settings.openai_model
        logger.info(
            "LLMService initialised — provider: %s, model: %s, base_url: %s",
            provider,
            self._model,
            settings.openai_base_url or "default (api.openai.com)",
        )

    def generate_grounded_response(self, prompt: str | list[dict[str, str]]) -> str:
        """Generate a final answer from grounded prompt messages.

        Args:
            prompt: Either a list of chat messages
                    (``[{"role": "system", ...}, {"role": "user", ...}]``)
                    or a flat string (legacy path — wrapped as a single
                    user message).

        Returns:
            The model's response text.
        """
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        logger.info(
            "Calling %s/%s — %d messages, %d chars total",
            self.provider,
            self._model,
            len(messages),
            sum(len(m["content"]) for m in messages),
        )

        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
        )
        elapsed = time.perf_counter() - start

        answer = response.choices[0].message.content or ""
        usage = response.usage

        logger.info(
            "LLM response in %.2fs — %d chars, tokens: %s prompt / %s completion",
            elapsed,
            len(answer),
            usage.prompt_tokens if usage else "?",
            usage.completion_tokens if usage else "?",
        )

        return answer
