import asyncio
import logging
from typing import Optional, Union, Dict

import litellm
from litellm.exceptions import AuthenticationError, APIError, BadRequestError, RateLimitError

from .schemas import EvaluationConfig, EvaluationResult

logger = logging.getLogger("LiteLLMEvaluator")
logging.basicConfig(level=logging.INFO)


class LiteLLMEvaluator:
    """Unified LiteLLM evaluator with proper provider support."""

    def __init__(self, config: EvaluationConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logger

        if not config.model_id.strip():
            raise ValueError("model_id must be provided and non-empty.")
        if not config.api_key.strip():
            raise ValueError("api_key must be provided and non-empty.")

        self.model = config.model_id.strip()
        self.api_key = config.api_key
        self.api_url = (config.api_url or "").strip() or "https://api.openai.com/v1"
        self.provider = config.provider or "openai"  # REQUIRED for LiteLLM
        litellm.drop_params = True

        self.logger.info(
            f"[LiteLLMEvaluator] Using model={self.model} api_base={self.api_url} provider={self.provider}"
        )

    def build_prompt(
        self,
        user_message: Optional[str],
        generated_text: str,
        expected_text: str,
    ) -> str:
        user_msg = user_message or "(no user message provided)"
        return (
            "You are an expert evaluator. Compare the generated text with the expected text. "
            "Score on semantic similarity, factual accuracy, completeness. "
            'Return JSON ONLY: {"match_level": <0-5>, "justification": "<=35 words>", "metadata": {}}\n\n'
            f'User Message: """{user_msg}"""\n'
            f'Expected: """{expected_text}"""\n'
            f'Generated: """{generated_text}"""'
        )

    async def call_llm(self, prompt: str) -> Union[Dict, str]:
        """Async call to LiteLLM using run_in_executor with provider included"""
        loop = asyncio.get_running_loop()

        def sync_call():
            kwargs = dict(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.llm_config.get("temperature", 0.0),
                max_tokens=self.config.llm_config.get("max_tokens", 150),
                api_base=self.api_url,
                api_key=self.api_key,
            )
            # Include provider explicitly for non-OpenAI
            if self.provider and self.provider.lower() != "openai":
                kwargs["provider"] = self.provider
            return litellm.completion(**kwargs)

        try:
            resp = await loop.run_in_executor(None, sync_call)
            return resp.choices[0].message.content.strip()
        except (AuthenticationError, APIError, BadRequestError, RateLimitError) as e:
            self.logger.error(f"[LiteLLM] API error: {e}")
            return {"error": str(e)}
        except Exception as e:
            self.logger.error(f"[LiteLLM] Unexpected error: {e}", exc_info=True)
            return {"error": str(e)}

    async def evaluate(
        self,
        generated_text: str,
        expected_text: str,
        user_message: Optional[str] = None,
    ) -> EvaluationResult:
        prompt = self.build_prompt(user_message, generated_text, expected_text)
        raw = await self.call_llm(prompt)

        try:
            import json
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            parsed = {"error": str(raw)}

        if isinstance(parsed, dict) and "match_level" in parsed:
            return EvaluationResult(**parsed)

        return EvaluationResult(
            match_level=0,
            justification="LiteLLM evaluation failed or returned invalid JSON.",
            metadata={"raw": raw if isinstance(raw, str) else str(raw)},
        )
