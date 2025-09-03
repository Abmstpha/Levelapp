import os
import logging
from typing import Optional
import json

from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import HumanMessage, SystemMessage

from .schemas import EvaluationConfig, EvaluationResult

logger = logging.getLogger("LiteLLMEvaluator")


class LiteLLMEvaluator:
    """
    Provides automatic fallback and unified interface to the available LLM providers.
    """

    def __init__(self, config: EvaluationConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger("LiteLLMEvaluator")
        
        # Setup model with fallback
        model = config.model_id.strip() if config.model_id else "gpt-4o-mini"
        
        # Configure LangChain ChatLiteLLM
        self.chat = ChatLiteLLM(
            model=model,
            api_key=config.api_key,
            api_base=config.api_url,
            temperature=config.llm_config.get("temperature", 0.0),
            max_tokens=config.llm_config.get("max_tokens", 150),
        )
        
        self.logger.info(f"[LiteLLMEvaluator] Using model={model}")

    async def evaluate(
        self,
        generated_text: str,
        expected_text: str,
        user_message: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate generated text against expected text using LangChain ChatLiteLLM.
        """
        try:
            # Build evaluation prompt
            user_msg = user_message or "(no user message provided)"
            
            system_prompt = (
                "You are an expert evaluator. Compare the generated text with the expected text. "
                "Score on semantic similarity, factual accuracy, completeness. "
                'Return JSON ONLY: {"match_level": <0-5>, "justification": "<=35 words", "metadata": {}}'
            )
            
            user_prompt = (
                f'User Message: """{user_msg}"""\n'
                f'Expected: """{expected_text}"""\n'
                f'Generated: """{generated_text}"""'
            )
            
            # Create messages
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            # Call LangChain ChatLiteLLM (handles fallbacks automatically)
            response = await self.chat.ainvoke(messages)
            content = response.content.strip()
            
            # Try to parse JSON response
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "match_level" in parsed:
                    return EvaluationResult(**parsed)
            except json.JSONDecodeError:
                pass
            
            # Fallback to heuristic parsing
            match_level = 1.0 if "yes" in content.lower() else 0.0
            return EvaluationResult(
                match_level=match_level,
                justification=content,
                metadata={"format": "heuristic_parsed"}
            )
            
        except Exception as e:
            self.logger.error(f"LiteLLM evaluation failed: {e}")
            return EvaluationResult(
                match_level=0.0,
                justification=f"Evaluation failed: {str(e)}",
                metadata={"error": str(e)}
            )
