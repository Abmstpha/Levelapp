import os
import logging
import asyncio
from typing import Dict, Any

# Import your LLM libraries
import litellm
from litellm.exceptions import AuthenticationError, APIError, RateLimitError

# If you have other provider SDKs, import them here
# from openai_sdk import OpenAIClient
# from ionos_sdk import IonosClient

logger = logging.getLogger("EvaluationService")
logging.basicConfig(level=logging.INFO)


# Example provider configs
providers = {
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "api_url": "https://api.openai.com/v1",
        "model_id": "gpt-4o-mini"
    },
    "ionos": {
        "api_key": os.getenv("IONOS_API_KEY"),
        "api_url": os.getenv("IONOS_ENDPOINT"),
        "model_id": "0b6c4a15-bb8d-4092-82b0-f357b77c59fd"
    },
    "litellm": {
        "provider": "huggingface",
        "api_key": os.getenv("LITELLM_API_KEY"),
        "api_url": "https://api-inference.huggingface.co/v1",
        "model_id": "huggingface/meta-llama/Llama-3.3-70B-Instruct:cerebras"
    }
}


def build_config(provider_cfg: dict) -> Dict[str, Any]:
    """
    Build LLM config.
    Only LiteLLM requires 'provider' field.
    """
    config = {
        "api_key": provider_cfg.get("api_key"),
        "api_url": provider_cfg.get("api_url"),
        "model_id": provider_cfg.get("model_id")
    }

    # Add LiteLLM-specific provider info
    if provider_cfg.get("provider"):
        config["provider"] = provider_cfg.get("provider")

    return config


class LiteLLMEvaluator:
    """
    LiteLLM evaluator wrapper
    """
    def __init__(self, config: dict):
        self.config = config
        self.client = litellm.LiteLLM(
            api_key=config["api_key"],
            api_url=config["api_url"],
            model=config["model_id"],
            provider=config.get("provider")  # Only LiteLLM uses this
        )

    def completion(self, prompt: str) -> str:
        try:
            response = self.client.completion(prompt)
            return response
        except AuthenticationError as e:
            logger.error(f"Authentication error: {e}")
            return str(e)
        except APIError as e:
            logger.error(f"API error: {e}")
            return str(e)
        except RateLimitError as e:
            logger.error(f"Rate limit error: {e}")
            return str(e)
        except Exception as e:
            logger.error(f"Unknown error: {e}")
            return str(e)


async def generic_llm_call(config: dict, prompt: str) -> str:
    """
    Placeholder for OpenAI / IONOS async call
    Replace with real SDK calls
    """
    # Example pseudo-call
    await asyncio.sleep(0.1)  # simulate network
    return f"Response from {config.get('model_id')} for prompt: {prompt}"


class EvaluationService:
    def __init__(self, providers: dict):
        self.providers = providers

    async def evaluate(self, provider_name: str, prompt: str) -> Dict[str, Any]:
        provider_cfg = self.providers[provider_name]
        cfg = build_config(provider_cfg)

        if provider_name.lower() == "litellm":
            evaluator = LiteLLMEvaluator(cfg)
            # Run sync completion in executor for async safety
            response = await asyncio.get_running_loop().run_in_executor(
                None, lambda: evaluator.completion(prompt)
            )
        else:
            # OpenAI / IONOS
            response = await generic_llm_call(cfg, prompt)

        return {
            "provider": provider_name,
            "model_id": cfg.get("model_id"),
            "response": response,
            "extra": cfg.get("provider")  # Only LiteLLM has this
        }


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    async def main():
        service = EvaluationService(providers)
        prompts = ["Hello world!", "Explain climate change."]

        for provider_name in providers:
            for prompt in prompts:
                result = await service.evaluate(provider_name, prompt)
                print(result)

    asyncio.run(main())
