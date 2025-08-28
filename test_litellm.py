import sys
import logging
import asyncio
import yaml
import os

from dotenv import load_dotenv

from level_core.evaluators.litellm_evaluator import LiteLLMEvaluator
from level_core.evaluators.ionos import IonosEvaluator
from level_core.evaluators.schemas import EvaluationConfig

# Optional datastore support
try:
    from level_core.datastore.registry import get_datastore
    from config.loader import get_database_config
except ImportError:
    get_datastore = None
    get_database_config = None

# Load .env so ${VAR} in YAML can resolve
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multi-eval")


def load_config(config_path: str):
    """Load YAML config with environment variable substitution."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Expand ${VAR} with os.environ
    expanded_text = os.path.expandvars(raw_text)
    return yaml.safe_load(expanded_text)


def build_config(provider_name: str, provider_cfg: dict) -> EvaluationConfig:
    """Build EvaluationConfig from provider settings."""
    return EvaluationConfig(
        model_id=provider_cfg.get("model_id"),
        api_key=provider_cfg.get("api_key"),
        api_url=provider_cfg.get("api_url"),
        llm_config={
            "temperature": float(provider_cfg.get("temperature", 0.0)),
            "max_tokens": int(provider_cfg.get("max_tokens", 150)),
        },
    )


async def run_eval_for_provider(provider_name: str, provider_cfg: dict, prompt: str, expected: str):
    """Run evaluation for a single provider config."""
    logger.info(f"Running evaluation for provider: {provider_name}")

    config = build_config(provider_name, provider_cfg)
    eval_logger = logging.getLogger(f"{provider_name.capitalize()}Evaluator")

    if provider_name == "ionos":
        evaluator = IonosEvaluator(config=config, logger=eval_logger)
    else:
        evaluator = LiteLLMEvaluator(config=config, logger=eval_logger)

    result = await evaluator.evaluate(prompt, expected)

    print("=" * 60)
    print(f"Provider: {provider_name}")
    print(f"Match Level: {result.match_level}")
    print(f"Justification: {result.justification}")
    print(f"Metadata: {result.metadata}")
    print("=" * 60)

    return {
        "provider": provider_name,
        "match_level": result.match_level,
        "justification": result.justification,
        "metadata": result.metadata,
    }


async def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python test_litellm.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)

    providers = config.get("providers", {})
    if not providers:
        logger.error("No providers found in YAML file.")
        sys.exit(1)

    prompt = "What is the capital of France?"
    expected = "Paris"

    all_results = []
    for provider_name, provider_cfg in providers.items():
        try:
            result = await run_eval_for_provider(provider_name, provider_cfg, prompt, expected)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Evaluation failed for provider {provider_name}: {e}")

    # Optionally save results to Firestore if database section exists
    if "database" in config and get_datastore and get_database_config:
        try:
            db_cfg = config["database"]
            backend = db_cfg.get("type", "firestore")
            project_id = db_cfg.get("project_id")

            firestore_service = get_datastore(backend=backend, config=db_cfg)

            import uuid
            batch_id = f"batch-{uuid.uuid4()}"

            firestore_service.save_batch_test_results(
                user_id="test-user",
                project_id=project_id,
                batch_id=batch_id,
                data={"results": all_results},
            )
            logger.info(f"Results saved to Firestore under project {project_id}, batch {batch_id}")
        except Exception as e:
            logger.error(f"Failed to save to Firestore: {e}")


if __name__ == "__main__":
    asyncio.run(main())
