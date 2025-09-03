"""Dynamic model provider resolver for LevelApp evaluation system."""
import os
import requests

TIMEOUT = 3

# Mapping from canonical model names to IONOS UUIDs
IONOS_MODEL_MAPPING = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "0b6c4a15-bb8d-4092-82b0-f357b77c59fd",
    "meta-llama/Llama-3.3-70B-Instruct": "0b6c4a15-bb8d-4092-82b0-f357b77c59fd",  # Add more as needed
}

class Spec(dict):
    __getattr__ = dict.get

def normalize_for_litellm(model_id: str) -> Spec:
    """Normalize model_id to provider-specific configuration for LiteLLM."""
    if model_id.startswith(("gpt-", "o", "text-")):
        return Spec(provider="openai", model=model_id, 
                   api_base="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY")
    
    if model_id.startswith("mistral") or model_id.startswith("mistral/"):
        model = model_id.split("/", 1)[-1] if "/" in model_id else model_id
        return Spec(provider="mistral", model=model,
                   api_base="https://api.mistral.ai/v1", api_key_env="MISTRAL_API_KEY")
    
    if model_id.startswith("claude-"):
        return Spec(provider="anthropic", model=model_id,
                   api_base="https://api.anthropic.com/v1", api_key_env="ANTHROPIC_API_KEY")
    
    if "/" in model_id:
        return Spec(provider="huggingface-router", model=model_id,
                   api_base="https://router.huggingface.co/v1", api_key_env="HUGGINGFACE_API_KEY")
    
    raise ValueError(f"Unrecognized model id pattern: {model_id}")

def _ionos_has(model_id: str) -> bool:
    """Check if IONOS hub has the specified model available."""
    key = os.getenv("IONOS_API_KEY")
    if not key:
        return False
    try:
        r = requests.get("https://openai.inference.de-txl.ionos.com/v1/models", 
                        headers={"Authorization": f"Bearer {key}"}, timeout=TIMEOUT)
        r.raise_for_status()
        return any(m.get("id") == model_id for m in r.json().get("data", []))
    except Exception:
        return False

def resolve_model(model_id: str) -> Spec:
    """Resolve model_id to the appropriate provider configuration."""
    # Prefer your infra if it carries the same canonical ID
    if _ionos_has(model_id):
        # Map canonical name to IONOS UUID for /predictions endpoint
        ionos_uuid = IONOS_MODEL_MAPPING.get(model_id)
        if not ionos_uuid:
            raise ValueError(f"IONOS model '{model_id}' found in catalog but no UUID mapping configured")
        
        # Use existing IonosEvaluator with UUID and exact config from main branch
        import os
        ionos_base = os.getenv("IONOS_ENDPOINT", "https://inference.de-txl.ionos.com/models")
        return Spec(provider="ionos", model=ionos_uuid, 
                   api_base=ionos_base, api_key_env="IONOS_API_KEY")
    
    try:
        return normalize_for_litellm(model_id)
    except ValueError:
        raise ValueError(f"Model '{model_id}' not found in IONOS or supported providers")