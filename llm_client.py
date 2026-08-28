import os
import json
import time
import logging
from typing import Union, Dict, Any

logger = logging.getLogger(__name__)

# Retry & Timeout configuration
MAX_RETRIES = 3
BASE_DELAY_SECONDS = 1.0
TIMEOUT_SECONDS = 30

# Default model ID (configurable via CONTROLPLANE_LLM_MODEL env var)
DEFAULT_MODEL = os.environ.get("CONTROLPLANE_LLM_MODEL", "qwen/qwen3.8-27b")


def _load_env_file():
    """Loads key-value pairs from .env file into os.environ."""
    for candidate in [".env", os.path.join(os.path.dirname(__file__), ".env")]:
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k and k not in os.environ:
                                os.environ[k] = v
                break
            except Exception as e:
                logger.debug(f"Failed reading {candidate}: {e}")


# Auto-load on import
_load_env_file()


def call_llm(
    prompt: str,
    system_instruction: str = "",
    json_mode: bool = False,
    model: str = DEFAULT_MODEL
) -> Union[str, Dict[str, Any]]:
    """
    Direct LLM caller with retry and timeout.
    - Defaults to 'qwen/qwen3.8-27b' (or CONTROLPLANE_LLM_MODEL env var).
    - Used for Primary Generation (PrimLLM), RAG NLI Entailment, and AI-as-a-Judge (AIJudge).
    - Automatically loads GROQ_API_KEY from .env.
    - Retries up to 3 times with exponential backoff on transient failures.
    - 30-second timeout per attempt to prevent pipeline hangs.
    """
    _load_env_file()
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set. Please set GROQ_API_KEY in .env.")

    base_url = os.environ.get("GROQ_BASE_URL") or os.environ.get("OPENAI_BASE_URL")

    from groq import Groq
    client = Groq(api_key=api_key, base_url=base_url, timeout=TIMEOUT_SECONDS) if base_url else Groq(api_key=api_key, timeout=TIMEOUT_SECONDS)

    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    response_format = {"type": "json_object"} if json_mode else None

    last_exception = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=response_format,
                temperature=0.2 if json_mode else 0.7
            )
            text = completion.choices[0].message.content or ""
            return json.loads(text) if json_mode else text

        except Exception as e:
            last_exception = e
            error_name = type(e).__name__
            is_transient = any(keyword in error_name.lower() or keyword in str(e).lower()
                               for keyword in ["rate", "timeout", "connection", "503", "429", "500"])
            
            if is_transient and attempt < MAX_RETRIES:
                delay = BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    f"LLM call attempt {attempt}/{MAX_RETRIES} failed ({error_name}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                logger.error(f"LLM call failed permanently after {attempt} attempt(s): {error_name}: {e}")
                raise

    raise last_exception
