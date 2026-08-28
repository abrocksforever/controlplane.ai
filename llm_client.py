import os
import json
import time
import logging
from typing import Union, Dict, Any

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
BASE_DELAY_SECONDS = 1.0
TIMEOUT_SECONDS = 30


def call_llm(
    prompt: str,
    system_instruction: str = "",
    json_mode: bool = False,
    model: str = "llama-3.3-70b-versatile"
) -> Union[str, Dict[str, Any]]:
    """
    Direct LLM caller for Groq API with retry and timeout.
    - Used for Primary Generation (PrimLLM) and AI-as-a-Judge (AIJudge).
    - Retries up to 3 times with exponential backoff on transient failures.
    - 30-second timeout per attempt to prevent pipeline hangs.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set. Please set your Groq API key.")

    from groq import Groq
    client = Groq(api_key=api_key, timeout=TIMEOUT_SECONDS)

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
            # Retry on transient errors (rate limit, timeout, connection issues)
            is_transient = any(keyword in error_name.lower() or keyword in str(e).lower()
                               for keyword in ["rate", "timeout", "connection", "503", "429", "500"])
            
            if is_transient and attempt < MAX_RETRIES:
                delay = BASE_DELAY_SECONDS * (2 ** (attempt - 1))  # Exponential backoff
                logger.warning(
                    f"LLM call attempt {attempt}/{MAX_RETRIES} failed ({error_name}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                logger.error(f"LLM call failed permanently after {attempt} attempt(s): {error_name}: {e}")
                raise

    raise last_exception  # Should not reach here, but safety net
