import os
import json
from typing import Union, Dict, Any

def call_llm(
    prompt: str,
    system_instruction: str = "",
    json_mode: bool = False,
    model: str = "llama-3.3-70b-versatile"
) -> Union[str, Dict[str, Any]]:
    """
    Direct LLM caller for Groq API.
    - Used for Primary Generation (PrimLLM) and AI-as-a-Judge (AIJudge).
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set. Please set your Groq API key.")

    from groq import Groq
    client = Groq(api_key=api_key)

    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    response_format = {"type": "json_object"} if json_mode else None

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=response_format,
        temperature=0.2 if json_mode else 0.7
    )
    
    text = completion.choices[0].message.content or ""
    return json.loads(text) if json_mode else text
