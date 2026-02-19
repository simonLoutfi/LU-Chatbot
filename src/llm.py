from __future__ import annotations

import os
from typing import List, Tuple, Optional

from .config import (
    OLLAMA_MODEL,
    MAX_CHUNK_TOKENS,
    MAX_CONTEXT_CHUNKS,
    MAX_CONTEXT_TOKENS,
    MAX_HISTORY_MESSAGES,
    MAX_HISTORY_TOKENS,
)
from .token_utils import get_token_counter, take_first_tokens, take_last_tokens


class LLMUnavailableError(RuntimeError):
    pass


def _get_ollama_client():
    try:
        import ollama
    except Exception as exc:
        raise LLMUnavailableError("ollama is not installed.") from exc

    host = os.environ.get("OLLAMA_HOST")
    api_key = os.environ.get("OLLAMA_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    if host or headers:
        return ollama.Client(host=host, headers=headers)
    return ollama.Client()


def _list_local_models(ollama_client) -> List[str]:
    try:
        payload = ollama_client.list()
    except Exception:
        return []

    models = payload.get("models", []) if isinstance(payload, dict) else []
    names = []
    for model in models:
        name = model.get("name") if isinstance(model, dict) else None
        if name:
            names.append(name)
    return names


def _resolve_model_name(ollama_client, requested: str) -> Tuple[str, List[str]]:
    available = _list_local_models(ollama_client)
    if not available:
        return requested, available

    if requested in available:
        return requested, available

    requested_base = requested.split(":", 1)[0]
    for name in available:
        if name.split(":", 1)[0] == requested_base:
            return name, available

    return requested, available


def _extract_chat_content(response) -> str:
    if response is None:
        return ""

    message = getattr(response, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if content:
            return content

    data = None
    if isinstance(response, dict):
        data = response
    elif hasattr(response, "model_dump"):
        data = response.model_dump()
    elif hasattr(response, "dict"):
        data = response.dict()

    if isinstance(data, dict):
        message_payload = data.get("message")
        if isinstance(message_payload, dict):
            content = message_payload.get("content")
            if content:
                return content

    return ""



def _format_history(history: Optional[List[dict]]) -> str:
    if not history:
        return ""

    filtered = [
        message
        for message in history
        if message.get("role") in {"user", "assistant"} and message.get("content")
    ]
    if not filtered:
        return ""

    if MAX_HISTORY_MESSAGES > 0:
        filtered = filtered[-MAX_HISTORY_MESSAGES:]

    lines = []
    for message in filtered:
        role = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{role}: {message['content']}")

    history_text = "\n".join(lines).strip()
    if not history_text:
        return ""

    if MAX_HISTORY_TOKENS <= 0:
        return history_text

    token_counter = get_token_counter()
    if token_counter(history_text) <= MAX_HISTORY_TOKENS:
        return history_text

    remaining = MAX_HISTORY_TOKENS
    kept_lines = []
    for line in reversed(lines):
        line_tokens = token_counter(line)
        if line_tokens > remaining:
            if not kept_lines:
                trimmed = take_last_tokens(line, remaining)
                if trimmed:
                    kept_lines.append(trimmed)
            break
        kept_lines.append(line)
        remaining -= line_tokens
        if remaining <= 0:
            break

    return "\n".join(reversed(kept_lines)).strip()


def build_prompt(query: str, chunks: List[dict], history: Optional[List[dict]] = None) -> Tuple[str, str]:
    context_lines = []
    token_counter = get_token_counter()
    max_chunks = MAX_CONTEXT_CHUNKS if MAX_CONTEXT_CHUNKS > 0 else len(chunks)
    remaining = MAX_CONTEXT_TOKENS if MAX_CONTEXT_TOKENS > 0 else None

    for idx, chunk in enumerate(chunks[:max_chunks], start=1):
        meta = chunk["metadata"]
        title = meta.get("section_title") or meta.get("document_id")
        text = chunk["text"]
        if MAX_CHUNK_TOKENS > 0:
            text = take_first_tokens(text, MAX_CHUNK_TOKENS)

        label = f"[{idx}] ({title}) "
        if remaining is not None:
            label_tokens = token_counter(label)
            if remaining <= label_tokens:
                break
            available_for_text = remaining - label_tokens
            text = take_first_tokens(text, available_for_text)

        line = f"{label}{text}"
        context_lines.append(line)
        if remaining is not None:
            remaining -= token_counter(line)
            if remaining <= 0:
                break

    context = "\n\n".join(context_lines)
    system = (
        "You are an admissions assistant for a Lebanese university. "
        "Answer only using the provided context. "
        "If the answer is not in the context, say you do not have that information. "
        "The conversation is only for resolving references, not for facts. "
        "Keep responses concise and admissions-focused."
    )
    history_text = _format_history(history)
    history_block = ""
    if history_text:
        history_block = f"Conversation so far (for resolving references only):\n{history_text}\n\n"
    user = f"{history_block}Question: {query}\n\nContext:\n{context}"

    return system, user




def generate_answer(query: str, chunks: List[dict], history: Optional[List[dict]] = None) -> str:
    ollama_client = _get_ollama_client()
    system, user = build_prompt(query, chunks, history=history)

    model_name, available = _resolve_model_name(ollama_client, OLLAMA_MODEL)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        response = ollama_client.chat(
            model=model_name,
            messages=messages,
            options={"temperature": 0},
        )
    except Exception as exc:
        message = str(exc).lower()
        if (
            "connect" in message
            or "connection" in message
            or "refused" in message
            or "failed to connect" in message
        ):
            raise LLMUnavailableError(
                "Ollama server not reachable. Ensure Ollama is running."
            ) from exc
        if "unauthorized" in message or "status code: 401" in message or "401" in message:
            raise LLMUnavailableError(
                "Ollama request unauthorized. Set OLLAMA_API_KEY for cloud models "
                "and OLLAMA_HOST if you are not using the local server."
            ) from exc
        if "not found" in message and "model" in message:
            hint = ""
            if available:
                preview = ", ".join(sorted(available)[:6])
                hint = f" Available models: {preview}."
            raise LLMUnavailableError(
                f"Model '{OLLAMA_MODEL}' not available locally.{hint} "
                f"Run `ollama pull {OLLAMA_MODEL}`."
            ) from exc
        raise
    return _extract_chat_content(response).strip()
