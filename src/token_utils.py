from __future__ import annotations

from typing import Callable


def _get_encoder():
    try:
        import tiktoken
    except Exception:
        return None

    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def get_token_counter() -> Callable[[str], int]:
    encoder = _get_encoder()
    if encoder is None:
        return lambda text: len(text.split())

    return lambda text: len(encoder.encode(text))


def take_last_tokens(text: str, token_count: int) -> str:
    if token_count <= 0:
        return ""

    encoder = _get_encoder()
    if encoder is None:
        words = text.split()
        if not words:
            return ""
        return " ".join(words[-token_count:])

    tokens = encoder.encode(text)
    if not tokens:
        return ""
    return encoder.decode(tokens[-token_count:])


def take_first_tokens(text: str, token_count: int) -> str:
    if token_count <= 0:
        return ""

    encoder = _get_encoder()
    if encoder is None:
        words = text.split()
        if not words:
            return ""
        return " ".join(words[:token_count])

    tokens = encoder.encode(text)
    if not tokens:
        return ""
    return encoder.decode(tokens[:token_count])
