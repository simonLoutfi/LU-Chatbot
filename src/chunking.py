from __future__ import annotations

import re
from typing import Iterable, List, Optional

from .token_utils import get_token_counter, take_last_tokens


def _is_section_header(line: str, prev_line: str, next_line: str) -> bool:
    if len(line) > 80:
        return False
    if not any(ch.isalpha() for ch in line):
        return False

    looks_like_title = line.isupper() or line.istitle()
    surrounded_by_blank = (not prev_line.strip()) and (not next_line.strip())
    return looks_like_title and surrounded_by_blank


def extract_sections(text: str) -> List[dict]:
    lines = [line.rstrip() for line in text.splitlines()]
    sections = []
    current_title: Optional[str] = None
    current_lines: List[str] = []

    def flush():
        if not current_lines:
            return
        section_text = "\n".join(current_lines).strip()
        if section_text:
            sections.append(
                {
                    "section_title": current_title,
                    "text": section_text,
                }
            )

    for idx, line in enumerate(lines):
        prev_line = lines[idx - 1] if idx > 0 else ""
        next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
        if line.strip() and _is_section_header(line.strip(), prev_line, next_line):
            flush()
            current_lines = []
            current_title = line.strip()
            continue

        current_lines.append(line)

    flush()
    return sections


def _split_into_units(text: str) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    units: List[str] = []
    sentence_pattern = re.compile(r"(?<=[.!?])\s+")

    for paragraph in paragraphs:
        if len(paragraph) < 200:
            units.append(paragraph)
            continue

        sentences = [s.strip() for s in sentence_pattern.split(paragraph) if s.strip()]
        units.extend(sentences or [paragraph])

    return units


def _split_long_unit(unit: str, max_tokens: int) -> Iterable[str]:
    words = unit.split()
    if not words:
        return []

    chunks = []
    for start in range(0, len(words), max_tokens):
        chunks.append(" ".join(words[start : start + max_tokens]))
    return chunks


def chunk_text(
    text: str, chunk_size: int, chunk_overlap: int
) -> List[dict]:
    token_counter = get_token_counter()
    sections = extract_sections(text)
    if not sections:
        sections = [{"section_title": None, "text": text.strip()}]

    chunks = []
    chunk_index = 0

    for section in sections:
        section_title = section["section_title"]
        section_text = section["text"]
        if section_title and not section_text.startswith(section_title):
            section_text = f"{section_title}\n{section_text}"

        units = _split_into_units(section_text)
        current_units: List[str] = []
        current_tokens = 0

        for unit in units:
            unit_tokens = token_counter(unit)
            if unit_tokens > chunk_size:
                for sub_unit in _split_long_unit(unit, chunk_size):
                    sub_tokens = token_counter(sub_unit)
                    if current_units and current_tokens + sub_tokens > chunk_size:
                        chunk_text_value = "\n\n".join(current_units).strip()
                        chunks.append(
                            {
                                "chunk_index": chunk_index,
                                "text": chunk_text_value,
                                "token_count": token_counter(chunk_text_value),
                                "section_title": section_title,
                            }
                        )
                        chunk_index += 1
                        overlap_text = take_last_tokens(
                            chunk_text_value, chunk_overlap
                        )
                        current_units = [overlap_text] if overlap_text else []
                        current_tokens = token_counter(overlap_text)

                    current_units.append(sub_unit)
                    current_tokens += sub_tokens
                continue

            if current_units and current_tokens + unit_tokens > chunk_size:
                chunk_text_value = "\n\n".join(current_units).strip()
                chunks.append(
                    {
                        "chunk_index": chunk_index,
                        "text": chunk_text_value,
                        "token_count": token_counter(chunk_text_value),
                        "section_title": section_title,
                    }
                )
                chunk_index += 1
                overlap_text = take_last_tokens(chunk_text_value, chunk_overlap)
                current_units = [overlap_text] if overlap_text else []
                current_tokens = token_counter(overlap_text)

            current_units.append(unit)
            current_tokens += unit_tokens

        if current_units:
            chunk_text_value = "\n\n".join(current_units).strip()
            if chunk_text_value:
                chunks.append(
                    {
                        "chunk_index": chunk_index,
                        "text": chunk_text_value,
                        "token_count": token_counter(chunk_text_value),
                        "section_title": section_title,
                    }
                )
                chunk_index += 1

    return chunks
