"""Preliminary cleaning pipeline for the Spanish subtitles corpus.

The script reads ``corpus/es.example.txt`` (or a custom input path) and applies
lightweight heuristics to keep phrases that are suitable as learning material.
It removes noisy metadata such as timestamps, URLs, credits, and extremely
short/long lines. The filtered lines can later be refined by an LLM-powered
step.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

SPANISH_LETTER_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.+-]+@\w+\.[\w.-]+")
TIMESTAMP_RE = re.compile(r"\d{2}:\d{2}(:\d{2})?")
HTML_TAG_RE = re.compile(r"<[^>]+>")
EXTRA_META_RE = re.compile(r"(subtitulado por|subtítulos|subtitles|créditos)", re.IGNORECASE)
BRACKETED_RE = re.compile(r"[\[{].+[\]}]")


class LineFilter:
    """Encapsulates filtering heuristics for Spanish subtitle lines."""

    def __init__(self, min_length: int = 15, max_length: int = 120) -> None:
        self.min_length = min_length
        self.max_length = max_length

    def normalize(self, line: str) -> str:
        cleaned = line.replace("\ufeff", "")
        cleaned = re.sub(r"^[-–—]\s+", "", cleaned)
        cleaned = re.sub(r"(\.{3,}|…)", ",", cleaned)
        cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"^[\s.]+|[\s.]+$", "", cleaned)
        return cleaned

    def too_many_digits(self, line: str) -> bool:
        digits = sum(char.isdigit() for char in line)
        letters = sum(char.isalpha() for char in line)
        return digits > letters

    def has_enough_spanish_letters(self, line: str) -> bool:
        return bool(SPANISH_LETTER_RE.search(line))

    def has_timestamp(self, line: str) -> bool:
        return "-->" in line or bool(TIMESTAMP_RE.search(line))

    def has_metadata(self, line: str) -> bool:
        return bool(URL_RE.search(line) or EMAIL_RE.search(line) or EXTRA_META_RE.search(line))

    def has_markup(self, line: str) -> bool:
        return bool(HTML_TAG_RE.search(line) or BRACKETED_RE.search(line))

    def has_weird_symbol_density(self, line: str) -> bool:
        allowed_extra = set("¡¿?!.,:;'\"-–—() «»… ")
        weird = sum(1 for ch in line if not (ch.isalnum() or ch in allowed_extra))
        return weird > 3

    def should_keep(self, line: str) -> tuple[bool, str | None]:
        normalized = self.normalize(line)
        if not normalized:
            return False, "empty"
        if self.has_timestamp(normalized):
            return False, "timestamp"
        if self.has_metadata(normalized):
            return False, "metadata"
        if self.has_markup(normalized):
            return False, "markup"
        if len(normalized) < self.min_length:
            return False, "too_short"
        if len(normalized) > self.max_length:
            return False, "too_long"
        if not self.has_enough_spanish_letters(normalized):
            return False, "no_spanish_letters"
        if self.too_many_digits(normalized):
            return False, "too_many_digits"
        if self.has_weird_symbol_density(normalized):
            return False, "weird_symbols"
        return True, None

    def filter_lines(self, lines: Iterable[str]) -> tuple[list[str], Counter[str]]:
        kept: list[str] = []
        dropped_reasons: Counter[str] = Counter()
        seen: set[str] = set()

        for raw_line in lines:
            normalized = self.normalize(raw_line)
            keep, reason = self.should_keep(normalized)
            if not keep:
                if reason:
                    dropped_reasons[reason] += 1
                continue
            if normalized in seen:
                dropped_reasons["duplicate"] += 1
                continue
            seen.add(normalized)
            kept.append(normalized)
        return kept, dropped_reasons


def read_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            yield line.rstrip("\n")


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-filter Spanish subtitle corpus")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("corpus/es.sample.txt"),
        help="Path to the raw corpus file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("corpus/es.filtered.txt"),
        help="Where to save the filtered lines",
    )
    parser.add_argument("--min-length", type=int, default=15, help="Minimum characters to keep a line")
    parser.add_argument("--max-length", type=int, default=120, help="Maximum characters to keep a line")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    line_filter = LineFilter(min_length=args.min_length, max_length=args.max_length)

    if not args.input.is_file():
        raise FileNotFoundError(
            f"Input file not found: {args.input}. "
            "Place the Spanish subtitles sample at this path before running the script."
        )

    kept, dropped = line_filter.filter_lines(read_lines(args.input))
    write_lines(args.output, kept)

    total = sum(dropped.values()) + len(kept)
    print(f"Total lines processed: {total}")
    print(f"Kept: {len(kept)}")
    if dropped:
        print("Dropped by reason:")
        for reason, count in dropped.most_common():
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
