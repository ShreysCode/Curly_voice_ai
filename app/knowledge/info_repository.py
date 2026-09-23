from __future__ import annotations

import json
import re
from pathlib import Path


class InfoRepository:
    """Conservative, single-pass speech/STT terminology normalizer.

    This file contains terminology variants only. It is not a source of
    institutional facts. Curly resolves the resulting canonical terms against
    the authoritative app/knowledge/data.json knowledge base.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.aliases: dict[str, str] = {}

        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            raw_aliases = data.get("aliases", {})
            if isinstance(raw_aliases, dict):
                for alias, canonical in raw_aliases.items():
                    alias_text = str(alias).strip().lower()
                    canonical_text = str(canonical).strip()
                    if alias_text and canonical_text:
                        self.aliases[alias_text] = canonical_text

        ordered = sorted(
            self.aliases.items(),
            key=lambda item: (-len(item[0]), item[0]),
        )
        self._ordered_aliases = ordered

        if ordered:
            alternatives = []
            for alias, _canonical in ordered:
                alternatives.append(
                    rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
                )
            self._alias_pattern: re.Pattern[str] | None = re.compile(
                "(?:" + "|".join(alternatives) + ")",
                re.IGNORECASE,
            )
        else:
            self._alias_pattern = None

    @staticmethod
    def _normalize_spaces(text: str) -> str:
        return " ".join(text.strip().split())

    @staticmethod
    def _normal_form(text: str) -> str:
        """Comparison form used only to identify alias/canonical identity."""
        text = text.lower().replace("–", "-").replace("—", "-")
        text = re.sub(r"[^a-z0-9\s-]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def normalize(self, text: str) -> tuple[str, list[dict[str, str]]]:
        """Normalize one utterance exactly once.

        Important property: replacements are performed against the original
        utterance in one regex pass. Newly generated canonical text is never
        scanned again, so aliases cannot recursively expand one another.
        """
        original = self._normalize_spaces(text)
        if not original or self._alias_pattern is None:
            return original, []

        matches: list[dict[str, str]] = []

        def replace(match: re.Match[str]) -> str:
            heard = match.group(0)
            alias_key = heard.lower()
            canonical = self.aliases.get(alias_key)

            if canonical is None:
                return heard

            # A canonical phrase must remain byte-for-byte unchanged, apart
            # from normal whitespace cleanup. It should also produce no match.
            if self._normal_form(heard) == self._normal_form(canonical):
                return heard

            matches.append(
                {
                    "heard": alias_key,
                    "canonical": canonical,
                }
            )
            return canonical

        normalized = self._alias_pattern.sub(replace, original)
        normalized = self._normalize_spaces(normalized)

        unique_matches: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for item in matches:
            key = (item["heard"], item["canonical"])
            if key not in seen:
                seen.add(key)
                unique_matches.append(item)

        return normalized, unique_matches

    @staticmethod
    def format_matches(matches: list[dict[str, str]]) -> str:
        if not matches:
            return ""

        return "\n".join(
            f'- "{item["heard"]}" → "{item["canonical"]}"'
            for item in matches
        )
