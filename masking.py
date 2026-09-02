"""
Module for extraction and local masking (Stages 1 & 2 of the ETL pipeline).

This module operates entirely within the trusted zone. It acts as a lightweight,
rule-based NER (Named Entity Recognition) engine that over-indexes on masking
potential PII locally before sending it to the cloud for contextual analysis.
"""

from __future__ import annotations
import re
from models import MaskedDocument, format_token


class LocalMaskingEngine:
    """
    Local masking engine.
    """

    STOPWORDS: frozenset[str] = frozenset(
        {
            # Weekdays & Months
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",

            # Honorifics (context cues should not be masked)
            "mr", "mrs", "ms", "miss", "dr", "prof", "sir", "madam",

            # Grammar & Connectives
            "the", "a", "an", "and", "but", "or", "if", "this", "that", "these", "those",
            "i", "we", "our", "your", "their", "his", "her", "its", "it", "they",

            # Generic business & banking vocabulary
            "invoice", "payment", "rent", "budget", "report", "team", "office",
            "quarter", "revenue", "meeting", "project", "branch", "department",
            "leadership", "marketing", "finance", "engineering", "sales", "legal",
            "please", "send", "contact", "regards", "hello", "hi", "dear", "thanks",
        }
    )

    # Matches capitalized words (2+ chars) to filter out single initials or things like 'Q3'
    CANDIDATE_RE = re.compile(r"[A-Z][a-zA-Z][a-zA-Z'’\-]*")

    # Characters that genuinely open a new sentence (excluding colons/semicolons)
    SENTENCE_ENDERS: frozenset[str] = frozenset({".", "!", "?", "\n", '"', "“", "(", ""})

    def mask(self, raw_text: str) -> MaskedDocument:
        """
        Replaces PII candidates with <TOKEN_N> placeholders.
        """
        matches = list(self.CANDIDATE_RE.finditer(raw_text))

        confirmed: set[str] = set()
        for match in matches:
            word = match.group(0)
            if word in confirmed:
                continue
            if self._is_candidate(word, raw_text, match.start()):
                confirmed.add(word)

        vault: dict[str, str] = {}
        surface_to_token: dict[str, str] = {}
        pieces: list[str] = []
        cursor = 0

        for match in matches:
            word = match.group(0)
            if word not in confirmed:
                continue

            token = surface_to_token.get(word)
            if token is None:
                token = format_token(len(surface_to_token) + 1)
                surface_to_token[word] = token
                vault[token] = word

            pieces.append(raw_text[cursor:match.start()])
            pieces.append(token)
            cursor = match.end()

        pieces.append(raw_text[cursor:])

        return MaskedDocument(
            masked_text="".join(pieces),
            tokens=list(vault.keys()),
            _vault=vault,
        )

    def _is_candidate(self, word: str, text: str, start: int) -> bool:
        """Determines if a capitalized word should be tokenized."""
        if word.isupper() and len(word) <= 5:  # Acronyms like VAT, USD
            return False
        if word.lower() in self.STOPWORDS:  # Known safe words
            return False
        return not self._is_sentence_initial(text, start)

    def _is_sentence_initial(self, text: str, start: int) -> bool:
        """Checks if the word is the first word in a sentence to avoid false positives."""
        idx = start - 1
        while idx >= 0 and text[idx] in " \t":
            idx -= 1
        preceding = text[idx] if idx >= 0 else ""
        return preceding in self.SENTENCE_ENDERS