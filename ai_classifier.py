"""
Module for context analysis (Stage 3 of the ETL pipeline).

This module handles the classification of masked tokens. It is the only part of the
pipeline that makes external network calls (to OpenAI). It receives a skeleton
string with <TOKEN_N> placeholders and determines if they represent PII based on grammar.

Components:
- OpenAIClassifier: Uses Structured Outputs for production classification.
- HeuristicClassifier: A deterministic fallback for offline QA testing.
- analyse_with_fail_closed: Ensures any API failures or unclassified tokens
  default to being redacted (fail-closed approach).
"""

from __future__ import annotations
import re
from typing import Sequence
from models import (
    MaskedDocument,
    SensitivityCategory,
    SensitivityReport,
    TokenAssessment,
    TokenClassifier,
    TOKEN_PATTERN,
    token_index,
)
from models import DEFAULT_MODEL

SYSTEM_PROMPT = """\
You are a privacy classification engine operating on ALREADY-REDACTED text.

Every `<TOKEN_N>` tag stands for a capitalised word that a local model flagged as a
possible entity. You cannot see the underlying values and must not attempt to guess
them. Judge each tag purely from the surrounding grammar and its discourse role.

Set is_sensitive = true when the tag occupies a slot that identifies a natural person:
  - PERSON       object of "to/from/contact/pay/wire", preceded by an honorific, part
                 of a name pair, or possessive ("<TOKEN_2>'s account")
  - LOCATION     object of "in/at/near/lives in" — a place tied to that person
  - ORGANIZATION a small or specific entity that narrows down who is meant
  - IDENTIFIER   an account, handle, or reference number

Set is_sensitive = false (NON_SENSITIVE) when the slot holds generic context: software
and product names ("migrated to <TOKEN_1>"), technologies, currencies, month or weekday
names, job titles, or common nouns that merely happen to be capitalised.

A sensitive verdict propagates across coordination: in "<TOKEN_1> <TOKEN_2> and
<TOKEN_3> <TOKEN_4>", if the first pair is a person then so is the second.

Never invent a tag that was not supplied. Return exactly one assessment
per tag listed in the request.\
"""


def build_masked_prompt(masked_text: str, tokens: Sequence[str]) -> str:
    """Prepare the user message with the masked string and requested tokens."""
    listed = "\n".join(f"- {t}" for t in tokens) or "- (none)"
    return (
        "Redacted document:\n"
        f"{masked_text}\n\n"
        "Placeholder tags requiring a verdict:\n"
        f"{listed}\n\n"
        "Return exactly one assessment per tag listed above."
    )


# --- API Handlers ---

class OpenAIClassifier:
    """Handles OpenAI API communication for token context analysis."""

    _NO_TEMPERATURE_PREFIXES = ("o1", "o3", "o4", "gpt-5")

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        # Lazy import to avoid requiring the API key for offline tests
        from openai import OpenAI

        self.model = model
        self.name = f"OpenAI:{model}"
        self._client = OpenAI()

    def classify(self, masked_text: str, tokens: Sequence[str]) -> SensitivityReport:
        if not tokens:
            return SensitivityReport(assessments=[])

        kwargs: dict[str, object] = {
            "model": self.model,
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_masked_prompt(masked_text, tokens)},
            ],
            "text_format": SensitivityReport,
        }

        if not self.model.startswith(self._NO_TEMPERATURE_PREFIXES):
            kwargs["temperature"] = 0

        response = self._client.responses.parse(**kwargs)

        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("API returned no parsed output.")

        return parsed


# --- Offline / QA Fallback ---

class HeuristicClassifier:
    """Deterministic offline fallback for QA and CI/CD pipelines."""

    name = "Heuristic:offline"

    PERSON_CUES = frozenset({
        "to", "from", "contact", "call", "email", "pay", "paid", "wire", "owes", "owe",
        "told", "met", "meet", "thank", "attn", "dear", "hi", "hello", "regards",
        "mr", "mrs", "ms", "miss", "dr", "prof", "sir", "reimburse", "invoice",
    })

    PLACE_CUES = frozenset({
        "in", "at", "near", "visited", "visiting", "lives", "live", "living",
        "based", "around", "across", "toward", "towards", "arriving",
    })

    TRANSPARENT = frozenset({"the", "a", "an", "my", "our", "their", "his", "her", "its"})
    CONNECTORS = frozenset({"and", "&", "or", "plus", ","})

    _UNIT_RE = re.compile(r"<TOKEN_\d+>|[A-Za-z']+|,|&")

    def classify(self, masked_text: str, tokens: Sequence[str]) -> SensitivityReport:
        units = self._UNIT_RE.findall(masked_text)
        assessments: list[TokenAssessment] = []
        verdicts: dict[str, TokenAssessment] = {}

        for i, unit in enumerate(units):
            # Process only the first occurrence of a tag
            if not TOKEN_PATTERN.fullmatch(unit) or unit in verdicts:
                continue

            category, is_sensitive, rationale = self._judge(units, i, verdicts)
            assessment = TokenAssessment(
                token=unit,
                category=category,
                is_sensitive=is_sensitive,
                rationale=rationale,
            )
            verdicts[unit] = assessment
            assessments.append(assessment)

        # Fail-safe: if a requested token was missed, default to sensitive.
        for token in tokens:
            if token not in verdicts:
                assessments.append(
                    TokenAssessment(
                        token=token,
                        category="PERSON",
                        is_sensitive=True,
                        rationale="No syntactic context resolved; defaulting to sensitive.",
                    )
                )

        assessments.sort(key=lambda a: token_index(a.token))
        return SensitivityReport(assessments=assessments)

    def _judge(
            self,
            units: list[str],
            i: int,
            verdicts: dict[str, TokenAssessment],
    ) -> tuple[SensitivityCategory, bool, str]:
        """Simple rules engine to simulate LLM context judgment."""
        cue = self._preceding_cue(units, i)

        if cue in self.PLACE_CUES:
            return "LOCATION", True, f"Object of locative '{cue}'."

        if cue in self.PERSON_CUES:
            return "PERSON", True, f"Object of person-directed '{cue}'."

        prev = units[i - 1] if i else None

        if prev is not None and TOKEN_PATTERN.fullmatch(prev):
            anchor = verdicts.get(prev)
            if anchor is not None and anchor.is_sensitive:
                return anchor.category, True, f"Name continuation of {prev}."

        if prev in self.CONNECTORS and i >= 2:
            candidate = units[i - 2]
            if TOKEN_PATTERN.fullmatch(candidate):
                anchor = verdicts.get(candidate)
                if anchor is not None and anchor.is_sensitive:
                    return anchor.category, True, f"Coordinated with {candidate} via '{prev}'."

        return "NON_SENSITIVE", False, "Occupies a generic object slot with no identifying cue."

    def _preceding_cue(self, units: list[str], i: int) -> str | None:
        j = i - 1
        while j >= 0 and units[j].lower() in self.TRANSPARENT:
            j -= 1
        return units[j].lower() if j >= 0 else None


def analyse_with_fail_closed(
        classifier: TokenClassifier,
        document: MaskedDocument,
) -> tuple[SensitivityReport, list[str]]:
    """Execute classification and normalize outputs for safety."""
    masked_text, tokens = document.exportable_payload()
    warnings: list[str] = []

    try:
        report = classifier.classify(masked_text, tokens)
    except Exception as exc:
        warnings.append(f"API Error ({type(exc).__name__}). Defaulting to full redaction.")
        return _redact_everything(tokens, "Fail-closed default after classifier error."), warnings

    requested = set(tokens)
    returned = report.verdict_map()

    hallucinated = sorted(t for t in returned if t not in requested)
    if hallucinated:
        warnings.append(f"Discarded hallucinated tags: {', '.join(hallucinated)}.")

    normalised = [a for a in report.assessments if a.token in requested]

    missing = requested - {a.token for a in normalised}
    for token in sorted(missing, key=token_index):
        warnings.append(f"{token} received no verdict; redacting (fail-closed).")
        normalised.append(
            TokenAssessment(
                token=token,
                category="PERSON",
                is_sensitive=True,
                rationale="Omitted by classifier; fail-closed default.",
            )
        )

    normalised.sort(key=lambda a: token_index(a.token))
    return SensitivityReport(assessments=normalised), warnings


def _redact_everything(tokens: Sequence[str], rationale: str) -> SensitivityReport:
    """Helper to generate a fully redacted report in case of API failure."""
    return SensitivityReport(
        assessments=[
            TokenAssessment(
                token=token,
                category="PERSON",
                is_sensitive=True,
                rationale=rationale,
            )
            for token in tokens
        ]
    )