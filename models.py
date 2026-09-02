"""
Data contracts for the ETL pipeline.

Contains all Pydantic schemas, dataclasses, and protocols used across modules.
Ensures modules can communicate without circular imports.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Literal, Protocol, Sequence, runtime_checkable
from pydantic import BaseModel, ConfigDict, Field


REDACTION_MARKER = "[REDACTED]"
DEFAULT_MODEL = "gpt-4o-mini"
TOKEN_PATTERN = re.compile(r"<TOKEN_(\d+)>")


def format_token(index: int) -> str:
    """Mints a placeholder tag (e.g. <TOKEN_1>)."""
    return f"<TOKEN_{index}>"


def token_index(token: str) -> int:
    """Returns the numeric ordinal of a tag for sorting."""
    match = TOKEN_PATTERN.fullmatch(token)
    return int(match.group(1)) if match else 10**9


@dataclass
class MaskedDocument:
    """
    Gatekeeper object holding both the safe exportable text and the local PII vault.
    """
    masked_text: str
    tokens: list[str]
    # repr=False prevents the vault from leaking into logs or tracebacks.
    _vault: dict[str, str] = field(repr=False, default_factory=dict)

    def exportable_payload(self) -> tuple[str, list[str]]:
        """Returns only the safe data for external API requests."""
        return self.masked_text, list(self.tokens)

    def original(self, token: str) -> str:
        """Resolves a tag back to its surface form (Trusted zone only)."""
        return self._vault[token]

    def has(self, token: str) -> bool:
        return token in self._vault

    @property
    def vault_values(self) -> list[str]:
        """Powers the QA leak-canary assertions."""
        return list(self._vault.values())


# --- Cloud Schemas (Pydantic / OpenAI Structured Outputs) ---

SensitivityCategory = Literal[
    "PERSON",
    "LOCATION",
    "ORGANIZATION",
    "IDENTIFIER",
    "NON_SENSITIVE",
]


class TokenAssessment(BaseModel):
    """One verdict for one placeholder tag."""
    model_config = ConfigDict(extra="forbid")

    token: str = Field(description="The placeholder tag verbatim, e.g. '<TOKEN_1>'.")
    category: SensitivityCategory = Field(description="Semantic category inferred from grammar.")
    is_sensitive: bool = Field(description="True if this tag stands for PII.")
    rationale: str = Field(description="Short justification for the verdict.")


class SensitivityReport(BaseModel):
    """The complete response object returned by the context-analysis stage."""
    model_config = ConfigDict(extra="forbid")

    assessments: list[TokenAssessment] = Field(
        description="Exactly one assessment per requested placeholder tag."
    )

    @property
    def sensitive_tokens(self) -> list[str]:
        """Flat list of tags marked for redaction."""
        return [a.token for a in self.assessments if a.is_sensitive]

    def verdict_map(self) -> dict[str, TokenAssessment]:
        return {a.token: a for a in self.assessments}


@runtime_checkable
class TokenClassifier(Protocol):
    """Protocol for the swappable analysis backend (Cloud vs Local QA)."""
    name: str
    def classify(self, masked_text: str, tokens: Sequence[str]) -> SensitivityReport: ...


@dataclass
class PipelineResult:
    """Holds all data generated during a single pipeline run."""
    raw_text: str
    masked_text: str
    final_text: str
    report: SensitivityReport
    document: MaskedDocument
    warnings: list[str]
    classifier_name: str

    def leaked_values(self) -> list[str]:
        """Returns any original vault values that accidentally ended up in the masked text."""
        return [v for v in self.document.vault_values if v in self.masked_text]