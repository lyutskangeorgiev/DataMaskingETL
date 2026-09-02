"""
Module for local reassembly and presentation (Stage 4).

Rebuilds the document locally based on the classifier's report.
Sensitive tokens are redacted, and safe tokens are restored from the local vault.
"""

from __future__ import annotations
import re

from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from models import (
    MaskedDocument,
    PipelineResult,
    REDACTION_MARKER,
    SensitivityReport,
    TOKEN_PATTERN,
)


# --- Core Logic ---

def reassemble(document: MaskedDocument, report: SensitivityReport) -> str:
    """
    Rebuilds the document locally.
    Uses a single regex substitution pass to prevent tag corruption.
    """
    sensitive = set(report.sensitive_tokens)

    def substitute(match: re.Match[str]) -> str:
        token = match.group(0)

        if token in sensitive:
            return REDACTION_MARKER

        if not document.has(token):
            # Fallback: Destroy token if it's missing from the vault.
            return REDACTION_MARKER

        return document.original(token)

    return TOKEN_PATTERN.sub(substitute, document.masked_text)


# --- CLI Presentation Utilities ---

def render_walkthrough(result: PipelineResult, console: Console) -> None:
    """Renders a stage-by-stage CLI view of the pipeline execution."""
    stages = Table.grid(padding=(0, 2))
    stages.add_column(style="bold cyan", justify="right", no_wrap=True)
    stages.add_column(overflow="fold")

    stages.add_row("[1] EXTRACT", Text(result.raw_text, style="white"))
    stages.add_row("[2] MASKED", Text(result.masked_text, style="yellow"))
    stages.add_row(
        "[3] SENT",
        Text("↑ line [2] verbatim — the vault never leaves this process", style="dim italic"),
    )
    stages.add_row("[4] REBUILT", Text(result.final_text, style="bold green"))

    body: list = [stages]

    if result.report.assessments:
        body += [Rule(style="dim"), _verdict_table(result)]
    if result.warnings:
        body += [Text("\n".join(f"⚠ {w}" for w in result.warnings), style="bold yellow")]

    leaked = result.leaked_values()
    if not leaked:
        body += [Text("✔ leak canary: no original value present in the outbound payload", style="green")]
    else:
        body += [Text(f"✘ LEAK: {leaked} present in the outbound payload", style="bold white on red")]

    console.print(
        Panel(
            Group(*body),
            title=f"[bold]Zero-Data-Leak Pipeline[/] · classifier=[cyan]{result.classifier_name}[/]",
            border_style="blue",
            padding=(1, 2),
        )
    )


def _verdict_table(result: PipelineResult) -> Table:
    """Builds the table showing the classifier's verdict for each token."""
    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Tag", no_wrap=True)
    table.add_column("Local value", style="dim")
    table.add_column("Category", no_wrap=True)
    table.add_column("Verdict", no_wrap=True)
    table.add_column("Rationale", overflow="fold")

    for assessment in result.report.assessments:
        local_value = (
            result.document.original(assessment.token)
            if result.document.has(assessment.token)
            else "—"
        )
        table.add_row(
            assessment.token,
            local_value,
            assessment.category,
            Text("REDACT", style="bold red") if assessment.is_sensitive else Text("restore", style="green"),
            assessment.rationale,
        )
    return table


def render_before_after(result: PipelineResult, expected: str | None, intent: str) -> Text:
    """Compact Before → Sent → After cell for the QA suite table."""
    detail = Text()
    detail.append(f"{intent}\n", style="dim italic")
    detail.append("BEFORE  ", style="bold cyan")
    detail.append(f"{result.raw_text}\n")
    detail.append("SENT    ", style="bold yellow")
    detail.append(f"{result.masked_text}\n", style="yellow")
    detail.append("AFTER   ", style="bold green")
    detail.append(result.final_text, style="green")

    if expected is not None and result.final_text != expected:
        detail.append("\nEXPECT  ", style="bold red")
        detail.append(expected, style="red")

    return detail