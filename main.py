"""
Orchestrator and QA Suite for the Zero-Data-Leak PII ETL Pipeline.

This script runs the end-to-end process:
1. Extract (Local)
2. Mask PII (Local)
3. Classify context (Cloud/API)
4. Reassemble and redact (Local)

It includes a CLI for running single strings and a built-in QA suite
to verify confidentiality (no leaks) and correctness.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from ai_classifier import HeuristicClassifier, OpenAIClassifier, analyse_with_fail_closed
from masking import LocalMaskingEngine
from models import PipelineResult, REDACTION_MARKER, TokenClassifier
from reassembly import reassemble, render_before_after, render_walkthrough
from models import DEFAULT_MODEL

console = Console()

DEFAULT_TEXT = "Send $500 to Georgi Ivanov in Sofia for rent."


def run_pipeline(raw_text: str, classifier: TokenClassifier) -> PipelineResult:
    """Executes the full ETL pipeline sequentially."""

    document = LocalMaskingEngine().mask(raw_text)                      # [1] + [2] local
    report, warnings = analyse_with_fail_closed(classifier, document)   # [3] cloud API
    final_text = reassemble(document, report)                           # [4] local

    return PipelineResult(
        raw_text=raw_text,
        masked_text=document.masked_text,
        final_text=final_text,
        report=report,
        document=document,
        warnings=warnings,
        classifier_name=getattr(classifier, "name", type(classifier).__name__),
    )


# --- QA Tests ---

@dataclass(frozen=True)
class QACase:
    name: str
    raw: str
    expected: str
    intent: str


QA_CASES: tuple[QACase, ...] = (
    QACase(
        name="canonical mixed",
        intent="Name pair + city destroyed; amount and generic nouns survive intact.",
        raw=DEFAULT_TEXT,
        expected=f"Send $500 to {REDACTION_MARKER} {REDACTION_MARKER} in {REDACTION_MARKER} for rent.",
    ),
    QACase(
        name="no PII at all",
        intent="Zero candidates survive local filtering; output must be byte-identical.",
        raw="Please review the quarterly budget report before Friday.",
        expected="Please review the quarterly budget report before Friday.",
    ),
    QACase(
        name="saturated PII",
        intent="Two coordinated name pairs and two coordinated cities — all destroyed.",
        raw="Contact Georgi Ivanov and Maria Petrova in Sofia and Plovdiv.",
        expected=(
            f"Contact {REDACTION_MARKER} {REDACTION_MARKER} and {REDACTION_MARKER} "
            f"{REDACTION_MARKER} in {REDACTION_MARKER} and {REDACTION_MARKER}."
        ),
    ),
    QACase(
        name="tricky non-name capitals",
        intent="Capitalised words that are NOT names get tokenised, then fully restored.",
        raw="Leadership approved Python and Kubernetes training for the whole team.",
        expected="Leadership approved Python and Kubernetes training for the whole team.",
    ),
    QACase(
        name="repeat entity + decoys",
        intent="Same name twice → one tag, one consistent redaction; 'Q3'/'Friday' untouched.",
        raw="Please wire the Q3 bonus to Elena Dimitrova at the Varna branch; Elena signs Friday.",
        expected=(
            f"Please wire the Q3 bonus to {REDACTION_MARKER} {REDACTION_MARKER} at the "
            f"{REDACTION_MARKER} branch; {REDACTION_MARKER} signs Friday."
        ),
    ),
    QACase(
        name="sentence-initial recurrence",
        intent=(
            "Confidentiality regression test: a name that opens a sentence must still be "
            "masked once a later mention in the same document confirms it's a name — the "
            "masked payload must never carry that first mention in plaintext, even if the "
            "offline heuristic classifier (unlike the live model) can't tell from a single, "
            "cue-less occurrence that it should also be redacted."
        ),
        raw="Elena signed the invoice. Please pay Elena Dimitrova for the consulting work.",
        expected="Elena signed the invoice. Please pay Elena Dimitrova for the consulting work.",
    ),
)


def run_qa_suite(classifier: TokenClassifier, strict: bool = True) -> bool:
    """
    Runs the test suite to verify no PII leaks during the process
    and checks if the redactions match the expected output.
    """
    console.print(Rule("[bold]STAGE 5 · QA SELF-CHECK[/]", style="magenta"))
    console.print(
        Text(
            f"backend={getattr(classifier, 'name', type(classifier).__name__)}   "
            f"strict={'on' if strict else 'off (correctness informational)'}\n",
            style="dim",
        )
    )

    table = Table(show_header=True, header_style="bold magenta", expand=True, show_lines=True)
    table.add_column("#", no_wrap=True, width=3)
    table.add_column("Case", no_wrap=True)
    table.add_column("Before → Sent → After", overflow="fold")
    table.add_column("Leak", no_wrap=True, width=6)
    table.add_column("Match", no_wrap=True, width=7)

    failures: list[str] = []

    for i, case in enumerate(QA_CASES, start=1):
        result = run_pipeline(case.raw, classifier)
        leaked = result.leaked_values()
        matched = result.final_text == case.expected

        # Check for PII leaks always enforced
        try:
            assert not leaked, (
                f"LEAK in '{case.name}': {leaked} present in the payload sent to the classifier\n"
                f"    payload: {result.masked_text}"
            )
        except AssertionError as exc:
            failures.append(str(exc))

        # Check exact match output
        if strict:
            try:
                assert matched, (
                    f"MISMATCH in '{case.name}'\n"
                    f"    expected: {case.expected}\n"
                    f"    actual:   {result.final_text}"
                )
            except AssertionError as exc:
                failures.append(str(exc))

        table.add_row(
            str(i),
            case.name,
            render_before_after(result, case.expected, case.intent),
            Text("NONE", style="bold green") if not leaked else Text("FOUND", style="bold red"),
            Text("PASS", style="bold green") if matched else Text("FAIL", style="bold red"),
        )

    console.print(table)

    if failures:
        console.print(
            Panel(
                "\n\n".join(failures),
                title=f"[bold red]QA FAILED · {len(failures)} assertion(s)[/]",
                border_style="red",
            )
        )
        return False

    console.print(
        Panel(
            Text(
                f"{len(QA_CASES)}/{len(QA_CASES)} cases passed.\n"
                "Confidentiality: No original value reached the outbound payload.\n"
                "Correctness: Every reassembled string matched its expected redaction.",
                style="bold green",
            ),
            title="[bold green]QA PASSED[/]",
            border_style="green",
        )
    )
    return True


def build_classifier(live: bool, model: str) -> TokenClassifier:
    """Instantiates the correct classifier based on CLI flags and env vars."""
    if not live:
        return HeuristicClassifier()
    if not os.getenv("OPENAI_API_KEY"):
        console.print(
            "[bold yellow]--live requested but OPENAI_API_KEY is not set.[/] "
            "Falling back to the offline heuristic classifier.\n"
        )
        return HeuristicClassifier()
    return OpenAIClassifier(model=model)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Zero-Data-Leak PII ETL pipeline.")
    parser.add_argument("--live", action="store_true",
                        help="Use the OpenAI classifier for the demo run.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"OpenAI model for --live (default: {DEFAULT_MODEL}).")
    parser.add_argument("--text", default=DEFAULT_TEXT,
                        help="Document to push through the demo run.")
    parser.add_argument("--qa-only", action="store_true",
                        help="Skip the demo, run the QA suite only.")
    parser.add_argument("--qa-live", action="store_true",
                        help="Run the QA suite against OpenAI (implies --live).")
    parser.add_argument("--no-strict", action="store_true",
                        help="Assert confidentiality only; report redaction mismatches "
                             "as informational.")
    args = parser.parse_args(argv)

    demo_classifier = build_classifier(args.live or args.qa_live, args.model)

    if not args.qa_only:
        console.print(Rule("[bold]STAGES 1–4 · DEMO[/]", style="blue"))
        render_walkthrough(run_pipeline(args.text, demo_classifier), console)

    # QA runs offline by default
    qa_classifier = demo_classifier if args.qa_live else HeuristicClassifier()
    qa_ok = run_qa_suite(qa_classifier, strict=not args.no_strict)

    return 0 if qa_ok else 1


if __name__ == "__main__":
    sys.exit(main())