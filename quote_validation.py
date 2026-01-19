"""
Verbatim quote validation for thematic analysis.

Validates that LLM-extracted quotes appear exactly in the source transcript.
Uses normalized exact matching: same words, same order, contiguous span.
"""

import copy
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class QuoteValidationResult:
    """Result of validating a single quote."""

    is_valid: bool
    quote_text: str
    normalized_quote: str
    reason: str
    position: Optional[Tuple[int, int]] = None  # (start, end) in normalized transcript


@dataclass
class ValidationSummary:
    """Summary of validation for entire analysis."""

    total_quotes: int
    valid_quotes: int
    invalid_quotes: int
    empty_subthemes: List[Tuple[str, str]]  # [(theme_title, subtheme_title), ...]


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    Transformations:
    - Convert to lowercase
    - Remove all punctuation
    - Collapse multiple whitespace to single space
    - Strip leading/trailing whitespace
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def validate_quote(
    quote_text: str, normalized_transcript: str
) -> QuoteValidationResult:
    """
    Validate a single quote against the transcript.

    Args:
        quote_text: The original quote text from LLM
        normalized_transcript: Pre-normalized transcript text

    Returns:
        QuoteValidationResult with validation details
    """
    normalized_quote = normalize_text(quote_text)

    if not normalized_quote:
        return QuoteValidationResult(
            is_valid=False,
            quote_text=quote_text,
            normalized_quote=normalized_quote,
            reason="Quote is empty after normalization",
        )

    # Find quote in transcript
    position = normalized_transcript.find(normalized_quote)

    if position == -1:
        return QuoteValidationResult(
            is_valid=False,
            quote_text=quote_text,
            normalized_quote=normalized_quote,
            reason="No matching text found in transcript",
        )

    return QuoteValidationResult(
        is_valid=True,
        quote_text=quote_text,
        normalized_quote=normalized_quote,
        reason="Quote verified",
        position=(position, position + len(normalized_quote)),
    )


def _print_invalid_quote(
    result: QuoteValidationResult, theme: str, subtheme: str
) -> None:
    """Print debug output for rejected quote."""
    print(f"\n[QUOTE VALIDATION] Invalid quote rejected")
    print(f'  Theme: "{theme}"')
    print(f'  Subtheme: "{subtheme}"')
    quote_display = result.quote_text[:100] + ("..." if len(result.quote_text) > 100 else "")
    print(f'  Quote: "{quote_display}"')
    print(f"  Reason: {result.reason}")
    normalized_display = result.normalized_quote[:80] + ("..." if len(result.normalized_quote) > 80 else "")
    print(f'  Normalized: "{normalized_display}"')


def _print_summary(summary: ValidationSummary) -> None:
    """Print validation summary."""
    print(f"\n[QUOTE VALIDATION] Summary")
    print(f"  Total quotes: {summary.total_quotes}")
    print(f"  Valid: {summary.valid_quotes}")
    print(f"  Invalid (removed): {summary.invalid_quotes}")
    if summary.empty_subthemes:
        print(
            f"  WARNING: {len(summary.empty_subthemes)} subtheme(s) have no valid quotes:"
        )
        for theme, subtheme in summary.empty_subthemes:
            print(f"    - {theme} > {subtheme}")


def validate_and_filter_quotes(
    analysis_data: dict, transcript: str, debug: bool = True
) -> Tuple[dict, ValidationSummary]:
    """
    Validate all quotes in analysis and filter out invalid ones.

    Args:
        analysis_data: The validated ThematicAnalysis dict
        transcript: Original transcript text
        debug: Whether to print debug output to console

    Returns:
        Tuple of (filtered_data, validation_summary)
    """
    # Pre-normalize transcript once for efficiency
    normalized_transcript = normalize_text(transcript)

    total_quotes = 0
    valid_quotes = 0
    invalid_quotes = 0
    empty_subthemes: List[Tuple[str, str]] = []

    # Deep copy to avoid mutating original
    filtered_data = copy.deepcopy(analysis_data)

    for theme in filtered_data.get("themes", []):
        theme_title = theme.get("theme_title", "Unknown Theme")

        for subtheme in theme.get("subthemes", []):
            subtheme_title = subtheme.get("subtheme_title", "Unknown Subtheme")
            quotes = subtheme.get("supporting_quotes", [])
            valid_quotes_list = []

            for quote in quotes:
                total_quotes += 1
                quote_text = quote.get("text", "")

                result = validate_quote(quote_text, normalized_transcript)

                if result.is_valid:
                    valid_quotes += 1
                    valid_quotes_list.append(quote)
                else:
                    invalid_quotes += 1
                    if debug:
                        _print_invalid_quote(result, theme_title, subtheme_title)

            # Replace quotes with only valid ones
            subtheme["supporting_quotes"] = valid_quotes_list

            if len(valid_quotes_list) == 0:
                empty_subthemes.append((theme_title, subtheme_title))

    summary = ValidationSummary(
        total_quotes=total_quotes,
        valid_quotes=valid_quotes,
        invalid_quotes=invalid_quotes,
        empty_subthemes=empty_subthemes,
    )

    if debug:
        _print_summary(summary)

    return filtered_data, summary


def format_quote_validation_error(summary: ValidationSummary) -> str:
    """
    Format validation summary as error message for LLM retry.

    Used when validation failure requires regeneration.
    """
    lines = ["Quote validation failed:"]
    lines.append(
        f"  - {summary.invalid_quotes} of {summary.total_quotes} quotes were not found verbatim in the transcript"
    )

    if summary.empty_subthemes:
        lines.append("  - The following subthemes have no valid quotes:")
        for theme, subtheme in summary.empty_subthemes:
            lines.append(f"    * {theme} > {subtheme}")

    lines.append("")
    lines.append("IMPORTANT: All quotes must be EXACT verbatim text from the transcript.")
    lines.append("Copy quotes character-for-character. Do not paraphrase or summarize.")

    return "\n".join(lines)
