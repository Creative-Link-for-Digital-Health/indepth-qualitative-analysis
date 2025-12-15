"""
API utilities for LLM calls and response processing.
"""

import json
import time

from constants import CODING_PROMPT


class AnalysisError(Exception):
    """Raised when transcript analysis fails after retries."""
    pass


def _extract_json(response_text: str) -> dict:
    """Extract and parse JSON from LLM response text."""
    # Try direct parsing first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON if wrapped in markdown
    if "```json" in response_text:
        json_str = response_text.split("```json")[1].split("```")[0]
        return json.loads(json_str)
    elif "```" in response_text:
        json_str = response_text.split("```")[1].split("```")[0]
        return json.loads(json_str)

    # If all extraction attempts fail, raise with original text
    raise json.JSONDecodeError("Could not extract valid JSON", response_text, 0)


def analyze_transcript(client, model: str, transcript: str, max_retries: int = 3) -> dict:
    """Call LLM to analyze transcript and return structured JSON.

    Retries up to max_retries times with exponential backoff on JSON parse failures.
    Raises AnalysisError if all retries fail.
    """
    full_prompt = f"{CODING_PROMPT}\n\nTRANSCRIPT DATA:\n{transcript}"
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.3,
            )

            response_text = response.choices[0].message.content
            return _extract_json(response_text)

        except json.JSONDecodeError as e:
            last_error = e
            if attempt < max_retries:
                # Exponential backoff: 2s, 4s
                time.sleep(2 ** attempt)
                continue
        except Exception as e:
            # For non-JSON errors (API errors, network issues), raise immediately
            raise AnalysisError(f"API error: {str(e)}") from e

    # All retries exhausted
    raise AnalysisError(
        f"Failed to get valid JSON after {max_retries} attempts. "
        f"Last error: {str(last_error)}"
    )


def format_results_as_text(results: dict) -> str:
    """Convert JSON results to readable text format."""
    lines = []

    # Document Summary
    lines.append("DOCUMENT SUMMARY")
    lines.append("=" * 50)
    lines.append(results.get("document_summary", "No summary available."))
    lines.append("")
    lines.append("")

    # Themes
    for theme in results.get("themes", []):
        theme_id = theme.get("id", "")
        theme_title = theme.get("theme_title", "Untitled Theme")

        lines.append(f"THEME {theme_id}: {theme_title}")
        lines.append("=" * 50)
        lines.append(theme.get("detailed_explanation", ""))
        lines.append("")

        # Subthemes
        for subtheme in theme.get("subthemes", []):
            subtheme_title = subtheme.get("subtheme_title", "Untitled Subtheme")

            lines.append(f"  Subtheme: {subtheme_title}")
            lines.append("  " + "-" * 40)
            lines.append(f"  Analysis: {subtheme.get('analysis', '')}")
            lines.append("")
            lines.append("  Quotes:")

            # Quotes
            for quote in subtheme.get("supporting_quotes", []):
                quote_text = quote.get("text", "")
                quote_explanation = quote.get("quote_explanation", "")

                lines.append(f'    - "{quote_text}"')
                lines.append(f"      Explanation: {quote_explanation}")
                lines.append("")

            lines.append("")

        lines.append("")

    return "\n".join(lines)
