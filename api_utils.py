"""
API utilities for LLM calls and response processing.
"""

import json

from constants import CODING_PROMPT


def analyze_transcript(client, model: str, transcript: str) -> dict:
    """Call LLM to analyze transcript and return structured JSON."""
    full_prompt = f"{CODING_PROMPT}\n\nTRANSCRIPT DATA:\n{transcript}"

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.7,
    )

    response_text = response.choices[0].message.content

    # Parse JSON from response
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON if wrapped in markdown
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0]
            return json.loads(json_str)
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0]
            return json.loads(json_str)
        raise


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
