"""
API utilities for LLM calls and response processing.
"""

import json
import re
import time

from constants import CODING_PROMPT_PREFIX, CODING_PROMPT_SUFFIX


class AnalysisError(Exception):
    """Raised when transcript analysis fails after retries."""
    pass


class SchemaValidationError(Exception):
    """Raised when JSON doesn't match expected schema."""
    pass


def _validate_schema(data: dict) -> None:
    """Validate that response matches expected schema. Raises SchemaValidationError if not."""
    if "document_summary" not in data and "themes" not in data:
        # Check if model used wrong field name
        if "summary" in data:
            raise SchemaValidationError(
                "Response used 'summary' instead of 'document_summary'. Missing 'themes' array."
            )
        raise SchemaValidationError("Response missing required fields: 'document_summary' and 'themes'")

    if "themes" not in data or not isinstance(data.get("themes"), list):
        raise SchemaValidationError("Response missing 'themes' array or themes is not a list")

    if len(data.get("themes", [])) == 0:
        raise SchemaValidationError("Response has empty 'themes' array - no themes were extracted")


def _extract_json(response_text: str) -> dict:
    """Extract and parse JSON from LLM response text."""
    # Try direct parsing first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON if wrapped in markdown (case-insensitive)
    if "```json" in response_text.lower():
        match = re.search(r"```[jJ][sS][oO][nN]\s*(.*?)```", response_text, re.DOTALL)
        if match:
            return json.loads(match.group(1).strip())
    elif "```" in response_text:
        json_str = response_text.split("```")[1].split("```")[0]
        return json.loads(json_str)

    # Try to find JSON object by matching braces
    first_brace = response_text.find("{")
    last_brace = response_text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_str = response_text[first_brace:last_brace + 1]
        return json.loads(json_str)

    # If all extraction attempts fail, raise with original text
    raise json.JSONDecodeError("Could not extract valid JSON", response_text, 0)


def analyze_transcript(client, model: str, transcript: str, max_retries: int = 3) -> dict:
    """Call LLM to analyze transcript and return structured JSON.

    Retries up to max_retries times with exponential backoff on JSON parse failures.
    Raises AnalysisError if all retries fail.
    """
    # Put JSON instruction at END of prompt - models pay attention to the end
    full_prompt = f"{CODING_PROMPT_PREFIX}\n\n{transcript}\n\n{CODING_PROMPT_SUFFIX}"
    last_error = None
    last_response = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.3,
            )

            response_text = response.choices[0].message.content
            last_response = response_text
            print(f"[DEBUG] Response length: {len(response_text)} chars")
            print(f"[DEBUG] First 500 chars: {response_text[:500]}")
            result = _extract_json(response_text)
            _validate_schema(result)
            return result

        except (json.JSONDecodeError, SchemaValidationError) as e:
            last_error = e
            print(f"[DEBUG] JSON parse error on attempt {attempt}: {e}")
            if attempt < max_retries:
                # Exponential backoff: 2s, 4s
                time.sleep(2 ** attempt)
                continue
        except Exception as e:
            # For non-JSON errors (API errors, network issues), raise immediately
            raise AnalysisError(f"API error: {str(e)}") from e

    # All retries exhausted - include response snippet in error
    snippet = last_response[:300] if last_response else "No response received"
    raise AnalysisError(
        f"Failed to get valid JSON after {max_retries} attempts. "
        f"Last error: {str(last_error)}\n"
        f"Response snippet: {snippet}..."
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
