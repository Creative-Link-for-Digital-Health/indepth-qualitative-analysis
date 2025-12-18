"""
API utilities for LLM calls and response processing.
"""

import json
import re
import time

from pydantic import ValidationError

from constants import CODING_PROMPT_PREFIX, CODING_PROMPT_SUFFIX
from schemas import ThematicAnalysis


class AnalysisError(Exception):
    """Raised when transcript analysis fails after retries."""
    pass


def _format_validation_error(error: ValidationError) -> str:
    """Format Pydantic validation error for LLM feedback.

    Converts Pydantic's ValidationError into a human-readable message
    that the LLM can understand and act on to fix specific issues.
    """
    lines = ["Validation errors found:"]
    for err in error.errors():
        location = " -> ".join(str(loc) for loc in err["loc"])
        lines.append(f"  - {location}: {err['msg']}")
    return "\n".join(lines)


def _validate_with_pydantic(data: dict) -> dict:
    """Validate data against ThematicAnalysis schema.

    Returns the validated data as a dict. Raises ValidationError if invalid.
    """
    validated = ThematicAnalysis.model_validate(data)
    return validated.model_dump()


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

    Uses Pydantic validation to ensure output matches the ThematicAnalysis schema.
    On validation failure, feeds the error back to the LLM for correction.
    Retries up to max_retries times with exponential backoff.
    Raises AnalysisError if all retries fail.
    """
    # Put JSON instruction at END of prompt - models pay attention to the end
    initial_prompt = f"{CODING_PROMPT_PREFIX}\n\n{transcript}\n\n{CODING_PROMPT_SUFFIX}"
    messages = [{"role": "user", "content": initial_prompt}]
    last_error = None
    last_response = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
            )

            response_text = response.choices[0].message.content
            last_response = response_text

            # Extract JSON from response
            result = _extract_json(response_text)

            # Validate with Pydantic - raises ValidationError on failure
            validated_result = _validate_with_pydantic(result)
            return validated_result

        except json.JSONDecodeError as e:
            last_error = e
            error_msg = f"JSON parsing failed: {str(e)}"

        except ValidationError as e:
            last_error = e
            error_msg = _format_validation_error(e)

        except Exception as e:
            # For non-JSON errors (API errors, network issues), raise immediately
            raise AnalysisError(f"API error: {str(e)}") from e

        # If we're here, validation failed - prepare retry with error feedback
        if attempt < max_retries:
            # Add the failed response and error feedback to conversation
            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "user",
                "content": f"""Your previous response had errors:

{error_msg}

Please fix these issues and respond with ONLY valid JSON matching the required schema.
Start your response with {{ and end with }}. No markdown, no explanation."""
            })
            # Exponential backoff: 2s, 4s
            time.sleep(2 ** attempt)

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
