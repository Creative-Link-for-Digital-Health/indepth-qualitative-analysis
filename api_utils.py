"""
API utilities for LLM calls and response processing.
"""

import json
import re
import time

from pydantic import ValidationError

from constants import (
    THEME_EXTRACTION_PREFIX,
    THEME_EXTRACTION_SUFFIX,
    QUOTE_EXTRACTION_PROMPT,
)
from schemas import ThematicAnalysis, ThemeExtraction
from quote_validation import validate_quote, normalize_text


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


def _extract_themes(client, model: str, transcript: str, max_retries: int = 3) -> dict:
    """Pass 1: Extract theme structure without quotes.

    Returns a ThemeExtraction dict (themes/subthemes without quotes).
    """
    initial_prompt = f"{THEME_EXTRACTION_PREFIX}\n\n{transcript}\n\n{THEME_EXTRACTION_SUFFIX}"
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

            # Validate with Pydantic against ThemeExtraction schema
            validated = ThemeExtraction.model_validate(result)
            return validated.model_dump()

        except json.JSONDecodeError as e:
            last_error = e
            error_msg = f"JSON parsing failed: {str(e)}"

        except ValidationError as e:
            last_error = e
            error_msg = _format_validation_error(e)

        except Exception as e:
            raise AnalysisError(f"API error during theme extraction: {str(e)}") from e

        # Retry with error feedback
        if attempt < max_retries:
            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "user",
                "content": f"""Your previous response had errors:

{error_msg}

Please fix these issues and respond with ONLY valid JSON matching the required schema.
Start your response with {{ and end with }}. No markdown, no explanation."""
            })
            time.sleep(2 ** attempt)

    snippet = last_response[:300] if last_response else "No response received"
    raise AnalysisError(
        f"Failed to extract themes after {max_retries} attempts. "
        f"Last error: {str(last_error)}\n"
        f"Response snippet: {snippet}..."
    )


def _extract_quotes_for_subtheme(
    client,
    model: str,
    transcript: str,
    theme_title: str,
    subtheme_title: str,
    subtheme_analysis: str,
    max_retries: int = 2,
) -> list:
    """Pass 2: Extract quotes for a specific subtheme.

    Returns a list of Quote dicts. May return empty list if no valid quotes found.
    """
    prompt = QUOTE_EXTRACTION_PROMPT.format(
        transcript=transcript,
        theme_title=theme_title,
        subtheme_title=subtheme_title,
        subtheme_analysis=subtheme_analysis,
    )
    messages = [{"role": "user", "content": prompt}]
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

            # Extract JSON array from response
            result = _extract_json_array(response_text)

            # Validate each quote has required fields
            validated_quotes = []
            for quote in result:
                if isinstance(quote, dict) and "text" in quote and "quote_explanation" in quote:
                    if quote["text"].strip() and quote["quote_explanation"].strip():
                        validated_quotes.append(quote)

            return validated_quotes

        except json.JSONDecodeError as e:
            last_error = e
            error_msg = f"JSON parsing failed: {str(e)}"

        except Exception as e:
            last_error = e
            error_msg = str(e)

        # Retry with error feedback
        if attempt < max_retries:
            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "user",
                "content": f"""Your previous response had errors:

{error_msg}

Please respond with ONLY a valid JSON array of quotes.
Start your response with [ and end with ]. No markdown, no explanation."""
            })
            time.sleep(1)

    # Return empty list on failure (quote extraction is best-effort)
    print(f"[QUOTE EXTRACTION] Failed for subtheme '{subtheme_title}' after {max_retries} attempts")
    return []


def _extract_json_array(response_text: str) -> list:
    """Extract and parse JSON array from LLM response text."""
    # Try direct parsing first
    try:
        result = json.loads(response_text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try to extract array if wrapped in markdown
    if "```json" in response_text.lower():
        match = re.search(r"```[jJ][sS][oO][nN]\s*(.*?)```", response_text, re.DOTALL)
        if match:
            result = json.loads(match.group(1).strip())
            if isinstance(result, list):
                return result
    elif "```" in response_text:
        json_str = response_text.split("```")[1].split("```")[0]
        result = json.loads(json_str)
        if isinstance(result, list):
            return result

    # Try to find JSON array by matching brackets
    first_bracket = response_text.find("[")
    last_bracket = response_text.rfind("]")
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        json_str = response_text[first_bracket:last_bracket + 1]
        result = json.loads(json_str)
        if isinstance(result, list):
            return result

    raise json.JSONDecodeError("Could not extract valid JSON array", response_text, 0)


def analyze_transcript(client, model: str, transcript: str, max_retries: int = 3) -> dict:
    """Analyze transcript using two-pass approach for better quote accuracy.

    Pass 1: Extract themes and subthemes (no quotes)
    Pass 2: For each subtheme, extract supporting quotes

    This approach reduces quote hallucination by having the LLM focus on
    one task at a time: first understanding the themes, then finding
    verbatim quotes to support each subtheme.
    """
    # Pass 1: Extract theme structure
    print("[ANALYSIS] Pass 1: Extracting themes and subthemes...")
    theme_structure = _extract_themes(client, model, transcript, max_retries)

    # Pre-normalize transcript for quote validation
    normalized_transcript = normalize_text(transcript)

    # Pass 2: Extract quotes for each subtheme
    total_subthemes = sum(len(t["subthemes"]) for t in theme_structure["themes"])
    current_subtheme = 0
    quote_retries = 2  # Additional retries for subthemes with no valid quotes

    for theme in theme_structure["themes"]:
        theme_title = theme["theme_title"]

        for subtheme in theme["subthemes"]:
            current_subtheme += 1
            subtheme_title = subtheme["subtheme_title"]
            print(f"[ANALYSIS] Pass 2: Extracting quotes for subtheme {current_subtheme}/{total_subthemes}: {subtheme_title}")

            valid_quotes = []

            # Try multiple times to get valid quotes
            for attempt in range(1, quote_retries + 1):
                # Get quotes from LLM
                raw_quotes = _extract_quotes_for_subtheme(
                    client,
                    model,
                    transcript,
                    theme_title,
                    subtheme_title,
                    subtheme["analysis"],
                )

                # Validate each quote against transcript
                for quote in raw_quotes:
                    result = validate_quote(quote["text"], normalized_transcript)
                    if result.is_valid:
                        valid_quotes.append(quote)
                    else:
                        print(f"[QUOTE VALIDATION] Rejected: '{quote['text'][:50]}...' - {result.reason}")

                if valid_quotes:
                    break  # Got at least one valid quote
                elif attempt < quote_retries:
                    print(f"[RETRY] No valid quotes found, retrying... (attempt {attempt + 1}/{quote_retries})")

            # Add validated quotes to subtheme
            subtheme["supporting_quotes"] = valid_quotes

            if not valid_quotes:
                print(f"[WARNING] No valid quotes found for subtheme: {subtheme_title}")

    # Remove subthemes with no valid quotes (required by schema)
    for theme in theme_structure["themes"]:
        theme["subthemes"] = [
            s for s in theme["subthemes"] if s.get("supporting_quotes")
        ]

    # Remove themes with no subthemes
    theme_structure["themes"] = [
        t for t in theme_structure["themes"] if t.get("subthemes")
    ]

    # Check if we have any themes left
    if not theme_structure["themes"]:
        raise AnalysisError("No valid themes with quotes could be extracted from the transcript.")

    # Final validation to ensure structure matches ThematicAnalysis
    validated = ThematicAnalysis.model_validate(theme_structure)
    return validated.model_dump()


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
