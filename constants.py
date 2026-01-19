"""
Constants and configuration for the Qualitative Theme Analysis app.
"""

from pathlib import Path

# File paths
SECRETS_PATH = ".secrets.toml"
OUTPUT_DIR = Path("outputs")

# CSS Styling for card-based UI
CARD_CSS = """
<style>
.theme-card {
    border: 2px solid #4a5568;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
    background-color: #1e1e1e;
}
.theme-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}
.theme-title {
    font-size: 1.2rem;
    font-weight: bold;
    color: #e2e8f0;
    margin: 0;
}
.theme-explanation {
    color: #a0aec0;
    margin-bottom: 1rem;
}
.subtheme-card {
    border: 1px solid #4a5568;
    border-radius: 6px;
    padding: 0.75rem;
    margin: 0.5rem 0 0.5rem 1rem;
    background-color: #2d2d2d;
}
.subtheme-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}
.subtheme-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: #cbd5e0;
}
.subtheme-analysis {
    color: #a0aec0;
    font-size: 0.9rem;
    margin-bottom: 0.75rem;
}
.quote-item {
    background-color: #383838;
    border-left: 3px solid #667eea;
    padding: 0.5rem 0.75rem;
    margin: 0.5rem 0;
    border-radius: 0 4px 4px 0;
}
.quote-text {
    color: #e2e8f0;
    font-style: italic;
    margin-bottom: 0.25rem;
}
.quote-explanation {
    color: #a0aec0;
    font-size: 0.85rem;
}
.icon-btn {
    background: none;
    border: none;
    cursor: pointer;
    padding: 0.25rem;
    font-size: 0.9rem;
    opacity: 0.7;
}
.icon-btn:hover {
    opacity: 1;
}
.add-btn {
    color: #667eea;
    font-size: 0.9rem;
    cursor: pointer;
    margin-top: 0.5rem;
}
</style>
"""

# Legacy single-pass prompts (kept for reference)
CODING_PROMPT_PREFIX = """You are a qualitative researcher. Analyze the following transcript for themes."""
CODING_PROMPT_SUFFIX = """
Based on the transcript above, provide a thematic analysis.

You MUST respond with ONLY a valid JSON object using this EXACT structure:
{
  "document_summary": "Brief 1-2 sentence summary of what this transcript is about",
  "themes": [
    {
      "id": 1,
      "theme_title": "Title of the Theme",
      "detailed_explanation": "Comprehensive paragraph explaining the theme's significance",
      "subthemes": [
        {
          "subtheme_title": "Title of the Subtheme",
          "analysis": "Overview of what the quotes illustrate",
          "supporting_quotes": [
            {
              "text": "Exact verbatim quote from transcript",
              "quote_explanation": "How this quote supports the subtheme"
            }
          ]
        }
      ]
    }
  ]
}

Requirements:
- Include 2-4 major themes
- Each theme: 1-3 subthemes
- Each subtheme: 2-3 supporting quotes (verbatim from transcript)

CRITICAL: Your response must be ONLY valid JSON. No markdown, no explanation, no text before or after.
Start your response with { and end with }
"""

# =============================================================================
# TWO-PASS PROMPTS (Reduces quote hallucination)
# =============================================================================

# Pass 1: Theme extraction (no quotes)
THEME_EXTRACTION_PREFIX = """You are a qualitative researcher. Analyze the following transcript and identify the major themes and subthemes."""

THEME_EXTRACTION_SUFFIX = """
Based on the transcript above, identify the thematic structure.

You MUST respond with ONLY a valid JSON object using this EXACT structure:
{
  "document_summary": "Brief 1-2 sentence summary of what this transcript is about",
  "themes": [
    {
      "id": 1,
      "theme_title": "Title of the Theme",
      "detailed_explanation": "Comprehensive paragraph explaining the theme's significance",
      "subthemes": [
        {
          "subtheme_title": "Title of the Subtheme",
          "analysis": "Overview of what participants discussed related to this subtheme"
        }
      ]
    }
  ]
}

Requirements:
- Include 2-4 major themes
- Each theme: 1-3 subthemes
- Do NOT include quotes yet - just identify the themes and subthemes

CRITICAL: Your response must be ONLY valid JSON. No markdown, no explanation.
Start your response with { and end with }
"""

# Pass 2: Quote extraction for a specific subtheme
QUOTE_EXTRACTION_PROMPT = """You are a qualitative researcher. Your task is to find EXACT VERBATIM quotes from the transcript that support a specific subtheme.

TRANSCRIPT:
{transcript}

---

Find quotes that support this subtheme:
Theme: {theme_title}
Subtheme: {subtheme_title}
Analysis: {subtheme_analysis}

---

Return 2-3 quotes that support this subtheme. You MUST respond with ONLY a valid JSON array:
[
  {{
    "text": "Copy the EXACT words from the transcript here - character for character",
    "quote_explanation": "How this quote supports the subtheme"
  }}
]

CRITICAL RULES:
1. Quotes must be EXACT VERBATIM text from the transcript above
2. Copy word-for-word - do not paraphrase, summarize, or combine quotes
3. If you cannot find good supporting quotes, return fewer quotes rather than making them up
4. Start your response with [ and end with ]
"""
