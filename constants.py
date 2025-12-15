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
    font-size: 1rem;
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

# Short intro BEFORE the transcript
CODING_PROMPT_PREFIX = """You are a qualitative researcher. Analyze the following transcript for themes."""

# Full JSON schema instruction AFTER the transcript (models pay attention to the end)
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
