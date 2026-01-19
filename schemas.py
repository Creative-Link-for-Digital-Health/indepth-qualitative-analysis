"""
Pydantic models for thematic analysis schema validation.

These models define the canonical structure for LLM-generated thematic analysis
outputs. All LLM responses are validated against these schemas before being
accepted or stored.
"""

from typing import List

from pydantic import BaseModel, Field


class Quote(BaseModel):
    """A supporting quote from the transcript."""

    text: str = Field(
        ...,
        min_length=1,
        description="Exact verbatim quote from transcript",
    )
    quote_explanation: str = Field(
        ...,
        min_length=1,
        description="How this quote supports the subtheme",
    )


class Subtheme(BaseModel):
    """A subtheme within a larger theme."""

    subtheme_title: str = Field(
        ...,
        min_length=1,
        description="Title of the subtheme",
    )
    analysis: str = Field(
        ...,
        min_length=1,
        description="Overview of what the quotes illustrate",
    )
    supporting_quotes: List[Quote] = Field(
        ...,
        min_length=1,
        description="Supporting quotes from the transcript",
    )


class Theme(BaseModel):
    """A major theme identified in the transcript."""

    id: int = Field(
        ...,
        ge=1,
        description="Unique theme identifier (1-indexed)",
    )
    theme_title: str = Field(
        ...,
        min_length=1,
        description="Title of the theme",
    )
    detailed_explanation: str = Field(
        ...,
        min_length=1,
        description="Comprehensive paragraph explaining the theme's significance",
    )
    subthemes: List[Subtheme] = Field(
        ...,
        min_length=1,
        description="Subthemes within this theme",
    )


class ThematicAnalysis(BaseModel):
    """
    Complete thematic analysis output from LLM.

    This is the root model that validates the entire LLM response structure.
    """

    document_summary: str = Field(
        ...,
        min_length=1,
        description="Brief 1-2 sentence summary of what the transcript is about",
    )
    themes: List[Theme] = Field(
        ...,
        min_length=1,
        description="Major themes identified in the transcript",
    )


# =============================================================================
# TWO-PASS SCHEMAS (for reduced hallucination approach)
# =============================================================================


class SubthemeStructure(BaseModel):
    """A subtheme without quotes (used in Pass 1)."""

    subtheme_title: str = Field(
        ...,
        min_length=1,
        description="Title of the subtheme",
    )
    analysis: str = Field(
        ...,
        min_length=1,
        description="Overview of what participants discussed related to this subtheme",
    )


class ThemeStructure(BaseModel):
    """A theme with subthemes but no quotes (used in Pass 1)."""

    id: int = Field(
        ...,
        ge=1,
        description="Unique theme identifier (1-indexed)",
    )
    theme_title: str = Field(
        ...,
        min_length=1,
        description="Title of the theme",
    )
    detailed_explanation: str = Field(
        ...,
        min_length=1,
        description="Comprehensive paragraph explaining the theme's significance",
    )
    subthemes: List[SubthemeStructure] = Field(
        ...,
        min_length=1,
        description="Subthemes within this theme",
    )


class ThemeExtraction(BaseModel):
    """
    Pass 1 output: Theme structure without quotes.

    Used in the two-pass approach to first identify themes/subthemes,
    then extract quotes separately.
    """

    document_summary: str = Field(
        ...,
        min_length=1,
        description="Brief 1-2 sentence summary of what the transcript is about",
    )
    themes: List[ThemeStructure] = Field(
        ...,
        min_length=1,
        description="Major themes identified in the transcript",
    )
