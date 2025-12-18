"""
CRUD operations for themes, subthemes, and quotes.

Uses Pydantic models from schemas.py to ensure type safety when creating new items.
"""

import streamlit as st

from schemas import Quote, Subtheme, Theme


def add_theme():
    """Add a new empty theme to results."""
    themes = st.session_state.results.get("themes", [])
    new_id = max([t.get("id", 0) for t in themes], default=0) + 1

    # Use Pydantic model for type-safe creation
    new_theme = Theme(
        id=new_id,
        theme_title="New Theme",
        detailed_explanation="Enter explanation here...",
        subthemes=[
            Subtheme(
                subtheme_title="New Subtheme",
                analysis="Enter analysis here...",
                supporting_quotes=[
                    Quote(
                        text="Enter quote text...",
                        quote_explanation="Enter explanation...",
                    )
                ],
            )
        ],
    )
    themes.append(new_theme.model_dump())
    st.session_state.results["themes"] = themes


def delete_theme(theme_idx: int):
    """Delete a theme by index."""
    themes = st.session_state.results.get("themes", [])
    if 0 <= theme_idx < len(themes):
        themes.pop(theme_idx)
        # Reindex theme IDs
        for i, theme in enumerate(themes):
            theme["id"] = i + 1
    st.session_state.editing = None


def add_subtheme(theme_idx: int):
    """Add a new empty subtheme to a theme."""
    themes = st.session_state.results.get("themes", [])
    if 0 <= theme_idx < len(themes):
        # Use Pydantic model for type-safe creation
        new_subtheme = Subtheme(
            subtheme_title="New Subtheme",
            analysis="Enter analysis here...",
            supporting_quotes=[
                Quote(
                    text="Enter quote text...",
                    quote_explanation="Enter explanation...",
                )
            ],
        )
        themes[theme_idx].setdefault("subthemes", []).append(new_subtheme.model_dump())


def delete_subtheme(theme_idx: int, subtheme_idx: int):
    """Delete a subtheme by index."""
    themes = st.session_state.results.get("themes", [])
    if 0 <= theme_idx < len(themes):
        subthemes = themes[theme_idx].get("subthemes", [])
        if 0 <= subtheme_idx < len(subthemes):
            subthemes.pop(subtheme_idx)
    st.session_state.editing = None


def add_quote(theme_idx: int, subtheme_idx: int):
    """Add a new empty quote to a subtheme."""
    themes = st.session_state.results.get("themes", [])
    if 0 <= theme_idx < len(themes):
        subthemes = themes[theme_idx].get("subthemes", [])
        if 0 <= subtheme_idx < len(subthemes):
            # Use Pydantic model for type-safe creation
            new_quote = Quote(
                text="Enter quote text...",
                quote_explanation="Enter explanation...",
            )
            subthemes[subtheme_idx].setdefault("supporting_quotes", []).append(
                new_quote.model_dump()
            )


def delete_quote(theme_idx: int, subtheme_idx: int, quote_idx: int):
    """Delete a quote by index."""
    themes = st.session_state.results.get("themes", [])
    if 0 <= theme_idx < len(themes):
        subthemes = themes[theme_idx].get("subthemes", [])
        if 0 <= subtheme_idx < len(subthemes):
            quotes = subthemes[subtheme_idx].get("supporting_quotes", [])
            if 0 <= quote_idx < len(quotes):
                quotes.pop(quote_idx)
    st.session_state.editing = None
