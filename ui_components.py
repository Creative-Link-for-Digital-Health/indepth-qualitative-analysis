"""
Streamlit UI components for rendering themes, subthemes, and quotes.
"""

import streamlit as st

from data_utils import (
    add_theme,
    delete_theme,
    add_subtheme,
    delete_subtheme,
    add_quote,
    delete_quote,
)


def render_quote(quote: dict, theme_idx: int, subtheme_idx: int, quote_idx: int):
    """Render a single quote with edit/delete functionality."""
    quote_key = f"quote-{theme_idx}-{subtheme_idx}-{quote_idx}"
    is_editing = st.session_state.get("editing") == quote_key

    if is_editing:
        # Edit mode
        new_text = st.text_area(
            "Quote text",
            value=quote.get("text", ""),
            key=f"edit-text-{quote_key}",
            height=100
        )
        new_explanation = st.text_input(
            "Explanation",
            value=quote.get("quote_explanation", ""),
            key=f"edit-expl-{quote_key}"
        )
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Save", key=f"save-{quote_key}", type="primary"):
                quote["text"] = new_text
                quote["quote_explanation"] = new_explanation
                st.session_state.editing = None
                st.rerun()
        with col2:
            if st.button("Cancel", key=f"cancel-{quote_key}"):
                st.session_state.editing = None
                st.rerun()
    else:
        # View mode
        col1, col2, col3 = st.columns([20, 1, 1])
        with col1:
            st.markdown(f'<div class="quote-item"><div class="quote-text">"{quote.get("text", "")}"</div><div class="quote-explanation">Explanation: {quote.get("quote_explanation", "")}</div></div>', unsafe_allow_html=True)
        with col2:
            if st.button("✏️", key=f"edit-{quote_key}", help="Edit quote"):
                st.session_state.editing = quote_key
                st.rerun()
        with col3:
            if st.button("🗑️", key=f"delete-{quote_key}", help="Delete quote"):
                delete_quote(theme_idx, subtheme_idx, quote_idx)
                st.rerun()


def render_subtheme(subtheme: dict, theme_idx: int, subtheme_idx: int):
    """Render a subtheme with its quotes."""
    subtheme_key = f"subtheme-{theme_idx}-{subtheme_idx}"
    is_editing = st.session_state.get("editing") == subtheme_key

    with st.container():
        if is_editing:
            # Edit mode
            new_title = st.text_input(
                "Subtheme title",
                value=subtheme.get("subtheme_title", ""),
                key=f"edit-title-{subtheme_key}"
            )
            new_analysis = st.text_area(
                "Analysis",
                value=subtheme.get("analysis", ""),
                key=f"edit-analysis-{subtheme_key}",
                height=100
            )
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Save", key=f"save-{subtheme_key}", type="primary"):
                    subtheme["subtheme_title"] = new_title
                    subtheme["analysis"] = new_analysis
                    st.session_state.editing = None
                    st.rerun()
            with col2:
                if st.button("Cancel", key=f"cancel-{subtheme_key}"):
                    st.session_state.editing = None
                    st.rerun()
        else:
            # View mode - header with edit/delete buttons
            col1, col2, col3 = st.columns([20, 1, 1])
            with col1:
                st.markdown(f"**{subtheme.get('subtheme_title', 'Untitled')}**")
            with col2:
                if st.button("✏️", key=f"edit-{subtheme_key}", help="Edit subtheme"):
                    st.session_state.editing = subtheme_key
                    st.rerun()
            with col3:
                if st.button("🗑️", key=f"delete-{subtheme_key}", help="Delete subtheme"):
                    delete_subtheme(theme_idx, subtheme_idx)
                    st.rerun()

            st.markdown(f"<div class='subtheme-analysis'>{subtheme.get('analysis', '')}</div>", unsafe_allow_html=True)

        # Render quotes
        quotes = subtheme.get("supporting_quotes", [])
        for q_idx, quote in enumerate(quotes):
            render_quote(quote, theme_idx, subtheme_idx, q_idx)

        # Add quote button
        if st.button("+ Add Quote", key=f"add-quote-{theme_idx}-{subtheme_idx}"):
            add_quote(theme_idx, subtheme_idx)
            st.rerun()


def render_theme_card(theme: dict, theme_idx: int):
    """Render a theme card with its subthemes."""
    theme_key = f"theme-{theme_idx}"
    is_editing = st.session_state.get("editing") == theme_key

    with st.container(border=True):
        if is_editing:
            # Edit mode
            new_title = st.text_input(
                "Theme title",
                value=theme.get("theme_title", ""),
                key=f"edit-title-{theme_key}"
            )
            new_explanation = st.text_area(
                "Explanation",
                value=theme.get("detailed_explanation", ""),
                key=f"edit-explanation-{theme_key}",
                height=150
            )
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Save", key=f"save-{theme_key}", type="primary"):
                    theme["theme_title"] = new_title
                    theme["detailed_explanation"] = new_explanation
                    st.session_state.editing = None
                    st.rerun()
            with col2:
                if st.button("Cancel", key=f"cancel-{theme_key}"):
                    st.session_state.editing = None
                    st.rerun()
        else:
            # View mode - header with edit/delete buttons
            col1, col2, col3 = st.columns([20, 1, 1])
            with col1:
                st.subheader(f"Theme {theme.get('id', theme_idx + 1)}: {theme.get('theme_title', 'Untitled')}", anchor=f"theme-{theme_idx}")
            with col2:
                if st.button("✏️", key=f"edit-{theme_key}", help="Edit theme"):
                    st.session_state.editing = theme_key
                    st.rerun()
            with col3:
                if st.button("🗑️", key=f"delete-{theme_key}", help="Delete theme"):
                    delete_theme(theme_idx)
                    st.rerun()

            st.markdown(f"<div class='theme-explanation'>{theme.get('detailed_explanation', '')}</div>", unsafe_allow_html=True)

        # Render subthemes
        subthemes = theme.get("subthemes", [])
        for s_idx, subtheme in enumerate(subthemes):
            with st.container(border=True):
                render_subtheme(subtheme, theme_idx, s_idx)

        # Add subtheme button
        if st.button("+ Add Subtheme", key=f"add-subtheme-{theme_idx}"):
            add_subtheme(theme_idx)
            st.rerun()


def render_results():
    """Main function to render all results with CRUD functionality."""
    if "results" not in st.session_state or not st.session_state.results:
        return

    results = st.session_state.results

    # Document summary
    st.subheader("Document Summary", anchor=False)
    st.markdown(results.get("document_summary", "No summary available."))
    st.divider()

    # Render themes
    themes = results.get("themes", [])
    for t_idx, theme in enumerate(themes):
        render_theme_card(theme, t_idx)

    # Add theme button
    if st.button("+ Add New Theme", key="add-theme"):
        add_theme()
        st.rerun()
