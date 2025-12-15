"""
Simple Qualitative Theme Analysis App
A minimal Streamlit app for analyzing interview transcripts using local Ollama.
"""

from pathlib import Path

import streamlit as st
from openai import OpenAI

from constants import CARD_CSS
from file_utils import load_api_params, parse_uploaded_file, save_results
from api_utils import analyze_transcript, format_results_as_text, AnalysisError
from ui_components import render_results


def main():
    st.set_page_config(
        page_title="Qualitative Analysis",
        page_icon="📝",
        layout="wide"
    )

    # Apply custom CSS
    st.markdown(CARD_CSS, unsafe_allow_html=True)

    st.title("Qualitative Theme Analysis", anchor=False)
    st.write("Upload a transcript to discover themes and extract supporting quotes.")

    # Initialize session state
    if 'client' not in st.session_state:
        api_params = load_api_params()
        st.session_state.api_params = api_params
        st.session_state.client = OpenAI(
            api_key=api_params['API_KEY'],
            base_url=api_params['API_URL']
        )
    if 'editing' not in st.session_state:
        st.session_state.editing = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'base_filename' not in st.session_state:
        st.session_state.base_filename = "analysis"

    # Sidebar - File upload and theme navigation
    with st.sidebar:
        # Collapsible upload section - expanded by default, collapsed after analysis
        upload_expanded = st.session_state.results is None
        with st.expander("Upload Transcript", expanded=upload_expanded):
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=["txt", "docx"],
                help="Upload a .txt or .docx transcript file"
            )

            if uploaded_file:
                st.success(f"Uploaded: {uploaded_file.name}")

            analyze_button = st.button(
                "Analyze Transcript",
                disabled=uploaded_file is None,
                type="primary"
            )

        # Theme navigation - only show when results exist
        if st.session_state.results:
            st.markdown("---")
            st.markdown("**Theme Navigation**")
            themes = st.session_state.results.get("themes", [])
            for idx, theme in enumerate(themes):
                theme_title = theme.get('theme_title', 'Untitled')
                theme_id = theme.get('id', idx + 1)
                st.markdown(f"[Theme {theme_id}: {theme_title}](#theme-{idx})")

    # Main content area
    if analyze_button and uploaded_file:
        with st.spinner("Analyzing transcript... This may take a minute."):
            # Parse file
            transcript_text = parse_uploaded_file(uploaded_file)

            try:
                # Analyze with automatic retry on JSON errors
                results = analyze_transcript(
                    st.session_state.client,
                    st.session_state.api_params['MODEL'],
                    transcript_text
                )

                # Save to outputs folder
                base_filename = Path(uploaded_file.name).stem
                saved_path = save_results(results, base_filename)

                # Store in session state
                st.session_state.results = results
                st.session_state.saved_path = saved_path
                st.session_state.base_filename = base_filename
                st.session_state.editing = None

            except AnalysisError:
                st.error(
                    "Analysis failed after multiple attempts. "
                    "The AI returned an invalid response. Please try again."
                )
                st.stop()

        st.success(f"Analysis complete! Results saved to: {saved_path}")
        st.rerun()

    # Display results with CRUD functionality
    if st.session_state.results:
        render_results()

        st.divider()

        # Download button - generates text from current (possibly edited) results
        download_text = format_results_as_text(st.session_state.results)
        download_filename = f"{st.session_state.base_filename}_results.txt"
        st.download_button(
            label="Download Results as Text",
            data=download_text,
            file_name=download_filename,
            mime="text/plain"
        )
    else:
        st.info("Upload a transcript file and click 'Analyze Transcript' to begin.")


if __name__ == "__main__":
    main()
