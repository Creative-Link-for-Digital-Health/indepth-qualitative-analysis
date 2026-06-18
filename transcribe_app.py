
import streamlit as st
import tempfile
import os
import toml
from pathlib import Path
import transcription_utils
import assemblyai_utils

def main():
    st.set_page_config(
        page_title="WhisperX Transcription",
        page_icon="🎙️",
        layout="wide"
    )

    st.title("Audio Transcription", anchor=False)
    st.write("Upload an audio file (MP3/WAV) to transcribe and identify speakers.")

    backend = st.radio(
        "Transcription backend",
        ["Local (WhisperX)", "AssemblyAI"],
        key="transcription_backend",
    )
    st.caption(
        "Local = slower but private, runs on your GPU/CPU. "
        "AssemblyAI = faster, requires internet and an API key."
    )

    # Audio File Uploader
    audio_file = st.file_uploader("Upload Audio", type=["mp3", "wav", "m4a"])
    
    # WhisperX settings
    model_size = "large-v3"
    
    # Load secrets
    try:
        secrets = toml.load(".secrets.toml")
        hf_token = secrets.get("HUGGINGFACE_API_KEY")
        assemblyai_key = secrets.get("ASSEMBLYAI_API_KEY")
    except Exception as e:
        hf_token = None
        assemblyai_key = None
        st.warning(f"Could not load .secrets.toml: {e}")
    
    if audio_file is not None:
        if st.button("Start Transcription"):
            use_assemblyai = backend == "AssemblyAI"

            if use_assemblyai and not assemblyai_key:
                st.error("Please provide an AssemblyAI key in .secrets.toml (ASSEMBLYAI_API_KEY).")
            elif not use_assemblyai and not hf_token:
                st.error("Please provide a Hugging Face Token in .secrets.toml (HUGGINGFACE_API_KEY).")
            else:
                spinner_msg = (
                    "Uploading to AssemblyAI and transcribing..."
                    if use_assemblyai
                    else "Processing audio... This involves loading models, transcribing, aligning, and diarizing. Please wait."
                )
                with st.spinner(spinner_msg):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.name).suffix) as tmp_file:
                            tmp_file.write(audio_file.getvalue())
                            tmp_file_path = tmp_file.name

                        if use_assemblyai:
                            result = assemblyai_utils.transcribe_with_assemblyai(
                                tmp_file_path, assemblyai_key
                            )
                        else:
                            result = transcription_utils.transcribe_and_diarize(
                                tmp_file_path,
                                hf_token=hf_token,
                                model_size=model_size,
                            )

                        os.remove(tmp_file_path)

                        if result:
                            formatted_text = transcription_utils.format_diarization_output(result)
                            st.session_state.transcription_result = formatted_text
                            st.success("Transcription complete!")
                        else:
                            st.error("Transcription failed. Check logs.")

                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        st.error(f"An error occurred: {e}")

    # Display Editable Result
    if "transcription_result" in st.session_state:
        st.subheader("Transcription Result")
        edited_transcription = st.text_area(
            "Edit Transcription", 
            value=st.session_state.transcription_result, 
            height=600
        )
        
        # Download button
        st.download_button(
            label="Download Transcription",
            data=edited_transcription,
            file_name="transcription.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
