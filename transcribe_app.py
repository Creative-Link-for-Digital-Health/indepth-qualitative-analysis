
import streamlit as st
import tempfile
import os
import toml
from pathlib import Path
import transcription_utils

def main():
    st.set_page_config(
        page_title="WhisperX Transcription",
        page_icon="🎙️",
        layout="wide"
    )

    st.title("Audio Transcription with WhisperX", anchor=False)
    st.write("Upload an audio file (MP3/WAV) to transcribe and identify speakers.")
    
    # Audio File Uploader
    audio_file = st.file_uploader("Upload Audio", type=["mp3", "wav", "m4a"])
    
    # WhisperX settings
    model_size = "large-v3"
    
    # Load secrets
    try:
        secrets = toml.load(".secrets.toml")
        hf_token = secrets.get("HUGGINGFACE_API_KEY")
    except Exception as e:
        hf_token = None
        st.warning(f"Could not load .secrets.toml: {e}")
    
    if audio_file is not None:
        # Check if we should process
        if st.button("Start Transcription"):
            if not hf_token:
                st.error("Please provide a Hugging Face Token in .secrets.toml (HUGGINGFACE_API_KEY).")
            else:
                with st.spinner("Processing audio... This involves loading models, transcribing, aligning, and diarizing. Please wait."):
                    try:
                        # Save temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.name).suffix) as tmp_file:
                            tmp_file.write(audio_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        # Run Transcription pipeline
                        # Note: We pass the HF token to the internal function
                        result = transcription_utils.transcribe_and_diarize(
                            tmp_file_path, 
                            hf_token=hf_token, 
                            model_size=model_size
                        )
                        
                        # Cleanup temp file
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
