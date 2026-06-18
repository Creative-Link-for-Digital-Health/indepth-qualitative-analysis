import assemblyai as aai


def transcribe_with_assemblyai(file_path: str, api_key: str) -> dict:
    """
    Transcribes audio with speaker diarization via AssemblyAI.

    Returns a dict with the same shape that
    transcription_utils.format_diarization_output expects:
        {"segments": [{"speaker": str, "text": str,
                       "start": float, "end": float}, ...]}
    """
    aai.settings.api_key = api_key

    config = aai.TranscriptionConfig(
        speaker_labels=True,
        speech_models=["universal-3-pro"],
    )
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file_path, config=config)

    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"AssemblyAI transcription failed: {transcript.error}")

    segments = [
        {
            "speaker": utt.speaker,
            "text": utt.text,
            "start": utt.start / 1000,
            "end": utt.end / 1000,
        }
        for utt in transcript.utterances
    ]

    return {"segments": segments}
