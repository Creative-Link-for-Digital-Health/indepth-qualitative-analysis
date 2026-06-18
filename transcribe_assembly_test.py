"""
Standalone AssemblyAI transcription test.
Run:  python transcribe_assembly_test.py <audio_file>
"""

import sys
import toml
import assemblyai as aai


def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe_assembly_test.py <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]

    secrets = toml.load(".secrets.toml")
    api_key = secrets.get("ASSEMBLYAI_API_KEY")
    if not api_key:
        print("ERROR: ASSEMBLYAI_API_KEY not found in .secrets.toml")
        sys.exit(1)

    aai.settings.api_key = api_key

    print(f"AssemblyAI SDK version: {aai.__version__}")
    print(f"Audio file: {audio_path}")
    print(f"Available SpeechModel values: best={aai.SpeechModel.best}, nano={aai.SpeechModel.nano}")
    print()

    config = aai.TranscriptionConfig(
        speaker_labels=True,
        speech_models=["universal-3-pro"],
    )

    print("Config: speaker_labels=True, speech_models=['universal-3-pro']")
    print("Starting transcription...")

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_path, config=config)

    if transcript.status == aai.TranscriptStatus.error:
        print(f"FAILED: {transcript.error}")
        sys.exit(1)

    print(f"Status: {transcript.status}")
    print(f"Utterances: {len(transcript.utterances) if transcript.utterances else 0}")
    print()

    if transcript.utterances:
        for utt in transcript.utterances:
            start_s = utt.start / 1000
            end_s = utt.end / 1000
            print(f"[{start_s:.1f}s - {end_s:.1f}s] Speaker {utt.speaker}: {utt.text}")
    else:
        print("No utterances returned. Raw text:")
        print(transcript.text)


if __name__ == "__main__":
    main()
