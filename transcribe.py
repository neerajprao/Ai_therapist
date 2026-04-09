import whisper
import torch
import os

class SpeechToText:
    def __init__(self):
        # We'll use the 'base' model—it's tiny, fast, and very accurate for English
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Loading Whisper on {self.device}...")
        self.model = whisper.load_model("base").to(self.device)

    def transcribe(self, audio_path):
        if not os.path.exists(audio_path):
            return "Error: File not found"
        
        # This will convert your .wav file into a text string
        result = self.model.transcribe(audio_path)
        return result['text'].strip()

if __name__ == "__main__":
    stt = SpeechToText()
    # Let's test it on your sad audio
    test_path = "data/raw_audio/sad.wav"
    transcript = stt.transcribe(test_path)
    print(f"\n--- TRANSCRIPTION TEST ---")
    print(f"File: {test_path}")
    print(f"Output: {transcript}")