import ollama
import whisper
import os
import re

class TherapistBrain:
    def __init__(self):
        # 1. Load Whisper locally on your M3 Pro (Base is fastest)
        print("Loading Whisper Speech-to-Text...")
        self.stt_model = whisper.load_model("base")
        
        # 2. Set the "Luna" Personality
        self.model_name = "llama3" # Ensure you have run 'ollama pull llama3'
        self.system_prompt = (
            "You are Luna, a deeply empathetic, soft-spoken therapist. "
            "Your goal is to provide comfort. "
            "CRITICAL RULES: "
            "1. NEVER use asterisks or stage directions like *sighs*, *pauses*, or [cries]. "
            "2. Never describe your own actions. Speak only the words you would say aloud. "
            "3. Keep responses very brief (1-2 sentences). "
            "4. Use '...' for a soft pause. "
            "5. If the user is poetic, be poetic. If they are brief, be brief."
        )
        self.chat_history = [{"role": "system", "content": self.system_prompt}]

    def transcribe_audio(self, audio_path):
        """Converts user speech to text locally."""
        if not os.path.exists(audio_path):
            return ""
        result = self.stt_model.transcribe(audio_path, fp16=False)
        return result['text'].strip()

    def detect_emotion(self, text):
        """
        Simple keyword-based logic for speed. 
        In 2026, LLMs are fast enough to detect this in the stream.
        """
        text = text.lower()
        if any(word in text for word in ["sad", "empty", "grief", "hurts", "lost", "pulls"]):
            return "Sad/Depressed"
        if any(word in text for word in ["scared", "anxious", "worry", "panic"]):
            return "Anxious"
        return "Neutral/Calm"

    def generate_streaming_response(self, audio_path):
        """Process audio -> Text -> Llama-3 Reasoning -> Clean Stream"""
        
        # 1. Get user text
        user_input = self.transcribe_audio(audio_path)
        if not user_input:
            yield "I'm listening. Please continue when you're ready."
            return

        self.chat_history.append({"role": "user", "content": user_input})
        
        # 2. Get LLM response via Ollama
        stream = ollama.chat(
            model=self.model_name,
            messages=self.chat_history,
            stream=True,
        )

        full_reply = ""
        for chunk in stream:
            content = chunk['message']['content']
            
            # 3. THE CLEANER: Strip out asterisks and stage directions on the fly
            # This prevents Luna from ever seeing a "*" or "[pause]"
            clean_content = re.sub(r'\*.*?\*', '', content) # Removes *anything*
            clean_content = re.sub(r'\[.*?\]', '', clean_content) # Removes [anything]
            
            full_reply += clean_content
            yield clean_content

        self.chat_history.append({"role": "assistant", "content": full_reply})

# To test this file independently:
if __name__ == "__main__":
    brain = TherapistBrain()
    print("Brain is online.")