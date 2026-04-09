import ollama
import whisper
import os
import re

class TherapistBrain:
    def __init__(self):
        print("Loading Whisper Speech-to-Text...")
        # Note: Using 'base' model for speed on M3 Pro
        self.stt_model = whisper.load_model("base")
        
        self.model_name = "llama3" 
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
        if not os.path.exists(audio_path):
            return ""
        # fp16=False used for CPU/MPS compatibility
        result = self.stt_model.transcribe(audio_path, fp16=False)
        return result['text'].strip()

    def detect_emotion(self, text):
        text = text.lower()
        if any(word in text for word in ["sad", "empty", "grief", "hurts", "lost", "pulls"]):
            return "Sad"
        if any(word in text for word in ["scared", "anxious", "worry", "panic"]):
            return "Anxious"
        if any(word in text for word in ["angry", "mad", "hate", "annoyed"]):
            return "Angry"
        return "Neutral"

    def generate_streaming_response(self, audio_path, pre_transcribed_text=None):
        user_input = pre_transcribed_text if pre_transcribed_text else self.transcribe_audio(audio_path)
        
        if not user_input:
            yield "I'm listening. Please continue when you're ready."
            return

        self.chat_history.append({"role": "user", "content": user_input})
        
        stream = ollama.chat(
            model=self.model_name,
            messages=self.chat_history,
            stream=True,
        )

        full_reply = ""
        for chunk in stream:
            content = chunk['message']['content']
            
            # Remove any unwanted LLM artifacts
            clean_content = re.sub(r'\*.*?\*', '', content) 
            clean_content = re.sub(r'\[.*?\]', '', clean_content) 
            
            full_reply += clean_content
            yield clean_content

        self.chat_history.append({"role": "assistant", "content": full_reply})

if __name__ == "__main__":
    brain = TherapistBrain()
    print("Brain is online.")