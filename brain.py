import ollama
import os
import json
from concurrent.futures import ThreadPoolExecutor
from transcribe import SpeechToText
from therapist_bot import TherapistBot

class TherapistBrain:
    def __init__(self):
        print("--- Initializing Neural Engines ---")
        self.stt = SpeechToText()
        self.emotion_bot = TherapistBot()
        self.model = "llama3"
        self.history_file = "data/history.json"
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        self.history = self.load_history()

    def load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"History load error: {e}")
                return []
        return []

    def save_history(self):
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=4)
        except Exception as e:
            print(f"History save error: {e}")

    def check_safety(self, text):
        """Standard CS Safety check for crisis keywords."""
        triggers = ["hurt myself", "suicide", "end my life", "self-harm", "kill myself"]
        cleaned_text = text.lower()
        return any(trigger in cleaned_text for trigger in triggers)

    def generate_response(self, audio_path):
        # --- PARALLEL INFERENCE BLOCK ---
        # We run STT and Emotion Detection simultaneously to reduce latency
        with ThreadPoolExecutor() as executor:
            future_text = executor.submit(self.stt.transcribe, audio_path)
            future_emotion = executor.submit(self.emotion_bot.analyze_voice, audio_path)
            
            transcript = future_text.result()
            emotion = future_emotion.result()

        # --- SAFETY OVERRIDE ---
        if self.check_safety(transcript):
            crisis_msg = (
                "I'm very concerned about what you're sharing. Please know that you're not alone. "
                "If you're feeling like you might hurt yourself, please reach out to a professional "
                "immediately. You can contact AASRA at +91-9820466726 or find local resources. "
                "I am here to listen, but your safety is the priority."
            )
            self.history.append({'role': 'user', 'content': f"[CRISIS TRIGGER] {transcript}"})
            self.history.append({'role': 'assistant', 'content': crisis_msg})
            self.save_history()
            return transcript, emotion, crisis_msg

        # --- STANDARD LLM LOGIC ---
        system_prompt = f"""
        You are a professional, empathetic therapist named SoulSync. 
        The user's current vocal tone is: {emotion}.
        
        GUIDELINES:
        - Acknowledge their tone ({emotion}) subtly if it adds value.
        - Reference their history to build a deep therapeutic connection.
        - Be concise, direct, and empathetic. Do not lecture.
        - Keep responses to 3-4 sentences maximum.
        """
        
        # Add context-aware message to history
        self.history.append({'role': 'user', 'content': f"[Tone: {emotion}] {transcript}"})
        
        # Build message payload (System + last 10 turns of history to prevent token bloat)
        messages = [{'role': 'system', 'content': system_prompt}] + self.history[-10:]
        
        print(f"[Brain] Processing {emotion} state via Llama-3...")
        
        try:
            response = ollama.chat(model=self.model, messages=messages)
            ai_reply = response['message']['content']
            
            # Record AI response and update persistence
            self.history.append({'role': 'assistant', 'content': ai_reply})
            self.save_history()
            
            return transcript, emotion, ai_reply
            
        except Exception as e:
            return transcript, emotion, f"Brain Error: {str(e)}"

if __name__ == "__main__":
    # Quick CLI test for developers
    brain = TherapistBrain()
    test_file = "data/raw_audio/test.wav"
    if os.path.exists(test_file):
        t, e, r = brain.generate_response(test_file)
        print(f"\nReport:\nText: {t}\nEmotion: {e}\nAI: {r}")