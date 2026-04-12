import ollama
import whisper
import os
import re
import json

class TherapistBrain:
    def __init__(self):
        print("Loading Whisper Speech-to-Text...")
        self.stt_model = whisper.load_model("base")
        
        self.model_name = "llama3" 
        # Updated system prompt with specific narrative and emotional exploration guidelines
        self.system_prompt = (
            "You are Luna, a professional and grounded therapist. Your tone is neutral, calm, and steady. "
            "Your goal is to listen deeply and validate the user's experience without mirroring their distress. "
            "CRITICAL: Keep responses to 2-3 concise sentences. Use '...' for a soft pause. Speak only words, no actions. "
            "When ending with an open-ended question or observation, avoid generic fillers. Instead, focus on: "
            "1. Clarifying the user's internal narrative (e.g., 'What does that situation say to you about...'). "
            "2. Identifying the physical or emotional weight of the moment. "
            "3. Exploring what the user needs in this space right now. "
            "Ensure your questions feel like a natural extension of their last sentence, rather than a standard clinical checklist."
        )

    def transcribe_audio(self, audio_path):
        if not os.path.exists(audio_path):
            return ""
        result = self.stt_model.transcribe(audio_path, fp16=False)
        return result['text'].strip()

    def detect_emotion(self, text):
        text = text.lower()
        if any(word in text for word in ["sad", "empty", "grief", "hurts", "lost"]): return "Sad"
        if any(word in text for word in ["scared", "anxious", "worry", "panic"]): return "Anxious"
        return "Neutral"

    def _get_history(self, session_id):
        history_file = f"data/history/{session_id}.json"
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                return json.load(f)["history"]
        return [{"role": "system", "content": self.system_prompt}]

    def _save_history(self, session_id, history):
        history_file = f"data/history/{session_id}.json"
        with open(history_file, 'w') as f:
            json.dump({"session_id": session_id, "history": history}, f, indent=4)

    def generate_streaming_response(self, audio_path, session_id, pre_transcribed_text=None):
        user_input = pre_transcribed_text if pre_transcribed_text else self.transcribe_audio(audio_path)
        
        if not user_input:
            yield "I'm listening. Please continue when you're ready."
            return

        # Load specific session history (Memory)
        history = self._get_history(session_id)
        history.append({"role": "user", "content": user_input})
        
        stream = ollama.chat(
            model=self.model_name,
            messages=history,
            stream=True,
        )

        full_reply = ""
        for chunk in stream:
            content = chunk['message']['content']
            # Cleaning logic to ensure TTS doesn't read stage directions
            clean_content = re.sub(r'\*.*?\*', '', content) 
            clean_content = re.sub(r'\[.*?\]', '', clean_content) 
            full_reply += clean_content
            yield clean_content

        # Append assistant response and save back to JSON
        history.append({"role": "assistant", "content": full_reply})
        self._save_history(session_id, history)

if __name__ == "__main__":
    brain = TherapistBrain()
    print("Brain is online with the updated Luna persona.")