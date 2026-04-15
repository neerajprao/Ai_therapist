import ollama
import whisper
import os
import re
import json
import torch
import numpy as np
from train_emotion import EmotionProcessor
from train_bridge import TherapistBridge

class TherapistBrain:
    def __init__(self):
        print("Loading Whisper STT & Emotion2Vec Neural Pipeline...")
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.stt_model = whisper.load_model("base")
        
        self.emotion_processor = EmotionProcessor()
        self.bridge = TherapistBridge(input_dim=1024, num_classes=5).to(self.device)
        
        checkpoint_path = "models/checkpoints/bridge_v1.pth"
        if os.path.exists(checkpoint_path):
            self.bridge.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.bridge.eval()

        self.model_name = "llama3" 
        
        # CHANGED: Added logic for handling [User Tone] tags in the instructions
        self.system_prompt = (
            "You are Samantha, a professional and grounded therapist. Your tone is neutral, calm, and steady. "
            "You will receive user messages formatted as '[User Tone: <Emotion>] <Message>'. "
            "Use the 'User Tone' to inform your level of empathy and validation, but NEVER explicitly "
            "mention the tag itself (e.g., do not say 'I see you are feeling anxious'). "
            "CRITICAL: Keep responses to 2-3 concise sentences. Use '...' for a soft pause. Speak only words, no actions. "
            "Focus on: 1. Clarifying the user's narrative. 2. Identifying emotional weight. 3. Exploring immediate needs."
        )

        self.keyword_map = {
            "Sad": ["sad", "unhappy", "depressed", "lonely", "hopeless", "broken", "empty", "tired"],
            "Anxious": ["anxious", "scared", "worry", "panic", "nervous", "jittery", "overwhelmed", "stressed"],
            "Angry": ["angry", "mad", "furious", "annoyed", "irritated", "frustrated", "resentful"],
            "Happy": ["happy", "joyful", "content", "delighted", "peaceful", "thrilled", "excited"]
        }

    def transcribe_audio(self, audio_path):
        if not os.path.exists(audio_path): return ""
        result = self.stt_model.transcribe(audio_path, fp16=False)
        return result['text'].strip()

    def detect_emotion_hybrid(self, text, audio_path):
        clean_text = text.lower()
        for emotion, keywords in self.keyword_map.items():
            if any(word in clean_text for word in keywords):
                return emotion

        try:
            _, embeddings = self.emotion_processor.get_results(audio_path)
            feat_tensor = torch.from_numpy(embeddings).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.bridge(feat_tensor)
                prediction = torch.argmax(output, dim=1).item()
            
            mapping = {0: "Anxious", 1: "Sad", 2: "Angry", 3: "Neutral", 4: "Happy"}
            return mapping.get(prediction, "Neutral")
        except:
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

    # CHANGED: Now accepts detected_emotion and prepends it to the user message
    def generate_streaming_response(self, user_input, session_id, detected_emotion=None):
        if not user_input:
            yield "I'm listening. Please continue when you're ready."
            return

        history = self._get_history(session_id)
        
        # Inject the emotion tag directly into the text seen by Ollama
        if detected_emotion:
            formatted_input = f"[User Tone: {detected_emotion}] {user_input}"
        else:
            formatted_input = user_input

        history.append({"role": "user", "content": formatted_input})
        
        stream = ollama.chat(model=self.model_name, messages=history, stream=True)

        full_reply = ""
        for chunk in stream:
            content = chunk['message']['content']
            clean_content = re.sub(r'[\*\[].*?[\*\]]', '', content) 
            full_reply += clean_content
            yield clean_content

        history.append({"role": "assistant", "content": full_reply})
        self._save_history(session_id, history)