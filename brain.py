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
        
        # Initialize Neural Voice Classifier
        self.emotion_processor = EmotionProcessor()
        self.bridge = TherapistBridge(input_dim=1024, num_classes=5).to(self.device)
        
        # Load your trained weights
        checkpoint_path = "models/checkpoints/bridge_v1.pth"
        if os.path.exists(checkpoint_path):
            self.bridge.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.bridge.eval()

        self.model_name = "llama3" 
        self.system_prompt = (
            "You are Luna, a professional and grounded therapist. Your tone is neutral, calm, and steady. "
            "Your goal is to listen deeply and validate the user's experience without mirroring their distress. "
            "CRITICAL: Keep responses to 2-3 concise sentences. Use '...' for a soft pause. Speak only words, no actions. "
            "When ending with an open-ended question or observation, avoid generic fillers. Instead, focus on: "
            "1. Clarifying the user's internal narrative. 2. Identifying the emotional weight. "
            "3. Exploring what the user needs right now. Ensure questions feel like a natural extension of their last sentence."
        )

        # Expanded Hardcoded Word Lists
        self.keyword_map = {
            "Sad": [
                "sad", "unhappy", "sorrowful", "dejected", "depressed", "downcast", "miserable", 
                "gloomy", "melancholy", "low", "heavy-hearted", "mournful", "disheartened", "blue", 
                "woeful", "heartbroken", "bereaved", "devastated", "crushed", "shattered", "lost", 
                "empty", "hollow", "aching", "torn", "grieved", "lamenting", "lonely", "alone", 
                "isolated", "abandoned", "forsaken", "lonesome", "unloved", "estranged", "solitary", 
                "rejected", "excluded", "hopeless", "despairing", "bleak", "pessimistic", "defeated", 
                "resigned", "worthless", "useless", "pointless", "futile", "burdened", "trapped", 
                "tired", "exhausted", "drained", "weary", "sluggish", "apathetic", "numb", "listless", 
                "heavy", "fatigued", "burned-out", "spiritless", "regretful", "sorry", "ashamed", 
                "guilty", "remorseful", "rueful", "self-pitying", "apologetic", "disappointed", "let-down"
            ],
            "Anxious": [
                "anxious", "scared", "worry", "panic", "nervous", "apprehensive", "fearful", 
                "uneasy", "jittery", "tense", "restless", "on-edge", "fretful", "alarmed", 
                "agitated", "distressed", "panicky", "shaky", "terrified", "petrified", 
                "frightened", "dread", "angst", "concerned", "hesitant", "suspicious", 
                "overwhelmed", "pressured", "stressed", "unsettled", "jittery", "twitchy", 
                "hyper-vigilant", "paranoid", "insecure", "defensive", "uncertain", 
                "indecisive", "panicked", "breathless", "suffocating", "trapped", 
                "threatened", "vulnerable", "exposed", "daunted", "intimidated", 
                "cowardly", "spineless", "weak-kneed", "hysterical", "overwrought", 
                "worried", "troubled", "disquieted", "fraught", "uptight"
            ],
            "Angry": [
                "angry", "mad", "furious", "irate", "enraged", "seething", "outraged", 
                "livid", "incensed", "agitated", "annoyed", "irritated", "vexed", 
                "exasperated", "frustrated", "resentful", "bitter", "spiteful", 
                "hostile", "aggressive", "belligerent", "antagonistic", "combative", 
                "fuming", "boiling", "lashing out", "wrathful", "vengeful", "vindictive", 
                "indignant", "offended", "affronted", "displeased", "provoked", 
                "hateful", "vicious", "cross", "grumpy", "testy", "irritable", 
                "short-tempered", "grouchy", "cranky", "choleric", "sullen", 
                "scowling", "ferocious", "fierce", "savage", "ballistic", 
                "explosive", "volatile", "huffy", "miffed", "pissed", "infuriated"
            ],
            "Happy": [
                "happy", "joyful", "cheerful", "content", "delighted", "ecstatic", "elated", 
                "glad", "jubilant", "lively", "merry", "overjoyed", "peaceful", "pleasant", 
                "pleased", "thrilled", "upbeat", "blissful", "radiant", "exuberant", 
                "satisfied", "sunny", "buoyant", "jovial", "lighthearted", "gleeful", 
                "animated", "spirited", "energetic", "enthusiastic", "optimistic", 
                "positive", "carefree", "untroubled", "euphoric", "rapturous", 
                "triumphant", "exultant", "gratified", "heartwarming", "wonderful", 
                "fantastic", "terrific", "marvelous", "glowing", "beaming", 
                "tickled", "amused", "jolly", "festive", "chipper", "peppy"
            ]
        }

    def transcribe_audio(self, audio_path):
        if not os.path.exists(audio_path): return ""
        result = self.stt_model.transcribe(audio_path, fp16=False)
        return result['text'].strip()

    def detect_emotion_hybrid(self, text, audio_path):
        # Phase 1: Hardcoded Keyword Check
        clean_text = text.lower()
        for emotion, keywords in self.keyword_map.items():
            if any(word in clean_text for word in keywords):
                return emotion

        # Phase 2: Neural Fallback (Emotion2Vec + Bridge)
        try:
            _, embeddings = self.emotion_processor.get_results(audio_path)
            feat_tensor = torch.from_numpy(embeddings).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.bridge(feat_tensor)
                prediction = torch.argmax(output, dim=1).item()
            
            # Map Neural Classes to UI Categories
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

    def generate_streaming_response(self, user_input, session_id):
        if not user_input:
            yield "I'm listening. Please continue when you're ready."
            return

        history = self._get_history(session_id)
        history.append({"role": "user", "content": user_input})
        
        stream = ollama.chat(model=self.model_name, messages=history, stream=True)

        full_reply = ""
        for chunk in stream:
            content = chunk['message']['content']
            clean_content = re.sub(r'[\*\[].*?[\*\]]', '', content) 
            full_reply += clean_content
            yield clean_content

        history.append({"role": "assistant", "content": full_reply})
        self._save_history(session_id, history)