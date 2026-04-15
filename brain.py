import ollama
import whisper
import os
import re
import json
import torch
import numpy as np
import librosa
import torch.nn.functional as F
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from train_emotion import EmotionProcessor

class TherapistBrain:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Loading Ensembled Brain on {self.device}...")
        
        # 1. Whisper STT
        self.stt_model = whisper.load_model("base")

        # 2. Load Fine-tuned Wav2Vec2
        model_path = "./custom_model/final_emotion_model" 
        self.w2v_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.w2v_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.w2v_model.eval()

        # 3. Load Emotion2Vec
        self.e2v_processor = EmotionProcessor()
        
        # 4. Global Emotion Map
        self.id2label = {0: 'Happy', 1: 'Sad', 2: 'Angry', 3: 'Neutral', 4: 'Anxious'}

        self.model_name = "llama3" 
        self.system_prompt = (
            "You are Samantha, a professional and grounded therapist. Your tone is neutral, calm, and steady. "
            "You will receive user input formatted as '[User Tone: <Emotion>] <Message>'. "
            "Use the 'User Tone' context to deeply validate the user's experience and adjust your empathy levels, "
            "but never explicitly mention the tone tag itself in your response. "
            "CRITICAL: Keep responses to 2-3 concise sentences. Use '...' for a soft pause. Speak only words, no actions. "
            "Focus on: 1. Clarifying the user's narrative. 2. Identifying emotional weight. 3. Exploring immediate needs."
        )

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
        # 1. Static/Silence Check
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            rms = librosa.feature.rms(y=y)[0]
            avg_rms = np.mean(rms)
            if avg_rms < 0.008: 
                return "Neutral"
        except Exception as e:
            print(f"Energy check error: {e}")

        # 2. Keyword Check
        clean_text = text.lower()
        for emotion, keywords in self.keyword_map.items():
            if any(word in clean_text for word in keywords):
                return emotion

        # 3. Neural Ensemble
        try:
            # Wav2Vec2
            speech, _ = librosa.load(audio_path, sr=16000)
            inputs = self.w2v_extractor(speech, sampling_rate=16000, return_tensors="pt").to(self.device)
            with torch.no_grad():
                w2v_logits = self.w2v_model(**inputs).logits
                w2v_probs = F.softmax(w2v_logits, dim=-1).cpu().numpy()[0] 
                # here is the prob distribution for all 5 emotions

            # Emotion2Vec
            e2v_labels, _ = self.e2v_processor.get_results(audio_path)
            e2v_probs = np.zeros(5)
            alignment = {"happy": 0, "sad": 1, "angry": 2, "neutral": 3, "anxious": 4}
            for lbl, score in e2v_labels.items():
                if lbl.lower() in alignment:
                    e2v_probs[alignment[lbl.lower()]] = score

            # Weighted Average
            final_probs = (w2v_probs * 0.6) + (e2v_probs * 0.4)
            prediction = np.argmax(final_probs)
            
            return self.id2label[prediction]
        except Exception as e:
            print(f"Neural Hybrid Error: {e}")
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

    def generate_streaming_response(self, user_input, session_id, detected_emotion=None):
        # Static/Empty input prompt
        if not user_input or (len(user_input.strip()) < 2 and detected_emotion == "Neutral"):
            yield "Hello, I'm Samantha... I noticed it's a bit quiet on your end. How are you feeling this wonderful day?"
            return

        history = self._get_history(session_id)
        formatted_input = f"[User Tone: {detected_emotion}] {user_input}" if detected_emotion else user_input
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