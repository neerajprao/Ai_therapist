import torch
import numpy as np
import os
from train_emotion import EmotionProcessor
from train_bridge import TherapistBridge

# 1. Categories for the Therapist
# We will map the Bridge outputs to these 5 states
EMOTION_MAP = {
    0: "Anxious/Stressed",
    1: "Sad/Depressed",
    2: "Angry/Frustrated",
    3: "Neutral/Calm",
    4: "Happy/Stable"
}

class TherapistBot:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Load the Phase 1 Encoder
        self.encoder = EmotionProcessor()
        
        # Load the Phase 2 Bridge
        self.bridge = TherapistBridge(input_dim=1024, num_classes=5).to(self.device)
        self.bridge.load_state_dict(torch.load("models/checkpoints/bridge_v1.pth", map_location=self.device))
        self.bridge.eval() # Set to evaluation mode

    def analyze_voice(self, audio_path):
        # Step A: Get Embeddings from Emotion2Vec
        _, embeddings = self.encoder.get_results(audio_path)
        
        # Step B: Pass through the Bridge
        # Embeddings come as (1024,), we need (1, 1024) for the network
        feat_tensor = torch.from_numpy(embeddings).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.bridge(feat_tensor)
            prediction = torch.argmax(output, dim=1).item()
        
        return EMOTION_MAP[prediction]

if __name__ == "__main__":
    bot = TherapistBot()
    test_file = "data/raw_audio/sad.wav"
    
    if os.path.exists(test_file):
        interpretation = bot.analyze_voice(test_file)
        print(f"\n--- THERAPIST INTERPRETATION ---")
        print(f"The Bridge interprets your state as: {interpretation}")
    else:
        print("Please ensure data/raw_audio/test.wav exists.")