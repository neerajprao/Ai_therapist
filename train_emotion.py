import torch
import librosa
import os
import numpy as np
from funasr import AutoModel

# 1. Setup Device for M3 Pro
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"--- System Status ---")
print(f"Device: {device.upper()}")

class EmotionProcessor:
    def __init__(self):
        # Path to your local model folder
        model_path = "models/emotion2vec"
        
        print("Loading Emotion2Vec into memory...")
        self.model = AutoModel(
            model=model_path,
            device=device,
            disable_update=True # Keeps it fast and offline
        )

    def get_results(self, audio_path):
        """
        Returns both the human-readable labels and the raw embeddings.
        """
        if not os.path.exists(audio_path):
            return None, None
        
        # We run inference once but extract two things:
        # 1. The classification (Labels)
        # 2. The internal features (Embeddings)
        res = self.model.generate(input=audio_path, extract_embedding=True)
        
        labels_dict = dict(zip(res[0]['labels'], res[0]['scores']))
        embeddings = np.array(res[0]['feats'])
        
        return labels_dict, embeddings

if __name__ == "__main__":
    processor = EmotionProcessor()
    
    # Path to your test file
    test_file = "data/raw_audio/test.wav"
    
    if os.path.exists(test_file):
        labels, feats = processor.get_results(test_file)
        
        print(f"\n--- INFERENCE REPORT ---")
        # Find the top emotion
        top_emotion = max(labels, key=labels.get)
        print(f"Primary Emotion: {top_emotion} ({labels[top_emotion]:.2%})")
        
        print(f"\n--- EMBEDDING DATA (For Phase 2) ---")
        print(f"Embedding Shape: {feats.shape}")
        print(f"Vector Preview: {feats[:5]}...") # Just show the first 5 numbers
        
        # Save embedding for later so we don't have to re-process
        np.save("data/test_embedding.npy", feats)
        print(f"\n[Success] Embedding saved to data/test_embedding.npy")
    else:
        print(f"\n[!] Error: {test_file} not found.")