import torch
import numpy as np
import os
from train_emotion import EmotionProcessor

def create_multi_class_dataset():
    processor = EmotionProcessor()
    
    # Define our anchors
    # 1: Sad/Depressed, 4: Happy/Stable (matching our EMOTION_MAP)
    targets = {
        "data/raw_audio/test.wav": 4, 
        "data/raw_audio/sad.wav": 1   
    }
    
    X, y = [], []

    for file_path, label in targets.items():
        if not os.path.exists(file_path):
            print(f"Skipping {file_path} - file not found.")
            continue

        print(f"Extracting embedding for label {label} from {file_path}...")
        _, emb = processor.get_results(file_path)
        
        # Generate 100 variations for each class to create 'clusters'
        for _ in range(100):
            noise = np.random.normal(0, 0.02, emb.shape) # Increased noise slightly
            X.append(emb + noise)
            y.append(label)
            
    X = np.array(X)
    y = np.array(y)
    
    np.save("data/X_train.npy", X)
    np.save("data/y_train.npy", y)
    print(f"\n[Success] Dataset created with {len(X)} samples.")

if __name__ == "__main__":
    create_multi_class_dataset()