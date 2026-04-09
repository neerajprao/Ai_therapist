import os
import pandas as pd

def create_metadata():
    # This creates a simple CSV to track your audio files and their emotions
    data = {
        "file_path": [],
        "label": [] # e.g., 'sad', 'happy', 'neutral'
    }
    df = pd.DataFrame(data)
    df.to_csv("data/metadata.csv", index=False)
    print("Metadata file created in data/metadata.csv")

if __name__ == "__main__":
    create_metadata()