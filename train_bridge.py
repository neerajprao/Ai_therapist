import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# 1. The "Bridge" Architecture (Remains the same)
class TherapistBridge(nn.Module):
    def __init__(self, input_dim=1024, num_classes=5):
        super(TherapistBridge, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        return self.network(x)

# 2. The Training Loop
def train_bridge():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Load the synthetic data you just generated
    if not os.path.exists("data/X_train.npy"):
        print("Error: No training data found. Run generate_synthetic_data.py first.")
        return

    X = torch.from_numpy(np.load("data/X_train.npy")).float().to(device)
    y = torch.from_numpy(np.load("data/y_train.npy")).long().to(device)

    model = TherapistBridge().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

    print(f"Training on {device}...")
    
    # 50 Epochs is plenty for 100 samples
    for epoch in range(50):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/50 | Loss: {loss.item():.4f}")

    # Save the SMART weights
    os.makedirs("models/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "models/checkpoints/bridge_v1.pth")
    print("\n[Success] Training complete. Smart weights saved to models/checkpoints/bridge_v1.pth")

if __name__ == "__main__":
    train_bridge()