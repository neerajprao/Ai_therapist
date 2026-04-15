import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

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

def train_bridge():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    if not os.path.exists("data/X_train.npy"):
        print("Error: No training data found. Run generate_synthetic_data.py first.")
        return

    # Load data
    X_np = np.load("data/X_train.npy")
    y_np = np.load("data/y_train.npy")
    X = torch.from_numpy(X_np).float().to(device)
    y = torch.from_numpy(y_np).long().to(device)

    model = TherapistBridge().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

    losses = []
    accuracies = []

    print(f"Training on {device}...")
    
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        # Calculate Epoch Accuracy
        predictions = torch.argmax(output, dim=1)
        correct = (predictions == y).sum().item()
        accuracy = correct / y.size(0)
        
        losses.append(loss.item())
        accuracies.append(accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/50 | Loss: {loss.item():.4f} | Acc: {accuracy:.2%}")

    # Save the weights
    os.makedirs("models/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "models/checkpoints/bridge_v1.pth")
    print("\n[Success] Training complete. Weights saved.")

    # --- FINAL EVALUATION & METRICS ---
    model.eval()
    with torch.no_grad():
        final_output = model(X)
        final_preds = torch.argmax(final_output, dim=1).cpu().numpy()
        y_true = y.cpu().numpy()

    # Classification Report
    target_names = ["Anxious", "Sad", "Angry", "Neutral", "Happy"]
    print("\n" + "="*30)
    print("FINAL ACCURACY METRICS")
    print("="*30)
    print(classification_report(y_true, final_preds, target_names=target_names, labels=range(5), zero_division=0))

    # Plotting Training Curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses, color='red', label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracies, color='blue', label='Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_bridge()