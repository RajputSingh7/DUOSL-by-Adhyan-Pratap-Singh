import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("landmarks_63_raw_mirrored.csv")

X = df.iloc[:, :63].values
Y = df.iloc[:, 63].values


label_map = {}
idx = 0
for label in sorted(set(Y)):
    label_map[label] = idx
    idx += 1

y = [label_map[label] for label in Y]
num_classes = len(label_map)

print("Label Map:", label_map)

def normalize_landmarks(sample_63):
    landmarks = sample_63.reshape(21, 3)

    wrist = landmarks[0]
    middle_mcp = landmarks[9]

    scale = np.linalg.norm(middle_mcp - wrist)
    if scale < 1e-6:
        scale = 1.0

    landmarks = (landmarks - wrist) / scale
    return landmarks.flatten()

class SignDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        x = normalize_landmarks(x) 

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)

        return x, y
def plot_accuracy_curve(train_acc, val_acc=None):
    epochs = range(1, len(train_acc) + 1)

    plt.figure()
    plt.plot(epochs, train_acc, label="Training Accuracy")
    
    if val_acc is not None:
        plt.plot(val_acc, label="Validation Accuracy")
    
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_loss_curve(train_losses, val_losses=None):
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Training Losses")
    
    if val_losses is not None:
        plt.plot(val_losses, label="Validation Loss")
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()


dataset = SignDataset(X, y)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)
class SignLanguageMLP(nn.Module):
    def __init__(self, input_size=63, num_classes=28):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = SignLanguageMLP(input_size=63, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 30
Accuracy = []
Losses = []
for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        optimizer.zero_grad()

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    Losses.append(total_loss)
    acc = correct / total
    Accuracy.append(acc)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Acc: {acc:.4f}")
plot_accuracy_curve(Accuracy)
plot_loss_curve(Losses)

torch.save(model.state_dict(), "sign_model.pth")

model.load_state_dict(torch.load("sign_model.pth"))
model.eval()

