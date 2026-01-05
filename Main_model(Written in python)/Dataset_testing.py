import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# from Data_preprocessing import generate_csv
# generate_csv()
df = pd.read_csv("landmarks_63_raw_mirrored.csv")

X = df.iloc[:, :63].values
Y = df.iloc[:, 63].values

label_map = {label: idx for idx, label in enumerate(sorted(set(Y)))}
inv_label_map = {v: k for k, v in label_map.items()}
num_classes = len(label_map)

y = [label_map[label] for label in Y]
def plot_confusion_matrix(model, dataloader, device, class_names):
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, xticklabels=class_names,
                yticklabels=class_names,
                annot=False, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

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
        x = normalize_landmarks(self.X[idx])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.y[idx])

dataset = SignDataset(X, y)

test_loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=False
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SignLanguageMLP(63, num_classes).to(device)
model.load_state_dict(torch.load("sign_model.pth", map_location=device))
model.eval()

test_epochs = 10

class_names = [inv_label_map[i] for i in range(num_classes)]
for epoch in range(test_epochs):
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    acc = correct / total
    if epoch % 5 == 0:
        plot_confusion_matrix(model, test_loader, device, class_names)
    print(f"Test Epoch {epoch+1}/{test_epochs} | Accuracy: {acc:.4f}")
