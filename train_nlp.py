import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# --------------------
# CONFIG
# --------------------
max_words = 5000
max_len = 15
batch_size = 32
epochs = 20


# --------------------
# LOAD DATA
# --------------------
df = pd.read_csv("commands.csv")

texts = df["text"].values
actions = df["action"].values
devices = df["device"].values


# --------------------
# LABEL ENCODING
# --------------------
action_encoder = LabelEncoder()
device_encoder = LabelEncoder()

y_action = action_encoder.fit_transform(actions)
y_device = device_encoder.fit_transform(devices)


# --------------------
# TOKENIZATION
# --------------------
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=max_len, padding="post")


# --------------------
# TRAIN / TEST SPLIT
# --------------------
X_train, X_test, y_action_train, y_action_test, y_device_train, y_device_test = train_test_split(
    X, y_action, y_device, test_size=0.2, random_state=42
)


# --------------------
# DATASET
# --------------------
class IntentDataset(Dataset):
    def __init__(self, X, y_action, y_device):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y_action = torch.tensor(y_action, dtype=torch.long)
        self.y_device = torch.tensor(y_device, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_action[idx], self.y_device[idx]


train_dataset = IntentDataset(X_train, y_action_train, y_device_train)
test_dataset = IntentDataset(X_test, y_action_test, y_device_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# --------------------
# MODEL
# --------------------
class IntentCNN(nn.Module):
    def __init__(self, vocab_size, num_actions, num_devices, embed_dim=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=5)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)

        self.action_head = nn.Linear(32, num_actions)
        self.device_head = nn.Linear(32, num_devices)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = torch.max(x, dim=2)[0]

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        action = self.action_head(x)
        device = self.device_head(x)

        return action, device


vocab_size = max_words
num_actions = len(action_encoder.classes_)
num_devices = len(device_encoder.classes_)

model = IntentCNN(vocab_size, num_actions, num_devices)


# --------------------
# LOSS + OPTIMIZER
# --------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# --------------------
# TRAINING LOOP
# --------------------
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for X_batch, y_a, y_d in train_loader:
        optimizer.zero_grad()

        out_a, out_d = model(X_batch)

        loss_a = criterion(out_a, y_a)
        loss_d = criterion(out_d, y_d)

        loss = loss_a + loss_d
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")


# --------------------
# EVALUATION
# --------------------
model.eval()

y_action_pred = []
y_device_pred = []

with torch.no_grad():
    for X_batch, _, _ in test_loader:
        out_a, out_d = model(X_batch)

        y_action_pred.extend(torch.argmax(out_a, dim=1).numpy())
        y_device_pred.extend(torch.argmax(out_d, dim=1).numpy())


# --------------------
# REPORTS
# --------------------
print("\nAction Report:")
print(classification_report(y_action_test, y_action_pred, target_names=action_encoder.classes_))

print("\nDevice Report:")
print(classification_report(y_device_test, y_device_pred, target_names=device_encoder.classes_))


# --------------------
# CONFUSION MATRIX
# --------------------
def plot_cm(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=labels,
                yticklabels=labels)

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


plot_cm(y_action_test, y_action_pred, action_encoder.classes_, "Action Confusion Matrix")
plot_cm(y_device_test, y_device_pred, device_encoder.classes_, "Device Confusion Matrix")


# --------------------
# SAVE MODEL + ARTIFACTS
# --------------------
torch.save(model.state_dict(), "model.pt")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("encoders.pkl", "wb") as f:
    pickle.dump((action_encoder, device_encoder), f)

print("\nTraining complete.")