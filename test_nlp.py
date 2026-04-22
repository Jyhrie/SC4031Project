import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ----------------------------
# MODEL DEFINITION (must match training)
# ----------------------------
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
        x = self.embedding(x)          # (B, T, E)
        x = x.permute(0, 2, 1)         # (B, E, T)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = torch.max(x, dim=2)[0]     # global max pooling

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        action = self.action_head(x)
        device = self.device_head(x)

        return action, device


# ----------------------------
# CONFIG
# ----------------------------
MAX_LEN = 15
MAX_WORDS = 5000


# ----------------------------
# DEVICE (GPU if available)
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# ----------------------------
# LOAD TOKENIZER + ENCODERS
# ----------------------------
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    action_encoder, device_encoder = pickle.load(f)


# ----------------------------
# LOAD MODEL
# ----------------------------
num_actions = len(action_encoder.classes_)
num_devices = len(device_encoder.classes_)

model = IntentCNN(MAX_WORDS, num_actions, num_devices).to(DEVICE)
model.load_state_dict(torch.load("model.pt", map_location=DEVICE))
model.eval()


# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def predict(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    x = torch.tensor(padded, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        action_logits, device_logits = model(x)

        action_probs = torch.softmax(action_logits, dim=1)[0].cpu().numpy()
        device_probs = torch.softmax(device_logits, dim=1)[0].cpu().numpy()

    action_idx = np.argmax(action_probs)
    device_idx = np.argmax(device_probs)

    action_conf = action_probs[action_idx]
    device_conf = device_probs[device_idx]

    action = action_encoder.inverse_transform([action_idx])[0]
    device = device_encoder.inverse_transform([device_idx])[0]

    return action, device, action_conf, device_conf


# ----------------------------
# INTERACTIVE LOOP
# ----------------------------
while True:
    text = input("Enter command: ")

    action, device, a_conf, d_conf = predict(text)

    print(f"Predicted: {action}_{device}")
    print(f"Confidence -> Action: {a_conf:.2f}, Device: {d_conf:.2f}")