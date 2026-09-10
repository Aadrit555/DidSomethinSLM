"""
RL-based Small Language Model (SLM) for decision making.

- Reads text input
- Uses a small neural network with attention
- Trained using reinforcement learning (REINFORCE)
- No text generation
- Robust to unseen characters via <UNK>
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

# Set random seed for reproducible training
torch.manual_seed(42)

# -----------------------------
# 1. Training data
# -----------------------------
texts = [
    "hello",
    "hi",
    "where is my order",
    "track my order",
    "cancel my order",
    "please cancel it",
    "refund please",
    "i want a refund",
    "bye",
    "goodbye"
]

# Intent labels
# 0: GREETING, 1: ORDER_STATUS, 2: CANCEL_ORDER, 3: REFUND, 4: GOODBYE
labels = torch.tensor([
    0, 0,
    1, 1,
    2, 2,
    3, 3,
    4, 4
])

actions = [
    "GREETING",
    "ORDER_STATUS",
    "CANCEL_ORDER",
    "REFUND",
    "GOODBYE"
]

# -----------------------------
# 2. Tokenizer (CHAR-level)
# -----------------------------
PAD = "<PAD>"
UNK = "<UNK>"

base_chars = sorted(set("".join(texts)))
vocab = [PAD, UNK] + base_chars

stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for c, i in stoi.items()}

pad_id = stoi[PAD]
unk_id = stoi[UNK]
vocab_size = len(vocab)

max_len = max(len(t) for t in texts)

def encode(text):
    text = text.lower().strip()
    ids = [stoi.get(c, unk_id) for c in text]
    if len(ids) < max_len:
        ids += [pad_id] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

X = torch.stack([encode(t) for t in texts])

# -----------------------------
# 3. Neural Network Policy
# -----------------------------
class AttentionPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 32, padding_idx=pad_id)
        self.attn_score = nn.Linear(32, 1)
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, len(actions))

    def forward(self, input_ids):
        mask = (input_ids != pad_id)

        emb = self.embedding(input_ids)              # (B, T, D)
        scores = self.attn_score(emb).squeeze(-1)    # (B, T)
        scores = scores.masked_fill(~mask, -1e9)

        weights = torch.softmax(scores, dim=1)
        attended = (emb * weights.unsqueeze(-1)).sum(dim=1)

        x = torch.relu(self.fc1(attended))
        logits = self.fc2(x)
        return logits, weights

model = AttentionPolicy()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -----------------------------
# 4. RL Training (REINFORCE)
# -----------------------------
def train_policy(epochs=300):
    model.train()
    for epoch in range(epochs):
        logits, _ = model(X)
        probs = torch.softmax(logits, dim=1)
        dist = Categorical(probs)

        sampled_actions = dist.sample()
        rewards = torch.where(sampled_actions == labels, 1.0, -1.0)

        loss = -(dist.log_prob(sampled_actions) * rewards).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Avg Reward: {rewards.mean().item():.2f}")

    print(f"Epoch {epochs:03d} | Final Reward: {rewards.mean().item():.2f}")
    print("\nTraining complete.\n")

# -----------------------------
# 5. Inference
# -----------------------------
def predict(text, return_attention=False):
    cleaned = text.lower().strip()
    if not cleaned:
        if return_attention:
            return "EMPTY_INPUT", 0.0, []
        return "EMPTY_INPUT", 0.0

    model.eval()
    with torch.no_grad():
        ids = encode(cleaned).unsqueeze(0)
        logits, weights = model(ids)
        probs = torch.softmax(logits, dim=1)
        action_id = probs.argmax().item()
        confidence = probs.max().item()

    if return_attention:
        attn_list = weights[0][:len(cleaned)].tolist()
        return actions[action_id], confidence, attn_list
    return actions[action_id], confidence

# -----------------------------
# 6. Interactive Demo
# -----------------------------
def interactive_demo():
    print("Enter text (Ctrl+C to exit)\n")
    try:
        while True:
            user_input = input(">> ").strip()
            if not user_input:
                continue
            intent, conf, attn = predict(user_input, return_attention=True)
            
            seen = set()
            char_focus = []
            for c, _ in sorted(zip(user_input, attn), key=lambda x: x[1], reverse=True):
                if c.isalnum() and c.lower() not in seen:
                    seen.add(c.lower())
                    char_focus.append(repr(c))
                    if len(char_focus) == 3:
                        break
            focus_str = f" | Top focus: {', '.join(char_focus)}" if char_focus else ""
            
            print(f"Intent: {intent} | Confidence: {conf:.2f}{focus_str}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")

if __name__ == "__main__":
    train_policy()
    interactive_demo()
