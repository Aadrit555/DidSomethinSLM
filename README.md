# RL-Based Small Language Model (SLM)

A small, reinforcement learning–based language model for **decision making**, not text generation.

This project was built as a learning exercise to understand how **language representations, attention mechanisms, and policy-gradient reinforcement learning** work together in a small, interpretable system.

---

## What This Project Does

- Reads free-form text input  
- Encodes text at the **character level**  
- Uses a **neural network with attention**  
- Selects an **intent (action)** from a fixed set  
- Learns using **reward feedback (REINFORCE)**  

**Example inputs**
- `"cancel my order"`
- `"shipment delayed"`
- `"i want my money back"`

**Example outputs**
- `CANCEL_ORDER`
- `ORDER_STATUS`
- `REFUND`

Each prediction includes a confidence score.

---

## What This Project Does NOT Do

- ❌ No text generation  
- ❌ No next-token prediction  
- ❌ No pretrained models  
- ❌ No external APIs  
- ❌ No keyword rules  

All behavior comes from **learned neural weights**.

---

## High-Level Architecture
Text Input
↓
Character Tokenizer (PAD, UNK)
↓
Embedding Layer
↓
Attention Mechanism
↓
Policy Network
↓
Intent Action
↓
Reward Feedback (+1 / −1)

---

## Tokenization

The model uses a **character-level tokenizer** to remain lightweight and dependency-free.

Special tokens:
- `<PAD>` for sequence padding  
- `<UNK>` for unseen characters  

This ensures the model **never fails on new or unexpected input**.

---

## Model Architecture

The neural network consists of:

1. **Embedding Layer**  
   Converts characters into trainable vectors.

2. **Attention Layer**  
   Assigns importance to different parts of the input.

3. **Policy Network**  
   Outputs action scores for each intent.

The model is intentionally small but fully trainable.

---

## Reinforcement Learning Setup

The task is framed as a **single-step RL problem**:

- **State:** input text  
- **Action:** predicted intent  
- **Reward:**  
  - +1 for correct action  
  - −1 for incorrect action  

Training uses **policy-gradient reinforcement learning (REINFORCE)**.

---

## Training Behavior

- Early training: near-random actions  
- Later training: stable, high average reward  

This shows the policy improving through feedback.

---

## Inference

During inference:
- The model selects the **highest-probability action**
- A confidence score is reported
- No action sampling is used

The model runs fully **on-device**.

---

## Why Small Language Models?

Small Language Models are often better suited for:
- Low-latency systems  
- Edge or privacy-sensitive applications  
- Interpretable ML pipelines  

This project explores those tradeoffs.

---

## AI Assistance Disclosure

AI tools were used selectively when stuck, mainly for:
- Structuring the policy-gradient loop  
- Handling unknown input safely  
- Sanity-checking the attention mechanism  

The design, implementation, and debugging were hands-on.

---

## How to Run

```bash
pip install torch
python rl_attention_slm.py


