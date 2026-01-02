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

