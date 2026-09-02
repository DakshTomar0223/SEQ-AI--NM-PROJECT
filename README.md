# SEQ-AI--NM-PROJECT

A unified exploration of sequence modeling across three major frontiers of AI: **Time Series Forecasting**, **Natural Language Processing**, and **Computer Vision**. Each frontier uses a purpose-built sequential architecture, tied together by a shared philosophy — combining recurrent memory with attention-based context to model dependencies over time and space.

---

## 🚀 Overview

This project demonstrates how sequential modeling techniques — from simple recurrence to hybrid attention-recurrence architectures — can be adapted across fundamentally different data modalities:

| Frontier | Task | Architecture |
|---|---|---|
| 📈 Time Series Forecasting | Predicting future values (stock/sensor/weather data) | Simple LSTM |
| 📝 Natural Language Processing | Sequence understanding/generation | LSTM Encoder + Transformer Decoder (Hybrid) |
| 🖼️ Computer Vision | Image-based recognition/classification | CNN |

---

## 🧠 Architectures

### 1. Time Series Forecasting — Simple LSTM
A straightforward LSTM-based sequence model that captures temporal dependencies in historical data to forecast future values.

- **Input:** Historical time-windowed sequences
- **Core:** Stacked LSTM layers
- **Output:** Forecasted value(s) for future timestep(s)

### 2. NLP — LSTM Encoder + Transformer Decoder (Hybrid)
An LSTM encoder compresses the input sequence into contextual representations, and a Transformer decoder attends over that encoded context to generate output sequences.

- **Encoder:** LSTM — captures sequential/recurrent structure of the input
- **Decoder:** Transformer — leverages self-attention and cross-attention over encoder outputs for generation
- **Output:** Generated/predicted sequence (task-dependent — e.g. text generation, translation, summarization)

### 3. Computer Vision — CNN
A convolutional architecture for extracting spatial hierarchies of features from image data.

- **Core:** Convolutional + pooling layers for feature extraction
- **Output:** Classification / recognition output (task-dependent)

---

## 📚 References

| # | Title | Link |
|---|---|---|
| 1 | Attention Is All You Need (Vaswani et al., 2017) — the original Transformer paper | https://arxiv.org/pdf/1706.03762 |
| 2 | Fusing Memory and Attention: A Study on LSTM, Transformer and Hybrid Architectures for Symbolic Music Generation (Ghoshal et al., 2026) | https://arxiv.org/html/2603.21282v1 |
| 3 | ReAct: Synergizing Reasoning and Acting in Language Models (Google Research Blog) | https://research.google/blog/react-synergizing-reasoning-and-acting-in-language-models/ |

---

## 👥 Contributors

Daksh Tomar
Arpit Goyal
Divyansh Gangwar

---
