# ScriptNet — Handwritten Word Recognition with Deep CRNN & CTC Loss

> IAM Handwriting Dataset · 26.9M parameter CRNN · 5-layer Stacked BiLSTM · CTC decoding · Trained on Kaggle GPU

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow)
![Dataset](https://img.shields.io/badge/Dataset-IAM%20Handwriting-green?style=flat-square)
![Params](https://img.shields.io/badge/Parameters-26.9M-purple?style=flat-square)
![Loss](https://img.shields.io/badge/Loss-CTC-orange?style=flat-square)

---

## Overview

Handwritten text recognition (HTR) is one of the hardest sequence recognition problems in computer vision — unlike printed OCR, handwriting varies enormously across writers in stroke width, slant, spacing, and style.

ScriptNet implements a full **CRNN (Convolutional Recurrent Neural Network)** pipeline on the **IAM Handwriting Word Database** — the standard benchmark for offline HTR research. The architecture combines a deep CNN for visual feature extraction with a 5-layer stacked Bidirectional LSTM for sequence modeling, trained end-to-end with **CTC (Connectionist Temporal Classification) loss** — no segmentation required.

---

## Architecture

```
Input Image (32 × 128 × 1 grayscale)
        │
        ▼
┌───────────────────────────────────┐
│         CNN Feature Extractor     │
│                                   │
│  Conv(32) → Pool                  │
│  Conv(64) → Pool                  │
│  Conv(128) × 2                    │
│  Conv(512) × 4 + Dropout(0.2)     │
│  Conv(256) × 2 + BatchNorm × 2    │
│  Pool → Conv(64)                  │
│                                   │
│  Output: feature maps (1×31×64)   │
└───────────────┬───────────────────┘
                │ squeeze → (31 × 64) time steps
                ▼
┌───────────────────────────────────┐
│     5-Layer Stacked BiLSTM        │
│                                   │
│  BiLSTM(128)  →  (31 × 256)       │
│  BiLSTM(512)  →  (31 × 1024)      │
│  BiLSTM(512)  →  (31 × 1024)      │
│  BiLSTM(512)  →  (31 × 1024)      │
│  BiLSTM(128)  →  (31 × 256)       │
└───────────────┬───────────────────┘
                │
                ▼
        Dense(128) → Dense(77)
                │
                ▼
         CTC Loss / Decoder
                │
                ▼
       Predicted word text
```

**Total parameters: 26,924,045 (102.71 MB)**

---

## Architecture?

**CNN → feature columns:** The spatial pooling reduces the image height to 1 while preserving width as time steps — each column of the final feature map becomes one time step fed into the LSTM.

**5-layer stacked BiLSTM:** Bidirectional LSTMs read the sequence both left-to-right and right-to-left — critical for handwriting where character recognition depends on neighboring strokes in both directions. Stacking 5 layers allows the model to learn increasingly abstract sequential representations.

**CTC loss:** Unlike standard cross-entropy, CTC handles variable-length output sequences without requiring pre-segmented character boundaries. The decoder uses greedy best-path decoding with repeat merging — standard for CTC outputs.

**SELU activation:** Used throughout the CNN instead of ReLU for self-normalizing properties — helps with gradient flow in deep networks without explicit normalization at every layer.

---

## Dataset

**IAM Handwriting Word Database** — the standard benchmark for offline HTR

| Property | Details |
|---|---|
| Source | IAM Handwriting Database (University of Bern) |
| Level | Word-level segmentation |
| Vocabulary | 77 characters (upper/lower/digits/punctuation) |
| Max label length | 19 characters |
| Split | 90% train / 10% validation |
| Image size | Resized to 32 × 128 (grayscale) |
| Corrupt images | 1 identified and removed (`a01-117-05-02.png`) |

---

## Training

| Config | Value |
|---|---|
| Epochs | 55 |
| Batch size | 20 |
| Optimizer | Adam (lr=0.001, β₁=0.9, β₂=0.999, clipnorm=1.0) |
| Checkpointing | Best model saved on val_loss |
| Hardware | Kaggle GPU (XLA-compiled) |
| Steps per epoch | 1,724 |
| Time per epoch | ~147 seconds |

Training loss converged from **13.27 → 1.31** over 55 epochs. Best validation loss: **6.46** (epoch 15), after which the model overfits — indicating the deep 5-layer BiLSTM architecture benefits from additional regularization or data augmentation.

---

**Two architectures explored:**
- `train1()` — lighter model (2 BiLSTM layers, 128 units)
- `train()` — full model (5 BiLSTM layers, up to 512 units) — final version

---

## Quick Start

```bash
git clone https://github.com/Drashti0913/Handwritten-Text-Prediction-using-Deep-Learning.git
cd Handwritten-Text-Prediction-using-Deep-Learning

pip install tensorflow opencv-python pillow tqdm pandas matplotlib

# Run on Kaggle (recommended — GPU required for CuDNNLSTM)
# Dataset: https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database
```

Open `handwritten-text-prediction.ipynb` and run all cells. Prediction visualization runs automatically every epoch via the `PlotPredictions` callback.

---

## Observations & What I'd Improve

The model trains well (loss: 13.27 → 1.31) but overfits after epoch 15 — val_loss diverges while train_loss keeps improving. With more time:

- **Data augmentation** — random rotation, elastic distortion, noise on handwriting images
- **Learning rate scheduling** — ReduceLROnPlateau to escape the plateau
- **CTC beam search decoder** — instead of greedy, for better accuracy on ambiguous characters
- **Word error rate (WER) metric** — more meaningful than raw accuracy for sequence tasks

---

## Tech Stack

| Component | Technology |
|---|---|
| Framework | TensorFlow 2.x / Keras |
| LSTM | `CuDNNLSTM` (GPU-optimized) |
| Data pipeline | `tf.data` with parallel mapping + prefetch |
| Image preprocessing | OpenCV, PIL, TensorFlow image ops |
| Training platform | Kaggle (GPU) |

---

## License

MIT
