# Self-Supervised Learning for ECG Time Series: AE, MAE, and VAE Approaches

This repository implements **self-supervised pretraining** of 12-lead ECG signals using three reconstruction-based methods — **Vanilla Autoencoder (AE)**, **Masked Autoencoder (MAE)**, and **Variational Autoencoder (VAE)** — followed by downstream **binary classification** using the learned encoder.

---

## 🔧 Pretraining Phase

We pretrain the encoder on a large **unlabeled ECG dataset** using the following self-supervised methods:

### 🟩 Autoencoder (AE)
A convolutional encoder-decoder architecture trained to reconstruct the input ECG using Mean Squared Error (MSE) loss. This captures a dense latent representation of the full input signal.

### 🟦 Masked Autoencoder (MAE)
A variant where parts of the input ECG are randomly masked (set to zero or replaced with noise), and the model is trained to reconstruct only the masked portions. This encourages the encoder to learn context-aware and robust representations.

### 🟨 Variational Autoencoder (VAE)
Extends the AE by learning a distribution (mean and variance) over latent features. The encoder outputs are regularized via KL divergence, enforcing a Gaussian prior on the latent space for better generalization.

---

## 🧠 Fine-Tuning Phase

After pretraining:
- A **classifier head** is added on top of the frozen or fine-tuned encoder.
- The entire model is trained end-to-end on a **limited labeled ECG dataset** to predict a **binary outcome** (e.g., CRT response).

---

## 🧪 Experimentation Strategy

Implemented in `train.py`, experiments follow a **repeated random sub-sampling validation** protocol:

1. Randomly split patients into train, validation, and test sets.
2. Train on the training set.
3. Use validation performance to select the best model.
4. Evaluate the selected model on the held-out test set.
5. Repeat the above K times with different random seeds.
6. Report **mean ± confidence interval** of test metrics across runs.
