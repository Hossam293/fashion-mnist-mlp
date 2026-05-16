# Fashion-MNIST Classification
### Using Multilayer Perceptron (MLP)
---

## Problem Description

This project addresses the task of **image classification** on the *Fashion-MNIST* dataset. The goal is to build a Multilayer Perceptron (MLP) that correctly identifies one of **10 clothing categories** from a 28×28 grayscale image.

### Classes

| Label | Class |
|-------|-------|
| 0 | T-shirt / Top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle Boot |

### Preprocessing
- Images converted to tensors via `transforms.ToTensor()`
- Pixel values normalized to `[-1, 1]` using mean = 0.5, std = 0.5
- Dataset split: **60,000** training samples / **10,000** test samples

---

## Dataset Link

The dataset is loaded automatically by PyTorch's `torchvision` library. No manual download is required.

- **Fashion-MNIST GitHub:** [https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)
- **Torchvision Docs:** [https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html)

---

## Experiments & Results

Three MLP configurations were trained and compared. Each model was evaluated on the **test set** (10,000 samples).

### Model Architectures

**Model 1 — Baseline (SGD + ReLU)**
- Architecture: 784 → 128 → 64 → 10
- Activation: ReLU
- Optimizer: SGD (lr = 0.01)
- Epochs: 15

**Model 2 — Regularized (Adam + ReLU + BatchNorm + Dropout)**
- Architecture: 784 → 128 → 64 → 10
- Activation: ReLU
- Regularization: BatchNorm1d + Dropout (0.3)
- Optimizer: Adam (lr = 0.001), StepLR scheduler
- Epochs: 15

**Model 3 — Deeper (Adam + Tanh + BatchNorm + Dropout)**
- Architecture: 784 → 512 → 256 → 10
- Activation: Tanh
- Regularization: BatchNorm1d + Dropout (0.3)
- Optimizer: Adam (lr = 0.01), StepLR scheduler
- Epochs: 20

### Comparison Table

| Model | Activation | Optimizer | Epochs | Regularization | Hidden Neurons | Test Accuracy |
|-------|-----------|-----------|--------|----------------|----------------|---------------|
| Model 1 (Baseline) | ReLU | SGD | 15 | None | 128, 64 | ~84% |
| Model 2 | ReLU | Adam | 15 | BatchNorm + Dropout | 128, 64 | ~87% |
| Model 3 | Tanh | Adam | 20 | BatchNorm + Dropout | 512, 256 | ~89% |

> **Note:** Replace the accuracy values with the exact numbers printed by the notebook after running all three models.

---

## Instructions for Running the Project

### Prerequisites
- Python 3.x
- PyTorch
- Torchvision
- Matplotlib
- NumPy

Install all dependencies:

```bash
pip install torch torchvision matplotlib numpy
```

### Steps

1. Clone the repository:
```bash
git clone https://github.com/Hossam293/fashion-mnist-mlp.git
cd fashion-mnist-mlp
```

3. Run all cells sequentially (*Kernel → Restart & Run All*)
4. The dataset will be downloaded automatically on first run to `./data/`
5. Training progress and test accuracy are printed after each experiment
6. Loss and accuracy plots are displayed inline after each model

### Expected Output

```
Epoch 1:  Loss = 0.8423,  Acc = 70.15%
...
Epoch 15: Loss = 0.3201,  Acc = 88.42%
Test Accuracy: 88.35%
```
