# Adapted ConvNeXt for MNIST

This project is a compact AI-assisted implementation of a ConvNeXt-inspired image classifier for MNIST. It is intentionally small enough for an 8 GB GPU and can also run on CPU.

## Experiment design

- Dataset: MNIST handwritten digits, 28 x 28 grayscale images
- Training subset: 10,000 images selected with seed 42
- Test subset: 2,000 images selected with seed 43
- Model: two-stage Adapted ConvNeXt with 24 and 48 channels
- Training: 3 epochs, batch size 64, AdamW, learning rate 0.001
- Output: metrics, model weights, learning curves, and sample predictions

## Run

```powershell
D:\conda_envs\aiasm\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
D:\conda_envs\aiasm\python.exe -m pip install "matplotlib>=3.8,<4"
D:\conda_envs\aiasm\python.exe main.py
```

For NVIDIA GPU training on this machine, the environment uses the CUDA 12.4 builds of PyTorch 2.6.0 and torchvision 0.21.0.

For a very quick check:

```powershell
python main.py --epochs 1 --train-size 512 --test-size 256 --output-dir results_smoke
```

## Files produced

The default run stores these files in `results/`:

- `metrics.json`
- `adapted_convnext_mnist.pt`
- `training_curves.png`
- `sample_predictions.png`

## Source code webpage

https://github.com/CookieWinnie/AIassignment1

## AI use

The algorithm choice, adaptation strategy, implementation, experiment design, and report drafting were completed with an AI coding agent. The prompt record is in `ai_prompts.md`.
