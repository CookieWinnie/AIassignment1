import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".matplotlib"))

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class LayerNorm2d(nn.Module):
    """LayerNorm for NCHW feature maps."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class AdaptedConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )
        self.norm = LayerNorm2d(channels)
        self.pointwise1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.activation = nn.GELU()
        self.pointwise2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.pointwise1(x)
        x = self.activation(x)
        x = self.pointwise2(x)
        return x + residual


class AdaptedConvNeXt(nn.Module):
    """Small ConvNeXt-inspired classifier adapted for 28x28 grayscale MNIST."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, stride=1, padding=1),
            LayerNorm2d(24),
        )
        self.stage1 = nn.Sequential(
            AdaptedConvNeXtBlock(24),
            AdaptedConvNeXtBlock(24),
        )
        self.downsample = nn.Sequential(
            LayerNorm2d(24),
            nn.Conv2d(24, 48, kernel_size=2, stride=2),
        )
        self.stage2 = nn.Sequential(
            AdaptedConvNeXtBlock(48),
            AdaptedConvNeXtBlock(48),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head_norm = nn.LayerNorm(48)
        self.head = nn.Linear(48, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.downsample(x)
        x = self.stage2(x)
        x = self.pool(x).flatten(1)
        x = self.head_norm(x)
        return self.head(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_subset(dataset, size: int, seed: int) -> Subset:
    size = min(size, len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


def build_loaders(args, device: torch.device):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_full = datasets.MNIST(
        args.data_dir, train=True, download=not args.no_download, transform=transform
    )
    test_full = datasets.MNIST(
        args.data_dir, train=False, download=not args.no_download, transform=transform
    )
    train_set = select_subset(train_full, args.train_size, args.seed)
    test_set = select_subset(test_full, args.test_size, args.seed + 1)
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)
    return train_loader, test_loader


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_examples += labels.size(0)

    return total_loss / total_examples, total_correct / total_examples


def save_curves(history, output_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train")
    axes[0].plot(epochs, history["test_loss"], marker="o", label="Test")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, history["train_accuracy"], marker="o", label="Train")
    axes[1].plot(epochs, history["test_accuracy"], marker="o", label="Test")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_predictions(model, loader, device, output_path: Path) -> None:
    images, labels = next(iter(loader))
    with torch.inference_mode():
        predictions = model(images.to(device)).argmax(1).cpu()
    fig, axes = plt.subplots(2, 5, figsize=(8, 3.6))
    for ax, image, label, prediction in zip(
        axes.flat, images[:10], labels[:10], predictions[:10]
    ):
        ax.imshow(image.squeeze().numpy(), cmap="gray")
        color = "#18794e" if label.item() == prediction.item() else "#b42318"
        ax.set_title(f"True {label.item()} | Pred {prediction.item()}", color=color, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Adapted ConvNeXt on MNIST")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-size", type=int, default=10_000)
    parser.add_argument("--test-size", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader, test_loader = build_loaders(args, device)
    model = AdaptedConvNeXt().to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    history = {key: [] for key in ["train_loss", "train_accuracy", "test_loss", "test_accuracy"]}
    started = time.perf_counter()

    print(f"Device: {device}")
    print(f"Parameters: {parameter_count:,}")
    print(f"Train/Test examples: {len(train_loader.dataset):,}/{len(test_loader.dataset):,}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        test_loss, test_acc = run_epoch(model, test_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_acc)
        print(
            f"Epoch {epoch}/{args.epochs} | train loss {train_loss:.4f} "
            f"acc {train_acc:.4f} | test loss {test_loss:.4f} acc {test_acc:.4f}"
        )

    elapsed = time.perf_counter() - started
    metrics = {
        "algorithm": "Adapted ConvNeXt",
        "dataset": "MNIST",
        "device": str(device),
        "parameters": parameter_count,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "train_size": len(train_loader.dataset),
        "test_size": len(test_loader.dataset),
        "elapsed_seconds": elapsed,
        "final_test_accuracy": history["test_accuracy"][-1],
        "best_test_accuracy": max(history["test_accuracy"]),
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    torch.save(model.state_dict(), args.output_dir / "adapted_convnext_mnist.pt")
    save_curves(history, args.output_dir / "training_curves.png")
    save_predictions(model, test_loader, device, args.output_dir / "sample_predictions.png")
    print(f"Finished in {elapsed:.1f} seconds. Results saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
