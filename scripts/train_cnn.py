"""Addestra SmallCNN su CIFAR-10 e salva il checkpoint. [INFRASTRUTTURA — già pronta]

Uso (Colab, GPU T4: ~5 minuti con 20 epoche):
    python scripts/train_cnn.py --epochs 20 --out checkpoints/smallcnn.pt
"""
import argparse
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from quantlab.data import cifar10_loaders  # noqa: E402
from quantlab.evaluate import accuracy  # noqa: E402
from quantlab.models import SmallCNN  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--out", default="checkpoints/smallcnn.pt")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = cifar10_loaders()
    model = SmallCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, epochs=args.epochs,
                                                steps_per_epoch=len(train_loader))
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss_fn(model(x), y).backward()
            opt.step()
            sched.step()
        print(f"epoca {epoch + 1:2d}/{args.epochs}  test acc {accuracy(model, test_loader, device):.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"salvato in {args.out}")


if __name__ == "__main__":
    main()
