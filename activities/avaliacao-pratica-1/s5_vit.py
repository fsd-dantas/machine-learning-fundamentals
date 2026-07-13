"""
Strategy 5 — Fine-tuning a Vision Transformer (ViT-B/16, ImageNet-21k pretrained).

The one script that leaves TensorFlow. ViT fine-tuning is far better supported in
PyTorch + HuggingFace, and the protocol survives the switch intact because of how
`common` is built: this script reads the *same* cached `.npz` splits with plain NumPy,
and hands its test predictions to the *same* `score_run`. Comparability lives in the
data and the evaluation, not in the framework.

Why a transformer is a fair thing to compare against a CNN here, and why it is not:

  Fair — it gets the identical 10,000-image budget, the identical validation set, and
         is scored once on the identical official test set.
  Not fair, and stated plainly in the report — ViT-B/16 carries 86M parameters
         pretrained on ImageNet-21k (14M images, 21k classes), against MobileNetV2's
         3.5M pretrained on ImageNet-1k. If it wins, the honest attribution is
         "more pretraining and more capacity", not "attention beats convolution".

A ViT has no convolutional inductive bias (no locality, no translation equivariance),
which is exactly why it is data-hungry and why *pretrained* is doing the heavy lifting:
trained from scratch on 10,000 images it would lose badly to strategy 1's small CNN.

Discriminative learning rates: the pretrained encoder is nudged at 2e-5 while the fresh
classification head — random init, nothing to preserve — trains 50x faster at 1e-3.

Usage:
    python s5_vit.py --seed 42
    python s5_vit.py --model facebook/deit-tiny-patch16-224   # cheaper variant
"""

from __future__ import annotations

import argparse

import numpy as np

import common

DEFAULT_MODEL = "google/vit-base-patch16-224-in21k"


class CifarTensorDataset:
    """uint8 32x32 HWC -> normalised float32 224x224 CHW, resized on the fly.

    Same reason as the TensorFlow pipeline: materialising 22,000 images at 224x224
    float32 would cost ~13 GB of RAM, and Colab gives ~12.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, size: int, augment: bool):
        import torch
        from torchvision import transforms

        self.x, self.y, self.torch = x, y, torch
        norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        steps = [transforms.ToPILImage(),
                 transforms.Resize(size, antialias=True)]
        if augment:
            # the winning policy from strategy 4, ported: flip + translation
            steps += [transforms.RandomHorizontalFlip(),
                      transforms.RandomAffine(degrees=0, translate=(0.125, 0.125))]
        steps += [transforms.ToTensor(), norm]
        self.tf = transforms.Compose(steps)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i):
        return self.tf(self.x[i]), int(self.y[i])


def assert_pretrained(loading_info: dict) -> None:
    """Fail loudly if `from_pretrained` did not actually load the pretrained encoder.

    This is not paranoia. transformers 5.x refactored the ViT implementation and renamed
    every encoder parameter (`encoder.layer.N.attention.attention.query` ->
    `vit.layers.N.attention.q_proj`). The old checkpoint's tensors then fail to map: the
    whole encoder is reported UNEXPECTED, silently discarded, and randomly initialised.
    `from_pretrained` raises nothing. Training proceeds. A number is produced and saved —
    and it is the number of a ViT trained *from scratch* on 10,000 images, which would
    have been written up as "the transformer is weak in the low-data regime": a
    conclusion that is false, and caused entirely by a library upgrade.

    The only key legitimately missing is the classification head, which is new by
    construction — ImageNet-21k's classes are replaced by CIFAR-10's ten.
    """
    missing = [k for k in loading_info.get("missing_keys", [])
               if not k.startswith("classifier.")]
    if missing:
        raise RuntimeError(
            f"pretrained weights did NOT load: {len(missing)} encoder parameters were "
            f"randomly initialised (e.g. {missing[:3]}).\n"
            "This is the transformers 5.x ViT rename. Install the pinned version:\n"
            "    pip install -q 'transformers>=4.40,<5'\n"
            "Refusing to train a randomly-initialised ViT and report it as pretrained."
        )


def evaluate(model, loader, device, torch) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits = model(pixel_values=images).logits
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=common.SEED)
    # A pretrained ViT converges on CIFAR-10 in very few epochs — most of the work was
    # done on ImageNet-21k. Early stopping usually fires before this budget is spent.
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-encoder", type=float, default=2e-5)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import ViTForImageClassification

    common.set_seed(args.seed)
    splits = common.load_splits()   # reads the cached .npz — no TensorFlow needed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = (torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
    print(f"device: {device_name} · {splits.summary()}")

    size = 224
    loaders = {
        "train": DataLoader(CifarTensorDataset(splits.x_train, splits.y_train, size,
                                               args.augment),
                            batch_size=args.batch_size, shuffle=True, num_workers=2,
                            pin_memory=True, drop_last=True),
        "val": DataLoader(CifarTensorDataset(splits.x_val, splits.y_val, size, False),
                          batch_size=args.batch_size, num_workers=2, pin_memory=True),
        "test": DataLoader(CifarTensorDataset(splits.x_test, splits.y_test, size, False),
                           batch_size=args.batch_size, num_workers=2, pin_memory=True),
    }

    model, loading_info = ViTForImageClassification.from_pretrained(
        args.model, num_labels=common.N_CLASSES,
        ignore_mismatched_sizes=True,   # the 21k-class head is discarded for a 10-class one
        output_loading_info=True,
    )
    assert_pretrained(loading_info)
    model = model.to(device)

    head_params = list(model.classifier.parameters())
    head_ids = {id(p) for p in head_params}
    encoder_params = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.lr_encoder},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=0.01)

    steps = args.epochs * len(loaders["train"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[args.lr_encoder, args.lr_head], total_steps=steps,
        pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    criterion = torch.nn.CrossEntropyLoss()

    best_state, best_val, best_epoch, stale = None, -1.0, 0, 0
    val_history: list[float] = []

    with common.Timer() as timer:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running, seen = 0.0, 0
            for images, labels in loaders["train"]:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                    loss = criterion(model(pixel_values=images).logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                running += loss.item() * labels.size(0)
                seen += labels.size(0)

            val_pred = evaluate(model, loaders["val"], device, torch)
            val_acc = float(np.mean(val_pred == splits.y_val))
            val_history.append(val_acc)
            print(f"epoch {epoch}/{args.epochs}  loss {running/seen:.4f}  "
                  f"val_acc {val_acc:.4f}")

            # Model selection on validation only. The test set is untouched until the
            # single scoring pass below — it is a held-out set, not a tuning signal.
            if val_acc > best_val:
                best_val, best_epoch, stale = val_acc, epoch, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"early stopping at epoch {epoch}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    y_pred = evaluate(model, loaders["test"], device, torch)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    common.score_run(
        strategy="s5_vit",
        label=args.model.split("/")[-1].replace("-", "_"),
        seed=args.seed,
        config={"model": args.model, "pretrained_on": "ImageNet-21k",
                "lr_encoder": args.lr_encoder, "lr_head": args.lr_head,
                "optimizer": "AdamW + OneCycle", "batch_size": args.batch_size,
                "augmentation": "flip+translate" if args.augment else None,
                "best_val_accuracy": best_val},
        y_true=splits.y_test,
        y_pred=y_pred,
        input_resolution=size,
        params_total=total,
        params_trainable=trainable,
        epochs_run=len(val_history),
        best_epoch=best_epoch,
        train_seconds=timer.seconds,
        framework="pytorch",
        device=device_name,
    )


if __name__ == "__main__":
    main()
