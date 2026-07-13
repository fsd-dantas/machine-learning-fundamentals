"""
BERT classifier — the Transformer arm (BERTimbau, `neuralmind/bert-base-portuguese-cased`).

Fine-tunes the pretrained encoder with a dense classification head, which is the first
option the assignment offers. The comparison against the LSTM is deliberately lopsided,
and saying so is part of the result rather than a caveat to bury:

    LSTM   ~1.4M parameters, embeddings learned from scratch on 1,450 headlines.
    BERT   ~109M parameters, pretrained on brWaC — 2.7 billion words of Portuguese.

The LSTM is asked to learn what words *mean* from 1,450 examples. BERT arrives already
knowing, and only has to learn what the labels mean. If BERT wins, the honest
attribution is *pretraining*, not *attention* — the architecture is not what is doing
the work. That distinction is the whole point of the comparison, and it is invisible to
anyone who only reports the two accuracies.

**Raw text, no stopword removal** — unlike the LSTM. BERT was pretrained on running
Portuguese, with its function words and its word order; deleting "não" from "não gostei"
inverts the sentiment, and stripping stopwords hands the model a token distribution it
has never seen. The preprocessing best for one model is destructive for the other, which
is itself a finding worth one line in the report.

**Discriminative learning rates**: the pretrained encoder is nudged at 2e-5, while the
randomly-initialised head — which has nothing worth preserving — trains 50x faster.

PyTorch, not Keras: HuggingFace's TensorFlow support is deprecated and BERTimbau ships
PyTorch weights. The protocol survives the switch because `common.score_run` evaluates
saved predictions, not models — the same design that let the ViT in Avaliação Prática 1
stay comparable to Keras CNNs.

Usage:
    python m2_bert.py --task binary
    python m2_bert.py --task multiclass --seed 7
"""

from __future__ import annotations

import argparse

import numpy as np

import common


class TextDataset:
    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.encodings = tokenizer(list(texts), truncation=True, padding="max_length",
                                   max_length=max_length, return_tensors="pt")
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i):
        import torch
        item = {k: v[i] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[i]))
        return item


def predict(model, loader, device, torch) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits = model(**batch).logits
            out.append(logits.float().argmax(dim=1).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="binary", choices=common.TASKS)
    parser.add_argument("--seed", type=int, default=common.SEED)
    parser.add_argument("--model", default=common.BERT_MODEL)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--lr-encoder", type=float, default=2e-5)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    common.set_seed(args.seed)
    corpus = common.load_corpus(args.task)
    splits = common.make_splits(corpus, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"device: {device_name} · {splits.summary()}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    loaders = {
        name: DataLoader(TextDataset(x, y, tokenizer, args.max_length),
                         batch_size=args.batch_size, shuffle=(name == "train"))
        for name, (x, y) in {
            "train": (splits.x_train, splits.y_train),
            "val": (splits.x_val, splits.y_val),
            "test": (splits.x_test, splits.y_test),
        }.items()
    }

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=corpus.n_classes).to(device)

    head = list(model.classifier.parameters())
    head_ids = {id(p) for p in head}
    encoder = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [{"params": encoder, "lr": args.lr_encoder},
         {"params": head, "lr": args.lr_head}], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[args.lr_encoder, args.lr_head],
        total_steps=args.epochs * len(loaders["train"]), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_state, best_val, stale, epochs_run = None, -1.0, 0, 0
    with common.Timer() as timer:
        for epoch in range(1, args.epochs + 1):
            model.train()
            total, seen = 0.0, 0
            for batch in loaders["train"]:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                    loss = model(**batch).loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                total += loss.item() * batch["labels"].size(0)
                seen += batch["labels"].size(0)

            val_pred = predict(model, loaders["val"], device, torch)
            val_acc = float(np.mean(val_pred == splits.y_val))
            epochs_run = epoch
            print(f"epoch {epoch}/{args.epochs}  loss {total/seen:.4f}  "
                  f"val_acc {val_acc:.4f}")

            # Selection on validation only; the test set stays sealed until below.
            if val_acc > best_val:
                best_val, stale = val_acc, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"early stopping at epoch {epoch}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    y_pred = predict(model, loaders["test"], device, torch)

    common.score_run(
        task=args.task, model="bert", seed=args.seed,
        config={"model": args.model, "pretrained_on": "brWaC (2.7B words, pt-BR)",
                "max_length": args.max_length, "batch_size": args.batch_size,
                "lr_encoder": args.lr_encoder, "lr_head": args.lr_head,
                "optimizer": "AdamW + OneCycle", "stopwords_removed": False,
                "best_val_accuracy": best_val},
        corpus=corpus, y_true=splits.y_test, y_pred=y_pred,
        params_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        epochs_run=epochs_run, train_seconds=timer.seconds,
        framework="pytorch", device=device_name,
    )


if __name__ == "__main__":
    main()
