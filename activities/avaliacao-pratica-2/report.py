"""
Report generator — turns the run JSONs into the tables, confusion matrices and
statistics for both write-ups:

    ../avaliacao-pratica-2.md            English  — repository content
    ../avaliacao-pratica-2-respostas.md  pt-BR    — the submitted PDF

Runs on CPU in seconds. Every run persists its test predictions, so the tables, the
confusion matrices and the paired McNemar test all regenerate from the committed JSONs
alone — no retraining, no GPU, no trust required.

Two columns appear next to every accuracy and are not decoration:

    floor    the majority-class rate. An accuracy is meaningless without it.
    Δ floor  how much the model actually learned.

Usage:
    python report.py
    python report.py --no-write
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

import common

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TARGETS = {
    "en": common.HERE.parent / "avaliacao-pratica-2.md",
    "pt": common.HERE.parent / "avaliacao-pratica-2-respostas.md",
}
IMG_DIR = common.HERE.parents[1] / "assets" / "img"

MODEL_NAMES = {
    "en": {
        "majority": "Majority class *(floor)*",
        "tfidf_svm": "TF-IDF + linear SVM *(classical baseline)*",
        "lstm": "LSTM",
        "bilstm": "BiLSTM",
        "bert": "BERT (BERTimbau, fine-tuned)",
    },
    "pt": {
        "majority": "Classe majoritária *(piso)*",
        "tfidf_svm": "TF-IDF + SVM linear *(baseline clássico)*",
        "lstm": "LSTM",
        "bilstm": "BiLSTM",
        "bert": "BERT (BERTimbau, ajuste fino)",
    },
}

L = {
    "en": {
        "model": "Model", "acc": "Accuracy", "f1": "Macro-F1", "wf1": "Weighted-F1",
        "ci": "95% CI", "floor": "Δ floor", "params": "Trainable params",
        "train": "Train", "seeds": "seeds", "none": "_(no runs yet)_",
        "cm_x": "Predicted", "cm_y": "True",
        "cm_title": "Confusion matrix — {model} · {task}\naccuracy {acc:.4f} · row-normalised",
        "sig_head": ["Discordant", "LSTM right / BERT wrong", "LSTM wrong / BERT right",
                     "p (exact McNemar)"],
        "sig_sig": ("The {gap:.2f} pp gap between **{winner}** and the other is "
                    "**significant** (α = 0.05, exact McNemar on paired test predictions)."),
        "sig_tie": ("The {gap:.2f} pp gap is **not significant** (α = 0.05, exact McNemar). "
                    "On this test set the two are statistically indistinguishable — a "
                    "**technical tie**, not a win."),
        "sens": ("| Mapping | LSTM | BERT | Note |\n|---|---|---|---|\n"),
        "missing_pair": "_(needs both an LSTM and a BERT run at the primary seed)_",
        "pair": "Comparison", "pvalue_short": "p (McNemar)", "verdict": "Significant?",
        "yes": "**yes**", "no": "no — technical tie", "class": "Class",
        "confusion": "Confusion", "rate": "Rate",
    },
    "pt": {
        "model": "Modelo", "acc": "Acurácia", "f1": "Macro-F1", "wf1": "F1 ponderado",
        "ci": "IC 95%", "floor": "Δ piso", "params": "Parâmetros treináveis",
        "train": "Treino", "seeds": "sementes", "none": "_(sem execuções ainda)_",
        "cm_x": "Predita", "cm_y": "Verdadeira",
        "cm_title": "Matriz de confusão — {model} · {task}\nacurácia {acc:.4f} · normalizada por linha",
        "sig_head": ["Discordâncias", "LSTM certo / BERT errado",
                     "LSTM errado / BERT certo", "p (McNemar exato)"],
        "sig_sig": ("A diferença de {gap:.2f} pp em favor do **{winner}** é "
                    "**significativa** (α = 0,05; McNemar exato sobre predições pareadas)."),
        "sig_tie": ("A diferença de {gap:.2f} pp **não é significativa** (α = 0,05; McNemar "
                    "exato). Neste conjunto de teste os dois são estatisticamente "
                    "indistinguíveis — **empate técnico**, não vitória."),
        "sens": ("| Mapeamento | LSTM | BERT | Observação |\n|---|---|---|---|\n"),
        "missing_pair": "_(exige uma execução de LSTM e uma de BERT na semente primária)_",
        "pair": "Comparação", "pvalue_short": "p (McNemar)", "verdict": "Significativa?",
        "yes": "**sim**", "no": "não — empate técnico", "class": "Classe",
        "confusion": "Confusão", "rate": "Taxa",
    },
}

TASK_LABEL = {
    "en": {"binary": "binary (positive/negative)",
           "multiclass": "multiclass (7 emotions)",
           "binary_valence": "binary, valence-clean"},
    "pt": {"binary": "binária (positivo/negativo)",
           "multiclass": "multiclasse (7 emoções)",
           "binary_valence": "binária, valência limpa"},
}

BEGIN, END = "<!-- BEGIN GENERATED: {} -->", "<!-- END GENERATED: {} -->"


def fmt(mean: float, std: float | None) -> str:
    return f"{mean:.4f}" if std is None else f"{mean:.4f} ± {std:.4f}"


def table_task(rows: list[dict], task: str, lang: str) -> str:
    t = L[lang]
    sel = [r for r in rows if r["task"] == task]
    if not sel:
        return t["none"]

    lines = [f"| {t['model']} | {t['acc']} | {t['floor']} | {t['f1']} | {t['wf1']} | "
             f"{t['ci']} | {t['params']} | {t['train']} |",
             "|---|---|---|---|---|---|---|---|"]
    runs = common.load_results(task)
    for row in sorted(sel, key=lambda r: r["accuracy_mean"], reverse=True):
        group = [r for r in runs if r["model"] == row["model"]]
        lo = min(r["accuracy_ci95"][0] for r in group)
        hi = max(r["accuracy_ci95"][1] for r in group)
        wf1 = float(np.mean([r["weighted_f1"] for r in group]))
        lift = (row["accuracy_mean"] - row["majority_floor"]) * 100
        name = MODEL_NAMES[lang].get(row["model"], row["model"])
        seeds = f" <sub>({row['n_seeds']} {t['seeds']})</sub>" if row["n_seeds"] > 1 else ""
        lines.append(
            f"| {name}{seeds} | **{fmt(row['accuracy_mean'], row['accuracy_std'])}** | "
            f"{lift:+.1f} pp | {fmt(row['macro_f1_mean'], row['macro_f1_std'])} | "
            f"{wf1:.4f} | {lo:.4f}–{hi:.4f} | {row['params_trainable']:,} | "
            f"{row['train_seconds_mean']:.0f}s |")
    return "\n".join(lines)


def significance(task: str, lang: str) -> str:
    """LSTM vs BERT, paired, on the same test texts — the comparison the task asks for."""
    t = L[lang]
    runs = {r["model"]: r for r in common.load_results(task) if r["seed"] == common.SEED}
    if "lstm" not in runs or "bert" not in runs:
        return t["missing_pair"]

    lstm, bert = runs["lstm"], runs["bert"]
    corpus = common.load_corpus(task, verbose=False)
    y_test = common.make_splits(corpus, common.SEED).y_test
    test = common.mcnemar(y_test, np.array(lstm["test_predictions"]),
                          np.array(bert["test_predictions"]))

    gap = abs(lstm["accuracy"] - bert["accuracy"]) * 100
    winner = "BERT" if bert["accuracy"] > lstm["accuracy"] else "LSTM"
    verdict = (t["sig_sig"].format(gap=gap, winner=winner)
               if test["significant_at_05"] else t["sig_tie"].format(gap=gap))

    head = t["sig_head"]
    return "\n".join([
        f"| {' | '.join(head)} |",
        "|---|---|---|---|",
        f"| {test['n_discordant']} | {test['n_a_correct_b_wrong']} | "
        f"{test['n_a_wrong_b_correct']} | {test['p_value']:.4g} |",
        "",
        verdict,
    ])


PAIRS = (("bert", "tfidf_svm"), ("tfidf_svm", "lstm"), ("bert", "lstm"),
         ("bilstm", "lstm"))


def pairwise(task: str, lang: str) -> str:
    """Every comparison that carries a claim, not just LSTM vs BERT.

    The assignment asks for LSTM vs Transformer, but the table also has a classical
    baseline in it, and 'TF-IDF beats the LSTM' is a claim — so it gets a test. Each
    pair is scored by exact McNemar on the primary seed's paired predictions.
    """
    t = L[lang]
    runs = {r["model"]: r for r in common.load_results(task) if r["seed"] == common.SEED}
    corpus = common.load_corpus(task, verbose=False)
    y_test = common.make_splits(corpus, common.SEED).y_test

    lines = [f"| {t['pair']} | Δ | {t['pvalue_short']} | {t['verdict']} |",
             "|---|---|---|---|"]
    any_row = False
    for a, b in PAIRS:
        if a not in runs or b not in runs:
            continue
        any_row = True
        test = common.mcnemar(y_test, np.array(runs[a]["test_predictions"]),
                              np.array(runs[b]["test_predictions"]))
        gap = (runs[a]["accuracy"] - runs[b]["accuracy"]) * 100
        name_a = MODEL_NAMES[lang].get(a, a).split(" *")[0]
        name_b = MODEL_NAMES[lang].get(b, b).split(" *")[0]
        verdict = t["yes"] if test["significant_at_05"] else t["no"]
        lines.append(f"| {name_a} vs. {name_b} | {gap:+.2f} pp | "
                     f"{test['p_value']:.3g} | {verdict} |")
    return "\n".join(lines) if any_row else t["none"]


def per_class(task: str, lang: str) -> str:
    """Per-class F1 of the best model — where a macro average hides the damage."""
    t = L[lang]
    runs = [r for r in common.load_results(task) if r["seed"] == common.SEED
            and r["model"] not in ("majority",)]
    if not runs:
        return t["none"]
    best = max(runs, key=lambda r: r["accuracy"])
    lines = [f"| {t['class']} | F1 |", "|---|---|"]
    for name, value in sorted(best["per_class_f1"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {name} | {value:.3f} |")
    return "\n".join(lines)


def confusions(task: str, lang: str, top: int = 6) -> str:
    """The off-diagonal — the only part of a confusion matrix that carries information."""
    t = L[lang]
    runs = [r for r in common.load_results(task) if r["seed"] == common.SEED
            and r["model"] not in ("majority",)]
    if not runs:
        return t["none"]
    best = max(runs, key=lambda r: r["accuracy"])
    names = best["class_names"]
    cm = np.array(best["confusion_matrix"], dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    np.fill_diagonal(norm, 0.0)

    pairs = sorted(((norm[i, j], i, j) for i in range(len(names))
                    for j in range(len(names)) if i != j), reverse=True)
    lines = [f"| {t['confusion']} | {t['rate']} |", "|---|---|"]
    for rate, i, j in pairs[:top]:
        lines.append(f"| {names[i]} → {names[j]} | {rate*100:.1f}% |")
    return "\n".join(lines)


def sensitivity(rows: list[dict], lang: str) -> str:
    """Does the binary conclusion survive dropping `neutro`/`surpresa`?"""
    t = L[lang]
    out = [t["sens"]]
    notes = {
        "binary": {"en": "instructor's map (`neutro`, `surpresa` → positive)",
                   "pt": "mapa do professor (`neutro`, `surpresa` → positivo)"},
        "binary_valence": {"en": "`neutro`/`surpresa` dropped",
                           "pt": "`neutro`/`surpresa` descartados"},
    }
    any_row = False
    for task in ("binary", "binary_valence"):
        cells = {r["model"]: r for r in rows if r["task"] == task}
        if not cells:
            continue
        any_row = True
        lstm = cells.get("lstm")
        bert = cells.get("bert")
        out.append(
            f"| {TASK_LABEL[lang][task]} | "
            f"{fmt(lstm['accuracy_mean'], lstm['accuracy_std']) if lstm else '—'} | "
            f"{fmt(bert['accuracy_mean'], bert['accuracy_std']) if bert else '—'} | "
            f"{notes[task][lang]} |\n")
    return "".join(out) if any_row else t["none"]


def confusion_png(run: dict, lang: str = "pt") -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = L[lang]
    names = run["class_names"]
    cm = np.array(run["confusion_matrix"], dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    paths = []

    for theme in ("light", "dark"):
        dark = theme == "dark"
        fg = "#e6edf3" if dark else "#1f2328"
        bg = "#0d1117" if dark else "#ffffff"
        size = 5.5 if len(names) == 2 else 8.0

        fig, ax = plt.subplots(figsize=(size + 1.5, size), facecolor=bg)
        ax.set_facecolor(bg)
        im = ax.imshow(norm, cmap="viridis" if dark else "Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel(t["cm_x"], color=fg, labelpad=8)
        ax.set_ylabel(t["cm_y"], color=fg, labelpad=8)
        ax.set_title(t["cm_title"].format(
            model=MODEL_NAMES[lang].get(run["model"], run["model"]).split(" *")[0],
            task=TASK_LABEL[lang][run["task"]], acc=run["accuracy"]), color=fg, pad=12)

        for i in range(len(names)):
            for j in range(len(names)):
                if norm[i, j] < 0.005:
                    continue
                ax.text(j, i, f"{norm[i, j]*100:.0f}", ha="center", va="center",
                        fontsize=9, color="white" if norm[i, j] > 0.5 else fg)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046)
        cbar.ax.tick_params(colors=fg)
        ax.tick_params(colors=fg)
        for spine in ax.spines.values():
            spine.set_color(fg)

        IMG_DIR.mkdir(parents=True, exist_ok=True)
        path = IMG_DIR / f"avaliacao-pratica-2-confusion-{run['task']}-{theme}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160, facecolor=bg)
        plt.close(fig)
        paths.append(path)
    return paths


def localise_decimals(text: str) -> str:
    """English number formatting -> pt-BR, for the submitted write-up.

    Both separators swap, and the order matters. `108,924,674` is a *thousands* comma in
    English and would read as a decimal in Portuguese — leaving it is worse than leaving
    the decimal point, because it silently changes the magnitude a reader perceives. So:

        108,924,674  ->  108.924.674     (thousands: comma -> dot)
        0.8285       ->  0,8285          (decimal:   dot   -> comma)
        8.25e-07     ->  8,25e-07

    Thousands groups are parked on a sentinel first, so the decimal pass cannot chew on
    the dots it just created. The lookarounds require digits on both sides, so `vs.`,
    `report.py` and URLs are left alone.
    """
    sentinel = "\x00"
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", sentinel, text)  # thousands
    text = re.sub(r"(?<=\d)\.(?=\d)", ",", text)               # decimals
    return text.replace(sentinel, ".")


def patch(markdown: str, key: str, content: str) -> str:
    begin, end = BEGIN.format(key), END.format(key)
    block = f"{begin}\n{content}\n{end}"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    return (pattern.sub(block, markdown) if pattern.search(markdown)
            else markdown + f"\n\n{block}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    runs = common.load_results()
    if not runs:
        raise SystemExit("no results found — run the models first")
    rows = common.aggregate(runs)
    print(f"{len(runs)} runs across {len({r['task'] for r in runs})} tasks")

    # The pictured model is the best *mean* over seeds, shown at the primary seed —
    # not the single luckiest run.
    best_run: dict[str, dict] = {}
    for task in ("binary", "multiclass"):
        candidates = [r for r in rows if r["task"] == task
                      and r["model"] not in ("majority",)]
        if not candidates:
            continue
        leader = max(candidates, key=lambda r: r["accuracy_mean"])
        for run in runs:
            if (run["task"] == task and run["model"] == leader["model"]
                    and run["seed"] == common.SEED):
                best_run[task] = run
                for path in confusion_png(run):
                    print(f"wrote {path.name}")
                break

    for lang, target in TARGETS.items():
        sections = {
            "table-binary": table_task(rows, "binary", lang),
            "table-multiclass": table_task(rows, "multiclass", lang),
            "significance-binary": significance("binary", lang),
            "significance-multiclass": significance("multiclass", lang),
            "sensitivity": sensitivity(rows, lang),
            "pairwise-binary": pairwise("binary", lang),
            "pairwise-multiclass": pairwise("multiclass", lang),
            "per-class-multiclass": per_class("multiclass", lang),
            "confusions-multiclass": confusions("multiclass", lang),
        }
        if args.no_write:
            for key, content in sections.items():
                print(f"\n===== [{lang}] {key} =====\n{content}")
            continue
        if not target.exists():
            print(f"skipped {target.name} (not found)")
            continue
        markdown = target.read_text(encoding="utf-8")
        for key, content in sections.items():
            markdown = patch(markdown, key,
                             localise_decimals(content) if lang == "pt" else content)
        target.write_text(markdown, encoding="utf-8")
        print(f"patched {target.name}")


if __name__ == "__main__":
    main()
