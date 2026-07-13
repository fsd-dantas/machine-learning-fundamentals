"""
Report generator — turns the persisted run JSONs into the tables, the confusion
matrix and the statistics that go into the two write-ups:

    ../avaliacao-pratica-1.md            English  — repository content
    ../avaliacao-pratica-1-respostas.md  pt-BR    — the submitted PDF

Both are generated from the same numbers, so they cannot drift apart. Content lands
between `<!-- BEGIN GENERATED: key -->` / `<!-- END GENERATED: key -->` markers;
prose around the markers is hand-written and never touched.

Runs on CPU in seconds and needs no GPU, no TensorFlow and no retraining: every number
is recomputed from the committed `results/*.json`, each of which carries its own 10,000
test predictions. That is what makes the report auditable — a reader can regenerate
every claim, including the significance test, from the repository alone.

The comparison table deliberately reports four columns next to accuracy: input
resolution, trainable parameters, training minutes, and the 95% confidence interval.
An accuracy ranking without them invites the reader to celebrate a 0.3 pp win that cost
50x the compute and sits inside the noise.

Usage:
    python report.py                 # write PNGs, patch both write-ups
    python report.py --no-write      # print only
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

import common

# The tables carry α, Δ, ± and em-dashes. Windows consoles still default to cp1252,
# which cannot encode them — and this script is meant to run locally on CPU, so the
# crash would be routine rather than exotic.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TARGETS = {
    "en": common.HERE.parent / "avaliacao-pratica-1.md",
    "pt": common.HERE.parent / "avaliacao-pratica-1-respostas.md",
}
IMG_DIR = common.HERE.parents[1] / "assets" / "img"

STRATEGY_NAMES = {
    "en": {
        "s1_scratch": "1 — CNN from scratch",
        "s2_features": "2 — Feature extraction",
        "s3_finetune": "3 — Fine-tuning",
        "s4_augment": "4 — Fine-tuning + augmentation",
        "s5_vit": "5 — ViT fine-tuning",
    },
    "pt": {
        "s1_scratch": "1 — CNN do zero",
        "s2_features": "2 — Extração de características",
        "s3_finetune": "3 — Ajuste fino",
        "s4_augment": "4 — Ajuste fino + aumento de dados",
        "s5_vit": "5 — Ajuste fino de ViT",
    },
}

L = {
    "en": {
        "strategy": "Strategy", "config": "Configuration", "accuracy": "Accuracy (test)",
        "ci": "95% CI", "f1": "Macro-F1", "resolution": "Resolution",
        "params": "Trainable params", "train": "Training",
        "acc_seeds": "Accuracy (mean ± sd, {n} seeds)", "delta": "Δ vs. best",
        "missing": "_(no results for `{p}*` — run the corresponding ablation)_",
        "min": "min", "backbone": "Backbone + classifier", "head": "Head (pooling)",
        "optimizer": "Optimiser", "policy": "Augmentation policy",
        "confusion": "Confusion", "rate": "Rate", "reading": "Reading",
        "discordant": "Discordant", "a_right": "1st right / 2nd wrong",
        "b_right": "1st wrong / 2nd right", "pvalue": "p (exact McNemar)",
        "need_two": "_(at least two runs at the primary seed are required)_",
        "significant": "**significant**", "not_significant": "**not significant**",
        "verdict_sig": ("The {gap:.2f} pp gap is {verdict} (α = 0.05). The top-ranked "
                        "model is therefore the best model in this comparison."),
        "verdict_tie": ("The {gap:.2f} pp gap is {verdict} (α = 0.05). The two models are "
                        "statistically indistinguishable on this test set; reporting either "
                        "as “the winner” would over-interpret a difference smaller than the "
                        "noise. We declare a **technical tie**."),
        "cm_title": "Confusion matrix — {label}\naccuracy {acc:.4f} · row-normalised",
        "cm_x": "Predicted class", "cm_y": "True class",
        "pair": "Comparison", "pvalue_short": "p (McNemar)",
        "verdict_col": "Significant?", "yes": "**yes**", "no": "no — technical tie",
    },
    "pt": {
        "strategy": "Estratégia", "config": "Configuração", "accuracy": "Acurácia (teste)",
        "ci": "IC 95%", "f1": "Macro-F1", "resolution": "Resolução",
        "params": "Parâmetros treináveis", "train": "Treino",
        "acc_seeds": "Acurácia (média ± dp, {n} seeds)", "delta": "Δ vs. melhor",
        "missing": "_(sem resultados para `{p}*` — execute a ablação correspondente)_",
        "min": "min", "backbone": "Backbone + classificador", "head": "Cabeça (pooling)",
        "optimizer": "Otimizador", "policy": "Política de aumento de dados",
        "confusion": "Confusão", "rate": "Taxa", "reading": "Leitura",
        "discordant": "Discordâncias", "a_right": "1º certo / 2º errado",
        "b_right": "1º errado / 2º certo", "pvalue": "p (McNemar exato)",
        "need_two": "_(são necessários ao menos dois runs na *seed* primária)_",
        "significant": "**significativa**", "not_significant": "**não significativa**",
        "verdict_sig": ("A diferença de {gap:.2f} pp é {verdict} (α = 0,05). O primeiro "
                        "colocado é, portanto, o melhor modelo desta comparação."),
        "verdict_tie": ("A diferença de {gap:.2f} pp é {verdict} (α = 0,05). Os dois modelos "
                        "são estatisticamente indistinguíveis neste conjunto de teste; "
                        "reportar um deles como “o vencedor” superinterpretaria uma diferença "
                        "menor que o ruído. Declaramos **empate técnico**."),
        "cm_title": "Matriz de confusão — {label}\nacurácia {acc:.4f} · normalizada por linha",
        "cm_x": "Classe predita", "cm_y": "Classe verdadeira",
        "pair": "Comparação", "pvalue_short": "p (McNemar)",
        "verdict_col": "Significativa?", "yes": "**sim**", "no": "não — empate técnico",
    },
}

BEGIN = "<!-- BEGIN GENERATED: {} -->"
END = "<!-- END GENERATED: {} -->"


def fmt(mean: float, std: float | None) -> str:
    """A single run reports no dispersion at all rather than '± 0.0000' — those are
    different statements, and the second one claims a stability never measured."""
    return f"{mean:.4f}" if std is None else f"{mean:.4f} ± {std:.4f}"


def best_per_strategy(rows: list[dict]) -> list[dict]:
    """The strongest configuration of each strategy — the headline comparison.

    Ranked by mean test accuracy. This is a *reporting* choice, not a tuning one: the
    configurations were selected on validation data, and each was scored on the test
    set exactly once.
    """
    best: dict[str, dict] = {}
    for row in rows:
        current = best.get(row["strategy"])
        if current is None or row["accuracy_mean"] > current["accuracy_mean"]:
            best[row["strategy"]] = row
    return sorted(best.values(), key=lambda r: r["accuracy_mean"], reverse=True)


def table_main(rows: list[dict], runs: list[dict], lang: str) -> str:
    t = L[lang]
    lines = [
        f"| {t['strategy']} | {t['config']} | {t['accuracy']} | {t['ci']} | "
        f"{t['f1']} | {t['resolution']} | {t['params']} | {t['train']} |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        group = [r for r in runs if r["strategy"] == row["strategy"]
                 and r["label"] == row["label"]]
        lo = min(r["accuracy_ci95"][0] for r in group)
        hi = max(r["accuracy_ci95"][1] for r in group)
        name = STRATEGY_NAMES[lang].get(row["strategy"], row["strategy"])
        lines.append(
            f"| {name} | `{row['label']}` | "
            f"**{fmt(row['accuracy_mean'], row['accuracy_std'])}** | {lo:.4f}–{hi:.4f} | "
            f"{fmt(row['macro_f1_mean'], row['macro_f1_std'])} | {row['input_resolution']}px | "
            f"{row['params_trainable']:,} | {row['train_minutes_mean']:.1f} {t['min']} |")
    return "\n".join(lines)


def table_ablation(rows: list[dict], strategy: str, prefix: str, variable: str,
                   lang: str) -> str:
    """One ablation: a single variable moves, everything else is pinned.

    An ablation needs at least two arms to exist. With only the `core` stage run, the
    prefix filter would otherwise match that stage's single configuration and render a
    one-row table under an ablation heading — presenting a result that was never
    measured. A partially-run experiment must look partially-run.
    """
    t = L[lang]
    sel = [r for r in rows if r["strategy"] == strategy
           and r["label"].startswith(prefix)]
    if len({r["label"] for r in sel}) < 2:
        return t["missing"].format(p=prefix or variable)

    sel.sort(key=lambda r: r["accuracy_mean"], reverse=True)
    top = sel[0]["accuracy_mean"]
    header = t["acc_seeds"].format(n=sel[0]["n_seeds"])
    lines = [f"| {variable} | {header} | {t['delta']} | {t['f1']} | {t['train']} |",
             "|---|---|---|---|---|"]
    for row in sel:
        name = row["label"][len(prefix):].lstrip("_") or row["label"]
        lines.append(
            f"| `{name}` | {fmt(row['accuracy_mean'], row['accuracy_std'])} | "
            f"{(row['accuracy_mean'] - top) * 100:+.2f} pp | {row['macro_f1_mean']:.4f} | "
            f"{row['train_minutes_mean']:.1f} {t['min']} |")
    return "\n".join(lines)


def primary_run(runs: list[dict], row: dict) -> dict | None:
    """The primary-seed run of a configuration.

    Never `max(runs, key=accuracy)`: that would report the luckiest seed of the
    luckiest configuration, which is the precise form of cherry-picking the seed
    sweep exists to prevent. The configuration is chosen by its mean across seeds;
    the artefacts (confusion matrix, McNemar) then come from its seed-42 run.
    """
    for run in runs:
        if (run["strategy"] == row["strategy"] and run["label"] == row["label"]
                and run["seed"] == common.SEED):
            return run
    return None


def top2_mcnemar(runs: list[dict], rows: list[dict], y_test: np.ndarray,
                 lang: str) -> str:
    """Paired significance test between the two best *strategies*.

    The two arms are the best configuration of each of the two leading strategies —
    not simply the two highest-scoring runs, which could easily be two variants of
    the same strategy and would answer a question nobody asked.

    McNemar, not a t-test over runs: both models are scored on the *same* 10,000
    images, so their errors are paired, and the correct question is whether the
    discordant predictions split more unevenly than a fair coin would produce.
    """
    t = L[lang]
    leaders = best_per_strategy(rows)[:2]
    if len(leaders) < 2:
        return t["need_two"]

    a, b = (primary_run(runs, leaders[0]), primary_run(runs, leaders[1]))
    if a is None or b is None:
        return t["need_two"]
    test = common.mcnemar(y_test, np.array(a["test_predictions"]),
                          np.array(b["test_predictions"]))
    gap = (a["accuracy"] - b["accuracy"]) * 100
    verdict = t["significant"] if test["significant_at_05"] else t["not_significant"]
    key = "verdict_sig" if test["significant_at_05"] else "verdict_tie"

    return "\n".join([
        f"- **1º** `{a['strategy']} / {a['label']}` — {a['accuracy']:.4f}",
        f"- **2º** `{b['strategy']} / {b['label']}` — {b['accuracy']:.4f}",
        "",
        f"| {t['discordant']} | {t['a_right']} | {t['b_right']} | {t['pvalue']} |",
        "|---|---|---|---|",
        f"| {test['n_discordant']} | {test['n_a_correct_b_wrong']} | "
        f"{test['n_a_wrong_b_correct']} | {test['p_value']:.4g} |",
        "",
        t[key].format(gap=gap, verdict=verdict),
    ])


def pairwise_strategies(runs: list[dict], rows: list[dict], y_test: np.ndarray,
                        lang: str) -> str:
    """Every strategy against every other, paired, at the primary seed.

    The top-2 test alone answers "who won". These pairs answer the questions the
    assignment is really about — each one is an increment of transfer, and each has a
    price:

        s2 vs s1   does a frozen ImageNet backbone beat learning from scratch?
        s3 vs s2   does unfreezing the top block buy anything over frozen features?
        s4 vs s3   does augmentation buy anything over plain fine-tuning?
        s5 vs s4   does a transformer beat the convolutional stack?

    A gap that fails McNemar is a gap that was paid for and not received. That is the
    most useful thing this report can tell a reader, and it is invisible in an accuracy
    ranking.
    """
    t = L[lang]
    leaders = best_per_strategy(rows)
    by_strategy = {row["strategy"]: primary_run(runs, row) for row in leaders}
    order = ["s1_scratch", "s2_features", "s3_finetune", "s4_augment", "s5_vit"]
    present = [s for s in order if by_strategy.get(s) is not None]
    if len(present) < 2:
        return t["need_two"]

    lines = [f"| {t['pair']} | Δ | {t['pvalue_short']} | {t['verdict_col']} |",
             "|---|---|---|---|"]
    for i, later in enumerate(present):
        for earlier in present[:i]:
            a, b = by_strategy[later], by_strategy[earlier]
            test = common.mcnemar(y_test, np.array(a["test_predictions"]),
                                  np.array(b["test_predictions"]))
            gap = (a["accuracy"] - b["accuracy"]) * 100
            name_a = STRATEGY_NAMES[lang][later].split(" — ")[0]
            name_b = STRATEGY_NAMES[lang][earlier].split(" — ")[0]
            verdict = t["yes"] if test["significant_at_05"] else t["no"]
            lines.append(f"| Estratégia {name_a} vs. {name_b} | {gap:+.2f} pp | "
                         f"{test['p_value']:.3g} | {verdict} |"
                         if lang == "pt" else
                         f"| Strategy {name_a} vs. {name_b} | {gap:+.2f} pp | "
                         f"{test['p_value']:.3g} | {verdict} |")
    return "\n".join(lines)


def cost_table(rows: list[dict], lang: str) -> str:
    """Accuracy per GPU-minute. The ranking nobody asks for and everybody needs.

    Reported because the assignment's question ("which strategy is best?") has no answer
    without a cost axis: a strategy that wins by 0.1 pp at 20x the compute has not won
    anything a practitioner would buy.
    """
    t = L[lang]
    leaders = best_per_strategy(rows)
    if not leaders:
        return t["need_two"]

    head = ("| Estratégia | Acurácia | Treino | pp acima da CNN do zero | pp por minuto |"
            if lang == "pt" else
            "| Strategy | Accuracy | Training | pp over from-scratch CNN | pp per minute |")
    lines = [head, "|---|---|---|---|---|"]

    scratch = next((r for r in leaders if r["strategy"] == "s1_scratch"), None)
    floor = scratch["accuracy_mean"] if scratch else 0.0

    for row in leaders:
        minutes = max(row["train_minutes_mean"], 0.01)
        lift = (row["accuracy_mean"] - floor) * 100
        name = STRATEGY_NAMES[lang].get(row["strategy"], row["strategy"])
        lines.append(f"| {name} | {row['accuracy_mean']:.4f} | "
                     f"{minutes:.1f} min | {lift:+.2f} pp | {lift / minutes:+.2f} |")
    return "\n".join(lines)


def localise_decimals(text: str) -> str:
    """English number formatting -> pt-BR. Both separators swap, and order matters:
    `2,171,722` is an English thousands comma that a Portuguese reader parses as a
    decimal. Thousands groups are parked on a sentinel so the decimal pass cannot chew
    the dots it just created."""
    sentinel = "\x00"
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", sentinel, text)
    text = re.sub(r"(?<=\d)\.(?=\d)", ",", text)
    return text.replace(sentinel, ".")


def hardest_classes(run: dict, lang: str) -> str:
    """Where the best model actually fails. The off-diagonal is the informative part
    of a confusion matrix; the diagonal only restates the score."""
    t = L[lang]
    cm = np.array(run["confusion_matrix"], dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True)
    np.fill_diagonal(norm, 0.0)

    pairs = sorted(((norm[i, j], i, j) for i in range(common.N_CLASSES)
                    for j in range(common.N_CLASSES) if i != j), reverse=True)
    lines = [f"| {t['confusion']} | {t['rate']} | {t['reading']} |", "|---|---|---|"]
    for rate, i, j in pairs[:5]:
        lines.append(f"| {common.CLASS_NAMES[i]} → {common.CLASS_NAMES[j]} | "
                     f"{rate * 100:.1f}% | |")
    return "\n".join(lines)


def confusion_matrix_png(run: dict, theme: str, lang: str = "pt") -> Path:
    """Confusion matrix of the best model — the artefact the assignment asks for.

    Row-normalised, so each cell reads as 'of all true X, this fraction was called Y'.
    Raw counts would let the eye compare classes of different sizes, which here they
    are not (CIFAR-10's test set is perfectly balanced) but which is a habit worth
    not forming.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = L[lang]
    dark = theme == "dark"
    fg = "#e6edf3" if dark else "#1f2328"
    bg = "#0d1117" if dark else "#ffffff"

    cm = np.array(run["confusion_matrix"], dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8.5, 7.5), facecolor=bg)
    ax.set_facecolor(bg)
    im = ax.imshow(norm, cmap="viridis" if dark else "Blues", vmin=0, vmax=1)

    ax.set_xticks(range(common.N_CLASSES), common.CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(common.N_CLASSES), common.CLASS_NAMES)
    ax.set_xlabel(t["cm_x"], color=fg, labelpad=10)
    ax.set_ylabel(t["cm_y"], color=fg, labelpad=10)
    ax.set_title(t["cm_title"].format(
        label=f"{run['strategy']}/{run['label']}", acc=run["accuracy"]), color=fg, pad=14)

    for i in range(common.N_CLASSES):
        for j in range(common.N_CLASSES):
            if norm[i, j] < 0.005:
                continue
            ax.text(j, i, f"{norm[i, j] * 100:.0f}", ha="center", va="center",
                    fontsize=8, color="white" if norm[i, j] > 0.5 else fg)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046)
    cbar.ax.tick_params(colors=fg)
    ax.tick_params(colors=fg)
    for spine in ax.spines.values():
        spine.set_color(fg)

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    path = IMG_DIR / f"avaliacao-pratica-1-confusion-{'dark' if dark else 'light'}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=bg)
    plt.close(fig)
    return path


def patch(markdown: str, key: str, content: str) -> str:
    begin, end = BEGIN.format(key), END.format(key)
    block = f"{begin}\n{content}\n{end}"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    if pattern.search(markdown):
        return pattern.sub(block, markdown)
    return markdown + f"\n\n{block}\n"


def build_sections(runs: list[dict], rows: list[dict], y_test: np.ndarray,
                   best_run: dict, lang: str) -> dict[str, str]:
    t = L[lang]
    return {
        "main-table": table_main(best_per_strategy(rows), runs, lang),
        "ablation-backbone": table_ablation(rows, "s2_features", "", t["backbone"], lang),
        "ablation-head": table_ablation(rows, "s3_finetune", "", t["head"], lang),
        "ablation-optimizer": table_ablation(rows, "s4_augment", "optimizer_",
                                             t["optimizer"], lang),
        "ablation-policy": table_ablation(rows, "s4_augment", "policy_", t["policy"], lang),
        "significance": top2_mcnemar(runs, rows, y_test, lang),
        "pairwise": pairwise_strategies(runs, rows, y_test, lang),
        "cost": cost_table(rows, lang),
        "hardest-classes": hardest_classes(best_run, lang),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    runs = common.load_results()
    if not runs:
        raise SystemExit("no results found — run the strategies first")

    rows = common.aggregate(runs)
    y_test = common.load_splits().y_test

    # The reported model is the configuration with the best *mean* accuracy, shown at
    # the primary seed — not the single luckiest run in the sweep.
    leader = best_per_strategy(rows)[0]
    best_run = primary_run(runs, leader) or max(runs, key=lambda r: r["accuracy"])
    print(f"{len(runs)} runs · best: {best_run['strategy']}/{best_run['label']} "
          f"@ {best_run['accuracy']:.4f} (seed {best_run['seed']}; "
          f"mean {leader['accuracy_mean']:.4f})")

    for theme in ("light", "dark"):
        print(f"wrote {confusion_matrix_png(best_run, theme)}")

    for lang, target in TARGETS.items():
        sections = build_sections(runs, rows, y_test, best_run, lang)
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
