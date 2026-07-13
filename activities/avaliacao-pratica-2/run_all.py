"""
Orchestrator — every model, every task, in dependency order.

This corpus is small (2,436 texts), so the whole sweep is cheap: roughly 25 minutes on a
T4, and the baselines alone run on a CPU in seconds. Unlike Avaliação Prática 1, nothing
here needs rationing.

    core        both tasks x {majority, TF-IDF+SVM, LSTM, BERT}, primary seed   ~8 min
    seeds       the same, at seeds 7 and 2024                                   ~15 min
    sensitivity the binary task without `neutro`/`surpresa`                     ~3 min
    extras      BiLSTM — does bidirectionality rescue the recurrent arm?        ~3 min
    report      tables, confusion matrices, McNemar                             ~5 s

Results accumulate in `results/` as JSON; a stage that already ran is never repeated.

Usage:
    python run_all.py --stage core
    python run_all.py --all
    python run_all.py --all --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

import common

TASKS = ("binary", "multiclass")

STAGES: dict[str, list[list[str]]] = {
    "core": [
        cmd
        for task in TASKS
        for cmd in (
            ["m0_baselines.py", "--task", task],   # majority + TF-IDF/SVM together
            ["m1_lstm.py", "--task", task],
            ["m2_bert.py", "--task", task],
        )
    ],
    # Dispersion. A single 70/30 holdout on ~730 test texts has a ~1.5 pp standard
    # error: one split can rank two models by luck. Three seeds is what lets the report
    # separate a real gap from a lucky one.
    "seeds": [
        cmd + ["--seed", str(seed)]
        for seed in common.SEEDS[1:]
        for task in TASKS
        for cmd in (
            ["m0_baselines.py", "--task", task],
            ["m1_lstm.py", "--task", task],
            ["m2_bert.py", "--task", task],
        )
    ],
    # Does the binary conclusion depend on the instructor's decision to call `neutro`
    # and `surpresa` positive? If the LSTM-vs-BERT verdict is the same without them,
    # the finding is robust to a labelling choice we did not make.
    "sensitivity": [
        cmd
        for cmd in (
            ["m0_baselines.py", "--task", "binary_valence"],
            ["m1_lstm.py", "--task", "binary_valence"],
            ["m2_bert.py", "--task", "binary_valence"],
        )
    ],
    "extras": [
        ["m1_lstm.py", "--task", task, "--bidirectional"] for task in TASKS
    ],
    "report": [["report.py"]],
}

ORDER = ["core", "seeds", "sensitivity", "extras", "report"]


def run(cmd: list[str], *, dry_run: bool) -> bool:
    print(f"\n{'='*78}\n>>> python {' '.join(cmd)}\n{'='*78}", flush=True)
    if dry_run:
        return True
    started = time.perf_counter()
    result = subprocess.run([sys.executable, *cmd], cwd=common.HERE, check=False)
    elapsed = (time.perf_counter() - started) / 60
    if result.returncode != 0:
        print(f"!!! FAILED after {elapsed:.1f} min", file=sys.stderr)
        return False
    print(f"<<< done in {elapsed:.1f} min", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=ORDER, action="append", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    stages = ORDER if args.all else (args.stage or ["core"])
    started = time.perf_counter()
    failures: list[str] = []

    for stage in stages:
        print(f"\n########## STAGE: {stage} ##########")
        for cmd in STAGES[stage]:
            if not run(cmd, dry_run=args.dry_run):
                failures.append(" ".join(cmd))
                if not args.keep_going:
                    raise SystemExit(1)

    print(f"\n{'='*78}\ntotal: {(time.perf_counter()-started)/60:.1f} min · "
          f"stages: {', '.join(stages)}")
    if failures:
        for failure in failures:
            print(f"  FAILED: {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
