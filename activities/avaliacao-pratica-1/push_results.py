"""
Push the result artefacts straight to GitHub from inside the run.

Written because the recovery loop — wait for the run, download a zip, hand it over — needs
a human awake at the end of it, and this experiment has already lost two hours of GPU to a
session reclaimed while nobody was watching.

With this, a Kaggle **batch** run (Save Version → Save & Run All) executes on Kaggle's
servers with the browser closed, and commits its own results to a branch. Nothing needs to
be downloaded, and nothing is lost if the session is reclaimed after the push.

Only the JSONs and the confusion-matrix PNGs are pushed: small, they are the evidence, and
they regenerate every table and test on a CPU. Weights are not — nothing in the report
depends on them.

Requires a GitHub token with `repo` scope, supplied as the environment variable
`GITHUB_TOKEN` (on Kaggle: Add-ons → Secrets). The token is never written to disk, never
committed, and is stripped from the remote URL before the script exits.

    python push_results.py --branch results/ap1-ablations
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import common

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_SLUG = "fsd-dantas/machine-learning-fundamentals"
ROOT = common.HERE.parents[1]


def git(*args: str, check: bool = True) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True,
                            capture_output=True, check=False)
    if check and result.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{result.stderr}")
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", default="results/ap1-ablations")
    parser.add_argument("--message", default=None)
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "GITHUB_TOKEN not set.\n"
            "On Kaggle: Add-ons → Secrets → add GITHUB_TOKEN (a GitHub PAT with `repo` "
            "scope), then attach it to this notebook.")

    artefacts = sorted(common.RESULTS_DIR.glob("*.json"))
    if not artefacts:
        raise SystemExit("results/ is empty — nothing to push")

    git("config", "user.email", "kaggle@runner.local")
    git("config", "user.name", "Kaggle batch runner")

    git("checkout", "-B", args.branch)
    git("add", "activities/avaliacao-pratica-1/results")
    git("add", "assets/img", check=False)

    if not git("status", "--porcelain"):
        print("nothing new to commit — the artefacts already match the repository")
        return

    message = args.message or (
        f"results: Avaliação Prática 1 — {len(artefacts)} artefacts from a Kaggle batch run"
    )
    git("commit", "-m", message)

    # The token lives only in this process's argv for the length of the push, and the
    # remote is reset immediately afterwards — a pushed URL containing a credential is a
    # credential leaked into the repository's config.
    authed = f"https://x-access-token:{token}@github.com/{REPO_SLUG}.git"
    try:
        git("push", "--force-with-lease", authed, f"HEAD:{args.branch}")
    finally:
        git("remote", "set-url", "origin",
            f"https://github.com/{REPO_SLUG}.git", check=False)

    print(f"\npushed {len(artefacts)} artefacts to branch `{args.branch}`")
    print(f"  https://github.com/{REPO_SLUG}/tree/{args.branch}")


if __name__ == "__main__":
    main()
