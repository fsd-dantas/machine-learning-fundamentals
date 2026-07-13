"""
Package the result artefacts for download — run this OFTEN, not once at the end.

A free-tier session can be reclaimed at any moment, and everything not yet downloaded dies
with it. That is not hypothetical: this experiment lost 23 artefacts and roughly two hours
of GPU time to exactly that, because the download lived in the last cell of the notebook.

Only JSONs and PNGs are packaged. They are small, they are the evidence, and they are
enough to regenerate every table, confusion matrix and significance test on a CPU. Trained
weights are not: they are large, and nothing in the report depends on them.

    %run -i baixar.py     (in the notebook)
    python baixar.py      (from a shell)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import common

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IMG_DIR = common.HERE.parents[1] / "assets" / "img"


def main() -> None:
    artefacts = sorted(common.RESULTS_DIR.glob("*.json"))
    if not artefacts:
        print("nothing to package — results/ is empty")
        return

    # Kaggle exposes /kaggle/working in the Output tab; Colab downloads from /content.
    root = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("/content")
    if not root.is_dir():
        root = common.HERE

    staging = root / "ap1-results"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "results").mkdir(parents=True)

    for path in artefacts:
        shutil.copy(path, staging / "results" / path.name)

    images = list(IMG_DIR.glob("avaliacao-pratica-1-*.png"))
    if images:
        (staging / "img").mkdir()
        for path in images:
            shutil.copy(path, staging / "img" / path.name)

    archive = shutil.make_archive(str(root / "ap1-results"), "zip", staging)
    size = Path(archive).stat().st_size / 1024

    print(f"{archive}  ({size:.0f} KB)")
    print(f"  {len(artefacts)} result artefacts · {len(images)} images")

    by_strategy: dict[str, int] = {}
    for path in artefacts:
        by_strategy[path.name.split("__")[0]] = by_strategy.get(
            path.name.split("__")[0], 0) + 1
    for strategy, count in sorted(by_strategy.items()):
        print(f"    {strategy:14s} {count}")

    if root.name == "content":   # Colab can push the file straight to the browser
        try:
            from google.colab import files
            files.download(archive)
        except ImportError:
            pass
    else:
        print("\n  download it from the Output tab (right-hand panel)")


if __name__ == "__main__":
    main()
