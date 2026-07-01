from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVITY_SCRIPT = ROOT / "activities" / "atividade-1.py"
ACTIVITY_REPORT = ROOT / "activities" / "atividade-1.md"


EXPECTED_RESULT_ANCHORS = [
    ("SVM/MLP accuracy", "0.9772", "0.9772"),
    ("SVM F1", "0.9754", "0.9754"),
    ("Ridge R2", "0.4791", "0.4791"),
    ("MLP MAE", "44.0493", "44.05"),
    ("Wilcoxon p-value", "p=0.8750", "p = 0.875"),
]


def main() -> int:
    result = subprocess.run(
        [sys.executable, str(ACTIVITY_SCRIPT)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        return result.returncode

    report = ACTIVITY_REPORT.read_text(encoding="utf-8")
    missing = []
    for label, output_value, report_value in EXPECTED_RESULT_ANCHORS:
        if output_value not in result.stdout:
            missing.append(f"{label} missing from generated output")
        if report_value not in report:
            missing.append(f"{label} missing from Markdown report")

    if missing:
        sys.stderr.write(
            "Activity result validation failed. Missing expected values: "
            + ", ".join(missing)
            + "\n"
        )
        return 1

    print("Activity results regenerated and report anchors are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
