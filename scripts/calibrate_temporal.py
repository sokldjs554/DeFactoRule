#!/usr/bin/env python3
"""T-serial 조건의 clean-dev 선례 위험도를 재보정한다."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.temporal_calibration import calibrate_temporal  # noqa: E402
from app.core.io import load_jsonl  # noqa: E402
from app.core.paths import EVAL, PROCESSED, RESULTS  # noqa: E402
from app.evaluation.confusable import idf_table  # noqa: E402


def main() -> None:
    dev_path = EVAL / "nonaction_dev_clean.jsonl"
    corpus_path = PROCESSED / "cases_nonaction.jsonl"
    output = RESULTS / "trap_risk_clean_temporal.json"

    dev = [row for row in load_jsonl(dev_path) if row.get("label")]
    cases = load_jsonl(corpus_path)
    texts = [
        case["fields"].get("요청대상행위")
        or case["fields"].get("질의요지")
        or ""
        for case in cases
    ]
    report = calibrate_temporal(
        dev,
        idf_table([text for text in texts if text]),
        policy="serial",
    )
    report["idf_source"] = f"{corpus_path.name} {len(cases)}건"

    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"dev={report['n_dev']} eligible={report['n_with_eligible_precedent']} "
        f"missing={report['n_without_eligible_precedent']}"
    )
    for band, cell in report["by_band"].items():
        print(
            f"{band}: n={cell['n']} wrong={cell['wrong']} "
            f"risk={cell['risk']} ci95={cell['ci95']}"
        )
    print(f"bands_separable={report['bands_separable']}")
    print(f"-> {output}")


if __name__ == "__main__":
    main()
