"""사례집 서식의 지문(fingerprint)을 뜬다.

2024년판을 투입한 순간 결론 검출률이 97%에서 38%로 떨어졌다. 원인은
2025년판 한 권만 보고 체크 표시를 U+2611 하나로 가정한 것이었고, 실제로는
한 권 안에 여섯 가지 서식이 섞여 있었다.

새 연도를 넣을 때마다 같은 일이 조용히 일어날 수 있다. 파서는 예외를 내지
않고 그냥 빈 값을 남기기 때문이다. 그래서 서식을 지문으로 떠서 고정해 두고,
지문이 달라지면 테스트가 깨지게 한다.

    python scripts/format_inventory.py \
        --input data/raw/casebooks --output tests/regression/baseline.json
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata as ud
from collections import Counter
from pathlib import Path

import pymupdf
from parse_casebook import (  # noqa: E402
    INTERP_FIELDS,
    NONACTION_FIELDS,
    parse_pdf,
)

# 체크 표시로 쓰일 수 있는 글자. 새 글자가 나오면 잡아낸다.
CHECK_GLYPHS = "☑☒þ■▣◉√✓Vv●"
RE_HEAD_BLOCK = re.compile(r"비조치의견서\s*\(([^)]{0,60})")


NO_CHECKBOX = "(헤더에 체크박스 없음)"


def normalize_pattern(raw: str) -> str:
    """헤더에서 체크박스 삼항(비조치/조치/기타)만 잘라낸다.

    닫는 괄호가 빠진 헤더가 있어 뒤따르는 일련번호까지 딸려 온다.
    '기타' 에서 끊지 않으면 일련번호마다 다른 서식으로 집계된다.
    """
    squeezed = re.sub(r"[\s\x00-\x1f]", "", raw)
    idx = squeezed.find("기타")
    if idx == -1:
        return NO_CHECKBOX
    return squeezed[: idx + 2]


def checkbox_patterns(path: Path) -> Counter:
    """헤더의 체크박스 서식을 원형 그대로 센다."""
    name = ud.normalize("NFC", path.name)
    if "비조치" not in name:
        return Counter()
    doc = pymupdf.open(path)
    full = "".join(p.get_text() for p in doc)
    return Counter(
        normalize_pattern(m.group(1)) for m in RE_HEAD_BLOCK.finditer(full)
    )


def build(input_dir: Path) -> dict:
    pdfs = [p for p in sorted(input_dir.rglob("*.pdf")) if "__MACOSX" not in str(p)]
    sources: dict[str, dict] = {}
    all_patterns: Counter = Counter()

    for path in pdfs:
        name = ud.normalize("NFC", path.name)
        cases = parse_pdf(path)
        doc_type = cases[0].doc_type if cases else "unknown"
        fields = NONACTION_FIELDS if doc_type == "nonaction" else INTERP_FIELDS

        n = len(cases)
        field_rates = {
            f: round(sum(1 for c in cases if c.fields.get(f, "").strip()) / n, 4)
            for f in fields
        }
        entry = {
            "doc_type": doc_type,
            "case_count": n,
            "serial_rate": round(sum(1 for c in cases if c.serial) / n, 4),
            "sector_rate": round(sum(1 for c in cases if c.sector) / n, 4),
            "field_rates": field_rates,
        }
        if doc_type == "nonaction":
            entry["decision_rate"] = round(
                sum(1 for c in cases if c.decision) / n, 4
            )
            pats = checkbox_patterns(path)
            entry["checkbox_patterns"] = dict(pats.most_common())
            all_patterns.update(pats)
        sources[name] = entry

    return {
        "sources": sources,
        "known_checkbox_patterns": sorted(all_patterns),
        "known_check_glyphs": sorted(
            {ch for pat in all_patterns for ch in pat if ch in CHECK_GLYPHS}
        ),
        "field_names": {
            "nonaction": NONACTION_FIELDS,
            "interpretation": INTERP_FIELDS,
        },
        # 회귀 판정 기준. 실측치보다 약간 낮게 잡아 잡음에는 안 깨지고
        # 서식 변화에는 깨지도록 한다.
        "thresholds": {
            "decision_rate_min": 0.90,
            "serial_rate_min": 0.95,
            "sector_rate_min": 0.95,
            "field_rate_min": 0.80,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    inv = build(Path(args.input))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(inv, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"자료 {len(inv['sources'])}건")
    for name, e in inv["sources"].items():
        extra = (
            f" 결론 {e['decision_rate']:.0%}" if "decision_rate" in e else ""
        )
        print(f"  {e['case_count']:4d}건  {name}{extra}")
    print(f"\n알려진 체크박스 서식 {len(inv['known_checkbox_patterns'])}종")
    for p in inv["known_checkbox_patterns"]:
        print(f"  {p}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
