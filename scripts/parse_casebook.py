"""금융당국 사례집 PDF를 사례 단위 JSONL 로 구조화한다.

두 종류의 서식을 다룬다.

  비조치의견서 사례집 (금융감독원)
      비조치의견서(☑비조치□조치□기타)
      (일련번호 250027)
      요청대상 행위 / 판단 / 판단 이유

  법령해석 회신문 사례집 (금융위원회)
      법령해석 회신문(230339)
      질의요지 / 회답 / 이유

결론 라벨의 출처가 서로 다르다. 비조치의견서는 체크박스가 문서에 박혀 있어
그대로 읽으면 되지만, 법령해석은 회답 본문의 문장에서 판단해야 한다.
이 스크립트는 전자만 확정 라벨로 기록하고, 후자는 원문만 보존한다 —
회답의 함의 판정은 별도 단계(LLM)의 일이며, 여기서 규칙으로 때려 넣으면
그 단계의 성능을 측정할 수 없게 된다.

    python scripts/parse_casebook.py --input data/raw/casebooks --output data/processed
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata as ud
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf

# 사례 헤더
RE_NONACTION_HEAD = re.compile(r"비조치의견서\s*\(([^)]{0,40})")
RE_INTERP_HEAD = re.compile(r"법령해석\s*회신문\s*\(\s*(\d{5,7})\s*\)")
# 일련번호 사이에 제어문자가 끼어 있는 경우가 있다
RE_SERIAL = re.compile(r"일련번호[\s\x00-\x1f]*(\d{5,7})")
# 체크 표시 서식이 발간 연도마다 다르다. 2024년판 한 권 안에만 여섯 가지가 섞여 있다.
#   ☑비조치□조치□기타      유니코드 체크박스
#   þ비조치□조치□기타      Wingdings 의 체크된 상자가 'þ' 로 추출됨
#   √□비조치□조치□기타    체크 표시가 상자 '앞'에 온다
#   □비조치√□조치□기타    〃
#   □비조치□조치☑기타
# '조치' 는 '비조치' 의 부분문자열이므로 대안 순서를 길이 내림차순으로 둔다.
RE_DECISION = re.compile(r"(?:[☑☒þ■▣◉]|[√✓Vv]\s*□)\s*(비조치|조치|기타)")

# 업권 구분 페이지. 발간 주체마다 서식이 다르다.
#   금융위 대분류: "공통\n1\n법령해석 회신문 사례집[2025년도]"   (이름 → 번호)
#   금융위 소분류: "•∙<장식문자>∙•\n금융정책 일반"
#   금감원:        "1. 공통"                                     (번호 → 이름)
RE_SECTOR_MAJOR_NAME_FIRST = re.compile(r"^\s*([가-힣]{2,10})\s*\n\s*(\d)\s*$", re.MULTILINE)
# 2022년 비조치의견서판: "공 통\n2022년\n비조치의견서 사례집" — 이름 뒤에 연도가 온다.
# 제목이 자간을 벌려 조판돼 "공 통" 처럼 글자 사이에 공백이 들어간다.
RE_SECTOR_NAME_THEN_YEAR = re.compile(
    r"^\s*([가-힣][가-힣\s]{1,14})\s*\n\s*\d{4}\s*년\s*$", re.MULTILINE
)
RE_SECTOR_NUM_FIRST = re.compile(r"^\s*\d\s*[.．]\s*([가-힣·∙\s]{2,20})\s*$", re.MULTILINE)
RE_SECTOR_MINOR = re.compile(
    r"^[•∙\s\ue000-\uf8ff]+\n\s*([가-힣]{2,12}(?:\s[가-힣]{2,6})?)\s*$", re.MULTILINE
)
# 구분 페이지는 아주 짧다. 넉넉히 잡으면 본문 첫 페이지가 업권으로 오인된다
# ("1. 기초서류에서 정하는 방법에 따른 경우" 가 업권으로 잡힌 사례가 있었다).
SECTOR_PAGE_MAX_CHARS = 90

# 같은 업권을 연도마다 다르게 적는다. 표기를 하나로 모은다.
SECTOR_ALIASES = {
    "상호저축은행": "상호저축은행업",
    "여신전문금융": "여신전문금융업",
    "온라인투자연계금융": "온라인투자연계금융업",
}


def canonical_sector(name: str | None) -> str | None:
    return SECTOR_ALIASES.get(name, name) if name else name


NONACTION_FIELDS = ["요청대상행위", "판단", "판단이유"]
INTERP_FIELDS = ["질의요지", "회답", "이유"]


@dataclass
class Case:
    source: str
    doc_type: str  # nonaction | interpretation
    serial: str | None
    sector: str | None
    subsector: str | None
    page: int
    decision: str | None  # 단일 체크일 때만. 복수 체크면 None
    decisions: list[str] = field(default_factory=list)  # 체크된 것 전부
    fields: dict[str, str] = field(default_factory=dict)
    raw: str = ""
    warnings: list[str] = field(default_factory=list)


def squeeze(s: str) -> str:
    """제어문자를 걷어내고 공백을 정리한다. 글자 자체는 건드리지 않는다."""
    s = "".join(ch for ch in s if ch == "\n" or ord(ch) >= 32)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def spacing_lost(text: str, sample: int = 4000) -> bool:
    """추출 과정에서 띄어쓰기가 소실됐는지 판정한다.

    비조치의견서 PDF 는 한글 사이 공백이 통째로 사라진 채 추출된다.
    다운스트림 토크나이저에 그대로 넣으면 안 되므로 사례마다 표시해 둔다.
    """
    head = text[:sample]
    hangul = sum(1 for c in head if "가" <= c <= "힣")
    if hangul < 200:
        return False
    return (head.count(" ") / hangul) < 0.05


def page_sector_map(doc: pymupdf.Document) -> dict[int, tuple[str | None, str | None]]:
    """업권 구분 페이지를 찾아 이후 본문 페이지에 (대분류, 소분류)를 상속시킨다.

    소분류 구분 페이지를 만나면 소분류만 갈아끼우고, 대분류를 만나면 소분류는 지운다.
    """
    major_marks: dict[int, str] = {}
    minor_marks: dict[int, str] = {}
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if len(text) > SECTOR_PAGE_MAX_CHARS:  # 본문 페이지
            continue
        m = RE_SECTOR_MAJOR_NAME_FIRST.search(text)
        if m:
            major_marks[i] = re.sub(r"\s+", "", m.group(1))
            continue
        m = RE_SECTOR_NAME_THEN_YEAR.search(text)
        if m:
            major_marks[i] = re.sub(r"\s+", "", m.group(1))
            continue
        m = RE_SECTOR_NUM_FIRST.search(text)
        if m:
            major_marks[i] = re.sub(r"\s+", "", m.group(1))
            continue
        m = RE_SECTOR_MINOR.search(text)
        if m:
            minor_marks[i] = re.sub(r"\s+", " ", m.group(1)).strip()

    out: dict[int, tuple[str | None, str | None]] = {}
    major = minor = None
    for i in range(doc.page_count):
        if i in major_marks:
            major, minor = major_marks[i], None
        if i in minor_marks:
            minor = minor_marks[i]
        out[i] = (canonical_sector(major), canonical_sector(minor))
    return out


def split_fields(body: str, names: list[str]) -> tuple[dict[str, str], list[str]]:
    """항목명이 단독 줄로 등장하는 것을 기준으로 본문을 쪼갠다.

    항목명 안에 줄바꿈이 들어간 경우("요청대상\\n행위")가 있어
    공백을 제거한 뒤 비교한다.
    """
    lines = body.split("\n")
    marks: list[tuple[int, str]] = []
    # "판단" 과 "판단이유" 가 함께 있을 때 짧은 쪽이 먼저 걸리면 긴 쪽을 영영 못 잡는다.
    # 줄 span 은 넓은 것부터, 이름은 긴 것부터 본다.
    ordered = sorted(names, key=len, reverse=True)
    i = 0
    while i < len(lines):
        matched = 0
        for span in (3, 2, 1):
            joined = re.sub(r"\s+", "", "".join(lines[i : i + span]))
            hit = next((n for n in ordered if joined == n), None)
            if hit:
                marks.append((i, hit))
                matched = span
                break
        i += matched if matched else 1

    result: dict[str, str] = {}
    warnings: list[str] = []
    for idx, (line_no, name) in enumerate(marks):
        end = marks[idx + 1][0] if idx + 1 < len(marks) else len(lines)
        # 항목명 줄 자체는 건너뛴다
        start = line_no + 1
        while start < end and not re.sub(r"\s+", "", lines[start]):
            start += 1
        value = squeeze("\n".join(lines[start:end]))
        if name in result:
            result[name] += "\n" + value
        else:
            result[name] = value
    for name in names:
        if name not in result:
            warnings.append(f"missing_field:{name}")
    return result, warnings


def parse_nonaction(doc: pymupdf.Document, source: str) -> list[Case]:
    sectors = page_sector_map(doc)
    pages = [(i, p.get_text()) for i, p in enumerate(doc)]
    full = "".join(t for _, t in pages)
    # 페이지 오프셋 → 페이지 번호 역산용
    offsets, acc = [], 0
    for i, t in pages:
        offsets.append((acc, i))
        acc += len(t)

    def page_of(pos: int) -> int:
        last = 0
        for off, idx in offsets:
            if off > pos:
                return last
            last = idx
        return last

    starts = [m.start() for m in RE_NONACTION_HEAD.finditer(full)]
    cases: list[Case] = []
    for n, s in enumerate(starts):
        e = starts[n + 1] if n + 1 < len(starts) else len(full)
        chunk = full[s:e]
        serial = RE_SERIAL.search(chunk)
        # 헤더 구간에서만 찾는다. 본문에 같은 낱말이 나와도 걸리지 않게 한다.
        header = chunk[:200]
        found = list(dict.fromkeys(m.group(1) for m in RE_DECISION.finditer(header)))
        fields, warns = split_fields(chunk, NONACTION_FIELDS)
        if not serial:
            warns.append("missing_serial")
        if not found:
            warns.append("missing_decision")
        elif len(found) > 1:
            # 한 건에 두 결론이 함께 표시된 사례가 실재한다. 임의로 하나를 고르면
            # 라벨이 조용히 오염되므로 단일 라벨은 비우고 표시만 남긴다.
            warns.append("multi_decision")
        if spacing_lost(chunk):
            warns.append("spacing_lost")
        pg = page_of(s)
        cases.append(
            Case(
                source=source,
                doc_type="nonaction",
                serial=serial.group(1) if serial else None,
                sector=sectors.get(pg, (None, None))[0],
                subsector=sectors.get(pg, (None, None))[1],
                page=pg + 1,
                decision=found[0] if len(found) == 1 else None,
                decisions=found,
                fields=fields,
                raw=squeeze(chunk),
                warnings=warns,
            )
        )
    return cases


def parse_interpretation(doc: pymupdf.Document, source: str) -> list[Case]:
    sectors = page_sector_map(doc)
    cases: list[Case] = []
    for i, page in enumerate(doc):
        text = page.get_text()
        heads = list(RE_INTERP_HEAD.finditer(text))
        for n, m in enumerate(heads):
            end = heads[n + 1].start() if n + 1 < len(heads) else len(text)
            chunk = text[m.start() : end]
            fields, warns = split_fields(chunk, INTERP_FIELDS)
            if spacing_lost(chunk):
                warns.append("spacing_lost")
            cases.append(
                Case(
                    source=source,
                    doc_type="interpretation",
                    serial=m.group(1),
                    sector=sectors.get(i, (None, None))[0],
                    subsector=sectors.get(i, (None, None))[1],
                    page=i + 1,
                    decision=None,  # 회답 본문에서 별도 판정 — 여기서 규칙으로 정하지 않는다
                    decisions=[],
                    fields=fields,
                    raw=squeeze(chunk),
                    warnings=warns,
                )
            )
    return cases


def parse_pdf(path: Path) -> list[Case]:
    name = ud.normalize("NFC", path.name)
    doc = pymupdf.open(path)
    if "비조치" in name:
        return parse_nonaction(doc, name)
    return parse_interpretation(doc, name)


def report(cases: list[Case]) -> None:
    from collections import Counter

    by_type = Counter(c.doc_type for c in cases)
    print(f"\n총 사례: {len(cases)}")
    for k, v in by_type.items():
        print(f"  {k}: {v}")

    na = [c for c in cases if c.doc_type == "nonaction"]
    if na:
        print("\n비조치의견서 결론 분포")
        for k, v in Counter(c.decision for c in na).most_common():
            print(f"  {k or '(단일 라벨 없음)'}: {v}")
        multi = [c for c in na if len(c.decisions) > 1]
        if multi:
            print(f"  └ 복수 체크 {len(multi)}건: " + ", ".join(
                f"{c.serial}={'+'.join(c.decisions)}" for c in multi[:5]
            ))

    sec = Counter(c.sector for c in cases)
    print("\n업권 분포")
    for k, v in sec.most_common():
        print(f"  {k or '(미분류)'}: {v}")

    warn = Counter(w for c in cases for w in c.warnings)
    if warn:
        print("\n경고")
        for k, v in warn.most_common():
            print(f"  {k}: {v}")

    for doc_type in ("nonaction", "interpretation"):
        subset = [c for c in cases if c.doc_type == doc_type and c.serial]
        if not subset:
            continue
        dupes = Counter(c.serial for c in subset)
        rep = {k: v for k, v in dupes.items() if v > 1}
        uniq = len(dupes)
        print(f"\n{doc_type}: 헤더 {len(subset)} → 고유 일련번호 {uniq}")
        if rep:
            print(f"  문서 내 중복: {list(rep.items())}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="사례집 PDF 가 있는 디렉토리")
    ap.add_argument("--output", required=True, help="JSONL 출력 디렉토리")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.input), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = [p for p in sorted(in_dir.rglob("*.pdf")) if "__MACOSX" not in str(p)]
    if not pdfs:
        raise SystemExit(f"PDF 를 찾지 못했습니다: {in_dir}")

    all_cases: list[Case] = []
    for p in pdfs:
        cases = parse_pdf(p)
        print(f"[{len(cases):4d}] {ud.normalize('NFC', p.name)}")
        all_cases.extend(cases)

    for doc_type in ("nonaction", "interpretation"):
        subset = [c for c in all_cases if c.doc_type == doc_type]
        if not subset:
            continue
        path = out_dir / f"cases_{doc_type}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for c in subset:
                fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        print(f"  -> {path} ({len(subset)}건)")

    report(all_cases)


if __name__ == "__main__":
    main()
