"""S5 deciding-factor analysis의 결정론적 Diff Coverage Gate.

이 모듈은 **API를 호출하지 않는다.** LLM이 만든 factor 구조가 주어졌을 때
요청/선례의 실제 텍스트 차이를 빠뜨렸는지, 근거가 원문에 접지되는지, 그리고
`applies`를 허용해도 되는지 보수적으로 계산한다.

PDF 줄바꿈이나 문장 전체를 difference로 오인하지 않도록, 대조용 정규화 문자열의
실제 변경 구간을 문자 단위로 정렬해 coverage를 계산한다. 의미 유사도 threshold는
도입하지 않는다.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

from app.core.text import normalize_for_match

Side = Literal["request", "precedent", "both"]
Basis = Literal[
    "identical_after_metadata",
    "no_decisive_difference",
    "decisive_difference",
    "unresolved_difference",
    "incomplete_analysis",
]

_METADATA_PATTERNS = [
    re.compile(r"(?:[’']?\d{2,4}[.년]\s*\d{1,2}[.월]\s*\d{0,2}일?\.?)"),
    re.compile(r"일련번호\s*[:：]?\s*\d+"),
    re.compile(r"(?:여신금융협회|은행연합회)\s*요청"),
    re.compile(r"^\s*※"),
]


@dataclass(frozen=True)
class Segment:
    text: str
    normalized: str
    side: Literal["request", "precedent"]
    span: tuple[int, int]


@dataclass(frozen=True)
class Factor:
    id: str
    text: str
    side: Side
    axis: str = ""
    value_in_request: str | None = None
    value_in_precedent: str | None = None
    decisive: bool = False
    why_not_decisive: str | None = None


@dataclass(frozen=True)
class GateResult:
    basis: Basis
    fired_rule: str
    substantive_differences: tuple[Segment, ...]
    uncovered_differences: tuple[Segment, ...]
    unresolved_differences: tuple[Segment, ...]
    grounded_shared_factor_ids: tuple[str, ...]
    grounded_factor_ids: tuple[str, ...]
    rejected_factor_ids: tuple[str, ...]
    decisive_confirmed_ids: tuple[str, ...]


def _is_metadata(segment: Segment) -> bool:
    return any(pattern.search(segment.text) for pattern in _METADATA_PATTERNS)


def _diff_regions(request: str, precedent: str) -> tuple[list[Segment], list[Segment]]:
    """공백/조판을 제거한 두 원문에서 실제로 달라진 문자 구간만 반환한다.

    기존 문장 단위 exact-difference는 PDF 줄바꿈이나 한 절의 변경 때문에 긴 문장
    전체를 difference로 만들었다. 여기서는 `normalize_for_match` 후 동일 부분열을
    정렬하므로 줄바꿈은 사라지고, 실제 insert/delete/replace 구간만 남는다.
    """
    a = normalize_for_match(request)
    b = normalize_for_match(precedent)
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    da: list[Segment] = []
    db: list[Segment] = []
    for tag, a1, a2, b1, b2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if a1 < a2:
            value = a[a1:a2]
            da.append(Segment(value, value, "request", (a1, a2)))
        if b1 < b2:
            value = b[b1:b2]
            db.append(Segment(value, value, "precedent", (b1, b2)))
    return da, db


def _locate(text: str, factor: Factor) -> list[tuple[int, int]]:
    needle = normalize_for_match(factor.text)
    haystack = normalize_for_match(text)
    if not needle:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        i = haystack.find(needle, start)
        if i < 0:
            break
        spans.append((i, i + len(needle)))
        start = i + 1
    return spans


def _shared_grounded(factor: Factor, request: str, precedent: str) -> bool:
    if factor.side != "both":
        return False
    needle = normalize_for_match(factor.text)
    if not needle:
        return False
    return needle in normalize_for_match(request) and needle in normalize_for_match(precedent)


def _inside(span: tuple[int, int], segment: Segment) -> int:
    if segment.span[0] < 0:
        return 0
    return max(0, min(span[1], segment.span[1]) - max(span[0], segment.span[0]))


def _covers(factor: Factor, factor_spans: list[tuple[int, int]], segment: Segment) -> bool:
    """factor가 실제 diff 구간의 절반 이상을 직접 인용하면 coverage로 인정한다.

    factor는 설명 가능한 조건절이므로 diff 주변의 공통 문맥을 함께 인용할 수 있다.
    따라서 factor 바깥 길이로 벌점을 주지 않고, 실제 diff 자체를 얼마나 덮는지만 본다.
    """
    if factor.side == "both" or factor.side != segment.side:
        return False
    required = math.ceil(len(segment.normalized) / 2)
    return any(_inside(span, segment) >= required for span in factor_spans)


def evaluate_diff_coverage(
    request: str,
    precedent: str,
    shared_factors: Iterable[Factor],
    difference_factors: Iterable[Factor],
    *,
    precedent_admissible: bool = False,
) -> GateResult:
    """설계서 G1~G6를 fail-closed로 계산한다.

    `identical_after_metadata`는 검색 적격성 A1~A4까지 함께 봐야 한다. 이 모듈은
    A1~A4 자체를 계산하지 않으므로 호출자가 `precedent_admissible=True`를
    명시한 경우에만 G5가 `applies` 성격의 basis를 낸다.
    """
    da, db = _diff_regions(request, precedent)
    substantive = [s for s in da + db if not _is_metadata(s)]

    shared = list(shared_factors)
    grounded_shared = [f.id for f in shared if _shared_grounded(f, request, precedent)]
    factors = list(difference_factors)
    grounded: dict[str, list[tuple[int, int]]] = {}
    rejected: list[str] = [f.id for f in shared if f.id not in grounded_shared]
    for factor in factors:
        if factor.side == "request":
            source = request
        elif factor.side == "precedent":
            source = precedent
        else:
            source = ""
        spans = _locate(source, factor) if source else []
        if spans:
            grounded[factor.id] = spans
        else:
            rejected.append(factor.id)

    uncovered: list[Segment] = []
    unresolved: list[Segment] = []
    decisive_confirmed: list[str] = []

    for segment in substantive:
        covering = [
            factor
            for factor in factors
            if factor.id in grounded and _covers(factor, grounded[factor.id], segment)
        ]
        if not covering:
            uncovered.append(segment)
            continue
        if any(
            (not factor.decisive) and not (factor.why_not_decisive or "").strip()
            for factor in covering
        ):
            unresolved.append(segment)

    for factor in factors:
        if not factor.decisive or factor.id not in grounded:
            continue
        covers_real_difference = any(
            _covers(factor, grounded[factor.id], segment) for segment in substantive
        )
        if (
            covers_real_difference
            and factor.side in {"request", "precedent"}
            and factor.axis.strip()
            and factor.value_in_request is not None
            and factor.value_in_precedent is not None
        ):
            decisive_confirmed.append(factor.id)

    if not grounded_shared:
        basis: Basis = "incomplete_analysis"
        rule = "G1"
    elif uncovered:
        basis = "incomplete_analysis"
        rule = "G2"
    elif unresolved:
        basis = "unresolved_difference"
        rule = "G3"
    elif decisive_confirmed:
        basis = "decisive_difference"
        rule = "G4"
    elif not substantive and precedent_admissible:
        basis = "identical_after_metadata"
        rule = "G5"
    elif not substantive:
        basis = "incomplete_analysis"
        rule = "G5-blocked"
    else:
        basis = "no_decisive_difference"
        rule = "G6"

    return GateResult(
        basis=basis,
        fired_rule=rule,
        substantive_differences=tuple(substantive),
        uncovered_differences=tuple(uncovered),
        unresolved_differences=tuple(unresolved),
        grounded_shared_factor_ids=tuple(sorted(grounded_shared)),
        grounded_factor_ids=tuple(sorted(grounded)),
        rejected_factor_ids=tuple(sorted(rejected)),
        decisive_confirmed_ids=tuple(sorted(decisive_confirmed)),
    )