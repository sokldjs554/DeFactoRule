"""S5 deciding-factor analysis의 결정론적 Diff Coverage Gate.

이 모듈은 **API를 호출하지 않는다.** LLM이 만든 factor 구조가 주어졌을 때
요청/선례의 실제 텍스트 차이를 빠뜨렸는지, 근거가 원문에 접지되는지, 그리고
`applies`를 허용해도 되는지 보수적으로 계산한다.

안전성은 비대칭이다. 원문에 접지되고 반대쪽에는 없는 결정적 차이 하나가 확인되면
그 선례의 적용을 거절할 근거는 충분하다. 반대로 `no_decisive_difference`처럼
선례를 회수하는 방향은 모든 실질 차이가 설명되어야만 허용한다. AG-13의 위험은
후자에서 결정적 차이를 누락한 채 회수하는 것이므로, recovery 쪽만 strict coverage를
요구한다.

PDF 줄바꿈이나 문장 전체를 difference로 오인하지 않도록, 대조용 정규화 문자열의
실제 변경 구간을 문자 단위로 정렬해 recovery coverage를 계산한다. 의미 유사도
threshold는 도입하지 않는다.
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

    이 차집합은 `applies`/`no_decisive_difference` recovery의 completeness gate에만
    사용한다. 이미 원문에 접지된 decisive difference가 있으면 다른 잔여 차이를
    모두 설명하지 못했다는 이유로 그 거절 근거까지 버리지는 않는다.
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


def _unique_to_declared_side(factor: Factor, request: str, precedent: str) -> bool:
    """decisive factor가 실제 한쪽에만 존재하는 literal difference인지 확인한다."""
    needle = normalize_for_match(factor.text)
    if not needle:
        return False
    if factor.side == "request":
        return needle in normalize_for_match(request) and needle not in normalize_for_match(precedent)
    if factor.side == "precedent":
        return needle in normalize_for_match(precedent) and needle not in normalize_for_match(request)
    return False


def _inside(span: tuple[int, int], segment: Segment) -> int:
    if segment.span[0] < 0:
        return 0
    return max(0, min(span[1], segment.span[1]) - max(span[0], segment.span[0]))


def _covers(factor: Factor, factor_spans: list[tuple[int, int]], segment: Segment) -> bool:
    """factor가 recovery용 실제 diff 구간의 절반 이상을 직접 인용하면 coverage다."""
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
    """S5 G1~G6를 fail-closed로 계산한다.

    G4는 안전한 rejection 경로다. 원문에 literal-grounded되고 반대쪽에는 없는
    decisive factor가 확인되면 다른 차이의 completeness와 무관하게 적용 거절 근거로
    인정한다. 반대로 G5/G6 recovery는 shared grounding과 전체 diff coverage를 모두
    통과해야 한다.

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
        if factor.axis.strip() and _unique_to_declared_side(factor, request, precedent):
            decisive_confirmed.append(factor.id)

    # Rejection과 recovery는 의도적으로 비대칭이다. 실재하는 결정적 차이 하나면
    # 선례를 적용하지 않을 근거는 충분하지만, 차이가 없다고 말하려면 전부 봐야 한다.
    if decisive_confirmed:
        basis: Basis = "decisive_difference"
        rule = "G4"
    elif not grounded_shared:
        basis = "incomplete_analysis"
        rule = "G1"
    elif uncovered:
        basis = "incomplete_analysis"
        rule = "G2"
    elif unresolved:
        basis = "unresolved_difference"
        rule = "G3"
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