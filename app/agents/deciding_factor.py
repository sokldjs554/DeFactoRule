"""S5 deciding-factor analysis의 결정론적 Diff Coverage Gate.

이 모듈은 **API를 호출하지 않는다.** LLM이 만든 factor 구조가 주어졌을 때
요청/선례의 실제 텍스트 차이를 빠뜨렸는지, 근거가 원문에 접지되는지, 그리고
`applies`를 허용해도 되는지 보수적으로 계산한다.

첫 구현에서는 의미 유사도를 쓰지 않는다. 설계서의 4-gram τ는 아직 보정되지
않았으므로 gate를 켜기 전까지 **정확 조각 일치만** 사용한다.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
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

_SEGMENT_SPLIT = re.compile(r"(?<=[다함?!])\.|\n+|[□○◦▪●•▶◆∙]+")
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


def _segments(text: str, side: Literal["request", "precedent"]) -> list[Segment]:
    """문장/줄 단위 조각과 normalize_for_match 원문 좌표를 만든다."""
    raw_parts = [p.strip() for p in _SEGMENT_SPLIT.split(text or "") if p.strip()]
    haystack = normalize_for_match(text)
    out: list[Segment] = []
    search_from = 0
    for part in raw_parts:
        norm = normalize_for_match(part)
        if not norm:
            continue
        start = haystack.find(norm, search_from)
        if start < 0:
            # 좌표를 지어내지 않는다. 음수 span은 어떤 factor도 덮지 못한다.
            out.append(Segment(part, norm, side, (-1, -1)))
            continue
        end = start + len(norm)
        out.append(Segment(part, norm, side, (start, end)))
        search_from = end
    return out


def _is_metadata(segment: Segment) -> bool:
    return any(pattern.search(segment.text) for pattern in _METADATA_PATTERNS)


def _exact_difference(a: list[Segment], b: list[Segment]) -> tuple[list[Segment], list[Segment]]:
    """τ 미보정 상태의 안전한 첫 구현: normalize된 조각의 정확 일치만 제거."""
    b_values = {s.normalized for s in b}
    a_values = {s.normalized for s in a}
    return (
        [s for s in a if s.normalized not in b_values],
        [s for s in b if s.normalized not in a_values],
    )


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
    if factor.side == "both" or factor.side != segment.side:
        return False
    required = math.ceil(len(segment.normalized) / 2)
    for span in factor_spans:
        inside = _inside(span, segment)
        outside = (span[1] - span[0]) - inside
        if inside >= required and outside <= inside:
            return True
    return False


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
    request_segments = _segments(request, "request")
    precedent_segments = _segments(precedent, "precedent")
    da, db = _exact_difference(request_segments, precedent_segments)
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