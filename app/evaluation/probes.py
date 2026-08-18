"""실패 케이스의 실행 가능한 재현 검사.

레지스트리의 각 케이스는 여기 있는 probe 하나를 가리킨다. probe 는
`(수정이 유지되는가, 설명)` 을 돌려준다. 회귀 테스트가 전부 돌리므로,
수정이 풀리면 그 자리에서 실패한다.

일부 probe 는 **옛 구현을 함께 들고 있다.** `_v0_` 로 시작하는 함수가 그것이며,
before 수치를 같은 입력에서 재기 위해서만 존재한다. 그래야 "고치기 전 얼마,
고친 뒤 얼마" 를 기억이 아니라 실행으로 말할 수 있다.
"""

from __future__ import annotations

import inspect
import re
import subprocess
import sys
import unicodedata as ud
from pathlib import Path

from app.core.paths import DEV_BASE_RATES, PROCESSED, ROOT, check_root

Result = tuple[bool, str]


def _cases(name: str) -> list[dict]:
    from app.core.io import load_jsonl

    path = PROCESSED / name
    return load_jsonl(path) if path.exists() else []


# ══ extraction ═══════════════════════════════════════════════════════
def checkbox_glyph_variants() -> Result:
    """2024년판 한 권에 체크 표시 서식이 여섯 종 있었다."""
    from app.extraction.casebook import RE_DECISION

    samples = {
        "☑ 비조치": "비조치", "☒ 조치": "조치", "■ 기타": "기타",
        "þ 비조치": "비조치", "▣ 조치": "조치", "√ □ 비조치": "비조치",
        "V□ 기타": "기타", "◉비조치": "비조치",
    }
    missed = [s for s, want in samples.items()
              if not (m := RE_DECISION.search(s)) or m.group(1) != want]
    return not missed, f"서식 {len(samples)}종 중 미검출 {len(missed)}: {missed}"


def check_before_box() -> Result:
    """체크 표시가 상자 '앞' 에 오는 서식. 상자만 찾으면 통째로 놓친다."""
    from app.extraction.casebook import RE_DECISION

    hit = RE_DECISION.search("√ □ 비조치의견")
    return bool(hit) and hit.group(1) == "비조치", f"매치: {hit.group(0) if hit else None!r}"


def wingdings_thorn() -> Result:
    """Wingdings 체크가 유니코드 þ(U+00FE) 로 뽑힌다. 눈으로는 체크로 보인다."""
    from app.extraction.casebook import RE_DECISION

    hit = RE_DECISION.search("þ 조치")
    return bool(hit), f"þ 검출: {bool(hit)}"


def field_name_longest_first() -> Result:
    """'판단' 이 '판단이유' 를 삼키던 문제.

    핵심은 **항목명 자체가 줄바꿈으로 쪼개져 들어온다**는 것이다. PDF 에서
    "판단이유" 가 "판단\n이유" 로 뽑히면, 한 줄씩만 보면서 짧은 이름을 먼저
    맞춰 보는 매칭은 첫 줄 "판단" 에 걸려 버리고 판단이유는 영영 못 잡는다.
    빈 값이 남을 뿐 예외는 나지 않는다.
    """
    from app.extraction.casebook import split_fields

    names = ["요청대상행위", "판단", "판단이유"]
    body = (
        "요청대상행위\n갑 행위를 하려 합니다\n"
        "판단\n비조치\n"
        "판단\n이유\n관련 규정상 문제 없음"   # 항목명이 줄바꿈으로 쪼개진 형태
    )

    def _v0_split(body: str, names: list[str]) -> dict:
        """한 줄씩만, 짧은 이름부터 맞춰 보던 옛 매칭."""
        lines = body.split("\n")
        marks = []
        for i, line in enumerate(lines):
            key = re.sub(r"\s+", "", line)
            hit = next((n for n in sorted(names, key=len) if key == n), None)
            if hit:
                marks.append((i, hit))
        out: dict[str, str] = {}
        for idx, (line_no, name) in enumerate(marks):
            end = marks[idx + 1][0] if idx + 1 < len(marks) else len(lines)
            out.setdefault(name, "\n".join(lines[line_no + 1 : end]).strip())
        return out

    before = _v0_split(body, names)
    now, warnings = split_fields(body, names)
    ok = now.get("판단이유", "").strip() == "관련 규정상 문제 없음" and not warnings
    return ok, (
        f"옛 매칭이 잡은 항목 {sorted(before)} (판단이유={before.get('판단이유')!r}) · "
        f"현재 {sorted(now)} (판단이유={now.get('판단이유')!r})"
    )


def field_name_tail_not_in_value() -> Result:
    """항목명의 뒷조각이 값 맨 앞에 남아 있는지 코퍼스 전체에서 센다.

    이 검사가 없어서 1,095건 중 406건이 오염된 채로 커밋되어 있었다. 값이
    비지 않고 그럴듯하게 채워지므로 건수만 보면 아무 문제가 없어 보인다.
    """
    from app.extraction.casebook import INTERP_FIELDS, NONACTION_FIELDS

    total = 0
    breakdown: dict[str, int] = {}
    for fname, names in (("cases_nonaction.jsonl", NONACTION_FIELDS),
                         ("cases_interpretation.jsonl", INTERP_FIELDS)):
        rows = _cases(fname)
        if not rows:
            return True, "산출물 없음 — 건너뜀"
        for row in rows:
            for name in names:
                value = (row["fields"].get(name) or "").lstrip()
                if not value:
                    continue
                head = re.split(r"[\n ]", value, maxsplit=1)[0]
                if head in [name[i:] for i in range(1, len(name))]:
                    total += 1
                    breakdown[f"{name}←{head}"] = breakdown.get(f"{name}←{head}", 0) + 1
    return total == 0, f"항목명 잔재 {total}건 {breakdown or ''}".strip()


def sector_divider_formats() -> Result:
    """업권 구분 페이지가 기관·연도마다 다른 네 가지 서식으로 나온다."""
    from app.extraction.casebook import (
        RE_SECTOR_MAJOR_NAME_FIRST,
        RE_SECTOR_MINOR,
        RE_SECTOR_NAME_THEN_YEAR,
        RE_SECTOR_NUM_FIRST,
    )

    checks = [
        ("공통\n1", RE_SECTOR_MAJOR_NAME_FIRST),
        ("공 통\n2022년\n비조치의견서 사례집", RE_SECTOR_NAME_THEN_YEAR),
        ("1. 공통", RE_SECTOR_NUM_FIRST),
        ("•∙∙∙•\n금융정책 일반", RE_SECTOR_MINOR),
    ]
    missed = [text for text, pattern in checks if not pattern.search(text)]
    return not missed, f"구분 서식 {len(checks)}종 중 미검출 {len(missed)}: {missed}"


def sector_alias_normalized() -> Result:
    """같은 업권이 '상호저축은행' 과 '상호저축은행업' 두 이름으로 온다."""
    from app.extraction.casebook import canonical_sector

    pairs = [("상호저축은행", "상호저축은행업"), ("여신전문금융", "여신전문금융업"),
             ("온라인투자연계금융", "온라인투자연계금융업"), ("중소", "중소금융")]
    bad = [(a, b) for a, b in pairs if canonical_sector(a) != canonical_sector(b)]
    return not bad, f"별칭 {len(pairs)}쌍 중 불일치 {len(bad)}: {bad}"


def sector_page_length_cap() -> Result:
    """구분 페이지 판정에 길이 상한이 없으면 본문이 업권으로 오인된다."""
    from app.extraction.casebook import SECTOR_PAGE_MAX_CHARS

    return SECTOR_PAGE_MAX_CHARS <= 120, f"상한 {SECTOR_PAGE_MAX_CHARS}자"


def unreadable_pdf_detected() -> Result:
    """ToUnicode 없는 CID 폰트는 예외 대신 쓰레기 문자열을 준다."""
    from app.extraction.casebook import MIN_TEXT_HEALTH, text_health

    garbage = "".join(chr(0xE000 + i % 200) for i in range(3000))
    normal = "금융위원회는 다음과 같이 회신합니다. " * 100
    bad, good = text_health(garbage), text_health(normal)
    ok = bad < MIN_TEXT_HEALTH <= good
    return ok, f"깨진 PDF {bad:.1%} · 정상 {good:.1%} · 문턱 {MIN_TEXT_HEALTH:.0%}"


def nfd_filename_collision() -> Result:
    """macOS zip 의 한글 파일명은 NFD 다. 정규화 없이 쓰면 두 파일이 겹친다."""
    nfc = "비조치의견서"
    nfd = ud.normalize("NFD", nfc)
    return nfc != nfd and ud.normalize("NFC", nfd) == nfc, (
        f"NFD 길이 {len(nfd)} vs NFC {len(nfc)} — 정규화 없이는 다른 경로"
    )


def control_chars_do_not_break_serial() -> Result:
    """일련번호와 숫자 사이에 제어문자가 끼어 정규식을 끊는다."""
    from app.extraction.casebook import RE_SERIAL

    hit = RE_SERIAL.search("일련번호\x00\x01 250123")
    return bool(hit) and hit.group(1) == "250123", f"추출: {hit.group(1) if hit else None}"


def answer_only_split_removed() -> Result:
    """회답 안의 순번은 요건 열거이지 질의 구분이 아니었다. 폴백을 없앴다."""
    pairs = _cases("qa_pairs.jsonl")
    if not pairs:
        return True, "산출물 없음 — 건너뜀"
    modes = {p.get("split_mode") for p in pairs}
    bad = modes - {"single", "paired"}
    return not bad, f"분할 모드 {sorted(modes)} · 허용 밖 {sorted(bad)}"


def record_key_is_unique() -> Result:
    """일련번호만으로 세면 다른 사례집의 같은 번호가 중복으로 잡힌다."""
    from app.core.io import key_of

    rows = _cases("qa_pairs.jsonl")
    if not rows:
        return True, "산출물 없음 — 건너뜀"
    serials = [r["serial"] for r in rows if r.get("serial")]
    keys = [key_of(r) for r in rows]
    dup_serial = len(serials) - len(set(serials))
    dup_key = len(keys) - len(set(keys))
    return dup_key == 0, f"일련번호 기준 중복 {dup_serial}건 · 전체 키 기준 {dup_key}건"


def boundary_labels_not_shifted() -> Result:
    """가장 비쌌던 버그. 옛 구현과 같은 입력에서 재구성 정확도를 잰다."""
    from app.extraction.spacing import iter_boundaries

    def _v0_iter_boundaries(line: str):
        """공백을 만나면 **직전** 경계를 세우던 옛 구현."""
        chars: list[str] = []
        labels: list[int] = []
        for ch in line:
            if ch == " ":
                if labels:
                    labels[-1] = 1
                continue
            if chars:
                labels.append(0)
            chars.append(ch)
        return "".join(chars), labels

    lines = [
        "금융위원회 및 금융감독원은",
        "금융회사 등이 대통령령으로 정하는 경우",
        "코로나19 펜데믹의 후유증으로 신용점수 하락이",
        "해당 행위는 자본시장법 제 8 조에 따른 것으로",
    ]

    def rebuild(fn, line):
        chars, labels = fn(line)
        if len(labels) != len(chars) - 1:
            return None
        return chars[0] + "".join(
            (" " if labels[i] else "") + chars[i + 1] for i in range(len(labels))
        )

    before = sum(1 for x in lines if rebuild(_v0_iter_boundaries, x) == x)
    after = sum(1 for x in lines if rebuild(iter_boundaries, x) == x)
    return after == len(lines), (
        f"원문 재구성 {len(lines)}줄 중 — 옛 구현 {before} · 현재 {after}"
    )


def restore_only_adds_spaces() -> Result:
    """복원기가 원문의 공백을 지우면 안 된다. 원문이 모델보다 믿을 만하다."""
    from app.extraction.spacing import restore_line

    line = "코로나19 펜데믹의후유증으로"
    kept = restore_line({"contexts": {}, "prior": -99.0}, line, threshold=0.0)
    return kept == line, f"입력 공백 {line.count(' ')} · 출력 {kept.count(' ')}"


# ══ labeling ═════════════════════════════════════════════════════════
def leak_words_removed() -> Result:
    """요청문에 '비조치' 가 그대로 남아 있으면 모델은 읽기만 하면 된다."""
    from app.evaluation.gold_nonaction import LEAK

    raw = _cases("cases_nonaction.jsonl")
    if not raw:
        return True, "산출물 없음 — 건너뜀"
    texts = [r["fields"].get("요청대상행위", "") or "" for r in raw]
    before = sum(1 for t in texts if LEAK.search(t))
    after = sum(1 for t in texts if LEAK.search(LEAK.sub(" ", t)))
    return after == 0, (
        f"요청문 {len(texts)}건 — 가리기 전 누출 {before}건 · 가린 뒤 {after}건"
    )


def leak_particles_removed() -> Result:
    """낱말만 지우면 '를요청' 이라는 흔적이 다시 신호가 된다."""
    from app.evaluation.gold_nonaction import LEAK

    text = "당사는 비조치를 요청드립니다"
    masked = re.sub(r"\s+", " ", LEAK.sub(" ", text)).strip()
    return "를" not in masked.split()[0], f"가린 뒤: {masked!r}"


def mask_token_is_not_a_new_leak() -> Result:
    """마스크 토큰 자체가 누출이었다. 옛 마스크와 나란히 계급별 출현을 센다."""
    from app.evaluation.gold_nonaction import LEAK, MASK

    raw = [r for r in _cases("cases_nonaction.jsonl") if r.get("decision")]
    if not raw:
        return MASK.strip() == "", "산출물 없음 — 마스크가 빈 문자열인지만 확인"

    def spread(mask: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        token = mask.strip()
        for r in raw:
            text = LEAK.sub(mask, r["fields"].get("요청대상행위", "") or "")
            if token and token in text:
                counts[r["decision"]] = counts.get(r["decision"], 0) + 1
        return counts

    before = spread("[결론표현]")
    after = spread(MASK)
    return not after, f"옛 마스크 계급별 출현 {before} · 현재 {after or '없음'}"


def unknown_is_not_abstain() -> Result:
    """'규칙이 못 읽음' 과 '당국이 판단을 유보함' 은 다른 사건이다."""
    from app.domain.labels import Verdict
    from app.rules.verdict import UNKNOWN

    return UNKNOWN != Verdict.ABSTAIN.value, f"{UNKNOWN!r} vs {Verdict.ABSTAIN.value!r}"


def multiple_checks_are_recorded() -> Result:
    """체크가 둘 이상인 사례를 조용히 하나로 접으면 안 된다."""
    rows = _cases("cases_nonaction.jsonl")
    if not rows:
        return True, "산출물 없음 — 건너뜀"
    has_field = sum(1 for r in rows if "decisions" in r)
    multi = [r for r in rows if len(r.get("decisions") or []) > 1]
    flagged = [r for r in multi if "multi_decision" in (r.get("warnings") or [])]
    return has_field == len(rows) and len(flagged) == len(multi), (
        f"decisions 필드 {has_field}/{len(rows)} · 복수 체크 {len(multi)}건 중 "
        f"경고 {len(flagged)}건"
    )


def split_is_deterministic() -> Result:
    """dev/test 를 난수로 나누면 실험을 다시 돌릴 수 없다."""
    from app.evaluation.gold_nonaction import build

    rows = _cases("cases_nonaction.jsonl")
    if not rows:
        return True, "산출물 없음 — 건너뜀"
    a_dev, a_test = build(rows)
    b_dev, b_test = build(list(reversed(rows)))
    same = [r["serial"] for r in a_dev] == [r["serial"] for r in b_dev] and \
           [r["serial"] for r in a_test] == [r["serial"] for r in b_test]
    return same, f"입력 순서를 뒤집어도 동일한 분할: {same} (dev {len(a_dev)} / test {len(a_test)})"


def dev_and_test_do_not_overlap() -> Result:
    from app.core.io import key_of
    from app.evaluation.gold_nonaction import build

    rows = _cases("cases_nonaction.jsonl")
    if not rows:
        return True, "산출물 없음 — 건너뜀"
    dev, test = build(rows)
    overlap = {key_of(r) for r in dev} & {key_of(r) for r in test}
    return not overlap, f"dev {len(dev)} · test {len(test)} · 교집합 {len(overlap)}"


# ══ evaluation ═══════════════════════════════════════════════════════
def missing_predictions_are_flagged() -> Result:
    """30건 예측을 170건 gold 로 재면서 커버리지 17.6%를 놓쳤다."""
    import json
    import tempfile
    from pathlib import Path

    gold = [
        {"source": "t", "page": i, "serial": str(i), "pair_index": 1,
         "request": "질의", "label": "비조치"}
        for i in range(1, 6)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        g = Path(tmp) / "g.jsonl"
        p = Path(tmp) / "p.jsonl"
        g.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in gold), encoding="utf-8")
        p.write_text(
            json.dumps({**gold[0], "predicted": "비조치"}, ensure_ascii=False),
            encoding="utf-8",
        )
        out = subprocess.run(
            [sys.executable, "scripts/evaluate.py", "--gold", str(g), "--pred", str(p),
             "--labels", "nonaction"],
            cwd=ROOT, capture_output=True, text=True,
        )
    warned = "예측이 없는" in out.stdout and "--limit" in out.stdout
    return warned, f"gold 5건 · 예측 1건 · 경고 출력: {warned}"


def macro_f1_exposes_majority_baseline() -> Result:
    """정확도를 대표로 삼으면 다수만 찍는 분류기가 좋아 보인다. 실제 test 로 잰다."""
    from app.core.io import load_jsonl
    from app.core.paths import EVAL
    from app.domain.labels import NON_ACTIONS
    from app.evaluation.metrics import macro_f1

    path = EVAL / "nonaction_test.jsonl"
    if not path.exists():
        return True, "평가셋 없음 — 건너뜀"
    gold = [r["label"] for r in load_jsonl(path) if r.get("label")]
    pairs = [(g, "비조치") for g in gold]
    accuracy = sum(1 for g, p in pairs if g == p) / len(pairs)
    macro, _ = macro_f1(pairs, NON_ACTIONS)
    return macro < accuracy - 0.30, (
        f"다수 클래스만 예측 — 정확도 {accuracy:.1%} · 매크로 F1 {macro:.3f} "
        f"(차이 {accuracy - macro:.3f})"
    )


def risk_coverage_exists_for_fair_comparison() -> Result:
    """커버리지가 다른 두 F1 을 나란히 놓는 것은 비교가 아니다."""
    from app.evaluation.selective import aurc, operating_points

    labels = ("비조치", "조치", "기타")
    # 신뢰도 신호가 없는 모델은 곡선이 한 점이어야 한다
    flat = [("비조치", "비조치", 1), ("조치", "비조치", 1), ("기타", "비조치", 1)]
    graded = [("비조치", "비조치", 3), ("조치", "비조치", 1), ("기타", "기타", 3)]
    pf, pg = operating_points(flat, labels), operating_points(graded, labels)
    ok = len(pf) == 1 and len(pg) > 1 and aurc(pg) < aurc(pf)
    return ok, (
        f"신호 없음 운영점 {len(pf)}개 AURC {aurc(pf):.3f} · "
        f"신호 있음 {len(pg)}개 AURC {aurc(pg):.3f}"
    )


def sector_lift_is_normalized() -> Result:
    """업권마다 다수 클래스 비율이 다르다. 절대 정확도로 줄세우면 진단이 틀린다."""
    from app.evaluation.sectors import MIN_N

    src = (ROOT / "app" / "evaluation" / "sectors.py").read_text(encoding="utf-8")
    has = "normalized_lift" in src and "items_vs_majority" in src
    return has and MIN_N >= 5, (
        f"정규화 lift {'있음' if 'normalized_lift' in src else '없음'} · "
        f"건수 병기 {'있음' if 'items_vs_majority' in src else '없음'} · 최소 표본 {MIN_N}"
    )


def bootstrap_is_seeded() -> Result:
    """시드가 고정되지 않으면 같은 실험이 다른 구간을 낸다."""
    from app.domain.labels import NON_ACTIONS
    from app.evaluation.metrics import bootstrap_macro_f1

    # 라벨 조합이 단조로우면 매크로 F1 이 몇 개의 값만 갖고, 서로 다른 시드가
    # 우연히 같은 백분위를 낸다. 세 클래스가 모두 오답을 갖도록 섞는다.
    pairs = (
        [("비조치", "비조치")] * 40
        + [("조치", "비조치")] * 8
        + [("조치", "조치")] * 5
        + [("기타", "비조치")] * 4
        + [("기타", "기타")] * 6
        + [("비조치", "기타")] * 3
    )
    a = bootstrap_macro_f1(pairs, NON_ACTIONS, rounds=400, seed=0)
    b = bootstrap_macro_f1(pairs, NON_ACTIONS, rounds=400, seed=0)
    c = bootstrap_macro_f1(pairs, NON_ACTIONS, rounds=400, seed=1)
    return a == b and a != c, (
        f"시드0 {a[0]:.3f}–{a[1]:.3f} · 재실행 동일 {a == b} · "
        f"시드1 {c[0]:.3f}–{c[1]:.3f} 달라짐 {a != c}"
    )


def absent_labels_do_not_dilute_macro() -> Result:
    from app.evaluation.metrics import macro_f1

    pairs = [("긍정", "긍정"), ("부정", "부정")]
    macro, per = macro_f1(pairs, ("긍정", "부정", "조건부", "판단유보"))
    return macro == 1.0 and per["조건부"]["support"] == 0, f"매크로 {macro:.3f}"


def check_completeness(gold_path, pred_dir) -> Result:
    """예측 파일이 gold 를 전부 덮는지. probe 가 테스트 가능하도록 분리했다."""
    from app.core.io import key_of, load_jsonl

    if not Path(gold_path).exists():
        return True, "평가셋 없음 — 건너뜀"
    gold = {key_of(r) for r in load_jsonl(Path(gold_path)) if r.get("label")}

    report = []
    incomplete = []
    for path in sorted(Path(pred_dir).glob("pred_nonaction_*.jsonl")):
        rows = load_jsonl(path)
        failed = sum(1 for r in rows if r.get("predicted") is None or "error" in r)
        absent = len(gold - {key_of(r) for r in rows})
        report.append(
            f"{path.name} {len(rows)}/{len(gold)} 실패 {failed} 누락 {absent}"
        )
        if failed or absent:
            incomplete.append(path.name)
    if not report:
        return False, "예측 파일이 하나도 없다"
    return not incomplete, " · ".join(report)


def predictions_are_complete_before_reporting() -> Result:
    """편향된 부분집합으로 비교하면 결론이 뒤집힌다. 결측이 있으면 보고 금지.

    결측은 두 가지다. **둘 다 세야 한다.**

      실패한 행   파일에는 있는데 predicted 가 없거나 error 가 붙은 것
      없는 행     gold 에는 있는데 파일에 아예 없는 것

    처음 구현은 앞의 것만 셌다. 그래서 156/170 짜리 파일이 "결측 0" 으로
    보고되고 통과했다 — EV-01(30건 예측을 170건 gold 로 채점)과 똑같은
    맹점이, 하필 그것을 막으려고 만든 검사 안에 들어 있었다.

    빠진 행은 무작위가 아닐 때가 많다. 실제로 sector 의 결측 39건은
    2025년 35건 + 2024년 4건이었고, llm·prior 의 14건은 파서 수정으로
    본문이 바뀐 사례였다. 어느 쪽도 무작위가 아니다.
    """
    from app.core.paths import EVAL

    return check_completeness(EVAL / "nonaction_test.jsonl", PROCESSED)


# ══ agent ════════════════════════════════════════════════════════════
def minority_rules_are_reachable() -> Result:
    """규칙 학습 문턱이 소수 클래스를 구조적으로 배제하지 않는가.

    라플라스 정밀도를 그대로 통과 문턱으로 쓰면 laplace(4,4,3)=0.714 이라,
    문턱 0.80 에서는 4건을 완벽히 덮는 규칙도 통과할 수 없다. 통과하려면 8건을
    완벽히 덮어야 하는데 dev 의 `조치` 는 전부 합쳐 8건이었다. 즉 소수 클래스
    규칙은 아무리 좋아도 나올 수 없었고, 학습기가 아무것도 못 찾는 것처럼 보였다.
    """
    from app.domain.labels import NON_ACTIONS
    from app.rules.induction import induce, laplace

    rows = (
        [{"source": "t", "page": i, "serial": str(i), "pair_index": 1, "sector": "공통",
          "request": f"내부망과 외부망의 망연계 구간 질의 {i}", "label": "조치"}
         for i in range(1, 7)]
        + [{"source": "t", "page": 10 + i, "serial": str(10 + i), "pair_index": 1,
            "sector": "공통", "request": f"겸영업무 신고 대상 여부 질의 {i}",
            "label": "비조치"} for i in range(1, 9)]
    )
    rules, _ = induce(rows, NON_ACTIONS, min_support=4, min_precision=0.80, max_depth=2)
    minority = [r for r in rules if r.label == "조치"]
    return bool(minority), (
        f"laplace(4,4,3)={laplace(4, 4, 3):.3f} (문턱으로 쓰면 0.80 미달) · "
        f"규칙 {len(rules)}개 중 소수 클래스 {len(minority)}개"
    )


def induced_rules_are_deduplicated() -> Result:
    """덮는 집합이 같은 n-gram 이 같은 규칙을 여러 벌 만들면 규칙 목록을 읽을 수 없다."""
    from app.rules.induction import Atom, coverage_masks

    rows = [
        {"source": "t", "page": i, "serial": str(i), "pair_index": 1, "sector": "공통",
         "request": f"망연계 구간 {i}", "label": "조치"}
        for i in range(1, 6)
    ]
    masks = coverage_masks(rows, [Atom("ngram", v) for v in ("망연계", "망연", "연계")])
    return len(masks) == 1, (
        f"같은 집합을 덮는 조건 3개 -> 대표 {len(masks)}개 "
        f"({[a.value for a in masks] if masks else '없음'})"
    )


def evidence_must_be_verbatim() -> Result:
    """근거 인용은 원문과 글자 단위로 대조한다. 그럴듯한 요약은 근거가 아니다."""
    from app.agents.classifier import evidence_is_grounded

    answer = "질의하신 행위는 전자금융거래법 제2조에 해당하지 않는 것으로 판단됩니다."
    cases = [
        ("해당하지 않는 것으로 판단됩니다", True),   # 그대로 인용
        ("해당하지  않는  것으로", True),            # 공백만 다름
        ("해당하는 것으로 판단됩니다", False),        # 뜻이 뒤집힌 위조
        ("", True),                                   # 인용 포기는 환각이 아니다
    ]
    wrong = [e for e, want in cases if evidence_is_grounded(e, answer) != want]
    return not wrong, f"{len(cases)}종 중 오판 {len(wrong)}: {wrong}"


def label_is_constrained_by_schema() -> Result:
    """자유 서술을 파싱하면 '아마 비조치' 같은 값이 들어온다. enum 으로 막는다."""
    from app.agents.classifier import _schema
    from app.domain.labels import NON_ACTIONS

    schema = _schema(NON_ACTIONS, "근거")["schema"]
    verdict = schema["properties"]["verdict"]
    conf = schema["properties"]["confidence"]
    ok = (
        verdict.get("enum") == list(NON_ACTIONS)
        and conf.get("enum") == ["high", "medium", "low"]
        and schema.get("additionalProperties") is False
        and set(schema["required"]) == {"verdict", "evidence", "confidence"}
    )
    return ok, (
        f"라벨 enum {verdict.get('enum')} · "
        f"추가 필드 허용 {schema.get('additionalProperties')}"
    )


def base_rates_come_from_dev_only() -> Result:
    """기저율을 test 에서 뽑아 프롬프트에 넣으면 정답을 흘리는 것이다."""
    import json

    if not DEV_BASE_RATES.exists():
        return True, "기저율 파일 없음 — 건너뜀"
    table = json.loads(DEV_BASE_RATES.read_text(encoding="utf-8"))
    return table.get("source") == "dev", f"source={table.get('source')!r} n={table.get('n')}"


def small_sector_falls_back_to_overall() -> Result:
    """3건짜리 분포를 100%라고 적어 주면 잡음을 신호로 위장하는 셈이다."""
    from app.domain.base_rates import MIN_SECTOR_N, describe_sector

    table = {
        "n": 85, "min_sector_n": MIN_SECTOR_N,
        "overall": {"비조치": 0.7, "조치": 0.2, "기타": 0.1},
        "sectors": {"희소업권": {"n": 3, "reliable": False,
                                 "rates": {"비조치": 1.0, "조치": 0.0, "기타": 0.0}}},
    }
    text = describe_sector(table, "희소업권")
    return "희소업권" not in text, f"표본 3건 업권 문장: {text[:40]}…"


def rule_baseline_emits_confidence() -> Result:
    """규칙 쪽에 기권 장치가 없으면 LLM 과의 비교 자체가 성립하지 않는다."""
    from app.rules.nonaction import classify

    _, _, majority = classify("무엇이든", "majority")
    _, _, hit = classify("망분리 관련", "keyword")
    _, _, miss = classify("관련 없는 문장", "keyword")
    ok = majority == "low" and hit == "high" and miss == "low"
    return ok, f"majority={majority} · 규칙적중={hit} · 폴백={miss}"


# ══ infrastructure ═══════════════════════════════════════════════════
def account_errors_abort_immediately() -> Result:
    """크레딧 소진은 다음 요청도 100% 실패한다. 항목 오류로 취급하면 전부 태운다."""
    from app.agents.classifier import FATAL_MARKERS

    fatal = "Your credit balance is too low to access the Anthropic API"
    retryable = "Overloaded: please retry"
    hits_fatal = any(m in fatal for m in FATAL_MARKERS)
    hits_retryable = any(m in retryable for m in FATAL_MARKERS)
    return hits_fatal and not hits_retryable, (
        f"치명 마커 {len(FATAL_MARKERS)}종 · 크레딧 오류 판정 {hits_fatal} · "
        f"일시 오류 오판 {hits_retryable}"
    )


def api_errors_keep_their_message() -> Result:
    """상태 코드만 남기면 39건이 왜 죽었는지 영영 모른다."""
    src = (ROOT / "app" / "agents" / "classifier.py").read_text(encoding="utf-8")
    keys = ["error_detail", "error_body", "prompt_chars"]
    missing = [k for k in keys if k not in src]
    return not missing, f"기록 항목 {keys} 중 누락 {missing}"


def repository_paths_resolve() -> Result:
    """파일을 옮기면 parents[N] 이 조용히 틀어진다. 루트를 한 곳에서만 센다."""
    return check_root() and DEV_BASE_RATES.parent.is_dir(), (
        f"ROOT={ROOT.name} · pyproject 존재 {check_root()} · "
        f"기저율 경로 {DEV_BASE_RATES.relative_to(ROOT)}"
    )


def ci_installs_an_existing_requirements_file() -> Result:
    """워크플로가 없는 파일을 설치하고 있었다. CI 는 한 번도 돈 적이 없다."""
    workflow = ROOT / ".github" / "workflows" / "ci.yml"
    if not workflow.exists():
        return False, "워크플로가 없다"
    text = workflow.read_text(encoding="utf-8")
    referenced = re.findall(r"-r\s+(\S+\.txt)", text)
    missing = [r for r in referenced if not (ROOT / r).exists()]
    return bool(referenced) and not missing, (
        f"참조 {referenced} · 없는 파일 {missing}"
    )


def ci_lints_the_whole_repository() -> Result:
    """린트 범위가 scripts tests 였다. app/ 이 통째로 빠져 있었다."""
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    lint = [ln.strip() for ln in text.splitlines() if "ruff check" in ln]
    covers_app = any(ln.endswith("ruff check .") or " app" in ln for ln in lint)
    return bool(lint) and covers_app, f"린트 명령 {lint}"


def code_parses_on_python_39() -> Result:
    """실행 문턱을 낮추려 3.9 문법으로 내려왔다. 그 약속이 유지되는지 본다."""
    out = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "app", "scripts", "tests"],
        cwd=ROOT, capture_output=True, text=True,
    )
    # 낱말이 나오는지가 아니라 **실제로 쓰는지**를 본다. labels.py 는 왜 쓰지
    # 않는지를 docstring 에 적어 두었고, 그것까지 걸리면 검사가 거짓말을 한다.
    used = re.compile(r"from\s+enum\s+import[^\n]*\bStrEnum\b|\(\s*StrEnum\s*\)")
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.glob("app/**/*.py")
        if used.search(path.read_text(encoding="utf-8"))
    ]
    return out.returncode == 0 and not offenders, (
        f"컴파일 {'통과' if out.returncode == 0 else '실패'} · "
        f"StrEnum 사용 파일 {offenders or '없음'}"
    )


def env_check_runs_on_python2() -> Result:
    """인터프리터가 틀렸다는 사실을 알려주려면 그 인터프리터로 파싱돼야 한다."""
    src = (ROOT / "scripts" / "check_env.py").read_text(encoding="utf-8")
    bad = [name for name, pat in
           (("f-string", r'\bf["\']'), ("타입 힌트", r"->"), ("월러스", r":="))
           if re.search(pat, src)]
    return not bad, f"Python 2 파서가 걸릴 문법: {bad or '없음'}"


def _is_probe(name: str, fn: object) -> bool:
    """probe 는 **인자 없이** 부를 수 있어야 한다.

    처음에는 모듈 안의 모든 호출 가능한 것을 그러모았고, 인자를 받는 보조
    함수까지 목록에 섞였다. 회귀 테스트가 그것을 인자 없이 부르면 TypeError 가
    나고, 그 실패는 "수정이 풀렸다" 로 읽힌다 — 거짓 경보가 진짜 경보를 덮는다.
    """
    if name.startswith("_") or not callable(fn):
        return False
    if getattr(fn, "__module__", None) != __name__:
        return False
    try:
        return not inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


PROBES = {name: fn for name, fn in sorted(globals().items()) if _is_probe(name, fn)}
