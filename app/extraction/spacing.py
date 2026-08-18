"""띄어쓰기가 소실된 사례의 공백을 복원한다.

비조치의견서 PDF 중 일부는 글자 배치 방식 때문에 한글 사이 공백이 통째로
사라진 채 추출된다.

    금융위원회및금융감독원은｢개인금융채권의관리및...

이 상태로 토크나이저에 넣으면 어절 경계가 사라져 다운스트림이 전부 망가진다.

외부 띄어쓰기 모델을 가져오지 않고 **이 코퍼스 자체로 학습한다.** 정상 추출된
사례에 82만여 자가 있고, 그것은 복원 대상과 같은 문서군·같은 문체·같은 법령
용어를 쓴다. 범용 모델보다 도메인이 정확히 일치하며, 학습·평가·적용이 모두
저장소 안에서 재현된다.

모델은 경계 문맥 n-gram 의 공백 출현 빈도를 세는 방식이다. 각 글자 사이 위치마다
경계를 걸치는 여러 문맥을 뽑아, 학습 코퍼스에서 그 문맥에 공백이 있었는지를
로그 오즈로 합산한다.

    python scripts/restore_spacing.py train --input data/processed --model models/spacing.json
    python scripts/restore_spacing.py eval  --input data/processed --model models/spacing.json
    python scripts/restore_spacing.py apply --input data/processed --model models/spacing.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

# 추출 과정에서 끼어드는 조판 잔재 — 제어문자, 폭 없는 문자, 글머리로 쓰인 U+2244.
# 줄바꿈과 탭은 남긴다.
JUNK = re.compile(
    "[\\x00-\\x08\\x0b-\\x1f\\x7f"      # 제어문자 (줄바꿈·탭은 남긴다)
    "\\u00ad\\u200b-\\u200f"            # soft hyphen, 폭 없는 문자
    "\\u2028\\u2029\\ufeff"             # 줄/문단 구분자, BOM
    "\\u2244]"                        # 글머리로 잘못 들어온 U+2244
)

# 경계를 걸치는 문맥 창. (왼쪽 글자수, 오른쪽 글자수)
WINDOWS = ((1, 1), (2, 1), (1, 2), (2, 2), (3, 2), (2, 3))

# 학습 코퍼스에서 이만큼도 안 나온 문맥은 버린다. 모델 크기와 잡음을 함께 줄인다.
MIN_COUNT = 3
# 문맥 하나가 판단을 독점하지 않게 로그 오즈를 자른다.
CLAMP = 4.0


def clean(text: str) -> str:
    return JUNK.sub("", text)


def contexts(chars: str, i: int) -> list[str]:
    """chars[i] 와 chars[i+1] 사이 경계의 문맥들."""
    out = []
    for left, right in WINDOWS:
        lo, hi = i - left + 1, i + 1 + right
        if lo < 0 or hi > len(chars):
            continue
        out.append(f"{left}_{right}:{chars[lo:hi]}")
    return out


def iter_boundaries(line: str) -> tuple[str, list[int]]:
    """공백이 있는 원문 한 줄에서 (공백 없는 글자열, 경계 라벨) 을 만든다.

    labels[i] 는 chars[i] 와 chars[i+1] 사이에 공백이 있었는지를 뜻한다.

    처음 구현은 공백을 만났을 때 labels[-1] 을 1로 세웠는데, 그것은 다가올
    경계가 아니라 **직전** 경계다. 모든 라벨이 한 칸 왼쪽으로 밀렸고,
    학습과 평가가 같은 함수를 쓴 탓에 지표는 버그와 함께 일관되어 멀쩡해
    보였다. 출력을 눈으로 보고서야 드러났다 ("후유증으 로", "것 이").
    """
    chars: list[str] = []
    labels: list[int] = []
    pending_space = False
    for ch in line:
        if ch == " ":
            pending_space = bool(chars)
            continue
        if chars:
            labels.append(1 if pending_space else 0)
        chars.append(ch)
        pending_space = False
    return "".join(chars), labels


def train(texts: list[str]) -> dict:
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [무공백, 공백]
    for text in texts:
        for raw_line in clean(text).split("\n"):
            line = re.sub(r"[ \t]+", " ", raw_line).strip()
            if len(line) < 4:
                continue
            chars, labels = iter_boundaries(line)
            for i, label in enumerate(labels):
                for ctx in contexts(chars, i):
                    stats[ctx][label] += 1

    model = {ctx: [n0, n1] for ctx, (n0, n1) in stats.items() if n0 + n1 >= MIN_COUNT}
    total0 = sum(v[0] for v in model.values())
    total1 = sum(v[1] for v in model.values())
    prior = math.log((total1 + 1) / (total0 + 1))
    return {"contexts": model, "prior": prior, "windows": [list(w) for w in WINDOWS]}


def boundary_score(model: dict, chars: str, i: int) -> float:
    ctxs = model["contexts"]
    score = 0.0
    used = 0
    for ctx in contexts(chars, i):
        hit = ctxs.get(ctx)
        if not hit:
            continue
        n0, n1 = hit
        odds = math.log((n1 + 0.5) / (n0 + 0.5))
        score += max(-CLAMP, min(CLAMP, odds))
        used += 1
    if used == 0:
        return model["prior"]
    return score / used


def restore_line(model: dict, line: str, threshold: float = 0.0) -> str:
    """공백을 넣기만 한다. 원문에 남아 있는 공백은 절대 지우지 않는다.

    소실됐다고 표시된 사례에도 공백이 3~4% 남아 있다(숫자·문장부호 주변).
    그것은 조판에서 살아남은 진짜 경계이므로 모델 예측보다 신뢰도가 높다.
    처음 구현은 전부 지우고 다시 예측해 맞던 공백까지 망가뜨렸다.
    """
    chars, kept = iter_boundaries(line)
    if len(chars) < 2:
        return line
    out = [chars[0]]
    for i in range(len(chars) - 1):
        if kept[i] or boundary_score(model, chars, i) > threshold:
            out.append(" ")
        out.append(chars[i + 1])
    return "".join(out)


def restore(model: dict, text: str, threshold: float = 0.0) -> str:
    return "\n".join(
        restore_line(model, line, threshold) for line in clean(text).split("\n")
    )


def score_text(model: dict, reference: str, threshold: float) -> tuple[int, int, int]:
    """공백 위치 기준 (맞춘 수, 예측 수, 정답 수)."""
    tp = pred = gold = 0
    for raw_line in clean(reference).split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if len(line) < 4:
            continue
        chars, labels = iter_boundaries(line)
        for i, label in enumerate(labels):
            hit = boundary_score(model, chars, i) > threshold
            pred += hit
            gold += label
            tp += hit and bool(label)
    return tp, pred, gold


def prf(tp: int, pred: int, gold: int) -> tuple[float, float, float]:
    p = tp / pred if pred else 0.0
    r = tp / gold if gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def load_cases(in_dir: Path) -> list[dict]:
    cases: list[dict] = []
    for path in sorted(in_dir.glob("cases_*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            cases.extend(json.loads(line) for line in fh if line.strip())
    return cases


def split_corpus(cases: list[dict]) -> tuple[list[dict], list[dict]]:
    """공백이 살아 있는 사례와 소실된 사례로 가른다."""
    lost = [c for c in cases if "spacing_lost" in c.get("warnings", [])]
    ok = [c for c in cases if "spacing_lost" not in c.get("warnings", [])]
    return ok, lost


def holdout(ok: list[dict], every: int = 10) -> tuple[list[dict], list[dict]]:
    """평가용을 결정론적으로 떼어낸다. 난수를 쓰지 않아 재현된다."""
    test = [c for i, c in enumerate(ok) if i % every == 0]
    train_set = [c for i, c in enumerate(ok) if i % every != 0]
    return train_set, test


def cmd_train(args: argparse.Namespace) -> None:
    cases = load_cases(Path(args.input))
    ok, lost = split_corpus(cases)
    train_set, test = holdout(ok)
    model = train([c["raw"] for c in train_set])
    out = Path(args.model)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")
    print(f"학습 {len(train_set)}건 · 평가 보류 {len(test)}건 · 복원 대상 {len(lost)}건")
    print(f"문맥 {len(model['contexts']):,}개  ({out.stat().st_size / 1e6:.1f} MB)")


def cmd_eval(args: argparse.Namespace) -> None:
    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    cases = load_cases(Path(args.input))
    ok, _ = split_corpus(cases)
    _, test = holdout(ok)

    # 복원 대상은 전부 비조치의견서다. 전체 평균을 보고하면 법령해석(다수)의
    # 성능에 가려져 실제 적용 성능을 과대평가하게 된다. 문체별로 따로 잰다.
    groups = {
        "전체": test,
        "비조치 (적용 대상)": [c for c in test if c["doc_type"] == "nonaction"],
        "법령해석": [c for c in test if c["doc_type"] == "interpretation"],
    }
    thresholds = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0)

    for label, subset in groups.items():
        if not subset:
            continue
        print(f"\n{label} — {len(subset)}건")
        print(f"{'임계값':>8}  {'정밀도':>7}  {'재현율':>7}  {'F1':>7}")
        best: tuple[float | None, float] = (None, -1.0)
        for threshold in thresholds:
            tp = pred = gold = 0
            for c in subset:
                a, b, g = score_text(model, c["raw"], threshold)
                tp, pred, gold = tp + a, pred + b, gold + g
            p, r, f = prf(tp, pred, gold)
            print(f"{threshold:>8.2f}  {p:>7.3f}  {r:>7.3f}  {f:>7.3f}")
            if f > best[1]:
                best = (threshold, f)
        print(f"  최적 {best[0]} · F1 {best[1]:.3f}")


def cmd_apply(args: argparse.Namespace) -> None:
    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    in_dir = Path(args.input)
    changed = 0
    for path in sorted(in_dir.glob("cases_*.jsonl")):
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
        for row in rows:
            if "spacing_lost" not in row.get("warnings", []):
                continue
            row["raw_restored"] = restore(model, row["raw"], args.threshold)
            row["fields_restored"] = {
                k: restore(model, v, args.threshold) for k, v in row["fields"].items()
            }
            changed += 1
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"복원 적용 {changed}건 — raw_restored / fields_restored 추가")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("train", cmd_train), ("eval", cmd_eval), ("apply", cmd_apply)):
        p = sub.add_parser(name)
        p.add_argument("--input", required=True)
        p.add_argument("--model", required=True)
        if name == "apply":
            p.add_argument("--threshold", type=float, default=0.0)
        p.set_defaults(func=fn)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
