"""라벨을 찾는 정규식이 고정돼 있는가.

## 왜 이 파일이 있는가

`조치` 는 `비조치` 의 부분문자열이다. 고정하지 않은 정규식으로 결과 표에서
`조치` 행을 찾으면 **`비조치` 행이 먼저 잡힌다.** 그러면 검사는 통과하는데
읽은 것은 다른 행이다.

실제로 두 번 당했다.

    docs/15  매크로 F1 표와 AURC 표가 모양이 같아 F1 자리에 AURC 값이 들어갔다.
             구획 표시를 넣어 고쳤다(EV-14).
    weights  `r"조치\\s+\\d+\\s+([\\d.]+)"` 가 비조치 행을 읽었다. 잡음 표본에서
             조치 재현율 0.964 가 나와 통과할 뻔했다(EV-21).

둘 다 **검사가 거짓으로 통과하는** 방향으로 틀렸다. 이 방향의 오류는 스스로
드러나지 않으므로 코드를 훑는 검사가 필요하다.

## 무엇을 보는가

`re.*` 에 **실제로 넘어가는** 문자열만 본다. docstring 에 라벨 이름이 있다고
잡으면 오탐이 80건 넘게 나와 아무도 읽지 않게 된다 — 실제로 그렇게 만들어 보고
버렸다.

## 무엇이 위험한가

`비조치` 안의 `조치` 앞에는 언제나 글자 `비` 가 있다. 그러므로 **어떤 글자든**
라벨 앞에 리터럴로 있으면 그것이 곧 고정이다. 위험한 것은 그 자리에 아무것도
없거나, 빈 문자열도 될 수 있는 것이 있는 경우다.

    조치\\s+\\d+          패턴 맨 앞      -> 위험. 비조치 행이 먼저 잡힌다
    [^?]{0,24}조치       수량자 뒤       -> 위험. 빈 문자열로 줄어들 수 있다
    \\s*조치              수량자 뒤       -> 위험
    ^  조치\\s+           공백 뒤         -> 안전
    (비조치|조치|기타)    분기 첫머리      -> 안전. 왼쪽 우선이라 비조치가 이긴다

처음 만들 때 "맨 앞은 고정된 것으로 본다" 고 적었다가, **우리를 실제로 속인
그 패턴을 놓치는** 검사를 만들었다. 규칙을 거꾸로 알고 있었다.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from app.core.paths import ROOT
from app.domain.labels import NON_ACTIONS

RE_FUNCS = frozenset({
    "compile", "search", "match", "fullmatch", "findall", "finditer", "sub", "split",
})
# 빈 문자열로 줄어들 수 있는 것들. 이 뒤에 오는 라벨은 고정되지 않은 것이다.
QUANTIFIERS = ("*", "+", "?", "}")

# 다른 라벨의 꼬리인 라벨만 위험하다. `조치` 는 `비조치` 의 꼬리라 헷갈리지만,
# `비조치` 와 `기타` 를 품는 라벨은 없으므로 고정하지 않아도 헷갈릴 수 없다.
AMBIGUOUS = tuple(
    label for label in NON_ACTIONS
    if any(other != label and other.endswith(label) for other in NON_ACTIONS)
)

# 일부러 고정하지 않은 자리. **이유를 적어야 들어올 수 있다.**
ALLOWED: dict[tuple[str, str], str] = {
    ("app/agents/criteria.py", "0,24}조치"):
        "결론을 되묻는 질문을 거르는 자리다. '비조치 대상인가' 도 결론을 되묻는 "
        "질문이므로 함께 잡히는 것이 맞다.",
}


def _pattern_of(node: ast.Call) -> str | None:
    """re.<fn>(pattern, ...) 의 첫 인자를 문자열로 되돌린다. 아니면 None."""
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr not in RE_FUNCS:
        return None
    if not isinstance(func.value, ast.Name) or func.value.id != "re":
        return None
    if not node.args:
        return None
    arg = node.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    if isinstance(arg, ast.JoinedStr):
        # 치환부는 알 수 없으므로 자리표시자로 둔다
        return "".join(
            p.value if isinstance(p, ast.Constant) else "\x00" for p in arg.values
        )
    return None


def unanchored_labels(pattern: str) -> list[tuple[str, int]]:
    """패턴 안에서 고정되지 않은 라벨 출현을 (라벨, 위치) 로 돌려준다."""
    found = []
    for label in AMBIGUOUS:
        for hit in re.finditer(re.escape(label), pattern):
            longer = [
                other for other in NON_ACTIONS
                if other != label and other.endswith(label)
                and pattern[hit.start() - (len(other) - len(label)):hit.end()] == other
            ]
            if longer:
                continue
            before = pattern[: hit.start()]
            if not before or before.endswith(QUANTIFIERS):
                found.append((label, hit.start()))
    return found


def find_unanchored_label_patterns(root: Path | None = None) -> list[str]:
    """저장소 전체에서 고정되지 않은 라벨 정규식을 찾는다."""
    root = Path(root or ROOT)
    problems = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        rel = path.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            pattern = _pattern_of(node)
            if not pattern:
                continue
            for label, at in unanchored_labels(pattern):
                snippet = pattern[max(0, at - 6):at + len(label)]
                if any(rel == f and key in pattern for f, key in ALLOWED):
                    continue
                problems.append(
                    f"{rel}:{node.lineno} — '{label}' 앞이 고정되지 않았다 "
                    f"(…{snippet}…). '비조치' 가 먼저 잡힐 수 있다."
                )
    return problems
