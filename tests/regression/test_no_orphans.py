"""배선하지 않은 정의가 남아 있는가 — **반쪽짜리 편집을 상시로 잡는다.**

## 왜 이 파일이 있는가

편집 스크립트가 중간에 실패한 적이 몇 번 있었다. 그때마다 앞부분은 적용되고
뒷부분은 안 적용된 상태로 커밋됐고, 결과는 늘 같았다 — **만들어 놓고 아무도
부르지 않는 코드.**

    check_fingerprint   답에 지문을 심어 놓고 **읽는 쪽이 없었다** (AG-11)
    validity.Claim      "모든 수치의 무효 기준 게이트" 인데 **사용처 0** (EV-23)

둘 다 테스트는 통과했다. 정의만으로는 아무것도 깨지지 않기 때문이다. 그래서
사람이 "그거 다 됐냐" 고 묻기 전까지 드러나지 않았다.

## 무엇을 보는가

`app/` 의 공개 정의(밑줄로 시작하지 않는 함수·클래스·상수)를 모아, 저장소
어디에서도 이름이 쓰이지 않는 것을 찾는다. probe 는 이름으로 디스패치되므로
`PROBES` 에 있으면 참조된 것으로 본다.

허용 목록은 **이유를 적어야** 들어올 수 있다. 이유를 쓰다 보면 대개 "사실
지금 배선해야 한다" 는 것이 드러난다.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from app.evaluation.probes import PROBES

ROOT = Path(__file__).resolve().parents[2]

# 아직 호출자가 없어도 두는 것 — **이유가 있어야 한다.**
ALLOWED: dict[str, str] = {
    "Retriever": "검색기 계약(Protocol). 형으로만 쓰이므로 이름이 호출되지 않는다.",
}


def _public_definitions() -> dict[str, tuple[str, int]]:
    found: dict[str, tuple[str, int]] = {}
    for path in sorted((ROOT / "app").rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(ROOT).as_posix()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not node.name.startswith("_"):
                    found[node.name] = (rel, node.lineno)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        found[target.id] = (rel, node.lineno)
    return found


def _corpus() -> list[tuple[str, str]]:
    out = []
    for pattern in ("app/**/*.py", "tests/**/*.py", "scripts/*.py", "docs/*.md", "*.md"):
        for path in ROOT.glob(pattern):
            if "__pycache__" in str(path):
                continue
            out.append((path.relative_to(ROOT).as_posix(),
                        path.read_text(encoding="utf-8")))
    return out


def test_every_public_definition_is_wired():
    """만들어 놓고 아무도 부르지 않는 것이 있는가."""
    definitions = _public_definitions()
    corpus = _corpus()

    orphans = []
    for name, (where, line) in sorted(definitions.items()):
        if name in PROBES or name in ALLOWED:
            continue
        uses = 0
        for path, text in corpus:
            hits = len(re.findall(rf"\b{re.escape(name)}\b", text))
            uses += hits - 1 if path == where else hits
        if uses <= 0:
            orphans.append(f"{where}:{line} {name}")

    assert not orphans, (
        "배선되지 않은 정의가 있습니다 — 배선하거나, 지우거나, "
        "ALLOWED 에 이유와 함께 적으세요:\n  " + "\n  ".join(orphans)
    )


def test_allowlist_entries_carry_a_reason():
    """허용 목록이 이유 없이 늘어나지 않는가."""
    for name, reason in ALLOWED.items():
        assert len(reason) >= 20, f"{name} 의 허용 이유가 너무 짧습니다: {reason!r}"
