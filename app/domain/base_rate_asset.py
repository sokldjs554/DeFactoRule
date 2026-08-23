"""기저율 표의 **출처를 파일 안에 못 박는다.**

## 왜 필요한가

`data/eval/dev_base_rates.json` 은 `source: "dev"` 라고만 적혀 있다. 그런데
이 저장소에는 이제 dev 가 둘이다 — legacy 85건과 clean 87건. 그리고 **clean
test 168건 중 54건이 legacy dev 안에 있었다**(`docs/25 §A.8`).

기저율 표는 LLM 프롬프트로 들어간다(`app/agents/classifier.py`). 그러므로
legacy 표를 그대로 두고 clean test 를 평가하면, **그 54건의 정답이 집계된
형태로 사전확률에 실려 프롬프트에 들어간다.**

기존 방지 장치(`probes.base_rates_come_from_dev_only`)는 이것을 못 잡는다.
`source == "dev"` 만 보고 **어느 dev 인지는 보지 않기 때문이다.**

## 그래서 이름표가 아니라 지문을 남긴다

`split: "clean"` 이라고 적어 두는 것만으로는 부족하다 — 손으로 고칠 수 있는
글자다. 그래서 **행 키의 지문**을 함께 적는다.

    row_key_digest   정렬한 (source, page, serial, pair_index) 목록의 SHA-256

검증은 재계산이다. clean dev 파일에서 지문을 다시 만들어 같은지 본다. 다르면
그 표는 clean dev 에서 나온 것이 아니다. 이름표는 거짓말할 수 있지만 지문은
못 한다.

`method_version` 도 같은 이유다. 계산 함수의 원문을 해시해 둔다. 셈법이
바뀌면 지문이 바뀌고, 옛 표가 새 코드의 산출물인 척할 수 없다.

## 이 파일이 하지 않는 일

**계산하지 않는다.** 분포 계산은 `app.domain.base_rates.compute` 가 그대로
한다 — 절차를 복제하면 두 값이 언젠가 갈린다. 여기서 하는 것은 그 결과에
출처를 붙이고, 붙인 것이 사실인지 다시 재는 것뿐이다.

## 왜 `domain` 에 있는가

`base_rates` 와 같은 이유다(`docs/11 §아키텍처`). 분류기(`agents`)와 채점
계층(`evaluation`)이 **둘 다** 이 검증을 부른다. `evaluation` 에 두면
`agents` 가 채점 계층을 임포트하게 되고, 그것은 의존 방향을 거스른다.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

from app.core.io import key_of
from app.domain.base_rates import compute

SCHEMA_VERSION = 1
METHOD = "app.domain.base_rates.compute"

# **짝이 맞아야 한다.** `source` 문자열 하나만 보고 통과시키면, dev 가 둘인
# 지금은 아무것도 막지 못한다.
ALLOWED_IDENTITY = {("dev", "legacy"), ("clean_dev", "clean")}

# 그리고 **이름표는 특정 행 집합에 묶여야 한다.** 이것이 없으면
# "legacy dev 로 만들었는데 split 은 clean 이라고 적은 표" 가 통과한다 —
# 내부적으로는 전부 앞뒤가 맞기 때문이다. 실제로 회귀 테스트가 그것을 잡았다.
SPLIT_DEV_FILE = {
    "legacy": "data/eval/nonaction_dev.jsonl",
    "clean": "data/eval/nonaction_dev_clean.jsonl",
}

REQUIRED_PROVENANCE = ("method", "method_version", "input", "input_sha256",
                       "row_key_digest", "n_rows_read")


class ProvenanceError(RuntimeError):
    """기저율 표의 출처를 확인하지 못했다. **읽지 않고 멈춘다.**"""


def row_key_digest(rows: list[dict]) -> str:
    """행 키의 지문. **어느 행에서 나왔는가**를 나중에 다시 잴 수 있게 한다."""
    keys = sorted(str(key_of(r)) for r in rows)
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def method_version() -> str:
    """계산 함수 원문의 지문. 셈법이 바뀌면 이 값이 바뀐다."""
    source = inspect.getsource(compute)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def build(rows: list[dict], input_path: Path, split: str, source: str) -> dict:
    """`compute` 의 결과에 출처를 붙인다. 분포 값은 손대지 않는다."""
    table = compute(rows)
    table["source"] = source
    table["split"] = split
    table["provenance"] = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "method_version": method_version(),
        "input": str(input_path).replace(str(Path.cwd()) + "/", ""),
        "input_sha256": file_digest(input_path),
        "row_key_digest": row_key_digest(rows),
        "n_rows_read": len(rows),
        "test_files_read": [],
    }
    return table


def _rows_of(path: Path) -> list[dict]:
    from app.core.io import load_jsonl

    return [r for r in load_jsonl(path) if r.get("label")]


def validate(table: dict, root: Path | None = None) -> list[str]:
    """표가 **스스로 밝힌 출처와 실제로 일치하는가.** 문제 목록을 돌려준다.

    이름표를 믿지 않는다. 표가 적어 둔 입력 파일을 열어 지문을 **다시 만들고**
    분포를 **다시 계산해서** 대조한다. 그래서 `source` 문자열이나 파일 이름만
    맞춘 표는 여기를 통과하지 못한다.
    """
    from app.core.paths import ROOT

    root = root or ROOT
    problems: list[str] = []

    identity = (table.get("source"), table.get("split"))
    if identity not in ALLOWED_IDENTITY:
        problems.append(
            f"source/split 짝이 허용 목록에 없다: {identity!r} "
            f"(허용 {sorted(ALLOWED_IDENTITY)})")

    prov = table.get("provenance") or {}
    missing = [k for k in REQUIRED_PROVENANCE if not prov.get(k)]
    if missing:
        return problems + [f"provenance 에 없는 항목: {missing}"]

    if prov["method"] != METHOD:
        problems.append(f"셈법이 {prov['method']!r} 이다 (기대 {METHOD!r})")
    if prov["method_version"] != method_version():
        problems.append("셈법 지문이 다르다 — 표를 만든 코드가 지금 코드와 다르다")

    input_path = root / prov["input"]
    if "test" in input_path.name:
        problems.append(f"test 파일에서 만든 표다: {prov['input']}")

    expected = SPLIT_DEV_FILE.get(table.get("split"))
    if expected and (root / expected).resolve() != input_path.resolve():
        problems.append(
            f"split {table['split']!r} 은 {expected} 에서 나와야 하는데 "
            f"{prov['input']} 에서 나왔다")

    if not input_path.exists():
        return problems + [f"적어 둔 입력 파일이 없다: {prov['input']}"]
    if file_digest(input_path) != prov["input_sha256"]:
        problems.append(f"입력 파일 지문이 다르다: {prov['input']}")

    rows = _rows_of(input_path)
    if row_key_digest(rows) != prov["row_key_digest"]:
        problems.append("행 지문이 다르다 — 이 표는 그 파일에서 나오지 않았다")
    if table.get("n") != len(rows):
        problems.append(f"n 이 {table.get('n')} 이다 (입력 {len(rows)}행)")
    if prov["n_rows_read"] != len(rows):
        problems.append(f"n_rows_read 가 {prov['n_rows_read']} 이다 (입력 {len(rows)}행)")

    fresh = compute(rows)
    if table.get("overall") != fresh["overall"]:
        problems.append("전체 분포가 재계산 결과와 다르다")
    if table.get("sectors") != fresh["sectors"]:
        problems.append("업권 분포가 재계산 결과와 다르다")
    if table.get("min_sector_n") != fresh["min_sector_n"]:
        problems.append("min_sector_n 이 재계산 결과와 다르다")
    return problems


def verify(table: dict, rows: list[dict], expect_split: str,
           expect_source: str) -> list[str]:
    """만든 쪽의 자기 점검. `validate` 에 **기대한 정체성**까지 얹는다."""
    problems = validate(table)
    if table.get("split") != expect_split:
        problems.append(f"split 이 {table.get('split')!r} 이다 (기대 {expect_split!r})")
    if table.get("source") != expect_source:
        problems.append(f"source 가 {table.get('source')!r} 이다 (기대 {expect_source!r})")
    if table.get("n") != len(rows):
        problems.append(f"n 이 {table.get('n')} 이다 (기대 {len(rows)})")
    prov = table.get("provenance") or {}
    if prov.get("row_key_digest") != row_key_digest(rows):
        problems.append("행 지문이 건네받은 행들과 다르다")
    return problems


def asset_record(table: dict, path: Path) -> dict:
    """실행 기록에 남길 한 줄. **어느 표를 썼는지 나중에 재현하기 위한 것이다.**"""
    from app.core.paths import ROOT

    prov = table["provenance"]
    # 저장소 기준 상대경로로 적는다. 절대경로를 남기면 기록이 그 기계에서만
    # 뜻이 통한다 — 기본값은 절대경로, 인자로 준 값은 상대경로였다.
    try:
        shown = str(path.resolve().relative_to(ROOT))
    except ValueError:
        shown = str(path)
    return {
        "path": shown,
        "source": table["source"],
        "split": table["split"],
        "n": table["n"],
        "row_key_digest": prov["row_key_digest"],
        "method_version": prov["method_version"],
        "input": prov["input"],
        "input_sha256": prov["input_sha256"],
    }


def load_validated(path: Path) -> tuple[dict, dict]:
    """읽고, 검증하고, 실행 기록을 함께 돌려준다. **검증에 실패하면 예외다.**

    호출부가 검사를 잊는 일이 없도록 읽기와 검사를 한 함수에 묶는다 — 예전
    가드가 `source == "dev"` 한 줄이었던 것은, 검사가 호출부에 흩어져 있었기
    때문이다.
    """
    if not path.exists():
        raise ProvenanceError(f"기저율 파일이 없습니다: {path}")
    table = json.loads(path.read_text(encoding="utf-8"))
    problems = validate(table)
    if problems:
        raise ProvenanceError(
            f"{path} 의 출처를 확인하지 못했습니다:\n  - " + "\n  - ".join(problems))
    return table, asset_record(table, path)


def main() -> None:
    import argparse

    from app.core.io import load_jsonl
    from app.core.paths import DEV_BASE_RATES, EVAL

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", default=str(EVAL / "nonaction_dev_clean.jsonl"))
    ap.add_argument("--out", default=str(EVAL / "dev_base_rates_clean.json"))
    ap.add_argument("--split", default="clean")
    ap.add_argument("--source", default="clean_dev")
    ap.add_argument("--write", action="store_true",
                    help="파일을 쓴다. 없으면 표만 보여 준다.")
    ap.add_argument("--restamp", action="store_true",
                    help="이미 있는 표에 출처만 다시 찍는다. **분포 값이 한 글자라도 "
                         "달라지면 쓰지 않는다.**")
    args = ap.parse_args()

    dev_path, out_path = Path(args.dev), Path(args.out)

    # 못 하게 막는 것 셋. 주석이 아니라 코드로 막는다.
    if "test" in dev_path.name:
        raise SystemExit(f"test 파일을 입력으로 받았습니다: {dev_path.name}")
    if out_path.resolve() == DEV_BASE_RATES.resolve() and not args.restamp:
        raise SystemExit("legacy 기저율 파일을 덮어쓰려 했습니다. "
                         "출처만 찍으려면 --restamp 를 쓰세요.")
    if out_path.exists() and args.write and not args.restamp:
        raise SystemExit(f"이미 있는 파일입니다: {out_path} — 지우고 다시 부르세요.")

    rows = [r for r in load_jsonl(dev_path) if r.get("label")]
    table = build(rows, dev_path, args.split, args.source)
    problems = verify(table, rows, args.split, args.source)

    if args.restamp:
        # **값은 손대지 않는다.** 옛 표와 분포가 다르면 그것은 재계산이지
        # 재각인이 아니다. 다르면 쓰지 않고 죽는다.
        if not out_path.exists():
            raise SystemExit(f"--restamp 인데 대상 파일이 없습니다: {out_path}")
        before = json.loads(out_path.read_text(encoding="utf-8"))
        for field in ("n", "min_sector_n", "overall", "sectors", "source"):
            if before.get(field) != table.get(field):
                raise SystemExit(
                    f"--restamp 인데 '{field}' 가 달라집니다. 쓰지 않았습니다.\n"
                    f"  전 {before.get(field)!r}\n  후 {table.get(field)!r}")
        print("재각인 — 분포 값은 그대로이고 출처만 붙습니다 "
              f"(추가되는 키 {sorted(set(table) - set(before))})")

    print(f"{dev_path.name} {len(rows)}건 · split {table['split']} · "
          f"source {table['source']}")
    print("전체: " + ", ".join(f"{k} {v:.1%}" for k, v in table["overall"].items()))
    print(f"\n{'업권':>12}  {'건수':>4}  {'사용':>4}  분포")
    for sector, info in sorted(table["sectors"].items(), key=lambda kv: -kv[1]["n"]):
        use = "업권" if info["reliable"] else "전체"
        dist = " · ".join(f"{k} {v:.0%}" for k, v in info["rates"].items() if v)
        print(f"{sector:>12}  {info['n']:>4}  {use:>4}  {dist}")

    print(f"\n행 지문   {table['provenance']['row_key_digest'][:16]}…")
    print(f"셈법 지문 {table['provenance']['method_version']}")
    print(f"검증      {'통과' if not problems else '실패'}")
    for problem in problems:
        print(f"  - {problem}")
    if problems:
        raise SystemExit("검증에 실패했습니다. 쓰지 않았습니다.")

    if not args.write:
        print("\n아직 쓰지 않았습니다. --write 를 붙이면 씁니다.")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(table, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"-> {out_path}")
