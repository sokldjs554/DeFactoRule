#!/usr/bin/env python3
"""검색기 비교 — CLI. 구현은 `app.retrieval.compare` 에 있다."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.io import load_jsonl  # noqa: E402
from app.core.paths import EVAL, PROCESSED, RESULTS  # noqa: E402
from app.domain.similarity import DOUBT  # noqa: E402
from app.retrieval.compare import compare  # noqa: E402
from app.retrieval.dense import DenseRetriever  # noqa: E402
from app.retrieval.hybrid import HybridRetriever  # noqa: E402
from app.retrieval.lexical import LexicalRetriever  # noqa: E402


def main() -> None:
    dev = [r for r in load_jsonl(EVAL / "nonaction_dev.jsonl") if r.get("label")]
    test = [r for r in load_jsonl(EVAL / "nonaction_test.jsonl") if r.get("label")]
    cases = load_jsonl(PROCESSED / "cases_nonaction.jsonl")
    corpus = [c["fields"].get("요청대상행위") or c["fields"].get("질의요지") or ""
              for c in cases]
    corpus = [t for t in corpus if t]

    retrievers = [LexicalRetriever(), DenseRetriever(),
                  HybridRetriever(LexicalRetriever(), DenseRetriever())]
    result = compare(retrievers, dev, test, corpus)

    print(f"선례 풀 dev {len(dev)}건 · 평가 test {len(test)}건 · "
          f"코퍼스 {len(corpus)}건 · 문턱 {DOUBT}\n")
    print(f"{'검색기':<12}{'순응':>5}{'함정':>5}{'선례없음':>9}{'함정 비율':>11}")
    for name, stats in result.items():
        rate = f"{stats['trap_rate']:.3f}" if stats["trap_rate"] is not None else "—"
        print(f"{name:<12}{stats['agree']:>5}{stats['trap']:>5}"
              f"{stats['unanchored']:>9}{rate:>11}")

    print("\n클래스별 — 선례를 찾은 비율과, 찾았을 때 함정일 비율")
    print(f"{'검색기':<12}{'라벨':<6}{'건수':>5}{'선례 있음':>10}{'함정':>6}{'함정 비율':>11}")
    for name, stats in result.items():
        for label, cell in stats["by_label"].items():
            rate = (f"{cell['trap_rate']:.3f}" if cell["trap_rate"] is not None
                    else "—")
            print(f"{name:<12}{label:<6}{cell['n']:>5}{cell['anchored']:>10}"
                  f"{cell['trap']:>6}{rate:>11}")

    path = RESULTS / "e8_retrievers.json"
    path.write_text(json.dumps({"floor": DOUBT, "n_dev": len(dev), "n_test": len(test),
                                "retrievers": result}, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
