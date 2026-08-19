"""저장소 안의 고정 경로.

`Path(__file__).parents[N]` 을 모듈마다 따로 쓰면, 파일을 옮기는 순간 N 이
틀어진다. 예외도 나지 않는다 — 존재하지 않는 경로가 조용히 만들어지고,
그 경로를 읽으려 할 때가 되어서야 터진다.

실제로 아키텍처 재편에서 `classify_llm.py` 를 `app/agents/classifier.py` 로
옮겼을 때 `parents[1]` 이 저장소 루트에서 `app/` 로 바뀌었고,
기저율 파일 경로가 `app/data/eval/...` 를 가리켰다. 테스트가 API 키를 요구하는
경로였던 탓에 어느 테스트도 이것을 건드리지 않았다.

그래서 루트를 한 곳에서만 계산하고, 그 계산이 맞는지 자체 검증한다.
"""

from __future__ import annotations

from pathlib import Path

# app/core/paths.py -> app/core -> app -> 저장소 루트
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
EVAL = DATA / "eval"
FAILURES = DATA / "failures"

MODELS = ROOT / "models"
EXPERIMENTS = ROOT / "experiments"
RESULTS = EXPERIMENTS / "results"

DEV_BASE_RATES = EVAL / "dev_base_rates.json"
SPACING_MODEL = MODELS / "spacing.json"


def check_root() -> bool:
    """ROOT 가 정말 저장소 루트인지. 파일이 옮겨지면 여기서 먼저 걸린다."""
    return (ROOT / "pyproject.toml").is_file() and (ROOT / "app").is_dir()
