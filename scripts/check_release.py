#!/usr/bin/env python3
"""배포 전에, 서비스가 읽을 산출물이 전부 제자리에 있는지 확인한다.

## 왜 health check 로 하지 않는가

`render.yaml` 의 `healthCheckPath` 는 배포 때만 도는 것이 아니다. Render 는 살아
있는 인스턴스에도 계속 요청을 보내고, 5xx 가 이어지면 **인스턴스를 재시작한다**
(15초 연속 실패 → 트래픽 차단, 60초 연속 실패 → 재시작).

산출물이 빠진 것은 재시작으로 고쳐지지 않는다. 그래서 `/health` 가 그 상황에서
503 을 돌려주게 만들면, 낫지 않는 상태를 낫게 하려고 무한히 재시작하다가 서비스가
아예 죽는다. 헬스체크는 **살아 있는가**를 묻는 자리이지 **준비됐는가**를 묻는
자리가 아니다.

준비 여부는 더 앞에서, 되돌릴 수 있을 때 막아야 한다. 이 스크립트를 빌드 명령에
붙이면 산출물이 없을 때 **빌드가 실패하고 배포 자체가 일어나지 않는다.** 그러면
이미 떠 있는 인스턴스가 그대로 서빙한다.

    buildCommand: pip install -r requirements.txt && python3 scripts/check_release.py

## 무엇을 보는가

`app.api.main` 을 실제로 임포트한다. 그것만으로도 문법 오류·의존성 누락·경로
계산 실패가 여기서 걸린다. 그다음 각 엔드포인트가 읽는 파일을 하나씩 확인한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    try:
        from app.api import main as api
        from app.core.paths import check_root
        from app.evaluation.failure_taxonomy import REGISTRY_PATH
    except Exception as exc:  # noqa: BLE001
        print(f"[X] app.api.main 을 임포트하지 못했습니다 — {type(exc).__name__}: {exc}")
        return 1
    print("[O] app.api.main 임포트")

    if not check_root():
        print("[X] 저장소 루트 계산이 어긋났습니다 (app/core/paths.py)")
        return 1
    print("[O] 저장소 루트 확인")

    # (표시할 이름, 경로, 그 파일을 읽는 엔드포인트)
    required = [
        ("평가셋", api.GOLD, "/evaluation/models, /evaluation/risk-coverage"),
        ("clean test", api.TEST_CLEAN, "/evaluation/summary"),
        ("최종 평가 산출물", api.FINAL_FREEZE, "/evaluation/summary"),
        ("질의·회답 쌍", api.QA_PAIRS, "/evaluation/summary"),
        ("업권별 기저율", api.DEV_BASE_RATES, "/base-rates"),
        ("실패 사례 레지스트리", REGISTRY_PATH, "/failures"),
        ("화면", api.STATIC / "index.html", "/"),
    ]
    for name in api.CORPUS_FILES:
        required.append(("사례 " + name, api.PROCESSED / name, "/evaluation/summary"))

    missing = []
    for label, path, used_by in required:
        if path.exists():
            print(f"[O] {label}")
        else:
            missing.append((label, path, used_by))
            print(f"[X] {label} 없음 — {path}  ({used_by} 가 읽습니다)")

    preds = sorted(api.PROCESSED.glob("pred_nonaction_*.jsonl"))
    if preds:
        print(f"[O] 예측 파일 {len(preds)}개")
    else:
        missing.append(("예측 파일", api.PROCESSED, "/evaluation/models"))
        print(f"[X] 예측 파일이 하나도 없습니다 — {api.PROCESSED}/pred_nonaction_*.jsonl")

    print()
    if missing:
        print(f"==> 산출물 {len(missing)}건이 빠져 배포할 수 없습니다.")
        print("    빌드를 실패시켜 기존 인스턴스가 계속 서빙하도록 둡니다.")
        return 1
    print("==> 배포 가능합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
