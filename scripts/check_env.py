"""실행 환경을 점검한다.

이 파일만은 **Python 2 에서도 문법 오류 없이 파싱된다.** f-string 도, 타입 힌트도
쓰지 않는다. 그래야 `python scripts/check_env.py` 를 잘못된 인터프리터로 실행했을
때도 무엇이 잘못됐는지 알려줄 수 있다.

macOS 에서 `python` 은 시스템에 남아 있는 2.7 을 가리키는 경우가 많고, 그 상태로
프로젝트 스크립트를 돌리면 파일을 찾기도 전에 죽거나 문법 오류만 뱉는다.

    python3 scripts/check_env.py
"""

import os
import sys

MIN_VERSION = (3, 9)
REQUIRED = ["pymupdf"]
OPTIONAL = {"anthropic": "LLM 분류기(classify_llm.py) 실행에 필요"}


def line(mark, text):
    sys.stdout.write(mark + " " + text + "\n")


def main():
    ok = True

    version = sys.version_info
    shown = "%d.%d.%d" % (version[0], version[1], version[2])
    if version[0] < 3 or (version[0], version[1]) < MIN_VERSION:
        ok = False
        line("[X]", "Python " + shown + " -- 이 프로젝트는 3.9 이상이 필요합니다.")
        line("   ", "macOS 라면 `python` 이 아니라 `python3` 로 실행하세요.")
        line("   ", "그래도 낮으면: brew install python@3.11")
    else:
        line("[O]", "Python " + shown)

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    if os.path.basename(os.getcwd()) and os.path.abspath(os.getcwd()) != root:
        line("[!]", "현재 위치: " + os.getcwd())
        line("   ", "저장소 루트에서 실행하세요: cd " + root)
    else:
        line("[O]", "저장소 루트에서 실행 중")

    for name in REQUIRED:
        try:
            __import__(name)
            line("[O]", name)
        except ImportError:
            ok = False
            line("[X]", name + " 없음 -- pip3 install -r requirements.txt")

    for name, why in OPTIONAL.items():
        try:
            __import__(name)
            line("[O]", name)
        except ImportError:
            line("[!]", name + " 없음 -- " + why)

    if os.environ.get("ANTHROPIC_API_KEY"):
        line("[O]", "ANTHROPIC_API_KEY 설정됨")
    else:
        line("[!]", "ANTHROPIC_API_KEY 없음 -- LLM 분류기 실행 시 필요")

    data = os.path.join(root, "data", "processed", "qa_pairs.jsonl")
    if os.path.exists(data):
        line("[O]", "data/processed/qa_pairs.jsonl")
    else:
        line("[!]", "data/processed/qa_pairs.jsonl 없음 -- 파서를 먼저 실행하세요")

    sys.stdout.write("\n")
    if ok:
        line("==>", "실행 가능합니다.")
        return 0
    line("==>", "위 [X] 항목을 먼저 해결하세요.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
