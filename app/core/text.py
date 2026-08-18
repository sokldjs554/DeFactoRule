"""문서 텍스트 위생 — 조판 잔재를 다루는 한 곳.

PDF 에서 뽑은 텍스트에는 눈에 보이지 않거나 엉뚱하게 매핑된 글자가 섞인다.
비조치의견서 판단이유 255건 중 **252건(98.8%)** 에 하나 이상 들어 있다.

    U+2244  NOT ASYMPTOTICALLY EQUAL TO   360회   글머리 기호가 깨진 것
    U+200C  ZERO WIDTH NON-JOINER         266회   보이지 않는다
    U+25A1  WHITE SQUARE                  362회   글머리 기호 (구조)
    U+00B7  MIDDLE DOT                    303회   글머리 기호 (구조)

두 가지를 구분해야 한다.

  **읽기용**   보이지 않는 잡티만 걷어내고 글머리 기호는 남긴다. 목록 구조가
               사라지면 사람도 모델도 문단을 잘못 읽는다.
  **대조용**   글머리 기호까지 전부 걷어낸다. 인용이 원문에 있는지 볼 때
               모델이 글머리 기호를 옮겨 적었는지는 중요하지 않다.

이 구분이 없으면 정상적인 인용이 "원문에 없다" 로 버려진다. 실측으로 확인했다 —
잔재를 뺀 인용을 원문과 대조하면 실패하고, 그러면 기준이 전부 폐기된다.
"""

from __future__ import annotations

import re

# 보이지 않거나 깨진 글자. 읽기용에서도 걷어낸다.
INVISIBLE = re.compile(
    "[\u0000-\u0008\u000b-\u001f\u007f\u00ad"
    "\u200b-\u200f\u2028\u2029\u2044\u2244\ufeff]"
)

# 글머리 기호. 읽기용에서는 남기고 대조용에서만 걷어낸다.
BULLETS = re.compile("[\u25a1\u25cb\u25e6\u25aa\u25cf\u00b7\u2022\u25b6\u25c6\u2219]")

WHITESPACE = re.compile(r"\s+")


def clean_for_prompt(text: str) -> str:
    """모델에게 보여줄 텍스트. 잡티만 걷어내고 구조는 남긴다."""
    text = INVISIBLE.sub("", text or "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_match(text: str) -> str:
    """인용 대조용. 공백·잡티·글머리 기호를 전부 걷어낸다.

    글자는 건드리지 않는다 — 뜻이 바뀐 인용은 걸러져야 하기 때문이다.
    """
    text = INVISIBLE.sub("", text or "")
    text = BULLETS.sub("", text)
    return WHITESPACE.sub("", text)
