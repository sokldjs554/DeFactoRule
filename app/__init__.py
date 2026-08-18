"""DeFactoRule — 금융 규제당국 회신문에서 공표된 적 없는 판단 기준을 찾는다.

계층은 아래에서 위로 흐른다. 위 계층은 아래를 알아도 되지만 그 반대는 안 된다.

    core            공용 입출력과 레코드 키. 도메인을 모른다.
    domain          라벨 체계와 판정 지침. 이 프로젝트가 무엇을 판단하는가.
    extraction      PDF → 사례 → 질의·회답 쌍. 결정론적.
    rules           결정론적 기준선. LLM 없이 어디까지 되는가.
    retrieval       근거 검색. 아직 비어 있다.
    agents          LLM 호출 계층. 후보 생성과 의미 해석만 맡는다.
    evaluation      채점·통계·오류 분석. 어떤 모델도 자기 점수를 매기지 않는다.
    infrastructure  외부 시스템(Anthropic API 등)과의 경계.
    api             서비스 진입점.
"""
