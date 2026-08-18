"""수집 대상 사이트의 구조를 정찰한다.

개발 컨테이너에서는 대상 도메인이 egress 차단되어 있어 파서를 추측으로 쓸 수 없다.
이 스크립트를 접근 가능한 로컬 환경에서 돌려 구조 요약을 얻은 뒤,
그 결과를 근거로 실제 파서를 작성한다.

    pip install requests beautifulsoup4
    python scripts/probe_site.py --target track_a_list
    python scripts/probe_site.py --url "https://..." --label my_page

출력:
    data/raw/probe/<label>.html   원본 HTML (파서 작성 근거)
    data/raw/probe/<label>.json   구조 요약 (사람이 읽고 공유하기 위한 것)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import requests
from bs4 import BeautifulSoup

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "probe"

# 정찰 대상. URL은 로컬에서 실제 화면을 보고 확정한다.
TARGETS = {
    "track_a_portal": "https://better.fsc.go.kr/",
    "track_a_archive": "https://better.fsc.go.kr/fsc_new/RecsroomList.do?stNo=11&muNo=146&muGpNo=75",
    "track_b_sandbox": "https://sandbox.fintech.or.kr/business/enterprise_intro.do?lang=ko",
    "track_b_press": "https://www.fsc.go.kr/no010101",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "ko-KR,ko;q=0.9",
}

DOC_SUFFIXES = (".pdf", ".hwp", ".hwpx", ".xlsx", ".xls", ".doc", ".docx", ".zip")


def summarize(html: str, url: str) -> dict:
    """파서를 쓰기 위해 알아야 하는 것만 뽑는다."""
    soup = BeautifulSoup(html, "html.parser")

    tables = []
    for i, t in enumerate(soup.find_all("table")):
        headers = [th.get_text(strip=True) for th in t.find_all("th")]
        rows = t.find_all("tr")
        first_row = (
            [td.get_text(strip=True)[:60] for td in rows[1].find_all("td")]
            if len(rows) > 1
            else []
        )
        tables.append(
            {
                "index": i,
                "class": t.get("class"),
                "id": t.get("id"),
                "header_cells": headers[:20],
                "row_count": len(rows),
                "first_data_row": first_row[:20],
            }
        )

    # 목록 페이지는 같은 class 를 가진 항목이 반복된다 — 그 반복 단위를 찾는다.
    class_counts = Counter()
    for el in soup.find_all(["li", "div", "tr", "article"]):
        for cls in el.get("class") or []:
            class_counts[f"{el.name}.{cls}"] += 1
    repeated = [
        {"selector": sel, "count": n}
        for sel, n in class_counts.most_common(25)
        if n >= 5
    ]

    links = [a for a in soup.find_all("a", href=True)]
    downloads = sorted(
        {
            a["href"]
            for a in links
            if a["href"].lower().endswith(DOC_SUFFIXES)
            or "getFile" in a["href"]
            or "download" in a["href"].lower()
        }
    )
    # 상세 페이지 링크 후보: 쿼리스트링에 숫자 id 가 붙은 것
    detail = sorted(
        {a["href"] for a in links if re.search(r"(No|Seq|Id|id|no)=\d+", a["href"])}
    )

    forms = [
        {
            "action": f.get("action"),
            "method": (f.get("method") or "get").lower(),
            "fields": [
                i.get("name") for i in f.find_all(["input", "select"]) if i.get("name")
            ][:25],
        }
        for f in soup.find_all("form")
    ]

    # 총 건수 표기 ("총 1,560건")
    text = soup.get_text(" ", strip=True)
    counts = re.findall(r"(?:총|전체)\s*([\d,]+)\s*(?:건|개)", text)

    return {
        "url": url,
        "title": soup.title.get_text(strip=True) if soup.title else None,
        "html_bytes": len(html),
        "declared_counts": counts[:5],
        "tables": tables[:10],
        "repeated_blocks": repeated,
        "forms": forms[:10],
        "download_links": downloads[:40],
        "detail_links_sample": detail[:20],
        "detail_links_total": len(detail),
        "pagination_hints": sorted(
            {a["href"] for a in links if re.search(r"page|Page|pg", a["href"])}
        )[:15],
    }


def probe(label: str, url: str, timeout: int = 30) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or resp.encoding

    (OUT_DIR / f"{label}.html").write_text(resp.text, encoding="utf-8")
    result = summarize(resp.text, url)
    result["status_code"] = resp.status_code
    result["final_url"] = resp.url
    (OUT_DIR / f"{label}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", choices=sorted(TARGETS), help="사전 정의된 대상")
    ap.add_argument("--url", help="임의 URL 직접 지정")
    ap.add_argument("--label", help="--url 사용 시 저장 파일명")
    ap.add_argument("--all", action="store_true", help="정의된 대상 전부 정찰")
    args = ap.parse_args()

    if args.all:
        jobs = list(TARGETS.items())
    elif args.target:
        jobs = [(args.target, TARGETS[args.target])]
    elif args.url:
        if not args.label:
            ap.error("--url 사용 시 --label 이 필요합니다")
        jobs = [(args.label, args.url)]
    else:
        ap.error("--target, --url, --all 중 하나를 지정하세요")

    for label, url in jobs:
        try:
            r = probe(label, url)
        except Exception as exc:  # 한 대상이 막혀도 나머지는 계속한다
            print(f"[FAIL] {label}: {type(exc).__name__}: {exc}")
            continue
        print(
            f"[OK]   {label}: {r['status_code']} · "
            f"표 {len(r['tables'])}개 · 반복블록 {len(r['repeated_blocks'])}종 · "
            f"상세링크 {r['detail_links_total']}개 · 첨부 {len(r['download_links'])}개"
        )
        if r["declared_counts"]:
            print(f"       페이지에 표기된 총 건수: {r['declared_counts']}")

    print(f"\n결과 위치: {OUT_DIR}")
    print("→ data/raw/probe/*.json 을 공유하면 이를 근거로 파서를 작성합니다.")


if __name__ == "__main__":
    main()
