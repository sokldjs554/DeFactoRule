"""수치를 내보내기 전에 통과해야 하는 **무효 기준**.

## 왜 이 파일이 있는가

이 저장소는 실험마다 **성공 기준**을 미리 적어 왔다. "조치 재현율 > 0.286",
"AURC < 0.124". 셋 다 *실험이 잘 됐는가* 를 묻는다.

한 번도 미리 적지 않은 것이 **무효 기준** — *이 수치를 믿어도 되는가* 다.
둘은 다르다. 결과가 좋든 나쁘든 측정 자체가 무효일 수 있다.

    표본이 계획한 전체가 아니다        26/85 에서 크레딧이 떨어졌다 (EV-22)
    표본이 모집단과 다르게 치우쳤다     조치 1.63배 · 기타 1.72배
    분모가 문턱을 판정하기에 모자란다   조치 8건으로는 기각이 불가능하다
    구간이 문턱을 걸친다               4건 중 3건 = [0.301, 0.954]

넷 다 결과와 무관하게 성립한다. 그래서 결과를 보기 전에 적을 수 있고,
**코드가 강제할 수 있다.**

## 기준은 한 종류가 아니다

여기서 한 번 더 틀릴 뻔했다. 넷을 전부 "위반이면 판정 불가" 로 묶었더니
dev 를 **끝까지 다 돌려도** 판정 불가가 나왔다. 조치 8건으로는 기각이
불가능하기 때문이다.

그러나 **기각할 수 없는 것과 확인할 수 없는 것은 다르다.** 5/8 의 구간
[0.306, 0.863] 은 하한이 문턱 위에 있으므로 "문턱보다 높다" 는 결론을
받친다. 한쪽으로만 힘이 있는 검정도 검정이다.

    무효 기준   측정 자체가 성립하지 않는다      -> 판정을 내주지 않는다
    한계 공시   측정은 성립하나 할 수 없는 말이 있다 -> 판정은 내되 반드시 밝힌다

부분 표본과 치우침은 무효 기준이다. 어느 쪽 판정도 나올 수 없는 분모도
무효 기준이다. 한쪽 판정만 나올 수 있는 분모는 **한계 공시**다.

## 무엇을 하는가

`Claim` 은 수치 하나와 그 수치의 출처를 함께 들고 다닌다. 분자·분모만으로는
만들 수 없다 — 쓴 표본과 쓰기로 했던 모집단을 함께 내야 한다. 그래야
"부분 표본인가" 를 물을 수 있다.

판정(`verdict`)은 무효 기준을 하나라도 어기면 나오지 않는다. 점추정을 적고
그 옆에 작은 글씨로 주의를 다는 방식은 이미 실패했다 — 사람은 굵은 숫자를
읽는다. 그러므로 **판정 자체를 내주지 않는다.**
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from app.evaluation.metrics import verdict_against, wilson_interval

# 기대 대비 이 배율을 벗어나면 치우친 것으로 본다. 기대가 1건 미만인 라벨은
# 배율이 요동치므로 보지 않는다.
SKEW_LOW, SKEW_HIGH = 0.67, 1.5


@dataclass(frozen=True)
class Claim:
    """수치 하나 + 그것을 믿어도 되는지에 대한 근거.

    분자·분모만으로는 만들 수 없다. 쓴 표본과 쓰기로 했던 모집단을 함께
    내야 한다 — 그것이 이 형의 존재 이유다.
    """

    name: str
    numerator: int
    denominator: int
    sample: Counter = field(default_factory=Counter)
    population: Counter = field(default_factory=Counter)
    threshold: float | None = None

    @property
    def value(self) -> float:
        return self.numerator / self.denominator if self.denominator else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.numerator, self.denominator)

    def skewed_labels(self) -> dict[str, float]:
        """모집단 대비 과대·과소표집된 라벨과 그 배율."""
        total_pop = sum(self.population.values())
        total_got = sum(self.sample.values())
        if not total_pop or not total_got:
            return {}
        out = {}
        for label, count in self.population.items():
            expected = count / total_pop * total_got
            if expected < 1:
                continue
            ratio = self.sample.get(label, 0) / expected
            if ratio >= SKEW_HIGH or ratio <= SKEW_LOW:
                out[label] = ratio
        return out

    def reachable_verdicts(self) -> set[str]:
        """이 분모로 애초에 낼 수 있는 판정들.

        보일 수 없는 것을 못 보였다고 실패로 적으면 그것도 거짓말이다.
        """
        if self.threshold is None or self.denominator <= 0:
            return set()
        return {
            verdict_against(self.threshold, *wilson_interval(k, self.denominator))
            for k in range(self.denominator + 1)
        }

    def problems(self) -> list[str]:
        """**무효 기준** 위반. 하나라도 있으면 판정을 내주지 않는다."""
        found = []
        total_pop, total_got = sum(self.population.values()), sum(self.sample.values())
        if total_pop and total_got < total_pop:
            found.append(f"부분 표본 — 계획 {total_pop}건 중 {total_got}건만 썼다")
        skew = self.skewed_labels()
        if skew:
            detail = " · ".join(f"{k} {v:.2f}배" for k, v in sorted(skew.items()))
            found.append(f"표본이 모집단과 다르게 치우쳤다 — {detail}")
        if self.threshold is not None and self.denominator > 0:
            reachable = self.reachable_verdicts() - {"판정 보류 — 구간이 문턱을 걸친다"}
            if not reachable:
                found.append(
                    f"분모 {self.denominator}건으로는 어느 쪽 판정도 나올 수 없다"
                )
        return found

    def limits(self) -> list[str]:
        """**한계 공시.** 측정은 성립하지만 이 표본으로 할 수 없는 말.

        판정을 막지는 않는다. 다만 적지 않고 넘어가면 읽는 사람이 이 수치가
        할 수 없는 말까지 했다고 여긴다.
        """
        if self.threshold is None or self.denominator <= 0:
            return []
        reachable = self.reachable_verdicts()
        out = []
        if "못 넘는다" not in reachable:
            out.append(
                f"이 분모({self.denominator}건)로는 '못 넘는다' 를 보일 수 없다 — "
                f"0/{self.denominator} 여도 상한이 "
                f"{wilson_interval(0, self.denominator)[1]:.3f} 다. "
                "기각하려면 표본이 더 필요하다."
            )
        if "넘는다" not in reachable:
            out.append(f"이 분모({self.denominator}건)로는 '넘는다' 를 보일 수 없다")
        return out

    def verdict(self) -> str:
        """무효 기준을 하나라도 어기면 판정하지 않는다."""
        if self.threshold is None:
            return "문턱 없음"
        if self.problems():
            return "판정 불가 — 무효 기준 위반"
        lo, hi = self.interval
        return verdict_against(self.threshold, lo, hi)

    def render(self, indent: str = "  ") -> str:
        lo, hi = self.interval
        lines = [
            f"{indent}{self.name}: {self.value:.3f} "
            f"({self.numerator}/{self.denominator}) · 95% CI [{lo:.3f}, {hi:.3f}]"
        ]
        if self.threshold is not None:
            lines.append(f"{indent}문턱 {self.threshold:.3f} 대비 판정: {self.verdict()}")
        for problem in self.problems():
            lines.append(f"{indent}  ✗ 무효 — {problem}")
        for limit in self.limits():
            lines.append(f"{indent}  · 한계 — {limit}")
        return "\n".join(lines)
