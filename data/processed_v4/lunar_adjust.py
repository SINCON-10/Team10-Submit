"""
설·추석 핵심 3일 보정 (lunar_adjust.py)  —  어떤 모델에도 붙일 수 있는 '예측 후 보정' 함수

[왜 필요한가]
  설·추석은 음력이라 해마다 날짜가 달라서, '364일 전 같은 요일' 기준의 작년 특징이 맞지 않는다.
  그래서 모델이 시험 구간의 추석 3일(2026-09-24~26)을 크게 높게 예측한다(추석 당일 약 25만 명, 실제는 15~19만 명 수준).
  이 함수는 그 3일만 '과거 설·추석이 평소 평일보다 얼마나 줄었는지'로 계산한 값과 섞어 준다.
  시간순 검증(F1)에서 추석 3일 RMSLE 0.46 → 0.31, F1 전체 0.2009 → 0.1937.

[사용법]  (모델이 어떤 것이든 상관없음. 예측값만 'log1p 단위'로 넘기면 된다)
    import pandas as pd, numpy as np
    from lunar_adjust import lunar_adjust

    train = pd.read_csv("train.csv", parse_dates=["date"])
    test  = pd.read_csv("test.csv",  parse_dates=["date"])
    # 모델 예측 (사람 수 단위) → log1p로 바꿔서 date, station_id, hour와 함께 넣는다
    pred = test[["date", "station_id", "hour"]].assign(p=np.log1p(내_예측_사람수))
    pred = lunar_adjust(train, pred)                 # p 열이 보정된 값으로 바뀐다
    boardings = np.expm1(pred["p"]).clip(lower=0)    # 다시 사람 수로

[주의]
  - 같은 폴더에 preprocess.py가 있어야 한다 (공휴일 목록 HOLIDAYS를 가져옴).
  - cutoff 이후 날짜의 실제 승차 인원은 쓰지 않는다 (누수 없음). train에는 cutoff까지만 들어 있어야 한다.
  - 검증(F1 등)에 쓸 때는 cutoff를 그 fold의 기준일로, train을 그 기준일까지로 잘라서 넣는다.
"""
import warnings

import numpy as np
import pandas as pd

from preprocess import HOLIDAYS

# 설·추석 핵심 3일 (is_major). 가운데 날이 명절 당일.
LUNAR_BLOCKS = [
    ("2025-01-28", "2025-01-29", "2025-01-30"),   # 설 2025
    ("2025-10-05", "2025-10-06", "2025-10-07"),   # 추석 2025
    ("2026-02-16", "2026-02-17", "2026-02-18"),   # 설 2026
    ("2026-09-24", "2026-09-25", "2026-09-26"),   # 추석 2026
]
LUNAR_W = 0.8   # 보정값 비중 (나머지 0.2는 모델 예측)


def _pivot(df: pd.DataFrame, value: str) -> pd.DataFrame:
    return df.pivot_table(index="date", columns=["station_id", "hour"], values=value, aggfunc="mean")


def _weekday_base(P: pd.DataFrame, center: pd.Timestamp, hol: set, n: int = 15) -> pd.Series:
    """명절 당일의 8일 전까지, 공휴일이 아닌 평일 중 값이 있는 최근 n일의 평균."""
    cand = [d for d in P.index
            if d < center - pd.Timedelta(days=8) and d.dayofweek < 5 and d not in hol and P.loc[d].notna().any()]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return P.loc[cand[-n:]].mean()


def lunar_adjust(train: pd.DataFrame, pred: pd.DataFrame, cutoff="2026-06-30", w: float = LUNAR_W) -> pd.DataFrame:
    """
    train : 원본 train.csv (date, station_id, hour, boardings) — cutoff까지의 실제 데이터
    pred  : date, station_id, hour, p  (p = log1p 예측, cutoff 이후 날짜)
    반환  : pred의 복사본 (설·추석 3일의 p만 보정됨)
    """
    cutoff = pd.Timestamp(cutoff)
    hol = {pd.Timestamp(d) for d in HOLIDAYS}
    tr = train[train["date"] <= cutoff].assign(ly=lambda x: np.log1p(x["boardings"]))
    Y = _pivot(tr, "ly")
    Pp = _pivot(pred, "p")
    P_all = pd.concat([Y, Pp]).sort_index()                       # 실제(≤cutoff) + 모델 예측(>cutoff)

    blocks = [tuple(pd.Timestamp(x) for x in b) for b in LUNAR_BLOCKS]
    known = [b for b in blocks if b[2] <= cutoff]                  # 이미 끝난 명절
    targets = [b for b in blocks if b[0] > cutoff and b[1] in Pp.index]
    if not known or not targets:
        return pred.copy()

    # 과거 명절: (명절 같은 날 - 그 명절 직전 평일 평균) 의 평균
    deltas = []
    for b in known:
        base = _weekday_base(P_all, b[1], hol)
        deltas.append([Y.loc[d] - base for d in b])
    dmean = [pd.concat([dl[j] for dl in deltas], axis=1).mean(axis=1) for j in range(3)]

    adj = Pp.copy()
    for b in targets:
        base = _weekday_base(P_all, b[1], hol)
        for j, d in enumerate(b):
            if d not in adj.index:
                continue
            analog = (base + dmean[j]).clip(lower=0)
            both = analog.notna() & adj.loc[d].notna()
            adj.loc[d, both[both].index] = (1 - w) * adj.loc[d, both[both].index] + w * analog[both]

    out = pred.copy()
    new = adj.stack(["station_id", "hour"]).rename("p_new").reset_index()
    out = out.merge(new, on=["date", "station_id", "hour"], how="left")
    out["p"] = out["p_new"].fillna(out["p"])
    return out.drop(columns="p_new")
