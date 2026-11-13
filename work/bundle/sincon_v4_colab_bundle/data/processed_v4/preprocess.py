"""
2026 SINCON: 인천 지하철 수요 예측 — 쉽조 데이터 전처리 (preprocess.py)

실행:  python preprocess.py --data_dir <train.csv 등이 있는 폴더> --out_dir ./processed [--stacked]
의존성: numpy, pandas (Kaggle·Colab 기본)

[산출물]
  1) train_processed.csv  : train 원본 + 달력·공휴일·역 메타·이벤트·이상치 플래그 + log1p 타깃
  2) test_processed.csv   : test 원본 + 같은 특징 (이벤트는 외부 일정표 기반)
  3) test_cutoff_features.csv : 2026-06-30 기준 '누수 없는' 수준·작년 계절 특징 (예측 입력용)
  4) (--stacked) train_cutoff_features.csv : 월말 cutoff마다 다음 123일을 쌓은 학습용 특징
     → 모델 담당은 3)·4)로 학습/예측. 타깃 추천: y - lvl8 (최근 8주 수준 대비 잔차)

[전처리 설계 요약]
  - 공휴일: 대체·임시·선거일 포함, 2026 노동절(5/1)·제헌절(7/17) 반영. 연휴 블록 길이/위치, 징검다리,
           연휴 전·후 평일, '평범한 날(normal_day)' 플래그
  - 요일유형 dt3: 0 평일 / 1 토요일 / 2 일요일·공휴일
  - 이상치: 검단 연장 3역(3107~3109) 개통 첫 주(2025-06-28~07-06)는 수준 계산에서 제외
  - 이벤트: 학습 기간은 '역·시간·요일유형 중앙값 대비 log 승차 급증'으로 자동 탐지(+귀가 피크 시각),
           예측 기간은 SSG 홈경기·공연·축제 일정표 사용
  - 결측: 검단 3역은 개통 전 행이 원래 없음(결측 아님). 0 승차는 76건뿐이라 그대로 둠
  - 누수 방지: 수준·작년 특징은 항상 cutoff(예측 기준일) 이전 데이터만 사용

[외부 데이터 출처] (조회일 2026-11-11)
  - 공휴일: 관공서의 공휴일에 관한 규정 / 2026 제헌절·노동절 공휴일 지정 (인사혁신처, 2026-06-28 보도)
  - SSG 랜더스 2026 홈경기·종료 시각: KBO·SSG 공식 일정, 나무위키 'SSG 랜더스/2026년' (CC BY-NC-SA 2.0 KR)
    ※ 팀원 수집본. 8/4 경기·8/5~6 폭염취소는 기사로 확인함. 나머지 경기도 KBO 기록과 대조 권장
  - 펜타포트 7/31~8/2 (인천시), 송도맥주축제 8/22~30 (NOL티켓), 싸이 흠뻑쇼 인천 7/11,
    서스테이너블 웨이브 8/29~30, PLAVE 9/12~13 (Weverse), GHOSTIVAL 10/24~25 (NOL티켓), OBJET K-POP FESTA 9/26
  ※ 대회 규칙 3절(수정): 7/1 이후 이용객 수 외 정보는 사용 가능
"""
import os
import sys
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

START = pd.Timestamp("2025-01-01")
TRAIN_END = pd.Timestamp("2026-06-30")
TEST_END = pd.Timestamp("2026-10-31")
HOURS = np.arange(6, 24)
HORIZON = 123

# =============================================================================
# 1. 공휴일 (대체공휴일·임시공휴일·선거일 포함)
# =============================================================================
HOLIDAYS = {
    # 2025
    "2025-01-01": "신정", "2025-01-27": "임시공휴일",
    "2025-01-28": "설", "2025-01-29": "설", "2025-01-30": "설",
    "2025-03-01": "삼일절", "2025-03-03": "대체(삼일절)",
    "2025-05-05": "어린이날·부처님오신날", "2025-05-06": "대체(어린이날)",
    "2025-06-03": "대통령선거", "2025-06-06": "현충일",
    "2025-08-15": "광복절", "2025-10-03": "개천절",
    "2025-10-05": "추석", "2025-10-06": "추석", "2025-10-07": "추석",
    "2025-10-08": "대체(추석)", "2025-10-09": "한글날", "2025-12-25": "성탄절",
    # 2026
    "2026-01-01": "신정",
    "2026-02-16": "설", "2026-02-17": "설", "2026-02-18": "설",
    "2026-03-01": "삼일절", "2026-03-02": "대체(삼일절)",
    "2026-05-01": "노동절",            # 2026년부터 공휴일
    "2026-05-05": "어린이날", "2026-05-24": "부처님오신날", "2026-05-25": "대체(부처님오신날)",
    "2026-06-03": "지방선거", "2026-06-06": "현충일",
    "2026-07-17": "제헌절",            # 2026년부터 공휴일 재지정
    "2026-08-15": "광복절", "2026-08-17": "대체(광복절)",
    "2026-09-24": "추석", "2026-09-25": "추석", "2026-09-26": "추석",
    "2026-10-03": "개천절", "2026-10-05": "대체(개천절)", "2026-10-09": "한글날",
}
# 공휴일은 아니지만 쉬는 사람이 많은 날 (근로자의날: 2025년엔 공휴일 아님)
SEMI_OFF = {"2025-05-01": "근로자의날"}


def build_calendar() -> pd.DataFrame:
    d = pd.DataFrame({"date": pd.date_range(START, TEST_END)})
    ds = d["date"].dt.strftime("%Y-%m-%d")
    d["dow"] = d["date"].dt.dayofweek
    d["month"] = d["date"].dt.month
    d["dom"] = d["date"].dt.day
    d["doy"] = d["date"].dt.dayofyear
    d["is_hol"] = ds.isin(HOLIDAYS.keys()).astype(int)
    d["is_major"] = ds.map(lambda x: HOLIDAYS.get(x, "") in ("설", "추석")).astype(int)
    d["is_semi"] = ds.isin(SEMI_OFF.keys()).astype(int)
    off = ((d["dow"] >= 5) | (d["is_hol"] == 1)).values.astype(int)
    d["offday"] = off

    # 연휴 블록 길이·블록 내 위치
    n = len(d)
    blen = np.zeros(n, int); bpos = np.zeros(n, int)
    i = 0
    while i < n:
        if off[i]:
            j = i
            while j + 1 < n and off[j + 1]:
                j += 1
            blen[i:j + 1] = j - i + 1
            bpos[i:j + 1] = np.arange(j - i + 1)
            i = j + 1
        else:
            i += 1
    d["block_len"] = blen
    d["block_pos"] = bpos
    d["block_rpos"] = np.where(blen > 0, blen - 1 - bpos, -1)
    # 평일: 다음 휴일 블록까지 남은 날, 그 블록 길이 / 직전 블록 이후 경과일
    nxt = np.full(n, 9); nxt_len = np.zeros(n, int); prv = np.full(n, 9); prv_len = np.zeros(n, int)
    for k in range(n):
        if off[k]:
            continue
        for t in range(1, 9):
            if k + t < n and off[k + t]:
                nxt[k] = t; nxt_len[k] = blen[k + t]; break
        for t in range(1, 9):
            if k - t >= 0 and off[k - t]:
                prv[k] = t; prv_len[k] = blen[k - t]; break
    d["days_to_off"] = nxt; d["next_off_len"] = nxt_len
    d["days_from_off"] = prv; d["prev_off_len"] = prv_len
    d["is_bridge"] = ((off == 0) & (nxt == 1) & (prv == 1)).astype(int)
    # 요일 유형: 0 평일, 1 토요일, 2 일요일·공휴일
    d["dt3"] = np.where(d["is_hol"] == 1, 2, np.where(d["dow"] == 6, 2, np.where(d["dow"] == 5, 1, 0)))
    # '평범한 날' (수준 계산용): 공휴일·연휴 인접·징검다리·준공휴일 제외
    d["normal_day"] = (
        (d["is_hol"] == 0) & (d["is_semi"] == 0) & (d["is_bridge"] == 0)
        & ~((off == 0) & (nxt == 1) & (nxt_len >= 3))
        & ~((off == 0) & (prv == 1) & (prv_len >= 3))
        & ~((off == 1) & (blen >= 3))
    ).astype(int)

    # 방학(근사): 대학 하계·동계, 인천 초중고 여름·겨울
    md = d["date"].dt.month * 100 + d["date"].dt.day
    d["univ_vac"] = ((md >= 621) & (md <= 831) | (md >= 1220) | (md <= 228)).astype(int)
    d["school_vac"] = ((md >= 724) & (md <= 817) | (md >= 1231) | (md <= 301)).astype(int)
    d["doy_sin"] = np.sin(2 * np.pi * d["doy"] / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * d["doy"] / 365.25)
    return d


# =============================================================================
# 2. 예측 기간(2026-07~10) 이벤트 일정
#    peak = 귀가 인파가 몰리는 '시간 구간' 시작 시각 (예: 21 → 21:00~22:00)
# =============================================================================
# SSG 랜더스 문학 홈경기 (date: (종료_시, 종료_분)); 8/5·8/6 폭염취소, 8/22·9/25 우천취소는 제외
SSG_HOME_2026 = {
    "2026-07-03": (21, 52), "2026-07-04": (21, 20), "2026-07-05": (21, 46),
    "2026-07-16": (21, 49), "2026-07-17": (21, 25), "2026-07-18": (21, 11), "2026-07-19": (21, 13),
    "2026-07-24": (21, 48), "2026-07-25": (21, 25), "2026-07-26": (21, 56),
    "2026-07-28": (21, 32), "2026-07-29": (21, 4), "2026-07-30": (21, 58),
    "2026-08-04": (22, 17), "2026-08-11": (21, 33), "2026-08-12": (22, 1), "2026-08-13": (21, 44),
    "2026-08-21": (22, 54), "2026-08-23": (22, 0), "2026-08-25": (21, 45), "2026-08-26": (21, 13),
    "2026-08-27": (21, 54),
    "2026-09-04": (20, 42), "2026-09-05": (19, 29), "2026-09-06": (20, 28), "2026-09-10": (21, 10),
    "2026-09-15": (21, 35), "2026-09-20": (17, 4), "2026-09-22": (21, 47), "2026-09-24": (20, 9),
    "2026-09-29": (21, 1), "2026-09-30": (21, 10), "2026-10-01": (21, 37), "2026-10-07": (21, 33),
    "2026-10-09": (16, 29),
}
# 기타 대형 행사: (station_id, date, peak_hour)
# 7/1 이후에 바뀌거나 새로 발표된 일정을 쓸지 여부.
#   대회 규칙 3절(수정본)은 '이용객 수 외 정보는 7/1 이후 것도 허용'이지만 Important Notes와 충돌하므로,
#   운영진 답변이 '불가'면 False로 바꾸고 전처리를 다시 돌린다.
USE_POST_0701_INFO = True

# (역, 날짜, 귀가 피크 시간대). 피크 = 종료 시각 + 10분이 속한 시간대.
#   조회일 2026-11-12. 일정 초안은 Gemini 조사표(incheon_subway_events_2026_07_10.csv), 기사로 일부 재확인.
#   2025년 같은 행사 때 승차 급증이 '자동 탐지'된 행사만 넣었다 (낮 시간 박람회·거리축제는 탐지된 적 없어 제외).
OTHER_EVENTS_2026 = [
    # 문학경기장 주경기장 (3127)
    (3127, "2026-08-29", 22), (3127, "2026-08-30", 22),   # Sustainable Wave Festival (14시~, 약 8시간)
    (3127, "2026-09-12", 22), (3127, "2026-09-13", 22),   # PLAVE World Tour (19시~)
    (3127, "2026-10-24", 21), (3127, "2026-10-25", 21),   # GHOSTIVAL (15~21시, NOL티켓 공식)
    # OBJET K-POP FESTA(9/26)는 재조사에서 문학경기장 개최가 확인되지 않아 뺐다.
    # 송도달빛축제공원 (3139) — 2025년 같은 행사 때도 22시 피크
    (3139, "2026-07-31", 22), (3139, "2026-08-01", 22), (3139, "2026-08-02", 22),  # 펜타포트
    *[(3139, f"2026-08-{dd:02d}", 22) for dd in range(22, 31)],                     # 송도맥주축제
    # 인천아시아드주경기장 (3209)
    (3209, "2026-07-11", 22),                             # 싸이 흠뻑쇼 (18시~, 약 4시간 30분)
    (3209, "2026-10-23", 15), (3209, "2026-10-24", 17),   # 인천과학대제전 (10~17시). 2025년 같은 행사: 금 15시·토 17시 피크
    # 인천대공원(3226) 시민의 날 대축제(10/10)는 뺐다: 이 역의 학습 행사일은 대부분 봄꽃 시즌(종일 붐빔)이라
    # 행사 표시를 하면 모델이 하루 종일 올려 버린다 (2025-10-10 실제로는 20시만 튐).
    # 부천종합운동장 (3754) — 부천FC 1995 K리그1 홈경기. 2025년 저녁 경기는 21시 피크
    (3754, "2026-07-19", 21),                             # 18R 서울 19:30
    (3754, "2026-07-22", 21),                             # 19R 안양 19:30
    (3754, "2026-08-08", 22 if USE_POST_0701_INFO else 21),  # 22R 광주. 19:30 → 20:00 (8/5 폭염 연기 발표)
    (3754, "2026-08-16", 21),                             # 23R 전북 19:30
    (3754, "2026-08-29", 21),                             # 26R 안양 19:30
    (3754, "2026-09-05", 21),                             # 27R 대전 19:00
    (3754, "2026-09-12", 18),                             # 제주 16:30
    (3754, "2026-09-19", 18),                             # 김천 16:30
    (3754, "2026-10-24", 16),                             # 33R 광주 14:00 (정규리그 마지막 홈경기)
]
if USE_POST_0701_INFO:
    OTHER_EVENTS_2026 += [
        (3754, "2026-07-29", 21),   # 코리아컵 3R 파주 19:00 (부천 홈)
        (3754, "2026-08-22", 21),   # 24R 포항 19:30 — 포항 스틸야드 안전 문제로 부천으로 장소 변경
    ]


def egress_hour(end_h: int, end_m: int) -> int:
    """종료 후 약 10분 뒤 승차가 몰리는 시간 구간."""
    return int((end_h * 60 + end_m + 10) // 60)


def test_event_table() -> pd.DataFrame:
    rows = [(3127, d, egress_hour(*hm)) for d, hm in SSG_HOME_2026.items()]
    rows += OTHER_EVENTS_2026
    ev = pd.DataFrame(rows, columns=["station_id", "date", "ev_peak"])
    ev["date"] = pd.to_datetime(ev["date"])
    return ev.drop_duplicates(["station_id", "date"], keep="first")


# =============================================================================
# 3. 데이터 → (날짜 × 역 × 시간) 큐브
# =============================================================================
class Cube:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, meta: pd.DataFrame, cal: pd.DataFrame):
        self.cal = cal
        self.dates = cal["date"].values
        self.D = len(cal)
        self.stations = np.sort(meta["station_id"].unique())
        self.S = len(self.stations)
        self.sidx = {s: i for i, s in enumerate(self.stations)}
        self.H = len(HOURS)
        self.line = meta.set_index("station_id").loc[self.stations, "canonical_line"].values

        Y = np.full((self.D, self.S, self.H), np.nan, np.float32)
        di = (train["date"] - START).dt.days.values
        si = train["station_id"].map(self.sidx).values
        hi = train["hour"].values - 6
        Y[di, si, hi] = np.log1p(train["boardings"].values)
        self.Y = Y
        self.n_train_days = (TRAIN_END - START).days + 1

        # 이상치: 검단 연장 개통 직후 1주
        anomaly = np.zeros((self.D, self.S), bool)
        for s in (3107, 3108, 3109):
            anomaly[: (pd.Timestamp("2025-07-07") - START).days, self.sidx[s]] = True
        self.anomaly = anomaly
        self._detect_events()
        self._attach_test_events()

    def _detect_events(self):
        """학습 기간: 역·시간·요일유형 중앙값 대비 log 승차가 크게 뛴 날 = 이벤트."""
        n = self.n_train_days
        Y = self.Y[:n]
        dt3 = self.cal["dt3"].values[:n]
        normal = self.cal["normal_day"].values[:n].astype(bool)
        med = np.full((3, self.S, self.H), np.nan, np.float32)
        for k in range(3):
            m = (dt3 == k) & normal
            med[k] = np.nanmedian(Y[m], axis=0)
        exc = Y - med[dt3]                                # (n,S,H)
        exc_ev = np.nanmax(np.where(np.isnan(exc), -9, exc)[:, :, 8:], axis=2)  # 14시 이후
        thr = np.full(self.S, 1.3, np.float32)
        for s, t in {3127: 1.0, 3139: 0.9, 3209: 0.9, 3754: 0.9}.items():
            thr[self.sidx[s]] = t
        is_ev = exc_ev > thr[None, :]
        is_ev &= ~self.anomaly[:n]
        peak = np.full((self.D, self.S), np.nan, np.float32)
        hrs = np.nanargmax(np.where(np.isnan(exc), -9, exc)[:, :, 8:], axis=2) + 8 + 6
        peak[:n][is_ev] = hrs[is_ev]
        ev = np.zeros((self.D, self.S), bool)
        ev[:n] = is_ev
        self.ev, self.ev_peak = ev, peak
        tab = pd.Series(is_ev.sum(0), index=self.stations)
        print("  - 학습 기간 자동 탐지 이벤트(역별 일수):", tab[tab > 0].to_dict())

    def _attach_test_events(self):
        for r in test_event_table().itertuples():
            d = (r.date - START).days
            s = self.sidx[r.station_id]
            self.ev[d, s] = True
            self.ev_peak[d, s] = r.ev_peak

    # ---- 시점 고정 통계 ----------------------------------------------------
    def level(self, c: int, weeks: int, by: str = "dt3"):
        """cutoff index c(포함)까지 직전 weeks주 '평범한 날' 평균. by=dt3 → (3,S,H), by=dow → (7,S,H)."""
        lo = max(0, c - weeks * 7 + 1)
        sl = slice(lo, c + 1)
        Y = self.Y[sl]
        ok = (self.cal["normal_day"].values[sl].astype(bool)[:, None]
              & ~self.ev[sl] & ~self.anomaly[sl])           # (w,S)
        key = self.cal[by].values[sl]
        K = 3 if by == "dt3" else 7
        out = np.full((K, self.S, self.H), np.nan, np.float32)
        cnt = np.zeros((K, self.S), np.float32)
        for k in range(K):
            m = (key == k)
            if not m.any():
                continue
            yk = np.where(ok[m][:, :, None], Y[m], np.nan)
            n_ok = np.sum(~np.isnan(yk), axis=0)
            with np.errstate(invalid="ignore"):
                out[k] = np.where(n_ok >= 2, np.nanmean(yk, axis=0), np.nan)
            cnt[k] = n_ok.max(axis=1)
        return out, cnt

    def window_mean(self, center: int, half: int, k_dt3: int, hi_limit: int = None):
        """center±half 범위에서 dt3==k 인 평범한 날 평균 (S,H)."""
        hi_lim = self.n_train_days - 1 if hi_limit is None else min(hi_limit, self.n_train_days - 1)
        lo, hi = max(0, center - half), min(hi_lim, center + half)
        if hi < lo:
            return np.full((self.S, self.H), np.nan, np.float32)
        sl = slice(lo, hi + 1)
        m = (self.cal["dt3"].values[sl] == k_dt3) & self.cal["normal_day"].values[sl].astype(bool)
        if not m.any():
            return np.full((self.S, self.H), np.nan, np.float32)
        ok = ~self.ev[sl][m] & ~self.anomaly[sl][m]
        yk = np.where(ok[:, :, None], self.Y[sl][m], np.nan)
        with np.errstate(invalid="ignore"):
            return np.nanmean(yk, axis=0)


CAL_FEATS = ["dow", "month", "dom", "doy_sin", "doy_cos", "is_hol", "is_major", "is_semi", "offday",
             "block_len", "block_pos", "block_rpos", "days_to_off", "next_off_len", "days_from_off",
             "prev_off_len", "is_bridge", "dt3", "univ_vac", "school_vac"]


def build_rows(cube: Cube, cutoff: pd.Timestamp, t_start: pd.Timestamp, t_end: pd.Timestamp,
               require_target: bool, avail_end: pd.Timestamp = None) -> pd.DataFrame:
    """cutoff 시점의 '최근 수준' + 다른 해(partner year) 같은 시기의 계절 변화량으로 특징 생성.

    partner year: 364일 전 데이터가 있으면 작년(-364), 없으면 내년(+364)을 사용.
      - 예측(2026-07~10) 행은 항상 작년(2025-07~10)을 씀
      - 학습 행 중 2025년 상반기는 2026년 상반기를 '짝'으로 써서, 계절 변화량을 얼마나 믿을지 모델이 배움
      - avail_end 이후 데이터는 절대 사용하지 않음 (검증 fold에서 누수 방지)
    """
    if avail_end is None:
        avail_end = TRAIN_END
    a_end = (avail_end - START).days
    c = (cutoff - START).days
    d0, d1 = (t_start - START).days, (t_end - START).days
    S, H = cube.S, cube.H
    nanSH = np.full((S, H), np.nan, np.float32)

    L4, _ = cube.level(c, 4)
    L8, cnt8 = cube.level(c, 8)
    L16, _ = cube.level(c, 16)
    L52, _ = cube.level(c, 52)
    # 평범한 날이 8주 안에 2일 미만인 경우(예: 문학경기장 주말은 거의 매번 야구) → 16주 → 52주 순으로 채움
    L8_filled = np.isnan(L8)
    L8 = np.where(np.isnan(L8), L16, L8)
    L8 = np.where(np.isnan(L8), L52, L8).astype(np.float32)
    L8dow, _ = cube.level(c, 8, by="dow")
    L8_prev, _ = cube.level(c - 56, 8) if c - 56 - 55 >= 0 else (np.full_like(L8, np.nan), None)
    # 짝 해의 cutoff 수준
    L8_p = {}
    for off in (-364, 364):
        cp = c + off
        if cp - 8 * 7 + 1 >= 0 and cp <= a_end:
            L8_p[off], _ = cube.level(cp, 8)

    cal = cube.cal
    blocks = []
    for d in range(d0, d1 + 1):
        crow = cal.iloc[d]
        k = int(crow["dt3"])
        dow = int(crow["dow"])
        y = cube.Y[d]
        if require_target and np.all(np.isnan(y)):
            continue
        off = 0
        for o in (-364, 364):
            dp = d + o
            if o in L8_p and 0 <= dp and dp + 7 <= a_end:
                off = o; break
        if off != 0:
            dp = d + off
            p_raw = cube.Y[dp]
            p_win = cube.window_mean(dp, 7, k, hi_limit=a_end)
            p_base = L8_p[off][k]
            p_ev = cube.ev[dp]
            p_hol = cal["is_hol"].values[dp]
            p_major = cal["is_major"].values[dp]
        else:
            p_raw = p_win = p_base = nanSH
            p_ev = np.zeros(S, bool); p_hol = -1; p_major = -1
        base = L8[k]
        f = {
            "sid": np.repeat(np.arange(S), H),
            "hour": np.tile(HOURS, S),
            "line": np.repeat(cube.line, H),
            "horizon": np.full(S * H, d - c),
            "lvl8": base.ravel(),
            "lvl8_filled": L8_filled[k].ravel().astype(np.float32),
            "lvl4": (L4[k] - base).ravel(),
            "lvl16": (L16[k] - base).ravel(),
            "lvl8_dow": (L8dow[dow] - base).ravel(),
            "lvl8_wk": (L8[0] - base).ravel(),
            "lvl8_sat": (L8[1] - base).ravel(),
            "lvl8_sun": (L8[2] - base).ravel(),
            "trend8": (L8[k] - L8_prev[k]).ravel(),
            "cnt8": np.repeat(cnt8[k], H),
            "yoy_dir": np.full(S * H, np.sign(off)),
            "yoy_raw": (p_raw - p_base).ravel(),        # 짝 해 같은 날(364일 차) - 짝 해 cutoff 수준
            "yoy_delta": (p_win - p_base).ravel(),      # 짝 해 ±7일 평균 - 짝 해 cutoff 수준 (계절 변화)
            "yoy_ev": np.repeat(p_ev.astype(np.float32), H),
            "yoy_hol": np.full(S * H, p_hol),
            "yoy_major": np.full(S * H, p_major),
            "ev": np.repeat(cube.ev[d].astype(np.float32), H),
            "ev_rel": (np.tile(HOURS, S) - np.repeat(cube.ev_peak[d], H)),
        }
        for cf in CAL_FEATS:
            f[cf] = np.full(S * H, crow[cf])
        f["cut_doy_sin"] = np.full(S * H, np.sin(2 * np.pi * cutoff.dayofyear / 365.25))
        f["cut_doy_cos"] = np.full(S * H, np.cos(2 * np.pi * cutoff.dayofyear / 365.25))
        if getattr(cube, "W", None) is not None:
            for wf, g in cube.W.items():
                f[wf] = np.tile(g[d], S)
        newst = np.isin(cube.stations, NEW_STATIONS)
        f["stn_age_cut"] = np.repeat(np.where(newst, np.clip(c - OPEN_IDX, 0, 400), 400), H).astype(np.float32)
        f["yoy_ramp"] = np.repeat((newst & (d - 364 < RAMP_END_IDX)).astype(np.float32), H)
        f["date_idx"] = np.full(S * H, d)
        f["y"] = y.ravel()
        blocks.append(pd.DataFrame(f))
    df = pd.concat(blocks, ignore_index=True)
    # 전체 역 평균 계절 변화량 (역별 값의 잡음 보완)
    df["yoy_delta_all"] = df.groupby(["date_idx", "hour"])["yoy_delta"].transform("mean")
    df["yoy_raw_all"] = df.groupby(["date_idx", "hour"])["yoy_raw"].transform("mean")
    if require_target:
        df = df[~df["y"].isna() & ~df["lvl8"].isna()]
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
    return df.reset_index(drop=True)


# =============================================================================
# 날씨 (선택): data_dir에 weather_incheon.csv가 있으면 자동으로 특징에 붙인다 (fetch_weather.py로 생성)
#   그날의 실측 날씨이므로 학습·예측 모두 '그 날짜'의 값을 쓴다 (대회 운영진: 7/1 이후 날씨 실측 사용 허용)
# =============================================================================
# 신설역(검단 3역, 2025-06-28 개통): 2026-03까지 이용객이 계속 늘던 증가기 → 작년 같은 날과 비교하는 yoy 특징이 왜곡됨
NEW_STATIONS = [3107, 3108, 3109]
OPEN_IDX = (pd.Timestamp("2025-06-28") - START).days
RAMP_END_IDX = (pd.Timestamp("2026-03-01") - START).days
STATION_FEATS = ["stn_age_cut", "yoy_ramp"]   # 개통 후 경과일(cutoff 기준, 400 상한) / 짝 해 날짜가 증가기면 1
WEATHER_FEATS = ["w_temp", "w_app", "w_rain", "w_rain3", "w_rain_day", "w_tmax", "w_snow_day"]


def attach_weather(cube: "Cube", path) -> bool:
    from pathlib import Path
    path = Path(path)
    cube.W = None
    if not path.exists():
        print(f"  - 날씨 파일 없음({path.name}) → 날씨 특징 없이 진행")
        return False
    w = pd.read_csv(path, parse_dates=["date"])
    w = w[w["hour"].between(6, 23)]
    di = (w["date"] - START).dt.days.values
    ok = (di >= 0) & (di < cube.D)
    def grid(col):
        g = np.full((cube.D, cube.H), np.nan, np.float32)
        g[di[ok], w["hour"].values[ok] - 6] = w[col].values[ok]
        return g
    temp, app, rain, snow = grid("temp"), grid("apparent_temp"), grid("precip"), grid("snow")
    rain0 = np.nan_to_num(rain)
    rain3 = rain0.copy(); rain3[:, 1:] += rain0[:, :-1]; rain3[:, 2:] += rain0[:, :-2]   # 이번+직전 2시간
    cube.W = {
        "w_temp": temp, "w_app": app, "w_rain": rain, "w_rain3": rain3,
        "w_rain_day": np.repeat(np.nansum(rain, 1, keepdims=True), cube.H, 1),
        "w_tmax": np.repeat(np.nanmax(np.where(np.isnan(temp), -99, temp), 1, keepdims=True), cube.H, 1),
        "w_snow_day": np.repeat(np.nansum(snow, 1, keepdims=True), cube.H, 1),
    }
    miss = int(np.isnan(temp[:cube.D]).all(1).sum())
    print(f"  - 날씨 붙임: {path.name} ({len(w):,}행), 날씨 없는 날 {miss}일")
    return True


FEATS = ["sid", "hour", "line", "horizon", "lvl8", "lvl4", "lvl16", "lvl8_dow", "lvl8_wk", "lvl8_sat",
         "lvl8_sun", "trend8", "cnt8", "yoy_dir", "yoy_raw", "yoy_delta", "yoy_ev", "yoy_hol", "yoy_major",
         "yoy_delta_all", "yoy_raw_all", "ev", "ev_rel", "cut_doy_sin", "cut_doy_cos"] + CAL_FEATS


def make_training_set(cube: Cube, last_target: pd.Timestamp, cut_freq: str = "MS") -> pd.DataFrame:
    """월말 cutoff마다 다음 최대 123일(단, last_target 이하) 행을 쌓음."""
    cuts = list(pd.date_range("2025-03-01", last_target, freq=cut_freq) - pd.Timedelta(days=1))
    cuts = [pd.Timestamp("2025-01-15")] + [c for c in cuts if c < last_target]  # 1/15: 설 연휴를 학습에 포함
    parts = []
    for cu in cuts:
        t1 = min(cu + pd.Timedelta(days=HORIZON), last_target)
        parts.append(build_rows(cube, cu, cu + pd.Timedelta(days=1), t1, require_target=True,
                                avail_end=last_target))
    return pd.concat(parts, ignore_index=True)


# =============================================================================
# 4. 실행: 전처리 결과 저장
# =============================================================================
ROW_FEATS = ["dow", "month", "dom", "doy", "doy_sin", "doy_cos", "is_hol", "is_major", "is_semi",
             "offday", "block_len", "block_pos", "block_rpos", "days_to_off", "next_off_len",
             "days_from_off", "prev_off_len", "is_bridge", "dt3", "normal_day", "univ_vac", "school_vac"]


def row_table(df: pd.DataFrame, cube: Cube, meta: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(cube.cal[["date"] + ROW_FEATS], on="date", how="left")
    out = out.merge(meta[["station_id", "station_name", "canonical_line"]], on="station_id", how="left")
    di = (out["date"] - START).dt.days.values
    si = out["station_id"].map(cube.sidx).values
    out["is_event"] = cube.ev[di, si].astype(int)
    out["ev_peak"] = cube.ev_peak[di, si]
    out["ev_rel"] = out["hour"] - out["ev_peak"]
    out["is_anomaly"] = cube.anomaly[di, si].astype(int)
    out["holiday_name"] = out["date"].dt.strftime("%Y-%m-%d").map(HOLIDAYS).fillna("")
    if getattr(cube, "W", None):
        hi = out["hour"].values - 6
        for wf, g in cube.W.items():
            out[wf] = g[di, hi]
    if "boardings" in out:
        out["log_boardings"] = np.log1p(out["boardings"])
        out["log_alightings"] = np.log1p(out["alightings"])
    return out


def cutoff_table(rows: pd.DataFrame, cube: Cube) -> pd.DataFrame:
    rows = rows.copy()
    rows["date"] = START + pd.to_timedelta(rows["date_idx"], unit="D")
    rows["station_id"] = cube.stations[rows["sid"].values]
    return rows.drop(columns=["date_idx"])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--out_dir", default="./processed")
    ap.add_argument("--stacked", action="store_true", help="학습용 cutoff 특징(약 250만 행)도 저장")
    args, _ = ap.parse_known_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("[1/4] 데이터 로드")
    train = pd.read_csv(data_dir / "train.csv", parse_dates=["date"])
    test = pd.read_csv(data_dir / "test.csv", parse_dates=["date"])
    meta = pd.read_csv(data_dir / "metaData.csv")
    assert len(test) == 157194, f"test 행 수 확인: {len(test)}"
    assert train["hour"].between(6, 23).all() and test["hour"].between(6, 23).all()
    print(f"  - train {len(train):,}행 ({train.date.min().date()}~{train.date.max().date()}), test {len(test):,}행")
    print(f"  - 검단 3역(3107~3109)은 2025-06-28 개통 → 이전 행 없음 (train 결측 {71*546*18-len(train):,}행 = 전부 이 경우)")
    print(f"  - 승차 0 인 행: {(train.boardings == 0).sum()}개 (대부분 23시) → 그대로 사용")

    print("[2/4] 달력·이벤트·이상치")
    cal = build_calendar()
    cube = Cube(train, test, meta, cal)
    attach_weather(cube, data_dir / "weather_incheon.csv")
    tev = test_event_table()
    print(f"  - 예측 기간 이벤트: {len(tev)}건 (역별 {tev.station_id.value_counts().to_dict()})")
    chk = cal[cal.date.isin(pd.to_datetime(["2026-07-17", "2026-08-17", "2026-09-25", "2026-10-05"]))]
    assert chk["is_hol"].all(), "예측 기간 공휴일 누락"

    tr_rows = row_table(train, cube, meta)
    te_rows = row_table(test, cube, meta)
    tr_rows.to_csv(out_dir / "train_processed.csv", index=False)
    te_rows.to_csv(out_dir / "test_processed.csv", index=False)
    print(f"  - 저장: train_processed.csv {tr_rows.shape}, test_processed.csv {te_rows.shape}")

    print("[3/4] 예측용 cutoff 특징 (2026-06-30 기준)")
    te_cut = build_rows(cube, TRAIN_END, TRAIN_END + pd.Timedelta(days=1), TEST_END,
                        require_target=False, avail_end=TRAIN_END)
    te_cut = cutoff_table(te_cut, cube).drop(columns=["y"])
    te_cut = test[["example_id", "date", "station_id", "hour"]].merge(
        te_cut, on=["date", "station_id", "hour"], how="left")
    assert len(te_cut) == 157194 and te_cut["lvl8"].notna().all()
    te_cut.to_csv(out_dir / "test_cutoff_features.csv", index=False)
    print(f"  - 저장: test_cutoff_features.csv {te_cut.shape}")

    if args.stacked:
        print("[4/4] 학습용 cutoff 특징 (월말 cutoff × 다음 123일)")
        tr_cut = cutoff_table(make_training_set(cube, TRAIN_END), cube)
        tr_cut.to_csv(out_dir / "train_cutoff_features.csv", index=False)
        print(f"  - 저장: train_cutoff_features.csv {tr_cut.shape}")
    else:
        print("[4/4] 학습용 cutoff 특징은 --stacked 옵션을 주면 생성")
    print(f"완료 ({time.time() - t0:.0f}s) → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
