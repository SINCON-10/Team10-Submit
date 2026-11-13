"""Weather of the reference days behind the level / year-over-year features (city-wide, Incheon).

The model sees the forecast day's weather, but not the weather of the days its baselines were averaged over:
  wx_l8   : mean log1p(daytime rain) over the current 8-week level window (cutoff-55..cutoff, same dt3, normal days)
  wx_pbase: same for the partner-year cutoff window (cutoff +- 364)
  wx_pwin : partner-year +-7 day window, same dt3, normal days, <= a_end   (the days behind yoy_delta)
  wx_pday : partner-year same day (the day behind yoy_raw)
Daytime rain = sum of hourly precip 08-20h. Only weather and the calendar are used (no ridership).
"""
from pathlib import Path
import numpy as np, pandas as pd

PV4 = Path(__import__('os').environ.get('PV4') or Path(__file__).resolve().parents[2] / 'data' / 'processed_v4')   # package: path made relative (was the author's local path)
_cal = pd.read_csv(PV4 / 'train_processed.csv.gz', usecols=['date', 'dt3', 'normal_day'], parse_dates=['date']).drop_duplicates('date')
_w = pd.read_csv(PV4 / 'weather_incheon.csv', parse_dates=['date'])
_rain = np.log1p(_w[_w.hour.between(8, 20)].groupby('date').precip.sum())
START = pd.Timestamp('2025-01-01')
_n = (_cal.date.max() - START).days + 1
DT3 = np.full(_n, -1); NORM = np.zeros(_n, bool); RAIN = np.full(_n, np.nan)
_i = (_cal.date - START).dt.days.to_numpy()
DT3[_i] = _cal.dt3.to_numpy(); NORM[_i] = _cal.normal_day.astype(bool).to_numpy()
_j = (_rain.index - START).days
_ok = (_j >= 0) & (_j < _n)
RAIN[_j[_ok]] = _rain.to_numpy()[_ok]
FEATS = ['wx_l8', 'wx_pbase', 'wx_pwin', 'wx_pday']


def _wmean(lo, hi, k):
    lo, hi = max(lo, 0), min(hi, _n - 1)
    if hi < lo:
        return np.nan
    s = slice(lo, hi + 1)
    m = (DT3[s] == k) & NORM[s]
    return float(np.nanmean(RAIN[s][m])) if m.any() else np.nan


def add_wx(df, a_end):
    """df needs date, horizon, dt3, yoy_dir. a_end = last day with known data for this dataset (fold origin or 2026-06-30)."""
    a = (pd.Timestamp(a_end) - START).days
    key = pd.DataFrame({'d': (pd.to_datetime(df.date) - START).dt.days.to_numpy(), 'h': df.horizon.to_numpy().astype(int),
                        'k': df.dt3.to_numpy().astype(int), 'o': df.yoy_dir.to_numpy().astype(int)})
    u = key.drop_duplicates().reset_index(drop=True)
    out = {f: [] for f in FEATS}
    for d, h, k, o in u.itertuples(index=False):
        c = d - h
        out['wx_l8'].append(_wmean(c - 55, c, k))
        if o == 0:
            for f in FEATS[1:]:
                out[f].append(np.nan)
            continue
        dp, cp = d + 364 * o, c + 364 * o
        out['wx_pbase'].append(_wmean(cp - 55, min(cp, a), k) if cp - 55 >= 0 and cp <= a else np.nan)
        out['wx_pwin'].append(_wmean(dp - 7, min(dp + 7, a), k))
        out['wx_pday'].append(RAIN[dp] if 0 <= dp < _n and dp <= a else np.nan)
    for f in FEATS:
        u[f] = out[f]
    m = key.merge(u, on=['d', 'h', 'k', 'o'], how='left', validate='many_to_one')
    for f in FEATS:
        df[f] = m[f].to_numpy(np.float64)
    return FEATS
