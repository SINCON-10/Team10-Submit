"""Time-ordered check of the lunar analog (lunar_adjust's analog part, w=1) on past holidays.
target chuseok2025 from {seol2025}; target seol2026 from {seol2025, chuseok2025}; (LOO seol2025 from the other two: info only)
Variants of the per-cell delta: V0 station x hour (current), shrinkage toward station-level / hour-band deltas."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, __import__('os').environ['S'] + '/data/processed_v4')   # package: path from paths.sh
from preprocess import HOLIDAYS
tr = pd.read_csv(__import__('os').environ['RAW'] + '/train.csv', usecols=['date', 'station_id', 'hour', 'boardings'], parse_dates=['date'])
Y = tr.assign(ly=np.log1p(tr.boardings)).pivot_table(index='date', columns=['station_id', 'hour'], values='ly')
hol = {pd.Timestamp(d) for d in HOLIDAYS}
B = {'seol25': ('2025-01-28', '2025-01-29', '2025-01-30'), 'chu25': ('2025-10-05', '2025-10-06', '2025-10-07'),
     'seol26': ('2026-02-16', '2026-02-17', '2026-02-18')}
B = {k: tuple(pd.Timestamp(x) for x in v) for k, v in B.items()}
def base(center, n=15):
    c = [d for d in Y.index if d < center - pd.Timedelta(days=8) and d.dayofweek < 5 and d not in hol and Y.loc[d].notna().any()]
    return Y.loc[c[-n:]].mean()
D = {k: [Y.loc[d] - base(b[1]) for d in b] for k, b in B.items()}   # delta per day position
hours = np.array(Y.columns.get_level_values(1)); band = np.select([hours < 10, hours < 17, hours < 20], [0, 1, 2], 3)
def variants(known, j):
    d = pd.concat([D[k][j] for k in known], axis=1).mean(axis=1)              # V0 station x hour
    st = d.groupby(level=0).transform('mean')                                  # station level
    bd = d.groupby([d.index.get_level_values(0), band]).transform('mean')      # station x hour band
    net_h = d.groupby(level=1).transform('mean')                               # network hour profile
    return {'V0 st x hour': d, 'V1 st x band': bd, 'V2 .5 V0 + .5 band': .5 * d + .5 * bd,
            'V3 st + (net_h - net)': st + net_h - d.mean(), 'V4 .5 V0 + .5 V3': .5 * d + .5 * (st + net_h - d.mean())}
cases = [('chu25', ['seol25']), ('seol26', ['seol25', 'chu25']), ('seol25 (LOO, info)', ['chu25', 'seol26'])]
for tgt, known in cases:
    k = tgt.split()[0]; b = B[k]; bs = base(b[1]); res = {}
    for j, day in enumerate(b):
        y = Y.loc[day]
        for name, dv in variants(known, j).items():
            e = (y - (bs + dv).clip(lower=0)).dropna()
            res.setdefault(name, []).append(e)
    print(tgt, 'from', known)
    for name, es in res.items():
        e = pd.concat(es); print(f'   {name:22s} rmsle {np.sqrt((e**2).mean()):.4f}  bias {e.mean():+.4f}  per day ' + ' '.join(f'{np.sqrt((x**2).mean()):.3f}' for x in es))
