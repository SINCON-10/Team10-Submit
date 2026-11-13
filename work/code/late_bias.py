"""Per-hour residual bias of the fixed C19 recipe on fold rows: is there a stable late-hour (20-23h) bias that an early-window
estimate (o0227 rows 2/28-3/31) would correct on later windows (o0331 4/1-6/30, o0430 5/1-6/30)? Ordinary rows only (not ev, not 3127).
args: OUT ATT_CSV SETS"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(sys.argv[1]); games = pd.read_csv(sys.argv[2], parse_dates=['date']).date; SETS = sys.argv[3].split(',')
R = {}
for o in ['o0227', 'o0331', 'o0430']:
    d = OUT / o; p = pd.read_parquet(d / 'preds.parquet')
    y, ev, V = p.y.to_numpy(), p.ev.eq(1).to_numpy(), p.V.to_numpy(); dt = pd.to_datetime(p.date)
    s27 = p.station_id.eq(3127).to_numpy(); game = s27 & dt.isin(games).to_numpy(); keep_old = s27 & ev & ~game
    def c9(A, E, C, H):
        Em = (E + C + H) / 3; M = .8 * Em + .2 * A; B = np.where(ev, M, .8 * M + .2 * V)
        return np.where(s27, np.where(~ev, A, (A + Em) / 2), B)
    old = c9(p.A.to_numpy(), p.E42.to_numpy(), pd.read_parquet(d / 'preds_base_C.parquet').C.to_numpy(), p.H.to_numpy())
    S = []
    for s in SETS:
        a, c = pd.read_parquet(d / f'preds_{s}_AEH.parquet'), pd.read_parquet(d / f'preds_{s}_C.parquet')
        S.append(c9(a.A.to_numpy(), a.E.to_numpy(), c.C.to_numpy(), a.H.to_numpy()))
    q = np.maximum(np.where(keep_old, old, np.where(game, np.mean(S, 0), (old + np.sum(S, 0)) / (1 + len(S)))), 0)
    R[o] = pd.DataFrame({'date': dt, 'hour': p.hour, 'off': p.offday if 'offday' in p else 0, 'y': y, 'q': q, 'm': ~ev & ~s27})
tab = {}
for o, r in R.items():
    x = r[r.m]; tab[o] = (x.y - x.q).groupby(x.hour).mean()
    if o == 'o0227':
        e = x[x.date < '2026-04-01']; tab['S 2/28-3/31'] = (e.y - e.q).groupby(e.hour).mean()
print('mean residual y - p by hour (ordinary rows):'); print(pd.DataFrame(tab).round(3).T.to_string())
e = R['o0227']; e = e[e.m & (e.date < '2026-04-01')]; b = (e.y - e.q).groupby(e.hour).mean()
for o in ['o0331', 'o0430']:
    r = R[o]; base = np.sqrt(((r.y - r.q) ** 2).mean())
    for lab, hrs in [('all hours', range(24)), ('20-23h', [20, 21, 22, 23])]:
        for k in [1., .5]:
            adj = r.hour.map(b).where(r.hour.isin(hrs) & r.m, 0).to_numpy() * k
            print(f'{o} S-bias correction {lab} x{k}: {base:.6f} -> {np.sqrt(((r.y - r.q - adj) ** 2).mean()):.6f}')
