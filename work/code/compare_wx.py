"""Weather-of-reference-days features (wxfeat.py) on the three origins.
old  = C9 models (preds.parquet A/E42/H + preds_base_C), att = + attendance, attwx = + attendance + reference-day weather.
C10 = old except 3127 game days (att);  C11 = attwx everywhere except 3127 non-game event days (old).
args: OUT ATT_CSV
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(sys.argv[1]); KEYS = ['date', 'station_id', 'hour']
games = pd.read_csv(sys.argv[2], parse_dates=['date']).date
rows = []
for o in ['o0227', 'o0331', 'o0430']:
    d = OUT / o
    p = pd.read_parquet(d / 'preds.parquet')
    b = pd.read_parquet(d / 'preds_base_C.parquet')
    a, ac = pd.read_parquet(d / 'preds_att_AEH.parquet'), pd.read_parquet(d / 'preds_att_C.parquet')
    w, wc = pd.read_parquet(d / 'preds_attwx_AEH.parquet'), pd.read_parquet(d / 'preds_attwx_C.parquet')
    for q in (b, a, ac, w, wc):
        assert (q[KEYS].astype(str).to_numpy() == p[KEYS].astype(str).to_numpy()).all()
    y, ev, V = p.y.to_numpy(), p.ev.eq(1).to_numpy(), p.V.to_numpy()
    s27 = p.station_id.eq(3127).to_numpy(); off = p.offday.eq(1).to_numpy()
    game = s27 & pd.to_datetime(p.date).isin(games).to_numpy()
    keep_old = s27 & ev & ~game

    def c9(A, E, C, H):
        A, E, C, H = (np.asarray(x) for x in (A, E, C, H))
        Em = (E + C + H) / 3
        M = .8 * Em + .2 * A; B = np.where(ev, M, .8 * M + .2 * V)
        return np.where(s27, np.where(~ev, A, (A + Em) / 2), B)

    old = c9(p.A, p.E42, b.C, p.H)
    att = c9(a.A, a.E, ac.C, a.H)
    wx = c9(w.A, w.E, wc.C, w.H)
    R = {'C9 (old)': old, 'C10 (game days att)': np.where(game, att, old),
         'att all, 3127 concerts old': np.where(keep_old, old, att), 'C11 = wx all, 3127 concerts old': np.where(keep_old, old, wx)}
    for k, args in {'A': (w.A, a.E, ac.C, a.H), 'E': (a.A, w.E, ac.C, a.H), 'C': (a.A, a.E, wc.C, a.H), 'H': (a.A, a.E, ac.C, w.H)}.items():
        R[f'wx in {k} only'] = np.where(keep_old, old, c9(*args))
    r = lambda q, m: float(np.sqrt(((y[m] - q[m]) ** 2).mean()))
    allm = np.ones(len(y), bool)
    for k, q in R.items():
        rows.append(dict(origin=o, recipe=k, rmse=r(q, allm), weekday=r(q, ~off), offday=r(q, off)))
t = pd.DataFrame(rows)
print(t[t.recipe.eq('C9 (old)')].round(5).to_string(index=False))
for c in ['rmse', 'offday', 'weekday']:
    t['d'] = t[c] - t.origin.map(t[t.recipe.eq('C9 (old)')].set_index('origin')[c])
    wv = t.pivot(index='recipe', columns='origin', values='d'); wv['worst'] = wv.max(axis=1)
    print(f'\n{c}: change vs C9 (negative = better)'); print(wv.round(5).to_string())
