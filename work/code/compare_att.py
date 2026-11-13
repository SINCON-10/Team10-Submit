"""C9 recipe with vs without the SSG attendance feature, on the three origins.
'all new' = every model retrained with att; '3127 only' = new models on 3127 rows, old C9 elsewhere (att is NaN off 3127).
args: OUT ATT_CSV
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(sys.argv[1]); KEYS = ['date', 'station_id', 'hour']
rows = []
for o in ['o0227', 'o0331', 'o0430']:
    p = pd.read_parquet(OUT / o / 'preds.parquet')
    C = (pd.read_parquet(OUT / 'fold_cat_preds.parquet').cat1500 if o == 'o0227' else pd.read_parquet(OUT / o / 'cat.parquet').C).to_numpy()
    n = pd.read_parquet(OUT / o / 'preds_att_AEH.parquet'); nc = pd.read_parquet(OUT / o / 'preds_att_C.parquet')
    for q in (n, nc):
        assert (q[KEYS].astype(str).to_numpy() == p[KEYS].astype(str).to_numpy()).all()
    y, ev, V = p.y.to_numpy(), p.ev.eq(1).to_numpy(), p.V.to_numpy()
    s27 = p.station_id.eq(3127).to_numpy()
    att = pd.read_csv(sys.argv[2], parse_dates=['date']).groupby('date').attendance.max()
    game = s27 & pd.to_datetime(p.date).isin(att.index).to_numpy()

    def c9(A, E, C, H):
        Em = (E + C + H) / 3
        M = .8 * Em + .2 * A; B = np.where(ev, M, .8 * M + .2 * V)
        return np.where(s27, np.where(~ev, A, (A + Em) / 2), B)

    old = c9(p.A.to_numpy(), p.E42.to_numpy(), C, p.H.to_numpy())
    new = c9(n.A.to_numpy(), n.E.to_numpy(), nc.C.to_numpy(), n.H.to_numpy())
    R = {'C9 (old)': old, 'all new': new, '3127 only': np.where(s27, new, old),
         'game days only': np.where(game, new, old)}
    # one model at a time (3127 rows only) to see where the gain comes from
    for k, args in {'A only': (n.A, p.E42, C, p.H), 'E only': (p.A, n.E, C, p.H), 'C only': (p.A, p.E42, nc.C, p.H),
                    'H only': (p.A, p.E42, C, n.H)}.items():
        R['3127 ' + k] = np.where(s27, c9(*[np.asarray(a) for a in args]), old)
    r = lambda q, m=slice(None): float(np.sqrt(((y[m] - q[m]) ** 2).mean()))
    for k, q in R.items():
        rows.append(dict(origin=o, recipe=k, rmse=r(q), r3127=r(q, s27), r_game=r(q, game), games=int(pd.Series(p.date[game]).nunique())))
t = pd.DataFrame(rows)
t['d'] = t.rmse - t.origin.map(t[t.recipe.eq('C9 (old)')].set_index('origin').rmse)
t['d_game'] = t.r_game - t.origin.map(t[t.recipe.eq('C9 (old)')].set_index('origin').r_game)
print(t.round(5).to_string(index=False))
w = t.pivot(index='recipe', columns='origin', values='d'); w['worst'] = w.max(axis=1)
print('\nchange in overall RMSE vs C9 (negative = better):'); print(w.round(5).to_string())
