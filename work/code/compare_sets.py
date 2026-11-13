"""Fold check of whole-model-set averaging (C12 recipe) against C9 / C10 on the three origins.
Each set = the four models (LGB direct A, LGB resid E, CatBoost C, HistGB H) put through the C9 blend.
old = preds.parquet (A, E42, H) + preds_base_C ; other sets = preds_{name}_AEH + preds_{name}_C.
Routing as in make_c12.py: 3127 non-game event days -> old; 3127 game days -> mean of new sets; else mean of all sets.
args: OUT ATT_CSV RECIPE [RECIPE ...]   RECIPE = comma list of new sets, e.g. att,attwx  att,attwx,att_s7
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(sys.argv[1]); KEYS = ['date', 'station_id', 'hour']
games = pd.read_csv(sys.argv[2], parse_dates=['date']).date
RECIPES = [r.split(',') for r in sys.argv[3:]]
rows = []
ORIGINS = os.environ.get('ORIGINS', 'o0227,o0331,o0430').split(',')   # partial check while folds are still training
for o in ORIGINS:
    d = OUT / o
    p = pd.read_parquet(d / 'preds.parquet')
    y, ev, V = p.y.to_numpy(), p.ev.eq(1).to_numpy(), p.V.to_numpy()
    s27 = p.station_id.eq(3127).to_numpy(); off = p.offday.eq(1).to_numpy()
    month = pd.to_datetime(p.date).dt.month.to_numpy()
    game = s27 & pd.to_datetime(p.date).isin(games).to_numpy()
    keep_old = s27 & ev & ~game

    def c9(A, E, C, H):
        A, E, C, H = (np.asarray(x) for x in (A, E, C, H))
        Em = (E + C + H) / 3
        M = .8 * Em + .2 * A; B = np.where(ev, M, .8 * M + .2 * V)
        return np.where(s27, np.where(~ev, A, (A + Em) / 2), B)

    def load(name):
        a, c = pd.read_parquet(d / f'preds_{name}_AEH.parquet'), pd.read_parquet(d / f'preds_{name}_C.parquet')
        for q in (a, c):
            assert (q[KEYS].astype(str).to_numpy() == p[KEYS].astype(str).to_numpy()).all()
        return c9(a.A, a.E, c.C, a.H)

    old = c9(p.A, p.E42, pd.read_parquet(d / 'preds_base_C.parquet').C, p.H)
    sets = {n: load(n) for n in sorted({n for r in RECIPES for n in r})}
    R = {'C9': old, 'C10': np.where(game, sets['att'], old)}
    for r in RECIPES:
        new = np.mean([sets[n] for n in r], axis=0)
        allm = (old + np.sum([sets[n] for n in r], axis=0)) / (1 + len(r))
        R['old+' + '+'.join(r)] = np.where(keep_old, old, np.where(game, new, allm))
    for k, q in R.items():
        q = np.maximum(q, 0)
        row = dict(origin=o, recipe=k, rmse=np.sqrt(((y - q) ** 2).mean()), weekday=np.sqrt(((y - q)[~off] ** 2).mean()),
                   offday=np.sqrt(((y - q)[off] ** 2).mean()), s3127=np.sqrt(((y - q)[s27] ** 2).mean()))
        for mo in np.unique(month):
            row[f'm{mo:02d}'] = np.sqrt(((y - q)[month == mo] ** 2).mean())
        rows.append(row)
t = pd.DataFrame(rows)
cols = [c for c in t.columns if c not in ('origin', 'recipe')]
base = t[t.recipe.eq('C10')].set_index('origin')
print('C10 scores'); print(base[cols].round(5).to_string())
for c in ['rmse', 'weekday', 'offday', 's3127']:
    t['d'] = t[c] - t.origin.map(base[c])
    wv = t.pivot(index='recipe', columns='origin', values='d'); wv['worst'] = wv.max(axis=1)
    print(f'\n{c}: change vs C10 (negative = better)'); print(wv.round(5).to_string())
mc = [c for c in cols if c.startswith('m')]
print('\nmonth cells: change vs C10, (better cells / cells) and worst cell')
for k in t.recipe.unique():
    if k in ('C9', 'C10'):
        continue
    dd = []
    for o in ORIGINS:
        r = t[(t.recipe == k) & (t.origin == o)].iloc[0]
        dd += [r[c] - base.loc[o, c] for c in mc if pd.notna(r[c])]
    dd = np.array(dd)
    print(f'{k:32s} {int((dd < 0).sum())}/{len(dd)} worst {dd.max():+.5f}')
