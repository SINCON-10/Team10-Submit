"""CatBoost1500 residual per origin (codex runtime), then score every recipe on every origin.

args: EDIR DATA ORIGIN_DIR...   (each has fold_train/fold_future/preds parquet; o0227 reuses EDIR/out/fold_cat_preds.parquet)
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

EDIR, DATA = Path(sys.argv[1]), Path(sys.argv[2]); DIRS = [Path(p) for p in sys.argv[3:]]; t0 = time.time()
KEYS = ['date', 'station_id', 'hour']
CATS = ['sid', 'line', 'dow', 'dt3']
P = dict(depth=6, learning_rate=.08, loss_function='RMSE', l2_leaf_reg=5, random_seed=42, thread_count=10,
         bootstrap_type='Bernoulli', subsample=.8, rsm=.9, border_count=128, max_ctr_complexity=1, one_hot_max_size=10,
         allow_writing_files=False, verbose=0)


def cat_preds(d):
    f = d / 'cat.parquet'
    if f.exists():
        return pd.read_parquet(f).C.to_numpy()
    if d.name == 'o0227':
        C = pd.read_parquet(EDIR / 'out' / 'fold_cat_preds.parquet').cat1500.to_numpy()
    else:
        from catboost import CatBoostRegressor, Pool
        tr = pd.read_parquet(d / 'fold_train.parquet'); fut = pd.read_parquet(d / 'fold_future.parquet')
        tp = pd.read_csv(DATA / 'train_processed.csv.gz', usecols=KEYS + ['is_event', 'ev_peak'], parse_dates=['date'])
        orc = fut[KEYS].merge(tp, on=KEYS, how='left', validate='one_to_one')     # same schedule proxy as fold_multi.py
        fut['ev'] = orc.is_event.fillna(0).astype(int).to_numpy()
        fut['ev_rel'] = np.where(fut.ev.eq(1), fut.hour - orc.ev_peak.to_numpy(), np.nan)
        drop = {'date', 'station_id', 'y', 'example_id', 'diagnostic_event', 'base_v2', 'base_minus_lvl8'}
        feats = [c for c in fut.columns if c not in drop]
        assert len(feats) == 55
        for c in CATS:
            tr[c] = tr[c].astype(int); fut[c] = fut[c].astype(int)
        sn_tr = (tr.lvl8 + tr.yoy_delta.fillna(0)).to_numpy(); sn_f = (fut.lvl8 + fut.yoy_delta.fillna(0)).to_numpy()
        m = CatBoostRegressor(iterations=1500, **P).fit(Pool(tr[feats], tr.y.to_numpy() - sn_tr, cat_features=CATS))
        C = np.maximum(sn_f + m.predict(Pool(fut[feats], cat_features=CATS)), 0)
    pd.DataFrame({'C': C}).to_parquet(f, index=False)
    print(f'{d.name} cat ready {time.time()-t0:.0f}s', flush=True)
    return C


import os
if os.environ.get('CAT_ONLY'):
    for d in DIRS:
        cat_preds(d)
    sys.exit(0)

rows = []
for d in DIRS:
    p = pd.read_parquet(d / 'preds.parquet'); C = cat_preds(d)
    y, ev, A, V = p.y.to_numpy(), p.ev.eq(1).to_numpy(), p.A.to_numpy(), p.V.to_numpy()
    s27 = p.station_id.eq(3127).to_numpy(); off = p.offday.eq(1).to_numpy()
    E, Es, H = p.E42.to_numpy(), (p.E42 + p.E1 + p.E2).to_numpy() / 3, p.H.to_numpy()

    def rec(Em, aw=.2):
        M = (1 - aw) * Em + aw * A
        B = np.where(ev, M, .8 * M + .2 * V)
        return np.where(s27, (A + Em) / 2, B)

    def c3(Em):
        B = np.where(ev, Em, .8 * Em + .2 * V)
        return np.where(s27, (A + Em) / 2, B)

    R = {'C3 (E)': c3(E), 'C4 (E+A20)': rec(E), 'C4 seed3': rec(Es), 'C7 (E+Cat)/2': rec((E + C) / 2),
         'C7 seed3': rec((Es + C) / 2), 'C8 (E+Cat+HGB)/3': rec((E + C + H) / 3), 'C8 seed3': rec((Es + C + H) / 3)}
    for k, v in R.items():
        e = y - v
        rows.append(dict(origin=d.name, recipe=k, rmse=np.sqrt((e ** 2).mean()), weekday=np.sqrt((e[~off] ** 2).mean()),
                         offday=np.sqrt((e[off] ** 2).mean()), bias=e.mean(), days=p.date.nunique()))
t = pd.DataFrame(rows)
ref = t[t.recipe.eq('C4 (E+A20)')].set_index('origin').rmse
t['vs_C4'] = t.rmse - t.origin.map(ref)
print(t.round(5).to_string(index=False))
print('\nchange vs C4 across origins:'); print(t.groupby('recipe').vs_C4.agg(['mean', 'min', 'max']).round(5).to_string())
