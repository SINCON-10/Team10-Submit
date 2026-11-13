"""Fold (Mar-Jun 2026): CatBoost on the seasonal residual target, alone and averaged with the LightGBM residual model (E).
Scored through the C4 recipe: M = .8*Emix + .2*A ; non-event .8*M + .2*v2 ; 3127 -> (A + Emix)/2.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostRegressor, Pool

OUT = Path(sys.argv[1]); DATA = Path(sys.argv[2]); t0 = time.time()
tr = pd.read_parquet(OUT / 'fold_train.parquet'); fut = pd.read_parquet(OUT / 'fold_future.parquet')
prev = pd.read_parquet(OUT / 'fold_lgb_preds.parquet')
tp = pd.read_csv(DATA / 'train_processed.csv.gz', usecols=['date', 'station_id', 'hour', 'is_event', 'ev_peak'], parse_dates=['date'])
orc = fut[['date', 'station_id', 'hour']].merge(tp, on=['date', 'station_id', 'hour'], how='left', validate='one_to_one')
fut['ev'] = orc.is_event.fillna(0).astype(int).to_numpy()
fut['ev_rel'] = np.where(fut.ev.eq(1), fut.hour - orc.ev_peak.to_numpy(), np.nan)
drop = {'date', 'station_id', 'y', 'example_id', 'diagnostic_event', 'base_v2', 'base_minus_lvl8'}
feats = [c for c in fut.columns if c not in drop]
CATS = ['sid', 'line', 'dow', 'dt3']
for c in CATS:
    tr[c] = tr[c].astype(int); fut[c] = fut[c].astype(int)
P = dict(depth=6, learning_rate=.08, loss_function='RMSE', l2_leaf_reg=5, random_seed=42, thread_count=8,
         bootstrap_type='Bernoulli', subsample=.8, rsm=.9, border_count=128, max_ctr_complexity=1, one_hot_max_size=10,
         allow_writing_files=False, verbose=0)
y = fut.y.to_numpy(); ev = fut.ev.eq(1).to_numpy(); s27 = fut.station_id.eq(3127).to_numpy()
E, A, V = prev.E_resid_seasonal.to_numpy(), prev.A_direct.to_numpy(), prev.v2.to_numpy()
sn_tr = (tr.lvl8 + tr.yoy_delta.fillna(0)).to_numpy(); sn_f = (fut.lvl8 + fut.yoy_delta.fillna(0)).to_numpy()
r = lambda p: float(np.sqrt(((y - p) ** 2).mean()))


def c4(Emix):
    M = .8 * Emix + .2 * A
    B = np.where(ev, M, .8 * M + .2 * V)
    return np.where(s27, (A + Emix) / 2, B)


print(f'reference LGB E: raw {r(E):.5f} | C4 recipe {r(c4(E)):.5f}', flush=True)
pool_tr = Pool(tr[feats], tr.y.to_numpy() - sn_tr, cat_features=CATS)
pool_f = Pool(fut[feats], cat_features=CATS)
res = {}
for name, it in [('cat500', 500), ('cat1500', 1500)]:
    m = CatBoostRegressor(iterations=it, **P).fit(pool_tr)
    Ec = np.maximum(sn_f + m.predict(pool_f), 0); res[name] = Ec
    print(f'{name}: raw {r(Ec):.5f} | C4 recipe alone {r(c4(Ec)):.5f} | avg with LGB {r(c4((E + Ec) / 2)):.5f} '
          f'| LGB 70/30 {r(c4(.7 * E + .3 * Ec)):.5f} | corr of errors {np.corrcoef(y - E, y - Ec)[0, 1]:.3f} [{time.time()-t0:.0f}s]', flush=True)
pd.DataFrame(res).to_parquet(OUT / 'fold_cat_preds.parquet', index=False)
