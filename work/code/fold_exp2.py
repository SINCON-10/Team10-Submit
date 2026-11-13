"""Round 2 on the Mar-Jun 2026 fold: base variants for the residual target + capacity; scored raw and through the C3 post-processing."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb

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
P = dict(objective='regression', learning_rate=.05, num_leaves=31, min_data_in_leaf=100, lambda_l2=5., feature_fraction=.9,
         bagging_fraction=.8, bagging_freq=1, max_bin=127, seed=42, deterministic=True, force_col_wise=True,
         num_threads=8, verbosity=-1)
y = fut.y.to_numpy(); ev = fut.ev.eq(1).to_numpy(); s27 = fut.station_id.eq(3127).to_numpy()
A, V = prev.A_direct.to_numpy(), prev.v2.to_numpy()


def post(E):  # C3 recipe
    B = np.where(ev, E, .8 * E + .2 * V)
    return np.where(s27, (A + E) / 2, B)


def r(p):
    return np.sqrt(((y - p) ** 2).mean())


def run(name, base_tr, base_f, params=P, rounds=500):
    m = lgb.train(params, lgb.Dataset(tr[feats], tr.y.to_numpy() - base_tr, categorical_feature=CATS), num_boost_round=rounds)
    E = np.maximum(base_f + m.predict(fut[feats]), 0)
    print(f'{name:40s} raw {r(E):.5f} | C3-post {r(post(E)):.5f}  bias {np.mean(y-E):+.4f}  [{time.time()-t0:.0f}s]', flush=True)
    return E


bases = {
    'E  lvl8 + yoy_delta': lambda d: d.lvl8 + d.yoy_delta.fillna(0),
    'E2 lvl8 + lvl8_dow + yoy_delta': lambda d: d.lvl8 + d.lvl8_dow.fillna(0) + d.yoy_delta.fillna(0),
    'E3 lvl8 + clip(yoy_delta, +-0.5)': lambda d: d.lvl8 + d.yoy_delta.fillna(0).clip(-.5, .5),
    'E4 lvl8 + yoy_delta_all': lambda d: d.lvl8 + d.yoy_delta_all.fillna(0),
}
print(f'reference: previous E raw {r(prev.E_resid_seasonal.to_numpy()):.5f} | C3-post {r(post(prev.E_resid_seasonal.to_numpy())):.5f}')
res = {}
for k, f in list(bases.items())[1:]:
    res[k] = run(k, f(tr).to_numpy(), f(fut).to_numpy())
f = bases['E  lvl8 + yoy_delta']
res['E5'] = run('E5 E, leaves 63, 1000 rounds', f(tr).to_numpy(), f(fut).to_numpy(), {**P, 'num_leaves': 63}, 1000)
res['E6'] = run('E6 E, lr .03, 1200 rounds', f(tr).to_numpy(), f(fut).to_numpy(), {**P, 'learning_rate': .03}, 1200)
pd.DataFrame({k: v for k, v in res.items()}).to_parquet(OUT / 'fold_exp2_preds.parquet', index=False)
