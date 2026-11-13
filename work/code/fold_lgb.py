"""LightGBM lever check on the Mar-Jun 2026 fold (future event flags = full-history detector, same for all variants)."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb

OUT = Path(sys.argv[1]); DATA = Path(sys.argv[2]); t0 = time.time()
tr = pd.read_parquet(OUT / 'fold_train.parquet'); fut = pd.read_parquet(OUT / 'fold_future.parquet')
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
y = fut.y.to_numpy(); v2 = np.maximum(fut.base_v2.to_numpy(), 0)


def fit(target_tr, base_fut):
    ds = lgb.Dataset(tr[feats], target_tr, categorical_feature=CATS, free_raw_data=False)
    m = lgb.train(P, ds, num_boost_round=500)
    return np.maximum(base_fut + m.predict(fut[feats]), 0)


def rep(name, p):
    e = y - p; off = fut.offday.eq(1).to_numpy(); ev = fut.ev.eq(1).to_numpy()
    print(f'{name:34s} {np.sqrt((e**2).mean()):.5f} bias {e.mean():+.4f} | weekday {np.sqrt((e[~off]**2).mean()):.4f} '
          f'offday {np.sqrt((e[off]**2).mean()):.4f} event {np.sqrt((e[ev]**2).mean()):.4f} [{time.time()-t0:.0f}s]', flush=True)


sn_tr = (tr.lvl8 + tr.yoy_delta.fillna(0)).to_numpy(); sn_f = (fut.lvl8 + fut.yoy_delta.fillna(0)).to_numpy()
preds = {
    'A_direct': fit(tr.y.to_numpy(), 0.),
    'D_resid_lvl8': fit((tr.y - tr.lvl8).to_numpy(), fut.lvl8.to_numpy()),
    'E_resid_seasonal': fit(tr.y.to_numpy() - sn_tr, sn_f),
}
rep('v2 structure', v2)
for k, p in preds.items():
    rep(k, p)
    for w in (.1, .2, .3):
        rep(f'  {k} + v2 {w:.1f}', (1 - w) * p + w * v2)
out = fut[['date', 'station_id', 'hour', 'y', 'ev', 'ev_rel', 'offday', 'dow']].copy()
for k, p in preds.items():
    out[k] = p
out['v2'] = v2
out.to_parquet(OUT / 'fold_lgb_preds.parquet', index=False)

print('\n[event days, E] residual by station x rel (mean y-pred, n station-days)')
e = out.loc[out.ev.eq(1)].assign(r=lambda d: d.y - d.E_resid_seasonal, rel=lambda d: d.ev_rel.round().clip(-6, 3))
t = e.groupby(['station_id', 'rel']).r.mean().unstack().round(2)
n = e.groupby('station_id').date.nunique()
print(t.assign(days=n).loc[n.sort_values(ascending=False).index[:6]].to_string())
