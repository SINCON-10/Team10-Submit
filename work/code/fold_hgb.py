"""Fold (Mar-Jun 2026): sklearn HistGradientBoosting on the seasonal residual target, as a third member next to LGB (E) and CatBoost1500."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

OUT = Path(sys.argv[1]); DATA = Path(sys.argv[2]); t0 = time.time()
tr = pd.read_parquet(OUT / 'fold_train.parquet'); fut = pd.read_parquet(OUT / 'fold_future.parquet')
prev = pd.read_parquet(OUT / 'fold_lgb_preds.parquet'); cat = pd.read_parquet(OUT / 'fold_cat_preds.parquet')
tp = pd.read_csv(DATA / 'train_processed.csv.gz', usecols=['date', 'station_id', 'hour', 'is_event', 'ev_peak'], parse_dates=['date'])
orc = fut[['date', 'station_id', 'hour']].merge(tp, on=['date', 'station_id', 'hour'], how='left', validate='one_to_one')
fut['ev'] = orc.is_event.fillna(0).astype(int).to_numpy()
fut['ev_rel'] = np.where(fut.ev.eq(1), fut.hour - orc.ev_peak.to_numpy(), np.nan)
drop = {'date', 'station_id', 'y', 'example_id', 'diagnostic_event', 'base_v2', 'base_minus_lvl8'}
feats = [c for c in fut.columns if c not in drop]
y = fut.y.to_numpy(); ev = fut.ev.eq(1).to_numpy(); s27 = fut.station_id.eq(3127).to_numpy()
E, A, V, C = prev.E_resid_seasonal.to_numpy(), prev.A_direct.to_numpy(), prev.v2.to_numpy(), cat.cat1500.to_numpy()
sn_tr = (tr.lvl8 + tr.yoy_delta.fillna(0)).to_numpy(); sn_f = (fut.lvl8 + fut.yoy_delta.fillna(0)).to_numpy()
r = lambda p: float(np.sqrt(((y - p) ** 2).mean()))


def c4(Emix):
    M = .8 * Emix + .2 * A
    B = np.where(ev, M, .8 * M + .2 * V)
    return np.where(s27, (A + Emix) / 2, B)


print(f'reference: LGB+Cat1500 50/50 C4 recipe {r(c4((E + C) / 2)):.5f}', flush=True)
for it in (500, 1500):
    m = HistGradientBoostingRegressor(learning_rate=.05, max_iter=it, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=5.,
                                      max_bins=127, early_stopping=False, categorical_features=[feats.index(c) for c in ['sid', 'line', 'dow', 'dt3']],
                                      random_state=42)
    m.fit(tr[feats].to_numpy(np.float32), tr.y.to_numpy() - sn_tr)
    H = np.maximum(sn_f + m.predict(fut[feats].to_numpy(np.float32)), 0)
    print(f'hgb{it}: raw {r(H):.5f} | alone C4 {r(c4(H)):.5f} | LGB+HGB 50/50 {r(c4((E + H) / 2)):.5f} | LGB+Cat+HGB 1/3 each {r(c4((E + C + H) / 3)):.5f}'
          f' | corr(err E,H) {np.corrcoef(y - E, y - H)[0, 1]:.3f} corr(err C,H) {np.corrcoef(y - C, y - H)[0, 1]:.3f} [{time.time()-t0:.0f}s]', flush=True)
