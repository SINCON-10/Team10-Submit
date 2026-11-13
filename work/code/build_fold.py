"""Error-budget fold: origin 2026-02-27 -> 2026-02-28..06-30, rebuilt from v4 with no post-origin labels.

Outputs (scratch only): fold_train.parquet, fold_future.parquet, preds.parquet
  pred_lgb_noev : LightGBM direct log1p, team-like params, future ev=0 (schedule unknown)
  pred_lgb_orev : same model, future ev/ev_rel from full-history detector (ORACLE upper bound, diagnostic only)
  base_v2       : v2 structural model (events excluded)
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

BUNDLE = Path(sys.argv[1]); DATA = Path(sys.argv[2]); OUT = Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BUNDLE))
from data_v4 import DataEngine  # noqa: E402

t0 = time.time()
eng = DataEngine(DATA, BUNDLE)
fold = eng.build('evaluation', '2026-02-27', {'smoke': False})
print(f'fold built {time.time()-t0:.0f}s train {len(fold.train):,} future {len(fold.future):,}', flush=True)
fut = fold.future.copy()
fut['base_v2'] = eng.structural_values(fold, fut)
print(f'v2 base {time.time()-t0:.0f}s', flush=True)
fold.train.to_parquet(OUT / 'fold_train.parquet', index=False)
fut.to_parquet(OUT / 'fold_future.parquet', index=False)

import lightgbm as lgb  # noqa: E402
feats = fold.features
params = dict(objective='regression', learning_rate=.05, num_leaves=31, min_data_in_leaf=100, lambda_l2=5.,
              feature_fraction=.9, bagging_fraction=.8, bagging_freq=1, max_bin=127, seed=42, deterministic=True,
              force_col_wise=True, num_threads=8, verbosity=-1)
tr = fold.train
ds = lgb.Dataset(tr[feats].astype(np.float32), tr.y, categorical_feature=['sid'], free_raw_data=True)
model = lgb.train(params, ds, num_boost_round=500)
print(f'lgb trained {time.time()-t0:.0f}s', flush=True)

# Oracle events (diagnostic only): full-history detector from train_processed
tp = pd.read_csv(DATA / 'train_processed.csv.gz', usecols=['date', 'station_id', 'hour', 'is_event', 'ev_peak'], parse_dates=['date'])
orc = fut[['date', 'station_id', 'hour']].merge(tp, on=['date', 'station_id', 'hour'], how='left', validate='one_to_one')
fut_or = fut.copy()
fut_or['ev'] = orc.is_event.fillna(0).astype(int).to_numpy()
fut_or['ev_rel'] = np.where(fut_or.ev.eq(1), fut_or.hour - orc.ev_peak.to_numpy(), np.nan)

out = fut[['date', 'station_id', 'hour', 'y', 'diagnostic_event', 'is_hol', 'is_major', 'offday', 'dt3', 'dow', 'lvl8', 'base_v2']].copy()
out['pred_lgb_noev'] = np.maximum(model.predict(fut[feats].astype(np.float32)), 0)
out['pred_lgb_orev'] = np.maximum(model.predict(fut_or[feats].astype(np.float32)), 0)
out['base_v2'] = np.maximum(out.base_v2, 0)
out.to_parquet(OUT / 'preds.parquet', index=False)
model.save_model(str(OUT / 'lgb_fold.txt'))
r = lambda p: float(np.sqrt(np.mean((out.y - out[p]) ** 2)))
print({p: round(r(p), 5) for p in ['pred_lgb_noev', 'pred_lgb_orev', 'base_v2']}, f'{time.time()-t0:.0f}s', flush=True)
