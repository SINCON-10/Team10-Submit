"""Extra validation origins (forecast through 2026-06-30) built exactly like DataEngine.build, then LGB (A direct, E residual x3 seeds) + HistGB1500 residual.

args: BUNDLE DATA OUTDIR ORIGIN [REUSE_DIR]
  REUSE_DIR: folder holding fold_train.parquet / fold_future.parquet for this origin (skip the build)
writes OUTDIR/{fold_train,fold_future,preds}.parquet
"""
import sys, time, types, warnings
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingRegressor

BUNDLE, DATA, OUT = map(Path, sys.argv[1:4]); ORIGIN = pd.Timestamp(sys.argv[4])
REUSE = Path(sys.argv[5]) if len(sys.argv) > 5 else None
OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
KEYS = ['date', 'station_id', 'hour']
if REUSE is None:
    sys.path.insert(0, str(BUNDLE))
    from data_v4 import DataEngine  # noqa: E402
    eng = DataEngine(DATA, BUNDLE)
    end = pd.Timestamp('2026-06-30')
    cube, history = eng.cube_at(ORIGIN)                     # passengers after ORIGIN removed (asserted inside)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        fut = eng.prep.cutoff_table(eng.prep.build_rows(cube, ORIGIN, ORIGIN + pd.Timedelta(days=1), end,
                                                        require_target=False, avail_end=ORIGIN), cube).drop(columns='y')
        tr = eng.prep.cutoff_table(eng.prep.make_training_set(cube, ORIGIN), cube)
    fut, cold = eng._cold_start(fut)
    assert fut.ev.eq(0).all() and tr.date.max() <= ORIGIN and (tr.horizon > 0).all()
    labels = eng.passengers.loc[eng.passengers.date.gt(ORIGIN) & eng.passengers.date.le(end), KEYS + ['boardings']].copy()
    labels['y'] = np.log1p(labels.pop('boardings'))
    fut = fut.merge(labels, on=KEYS, how='inner', validate='one_to_one')
    fut['base_v2'] = eng.structural_values(types.SimpleNamespace(cube=cube), fut)
    print(f'{ORIGIN.date()} built {time.time()-t0:.0f}s train {len(tr):,} future {len(fut):,} cold {cold}', flush=True)
    tr.to_parquet(OUT / 'fold_train.parquet', index=False); fut.to_parquet(OUT / 'fold_future.parquet', index=False)
else:
    tr = pd.read_parquet(REUSE / 'fold_train.parquet'); fut = pd.read_parquet(REUSE / 'fold_future.parquet')

tp = pd.read_csv(DATA / 'train_processed.csv.gz', usecols=KEYS + ['is_event', 'ev_peak'], parse_dates=['date'])
orc = fut[KEYS].merge(tp, on=KEYS, how='left', validate='one_to_one')
fut['ev'] = orc.is_event.fillna(0).astype(int).to_numpy()            # schedule proxy, same for all recipes
fut['ev_rel'] = np.where(fut.ev.eq(1), fut.hour - orc.ev_peak.to_numpy(), np.nan)
drop = {'date', 'station_id', 'y', 'example_id', 'diagnostic_event', 'base_v2', 'base_minus_lvl8'}
feats = [c for c in fut.columns if c not in drop]
assert len(feats) == 55, len(feats)
CATS = ['sid', 'line', 'dow', 'dt3']
P = dict(objective='regression', learning_rate=.05, num_leaves=31, min_data_in_leaf=100, lambda_l2=5., feature_fraction=.9,
         bagging_fraction=.8, bagging_freq=1, max_bin=127, seed=42, deterministic=True, force_col_wise=True,
         num_threads=8, verbosity=-1)
sn_tr = (tr.lvl8 + tr.yoy_delta.fillna(0)).to_numpy(); sn_f = (fut.lvl8 + fut.yoy_delta.fillna(0)).to_numpy()
out = fut[KEYS + ['y', 'ev', 'offday', 'horizon']].copy()
out['V'] = np.maximum(fut.base_v2.to_numpy(), 0)
out['A'] = np.maximum(lgb.train(P, lgb.Dataset(tr[feats], tr.y.to_numpy(), categorical_feature=CATS), 500).predict(fut[feats]), 0)
for s in (42,):   # seeds 1/2 dropped (seed averaging was rejected)
    m = lgb.train({**P, 'seed': s}, lgb.Dataset(tr[feats], tr.y.to_numpy() - sn_tr, categorical_feature=CATS), 500)
    out[f'E{s}'] = np.maximum(sn_f + m.predict(fut[feats]), 0)
print(f'lgb done {time.time()-t0:.0f}s', flush=True)
h = HistGradientBoostingRegressor(learning_rate=.05, max_iter=1500, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=5.,
                                  max_bins=127, early_stopping=False, categorical_features=[feats.index(c) for c in CATS], random_state=42)
h.fit(tr[feats].to_numpy(np.float32), tr.y.to_numpy() - sn_tr)
out['H'] = np.maximum(sn_f + h.predict(fut[feats].to_numpy(np.float32)), 0)
out.to_parquet(OUT / 'preds.parquet', index=False)
print(f'hgb done {time.time()-t0:.0f}s', flush=True)
