"""Fold check for the SSG attendance feature: att = attendance/1000 on 3127 game days (NaN elsewhere).
fold_att.py + E_CURVE/A_CURVE/H_CURVE (predictions after k trees), HGB_ITERS. Retrains A (LGB direct), E (LGB residual, seed 42) and H (HistGB1500 residual) with the 56th feature, same params as fold_multi.py.

args: TRAIN_DIR ORIGIN_DIR DATA ATT_CSV [MODELS]   (TRAIN_DIR holds fold_train/fold_future; MODELS default 'AEH', 'C' = CatBoost1500)
writes ORIGIN_DIR/preds_att_<MODELS>.parquet
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import os
SEED = int(os.environ.get('SEED', '42'))   # whole-set seed (LGB seed, HGB random_state, Cat random_seed)

TD, OD, DATA, ATT = map(Path, sys.argv[1:5]); MODELS = sys.argv[5] if len(sys.argv) > 5 else 'AEH'; t0 = time.time()
KEYS = ['date', 'station_id', 'hour']
CATS = ['sid', 'line', 'dow', 'dt3']
tr = pd.read_parquet(os.environ.get('TRAIN_FILE') or TD / 'fold_train.parquet')   # TRAIN_FILE: e.g. twice-a-month cutoffs (build_smc.py)
TAG = os.environ.get('TAG', '')
if os.environ.get('DROP_BREAK') == '1':   # same rule as final_att.py: 3110/3111 rows straddling the 2025-06-28 Geomdan opening
    b = pd.Timestamp('2025-06-28'); dd = pd.to_datetime(tr.date); cu = dd - pd.to_timedelta(tr.horizon, unit='D')
    brk = (tr.station_id.isin([3110, 3111]) & (dd >= b) & (cu - pd.Timedelta(days=56) < b)).to_numpy()
    tr = tr[~brk].reset_index(drop=True); del dd, cu
    print(f'dropped {int(brk.sum()):,} break rows of 3110/3111', flush=True)
fut = pd.read_parquet(TD / 'fold_future.parquet')
tp = pd.read_csv(DATA / 'train_processed.csv.gz', usecols=KEYS + ['is_event', 'ev_peak'], parse_dates=['date'])
orc = fut[KEYS].merge(tp, on=KEYS, how='left', validate='one_to_one')
fut['ev'] = orc.is_event.fillna(0).astype(int).to_numpy()            # schedule proxy, same as fold_multi.py
fut['ev_rel'] = np.where(fut.ev.eq(1), fut.hour - orc.ev_peak.to_numpy(), np.nan)
drop = {'date', 'station_id', 'y', 'example_id', 'diagnostic_event', 'base_v2', 'base_minus_lvl8'}
feats = [c for c in fut.columns if c not in drop]
assert len(feats) == 55, len(feats)
att = pd.read_csv(ATT, parse_dates=['date']).groupby('date').attendance.max() / 1000   # doubleheader -> one value


def add_att(df):
    return pd.to_datetime(df.date).map(att).where(df.station_id.eq(3127)).to_numpy(np.float64)


import os
NOATT, WX = os.environ.get('NOATT') == '1', os.environ.get('WX') == '1'
if not NOATT:
    tr['att'] = add_att(tr); fut['att'] = add_att(fut); feats = feats + ['att']
else:
    tr['att'] = np.nan; fut['att'] = np.nan
if WX:   # weather of the level / partner-year reference days (wxfeat.py)
    from wxfeat import add_wx
    origin = pd.Timestamp(f'2026-{OD.name[1:3]}-{OD.name[3:5]}')
    assert pd.to_datetime(tr.date).max() <= origin
    feats = feats + add_wx(tr, origin); add_wx(fut, origin)
    print(f'{OD.name} wx features added (origin {origin.date()}), non-null share train {tr.wx_pwin.notna().mean():.2f} future {fut.wx_pwin.notna().mean():.2f}', flush=True)
print(f'{OD.name}: att rows train {np.isfinite(tr.att).sum():,} future {np.isfinite(fut.att).sum():,} '
      f'(future game days {fut.loc[np.isfinite(fut.att), "date"].nunique()})', flush=True)
YB = 'yoy_delta_all' if os.environ.get('BASE_ALL') == '1' else 'yoy_delta'   # BASE_ALL=1: residual base uses the all-station partner-year delta
sn_tr = (tr.lvl8 + tr[YB].fillna(0)).to_numpy(); sn_f = (fut.lvl8 + fut[YB].fillna(0)).to_numpy()
out = fut[KEYS].copy()
if 'A' in MODELS or 'E' in MODELS:
    import lightgbm as lgb
    P = dict(objective='regression', learning_rate=.05, num_leaves=31, min_data_in_leaf=100, lambda_l2=5., feature_fraction=.9,
             bagging_fraction=.8, bagging_freq=1, max_bin=127, seed=SEED, deterministic=True, force_col_wise=True,
             num_threads=8, verbosity=-1)
    if 'A' in MODELS:
        mA = lgb.train(P, lgb.Dataset(tr[feats], tr.y.to_numpy(), categorical_feature=CATS), int(os.environ.get('LGB_ROUNDS', '500')))
        out['A'] = np.maximum(mA.predict(fut[feats]), 0)
        for k in [int(x) for x in os.environ.get('A_CURVE', '').split(',') if x]:   # learning curve: same model, first k trees
            out[f'A_{k}'] = np.maximum(mA.predict(fut[feats], num_iteration=k), 0)
    if 'E' in MODELS:
      m = lgb.train(P, lgb.Dataset(tr[feats], tr.y.to_numpy() - sn_tr, categorical_feature=CATS), int(os.environ.get('LGB_ROUNDS', '500')))
      out['E'] = np.maximum(sn_f + m.predict(fut[feats]), 0)
      for k in [int(x) for x in os.environ.get('E_CURVE', '').split(',') if x]:   # learning curve: same model, first k trees
          out[f'E_{k}'] = np.maximum(sn_f + m.predict(fut[feats], num_iteration=k), 0)
    print(f'{OD.name} lgb done {time.time()-t0:.0f}s', flush=True)
if 'H' in MODELS:
    from sklearn.ensemble import HistGradientBoostingRegressor
    h = HistGradientBoostingRegressor(learning_rate=.05, max_iter=int(os.environ.get('HGB_ITERS', '1500')), max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=5.,
                                      max_bins=127, early_stopping=False, categorical_features=[feats.index(c) for c in CATS], random_state=SEED)
    h.fit(tr[feats].to_numpy(np.float32), tr.y.to_numpy() - sn_tr)
    out['H'] = np.maximum(sn_f + h.predict(fut[feats].to_numpy(np.float32)), 0)
    hk = {int(x) for x in os.environ.get('H_CURVE', '').split(',') if x}
    if hk:   # learning curve: predictions after the first k iterations of the same model
        for i, ph in enumerate(h.staged_predict(fut[feats].to_numpy(np.float32)), 1):
            if i in hk:
                out[f'H_{i}'] = np.maximum(sn_f + ph, 0)
    print(f'{OD.name} hgb done {time.time()-t0:.0f}s', flush=True)
if 'C' in MODELS or 'D' in MODELS:   # D = CatBoost on the direct target y (no residual base)
    from catboost import CatBoostRegressor, Pool
    for c in CATS:
        tr[c] = tr[c].astype(int); fut[c] = fut[c].astype(int)
    Pc = dict(depth=int(os.environ.get('CAT_DEPTH', '6')), learning_rate=float(os.environ.get('CAT_LR', '.08')), loss_function='RMSE', l2_leaf_reg=5, random_seed=SEED, thread_count=8,
              bootstrap_type='Bernoulli', subsample=.8, rsm=.9, border_count=128, max_ctr_complexity=1, one_hot_max_size=10,
              allow_writing_files=False, verbose=0)
    if os.environ.get('CAT_GPU') == '1':   # CatBoost on the GPU: rsm is CPU-only for RMSE -> dropped
        Pc.pop('rsm'); Pc.pop('thread_count'); Pc.update(task_type='GPU', devices='0')
    if 'C' in MODELS:
        cm = CatBoostRegressor(iterations=int(os.environ.get('CAT_ITERS', '1500')), **Pc).fit(Pool(tr[feats], tr.y.to_numpy() - sn_tr, cat_features=CATS))
        out['C'] = np.maximum(sn_f + cm.predict(Pool(fut[feats], cat_features=CATS)), 0)
    if 'D' in MODELS:
        cd = CatBoostRegressor(iterations=int(os.environ.get('CAT_ITERS', '1500')), **Pc).fit(Pool(tr[feats], tr.y.to_numpy(), cat_features=CATS))
        out['D'] = np.maximum(cd.predict(Pool(fut[feats], cat_features=CATS)), 0)
    print(f'{OD.name} cat done {time.time()-t0:.0f}s', flush=True)
out.to_parquet(OD / f'preds_{"base" if NOATT else "att"}{"wx" if WX else ""}{"" if SEED == 42 else f"_s{SEED}"}{TAG}_{MODELS}.parquet', index=False)
