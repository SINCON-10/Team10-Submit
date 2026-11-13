"""Final models (origin 2026-06-30) with the SSG attendance feature + 5 extra test event days.
Same data/params as final_lgb.py / final_hgb.py / final_cat.py; only 'att' (56th feature) and the extra test flags differ.

args: PV4 ATT_CSV OUTDIR MODELS   (MODELS: any of A E H C; C needs the codex runtime)
writes OUTDIR/final_att_<m>.npy (log1p, test order) and OUTDIR/final_att_ev.npy (test ev flags incl. the 5 extra days)
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import os
SEED = int(os.environ.get('SEED', '42'))   # whole-set seed (LGB seed, HGB random_state, Cat random_seed)

PV4, ATT, OUT = map(Path, sys.argv[1:4]); MODELS = sys.argv[4]; t0 = time.time()
CATS = ['sid', 'line', 'dow', 'dt3']
TF = os.environ.get('TRAIN_FILE'); TAG = os.environ.get('TAG', '')   # TRAIN_FILE: twice-a-month cutoffs (build_smc.py)
tr = pd.read_parquet(TF) if TF else pd.read_csv(PV4 / 'train_cutoff_features.csv.gz')
TEF = os.environ.get('TEST_FILE')   # e.g. test_fix.csv.gz (fix_test_yoy.py); training is unchanged
te = pd.read_csv(TEF or PV4 / 'test_cutoff_features.csv.gz')
NOATT = os.environ.get('NOATT') == '1'   # old set (C0/C1/final_hgb): no att, no extra event flags
feats = [c for c in te.columns if c not in ('date', 'station_id', 'y', 'example_id')]
assert len(feats) == 55 and (len(tr) > 2391876 if TF else len(tr) == 2391876) and len(te) == 157194
BRK = os.environ.get('DROP_BREAK') == '1'
if BRK:   # 계양 3110 / 귤현 3111: drop rows whose target is on/after the 2025-06-28 Geomdan opening while the 8-week level window
    # (lvl8) still starts before it -- they teach the one-off -0.3..-0.55 drop as a Jul-Oct station effect (see fix_test_yoy.py)
    b = pd.Timestamp('2025-06-28'); dd = pd.to_datetime(tr.date); cu = dd - pd.to_timedelta(tr.horizon, unit='D')
    brk = (tr.station_id.isin([3110, 3111]) & (dd >= b) & (cu - pd.Timedelta(days=56) < b)).to_numpy()
    tr = tr[~brk].reset_index(drop=True); del dd, cu
    print(f'dropped {int(brk.sum()):,} break rows of 3110/3111', flush=True)

# 5 event days missing from the v4 test table: same station surged at the same event in 2025 (peak hour from 2025)
EXTRA = [(3138, '2026-08-01', 23),                                       # 펜타포트 토 (2025-08-02: 3138 peak 23)
         (3138, '2026-08-22', 22), (3138, '2026-08-23', 22), (3138, '2026-08-29', 22),   # 송도맥주축제 (2025 Sat/Sun/Sat: 22)
         (3210, '2026-07-11', 22)]                                       # 흠뻑쇼 @3209 (2025-06-28/29: 3210 peak 22)
d = pd.to_datetime(te.date)
for s, day, pk in ([] if NOATT else EXTRA):
    m = te.station_id.eq(s) & d.eq(pd.Timestamp(day))
    assert m.sum() == 18 and te.loc[m, 'ev'].eq(0).all(), (s, day)
    te.loc[m, 'ev'] = 1; te.loc[m, 'ev_rel'] = te.loc[m, 'hour'] - pk
if not NOATT:
    np.save(OUT / 'final_att_ev.npy', te.ev.to_numpy())

att = pd.read_csv(ATT, parse_dates=['date']).groupby('date').attendance.max() / 1000
for df in (tr, te):
    df['att'] = pd.to_datetime(df.date).map(att).where(df.station_id.eq(3127)).to_numpy(np.float64)
feats = feats + ([] if NOATT else ['att'])
WX = os.environ.get('WX') == '1'
if WX:   # weather of the level / partner-year reference days (wxfeat.py)
    from wxfeat import add_wx
    feats = feats + add_wx(tr, '2026-06-30'); add_wx(te, '2026-06-30')
    print(f'wx added, non-null share train {tr.wx_pwin.notna().mean():.2f} test {te.wx_pwin.notna().mean():.2f}', flush=True)
g = te.loc[np.isfinite(te.att), 'date'].nunique()
assert g == 35 or NOATT, g
print(f'loaded {time.time()-t0:.0f}s, att rows train {np.isfinite(tr.att).sum():,} test {np.isfinite(te.att).sum():,} ({g} games)', flush=True)
YB = 'yoy_delta_all' if os.environ.get('BASE_ALL') == '1' else 'yoy_delta'   # BASE_ALL=1: residual base uses the all-station partner-year delta
sn_tr = (tr.lvl8 + tr[YB].fillna(0)).to_numpy(); sn_te = (te.lvl8 + te[YB].fillna(0)).to_numpy()
pred = {}
if 'A' in MODELS or 'E' in MODELS:
    import lightgbm as lgb
    P = dict(objective='regression', learning_rate=.05, num_leaves=31, min_data_in_leaf=100, lambda_l2=5., feature_fraction=.9,
             bagging_fraction=.8, bagging_freq=1, max_bin=127, seed=SEED, deterministic=True, force_col_wise=True,
             num_threads=8, verbosity=-1)
    if 'A' in MODELS:
        pred['A'] = lgb.train(P, lgb.Dataset(tr[feats], tr.y.to_numpy(), categorical_feature=CATS), int(os.environ.get('LGB_ROUNDS', '500'))).predict(te[feats])
    if 'E' in MODELS:
        pred['E'] = sn_te + lgb.train(P, lgb.Dataset(tr[feats], tr.y.to_numpy() - sn_tr, categorical_feature=CATS), int(os.environ.get('LGB_ROUNDS', '500'))).predict(te[feats])
if 'H' in MODELS:
    from sklearn.ensemble import HistGradientBoostingRegressor
    h = HistGradientBoostingRegressor(learning_rate=.05, max_iter=1500, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=5.,
                                      max_bins=127, early_stopping=False, categorical_features=[feats.index(c) for c in CATS], random_state=SEED)
    h.fit(tr[feats].to_numpy(np.float32), tr.y.to_numpy() - sn_tr)
    pred['H'] = sn_te + h.predict(te[feats].to_numpy(np.float32))
if 'C' in MODELS or 'D' in MODELS:   # D = CatBoost on the direct target y (no residual base)
    from catboost import CatBoostRegressor, Pool
    for c in CATS:
        tr[c] = tr[c].astype(int); te[c] = te[c].astype(int)
    Pc = dict(depth=int(os.environ.get('CAT_DEPTH', '6')), learning_rate=float(os.environ.get('CAT_LR', '.08')), loss_function='RMSE', l2_leaf_reg=5, random_seed=SEED, thread_count=8,
              bootstrap_type='Bernoulli', subsample=.8, rsm=.9, border_count=128, max_ctr_complexity=1, one_hot_max_size=10,
              allow_writing_files=False, verbose=0)
    if os.environ.get('CAT_GPU') == '1':   # CatBoost on the GPU: rsm is CPU-only for RMSE -> dropped
        Pc.pop('rsm'); Pc.pop('thread_count'); Pc.update(task_type='GPU', devices='0')
    if 'C' in MODELS:
        cm = CatBoostRegressor(iterations=int(os.environ.get('CAT_ITERS', '1500')), **Pc).fit(Pool(tr[feats], tr.y.to_numpy() - sn_tr, cat_features=CATS))
        pred['C'] = sn_te + cm.predict(Pool(te[feats], cat_features=CATS))
    if 'D' in MODELS:
        cd = CatBoostRegressor(iterations=int(os.environ.get('CAT_ITERS', '1500')), **Pc).fit(Pool(tr[feats], tr.y.to_numpy(), cat_features=CATS))
        pred['D'] = cd.predict(Pool(te[feats], cat_features=CATS))
for k, v in pred.items():
    v = np.maximum(v, 0); assert np.isfinite(v).all() and len(v) == 157194
    np.save(OUT / f'final_{"base" if NOATT else "att"}{"wx" if WX else ""}{"" if SEED == 42 else f"_s{SEED}"}{TAG}_{k}.npy', v)
    print(f'{k} saved {time.time()-t0:.0f}s mean {v.mean():.4f}', flush=True)
