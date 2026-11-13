"""Final candidates (origin 2026-06-30) from the delivered v4 cutoff files. No post-06-30 passenger data exists in any input.

C0 lgb_direct          : LightGBM on log1p target (team-like reference)
C1 lgb_seasonal_resid  : LightGBM on y - (lvl8 + yoy_delta)   [fold Mar-Jun 2026: 0.1761 -> 0.1687]
C2 C1 85% + v2 15%     : log-space blend with v2 structural   [fold: 0.1675]
"""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb

BUNDLE, DATA, RAW, DEST = map(Path, sys.argv[1:5])
DEST.mkdir(parents=True, exist_ok=True)
t0 = time.time()
tr = pd.read_csv(DATA / 'train_cutoff_features.csv.gz')
te = pd.read_csv(DATA / 'test_cutoff_features.csv.gz')
test = pd.read_csv(RAW / 'test.csv')
feats = [c for c in te.columns if c not in ('date', 'station_id', 'y', 'example_id')]
assert len(feats) == 55 and len(tr) == 2391876 and len(te) == 157194
assert te.example_id.tolist() == test.example_id.tolist()
print(f'loaded {time.time()-t0:.0f}s', flush=True)
CATS = ['sid', 'line', 'dow', 'dt3']
P = dict(objective='regression', learning_rate=.05, num_leaves=31, min_data_in_leaf=100, lambda_l2=5., feature_fraction=.9,
         bagging_fraction=.8, bagging_freq=1, max_bin=127, seed=42, deterministic=True, force_col_wise=True,
         num_threads=8, verbosity=-1)


def fit(target, name):
    m = lgb.train(P, lgb.Dataset(tr[feats], target, categorical_feature=CATS), num_boost_round=500)
    m.save_model(str(DEST / f'{name}.txt'))
    print(f'{name} trained {time.time()-t0:.0f}s', flush=True)
    return m.predict(te[feats])


pred = {}
pred['C0_lgb_direct'] = np.maximum(fit(tr.y.to_numpy(), 'model_C0_lgb_direct'), 0)
sn_tr = (tr.lvl8 + tr.yoy_delta.fillna(0)).to_numpy(); sn_te = (te.lvl8 + te.yoy_delta.fillna(0)).to_numpy()
pred['C1_lgb_seasonal_resid'] = np.maximum(sn_te + fit(tr.y.to_numpy() - sn_tr, 'model_C1_lgb_seasonal_resid'), 0)

# v2 structural for the final origin, same construction as in the validation fold
sys.path.insert(0, str(BUNDLE))
from data_v4 import DataEngine  # noqa: E402
eng = DataEngine(DATA, BUNDLE)
fold = eng.build('final', '2026-06-30', {'smoke': False}, with_train=False)
v2 = pd.Series(np.maximum(eng.structural_values(fold, fold.future), 0), index=fold.future.example_id.to_numpy())
pred['v2_structure'] = v2.loc[te.example_id].to_numpy()
ev = te.ev.eq(1).to_numpy(); s27 = te.station_id.eq(3127).to_numpy()
E, A, V = pred['C1_lgb_seasonal_resid'], pred['C0_lgb_direct'], pred['v2_structure']
# v2 has no event component: blend it only on non-event rows so scheduled event peaks are kept  [fold 0.16748]
pred['C2_resid_v2_20_nonevent'] = np.where(ev, E, .8 * E + .2 * V)
# 3127 (Munhak): seasonal base double-counts baseball excess in the fold; hedge with the direct model  [fold 0.16400]
pred['C3_C2_3127_avg_direct'] = np.where(s27, (A + E) / 2, pred['C2_resid_v2_20_nonevent'])
print(f'v2 done {time.time()-t0:.0f}s', flush=True)

summary = {}
for name in ['C0_lgb_direct', 'C1_lgb_seasonal_resid', 'C2_resid_v2_20_nonevent', 'C3_C2_3127_avg_direct']:
    p = np.maximum(pred[name], 0)
    sub = pd.DataFrame({'example_id': te.example_id, 'boardings': np.expm1(p)})
    assert len(sub) == 157194 and sub.example_id.is_unique and sub.example_id.tolist() == test.example_id.tolist()
    assert np.isfinite(sub.boardings).all() and sub.boardings.ge(0).all()
    path = DEST / f'submission_{name}.csv'
    sub.to_csv(path, index=False, float_format='%.6f', lineterminator='\n')
    summary[name] = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), mean_log=float(p.mean()))
ref = pd.read_csv(RAW.parent.parent / 'codex_works/results/team_final_20261112_140730/submission.csv').set_index('example_id').loc[te.example_id]
rl = np.log1p(ref.boardings.to_numpy())
for name in summary:
    p = np.maximum(pred[name], 0)
    summary[name].update(vs_team_final_mean_diff=float((p - rl).mean()), vs_team_final_rmse=float(np.sqrt(((p - rl) ** 2).mean())))
(DEST / 'candidates.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding='utf-8')
print(json.dumps(summary, indent=1, ensure_ascii=False))
