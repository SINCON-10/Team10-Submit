"""C8 = Chuseok-adjusted C4 recipe with Emix = (LGB resid + CatBoost1500 resid + HistGB1500 resid) / 3   [fold 0.16168]
args: PV4 RAW CAND HGB_NPY
"""
import hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostRegressor, Pool

PV4, RAW, CAND, HNPY = map(Path, sys.argv[1:5])
sys.path.insert(0, str(PV4))
from lunar_adjust import lunar_adjust  # noqa: E402

te = pd.read_csv(PV4 / 'test_cutoff_features.csv.gz')
test = pd.read_csv(RAW / 'test.csv', parse_dates=['date'])
train = pd.read_csv(RAW / 'train.csv', usecols=['date', 'station_id', 'hour', 'boardings'], parse_dates=['date'])
assert te.example_id.tolist() == test.example_id.tolist() and train.date.max() == pd.Timestamp('2026-06-30')
feats = [c for c in te.columns if c not in ('date', 'station_id', 'y', 'example_id')]
CATS = ['sid', 'line', 'dow', 'dt3']
for c in CATS:
    te[c] = te[c].astype(int)
sn_te = (te.lvl8 + te.yoy_delta.fillna(0)).to_numpy()
cm = CatBoostRegressor(); cm.load_model(str(CAND / 'model_C7_cat1500_seasonal_resid.cbm'))
C = np.maximum(sn_te + cm.predict(Pool(te[feats], cat_features=CATS)), 0)
H = np.load(HNPY); assert H.shape == (157194,)


def load(name):
    s = pd.read_csv(CAND / f'submission_{name}.csv')
    assert s.example_id.tolist() == test.example_id.tolist()
    return np.log1p(s.boardings.to_numpy())


A, E, C2 = (load(n) for n in ['C0_lgb_direct', 'C1_lgb_seasonal_resid', 'C2_resid_v2_20_nonevent'])
V = (C2 - .8 * E) / .2
ev = te.ev.eq(1).to_numpy(); s27 = te.station_id.eq(3127).to_numpy()
Emix = (E + C + H) / 3
M = .8 * Emix + .2 * A
B = np.where(ev, M, .8 * M + .2 * V)
RULE27 = len(sys.argv) > 5 and sys.argv[5] == 'rule27'
# C9: 3127 rows without a scheduled event use the direct model only (better in all 3 validation origins and every month)
B27 = np.where(~ev, A, (A + Emix) / 2) if RULE27 else (A + Emix) / 2
B = np.maximum(np.where(s27, B27, B), 0)
out = lunar_adjust(train, test[['date', 'station_id', 'hour']].assign(p=B))
assert (out[['date', 'station_id', 'hour']].to_numpy() == test[['date', 'station_id', 'hour']].to_numpy()).all()
p = np.maximum(out.p.to_numpy(), 0)
sub = pd.DataFrame({'example_id': test.example_id, 'boardings': np.expm1(p)})
assert len(sub) == 157194 and sub.example_id.is_unique and np.isfinite(sub.boardings).all() and sub.boardings.ge(0).all()
path = CAND / ('submission_C9_C8_3127rule.csv' if RULE27 else 'submission_C8_C5_lgb_cat_hgb.csv')
sub.to_csv(path, index=False, float_format='%.6f', lineterminator='\n')
C5, C7 = load('C5_C4_chuseok'), load('C7_C5_with_cat50')
print(path.name, hashlib.sha256(path.read_bytes()).hexdigest()[:8], f'rmse vs C5 {np.sqrt(((p - C5) ** 2).mean()):.4f}',
      f'vs C7 {np.sqrt(((p - C7) ** 2).mean()):.4f}', f'mean diff vs C5 {(p - C5).mean():+.4f}',
      f'| resid models rmse: E-C {np.sqrt(((E - C) ** 2).mean()):.4f} E-H {np.sqrt(((E - H) ** 2).mean()):.4f}')
