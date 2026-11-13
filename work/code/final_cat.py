"""C7 = Chuseok-adjusted C4 recipe with Emix = (1-w)*C1(LGB residual) + w*CatBoost residual.

CatBoost: team CatBoost settings, target y - (lvl8 + yoy_delta), trained on the delivered v4 train_cutoff_features (<= 2026-06-30).
A (C0), E (C1) and v2 are read back from the written C0/C1/C2 submissions (v2 recovered on non-event rows).
args: PV4 RAW CAND ITERS W_CAT
"""
import hashlib, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostRegressor, Pool

PV4, RAW, CAND = map(Path, sys.argv[1:4]); ITERS, W = int(sys.argv[4]), float(sys.argv[5])
sys.path.insert(0, str(PV4))
from lunar_adjust import lunar_adjust  # noqa: E402

t0 = time.time()
tr = pd.read_csv(PV4 / 'train_cutoff_features.csv.gz')
te = pd.read_csv(PV4 / 'test_cutoff_features.csv.gz')
test = pd.read_csv(RAW / 'test.csv', parse_dates=['date'])
train = pd.read_csv(RAW / 'train.csv', usecols=['date', 'station_id', 'hour', 'boardings'], parse_dates=['date'])
feats = [c for c in te.columns if c not in ('date', 'station_id', 'y', 'example_id')]
assert len(feats) == 55 and len(tr) == 2391876 and len(te) == 157194 and train.date.max() == pd.Timestamp('2026-06-30')
assert te.example_id.tolist() == test.example_id.tolist()
CATS = ['sid', 'line', 'dow', 'dt3']
for c in CATS:
    tr[c] = tr[c].astype(int); te[c] = te[c].astype(int)
P = dict(depth=6, learning_rate=.08, loss_function='RMSE', l2_leaf_reg=5, random_seed=42, thread_count=8,
         bootstrap_type='Bernoulli', subsample=.8, rsm=.9, border_count=128, max_ctr_complexity=1, one_hot_max_size=10,
         allow_writing_files=False, verbose=0)
sn_tr = (tr.lvl8 + tr.yoy_delta.fillna(0)).to_numpy(); sn_te = (te.lvl8 + te.yoy_delta.fillna(0)).to_numpy()
m = CatBoostRegressor(iterations=ITERS, **P).fit(Pool(tr[feats], tr.y.to_numpy() - sn_tr, cat_features=CATS))
m.save_model(str(CAND / f'model_C7_cat{ITERS}_seasonal_resid.cbm'))
Ec = np.maximum(sn_te + m.predict(Pool(te[feats], cat_features=CATS)), 0)
print(f'catboost trained+predicted {time.time()-t0:.0f}s', flush=True)


def load(name):
    s = pd.read_csv(CAND / f'submission_{name}.csv')
    assert s.example_id.tolist() == test.example_id.tolist()
    return np.log1p(s.boardings.to_numpy())


A, E, C2 = (load(n) for n in ['C0_lgb_direct', 'C1_lgb_seasonal_resid', 'C2_resid_v2_20_nonevent'])
V = (C2 - .8 * E) / .2
ev = te.ev.eq(1).to_numpy(); s27 = te.station_id.eq(3127).to_numpy()
Emix = (1 - W) * E + W * Ec
M = .8 * Emix + .2 * A
B = np.where(ev, M, .8 * M + .2 * V)
B = np.where(s27, (A + Emix) / 2, B)
B = np.maximum(B, 0)
out = lunar_adjust(train, test[['date', 'station_id', 'hour']].assign(p=B))
assert (out[['date', 'station_id', 'hour']].to_numpy() == test[['date', 'station_id', 'hour']].to_numpy()).all()
p = np.maximum(out.p.to_numpy(), 0)
sub = pd.DataFrame({'example_id': test.example_id, 'boardings': np.expm1(p)})
assert len(sub) == 157194 and sub.example_id.is_unique and np.isfinite(sub.boardings).all() and sub.boardings.ge(0).all()
path = CAND / f'submission_C7_C5_with_cat{int(W*100)}.csv'
sub.to_csv(path, index=False, float_format='%.6f', lineterminator='\n')
C5 = load('C5_C4_chuseok')
print(path.name, hashlib.sha256(path.read_bytes()).hexdigest()[:8], f'rmse vs C5 {np.sqrt(((p - C5) ** 2).mean()):.4f}',
      f'mean diff vs C5 {(p - C5).mean():+.4f}', f'cat vs lgb resid rmse {np.sqrt(((Ec - E) ** 2).mean()):.4f}', f'{time.time()-t0:.0f}s')
