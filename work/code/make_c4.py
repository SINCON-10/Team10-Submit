"""C4 = C3 recipe applied to (0.8*C1 + 0.2*C0), built from the already-written C0/C1/C2 submissions (no retraining).

v2 structural is recovered from C2 on non-event rows (C2 = .8*C1 + .2*v2 there); event rows do not use v2.
3127 stays (C0+C1)/2 exactly as in C3. Fold (Mar-Jun 2026): C3 0.16400 -> C4 0.16349.
"""
import hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd

DATA, RAW, CAND = map(Path, sys.argv[1:4])
te = pd.read_csv(DATA / 'test_cutoff_features.csv.gz', usecols=['example_id', 'station_id', 'ev'])
test = pd.read_csv(RAW / 'test.csv', usecols=['example_id'])
assert te.example_id.tolist() == test.example_id.tolist()


def load(name):
    s = pd.read_csv(CAND / f'submission_{name}.csv')
    assert s.example_id.tolist() == test.example_id.tolist()
    return np.log1p(s.boardings.to_numpy())


A, E, C2, C3 = (load(n) for n in ['C0_lgb_direct', 'C1_lgb_seasonal_resid', 'C2_resid_v2_20_nonevent', 'C3_C2_3127_avg_direct'])
ev = te.ev.eq(1).to_numpy(); s27 = te.station_id.eq(3127).to_numpy()
V = (C2 - .8 * E) / .2
# self-check: rebuilding C3 from the recovered pieces must match the written C3 file
B3 = np.where(ev, E, .8 * E + .2 * V); B3 = np.where(s27, (A + E) / 2, B3)
print('C3 rebuild max abs diff (log):', float(np.abs(B3 - C3).max()))
assert np.abs(B3 - C3).max() < 1e-5

M = .8 * E + .2 * A
P = np.where(ev, M, .8 * M + .2 * V)
P = np.where(s27, (A + E) / 2, P)          # 3127 kept as in C3
P = np.maximum(P, 0)
sub = pd.DataFrame({'example_id': te.example_id, 'boardings': np.expm1(P)})
assert len(sub) == 157194 and sub.example_id.is_unique and np.isfinite(sub.boardings).all() and sub.boardings.ge(0).all()
path = CAND / 'submission_C4_C3_with_direct20.csv'
sub.to_csv(path, index=False, float_format='%.6f', lineterminator='\n')
print(path.name, hashlib.sha256(path.read_bytes()).hexdigest()[:8], 'mean log', round(float(P.mean()), 5),
      'mean diff vs C3', round(float((P - C3).mean()), 5), 'rmse vs C3', round(float(np.sqrt(((P - C3) ** 2).mean())), 5))
