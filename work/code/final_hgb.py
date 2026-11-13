"""Final HistGradientBoosting (1500 iters) on the seasonal residual target; saves test predictions (log1p scale) in test order."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

PV4, OUTNPY = Path(sys.argv[1]), Path(sys.argv[2]); t0 = time.time()
tr = pd.read_csv(PV4 / 'train_cutoff_features.csv.gz')
te = pd.read_csv(PV4 / 'test_cutoff_features.csv.gz')
feats = [c for c in te.columns if c not in ('date', 'station_id', 'y', 'example_id')]
assert len(feats) == 55 and len(tr) == 2391876 and len(te) == 157194
sn_tr = (tr.lvl8 + tr.yoy_delta.fillna(0)).to_numpy(); sn_te = (te.lvl8 + te.yoy_delta.fillna(0)).to_numpy()
m = HistGradientBoostingRegressor(learning_rate=.05, max_iter=1500, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=5.,
                                  max_bins=127, early_stopping=False, categorical_features=[feats.index(c) for c in ['sid', 'line', 'dow', 'dt3']],
                                  random_state=42)
m.fit(tr[feats].to_numpy(np.float32), tr.y.to_numpy() - sn_tr)
H = np.maximum(sn_te + m.predict(te[feats].to_numpy(np.float32)), 0)
assert np.isfinite(H).all()
np.save(OUTNPY, H)
print(f'hgb1500 final done {time.time()-t0:.0f}s mean {H.mean():.4f}')
