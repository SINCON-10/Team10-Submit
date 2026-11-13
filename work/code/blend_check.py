"""Fold check: C3-post with some direct-model (A) mixed into E on all rows; error split of C3-post by day type / hour."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(sys.argv[1])
d = pd.read_parquet(OUT / 'fold_lgb_preds.parquet')
y, E, A, V = d.y.to_numpy(), d.E_resid_seasonal.to_numpy(), d.A_direct.to_numpy(), d.v2.to_numpy()
ev = d.ev.eq(1).to_numpy(); s27 = d.station_id.eq(3127).to_numpy()
r = lambda p, m=slice(None): float(np.sqrt(((y[m] - p[m]) ** 2).mean()))


def post(E):
    B = np.where(ev, E, .8 * E + .2 * V)
    return np.where(s27, (A + E) / 2, B)


for w in (0, .1, .2, .3, .4):
    M = (1 - w) * E + w * A
    print(f'A weight {w:.1f}: C3-post {r(post(M)):.5f}')
for wv in (.1, .2, .3):
    B = np.where(ev, E, (1 - wv) * E + wv * V); B = np.where(s27, (A + E) / 2, B)
    print(f'v2 weight {wv:.1f}: {r(B):.5f}')
P = post(E)
off = d.offday.eq(1).to_numpy()
print(f'C3-post weekday {r(P, ~off):.4f} offday {r(P, off):.4f}  bias wk {np.mean((y-P)[~off]):+.4f} off {np.mean((y-P)[off]):+.4f}')
h = d.assign(e=(y - P) ** 2, b=y - P).groupby('hour').agg(rmse=('e', lambda s: np.sqrt(s.mean())), bias=('b', 'mean'))
print(h.round(3).T.to_string())
