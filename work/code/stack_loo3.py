"""Leave-one-origin-out of the C20w layout (A,E,C,H mean over sets + V) with optional extra columns. Nothing from Public.
args: OUT SETS [EXTRA ...]   EXTRA = file_stem:column, e.g. att_dcat_D:D  (reads ORIGIN/preds_<file_stem>.parquet)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
OUT = Path(sys.argv[1]); SETS = sys.argv[2].split(','); EXTRA = [e.split(':') for e in sys.argv[3:]]
D = {}
for o in ['o0227', 'o0331', 'o0430']:
    d = OUT / o; p = pd.read_parquet(d / 'preds.parquet'); m = ~p.ev.eq(1).to_numpy() & ~p.station_id.eq(3127).to_numpy()
    comp = [(p.A.to_numpy(), p.E42.to_numpy(), pd.read_parquet(d / 'preds_base_C.parquet').C.to_numpy(), p.H.to_numpy())]
    for s in SETS:
        a, c = pd.read_parquet(d / f'preds_{s}_AEH.parquet'), pd.read_parquet(d / f'preds_{s}_C.parquet')
        comp.append((a.A.to_numpy(), a.E.to_numpy(), c.C.to_numpy(), a.H.to_numpy()))
    cols = [np.mean([v[i] for v in comp], 0) for i in range(4)] + [p.V.to_numpy()]
    for stem, col in EXTRA:
        q = pd.read_parquet(d / f'preds_{stem}.parquet'); assert (q.station_id.to_numpy() == p.station_id.to_numpy()).all(); cols.append(q[col].to_numpy())
    D[o] = dict(y=p.y.to_numpy()[m], X=np.column_stack(cols)[m])
k = 5 + len(EXTRA); names = ['A', 'E', 'C', 'H', 'V'] + [c for _, c in EXTRA]
def fit(X, y, kk):
    n = len(y); XtX = X.T @ X / n; Xty = X.T @ y / n
    r = minimize(lambda w: w @ XtX @ w - 2 * w @ Xty, np.full(kk, 1 / kk), jac=lambda w: 2 * XtX @ w - 2 * Xty, method='SLSQP', bounds=[(0, 1)] * kk,
                 constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1, 'jac': lambda w: np.ones(kk)}, options={'ftol': 1e-14, 'maxiter': 1000})
    assert r.success, r.message
    return r.x
for t in D:
    tr = [o for o in D if o != t]; y = D[t]['y']; res = []
    for kk in (5, k):
        w = fit(np.vstack([D[o]['X'][:, :kk] for o in tr]), np.concatenate([D[o]['y'] for o in tr]), kk)
        res.append((np.sqrt(((y - np.maximum(D[t]['X'][:, :kk] @ w, 0)) ** 2).mean()), w))
    (e5, w5), (ek, wk) = res
    print(f'test {t}: C20w layout {e5:.5f} | +extra {ek:.5f} ({ek - e5:+.5f})  w ' + ' '.join(f'{n}{x:.2f}' for n, x in zip(names, wk)))
