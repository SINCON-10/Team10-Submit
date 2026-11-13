"""Leave-one-origin-out: richer weight layouts on ordinary rows (not event, not 3127). Nothing from Public.
layouts: one = A,E,C,H (mean over all sets) + V ; base = A,E,C,H separately for team-base sets and all-station-base sets + V ;
         perset = each set's C9 blend as one column (+ nothing else)
args: OUT SETS_TEAMBASE SETS_ALLBASE"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
OUT = Path(sys.argv[1]); TB = sys.argv[2].split(','); AB = sys.argv[3].split(',')
D = {}
for o in ['o0227', 'o0331', 'o0430']:
    d = OUT / o; p = pd.read_parquet(d / 'preds.parquet')
    y, ev, V = p.y.to_numpy(), p.ev.eq(1).to_numpy(), p.V.to_numpy(); m = ~ev & ~p.station_id.eq(3127).to_numpy()
    comp = {'old': (p.A.to_numpy(), p.E42.to_numpy(), pd.read_parquet(d / 'preds_base_C.parquet').C.to_numpy(), p.H.to_numpy())}
    for s in TB + AB:
        a, c = pd.read_parquet(d / f'preds_{s}_AEH.parquet'), pd.read_parquet(d / f'preds_{s}_C.parquet')
        comp[s] = (a.A.to_numpy(), a.E.to_numpy(), c.C.to_numpy(), a.H.to_numpy())
    tb = ['old'] + TB
    c9 = lambda A, E, C, H: .8 * (.8 * (E + C + H) / 3 + .2 * A) + .2 * V
    L = {'one': np.column_stack([np.mean([v[i] for v in comp.values()], 0) for i in range(4)] + [V]),
         'base': np.column_stack([np.mean([comp[s][i] for s in tb], 0) for i in range(4)] + [np.mean([comp[s][i] for s in AB], 0) for i in range(4)] + [V]),
         'perset': np.column_stack([c9(*v) for v in comp.values()])}
    D[o] = dict(y=y[m], q0=np.mean([c9(*v) for v in comp.values()], 0)[m], L={k: v[m] for k, v in L.items()})
def fit(X, y):
    k = X.shape[1]; n = len(y); XtX = X.T @ X / n; Xty = X.T @ y / n; w0 = np.full(k, 1 / k)
    r = minimize(lambda w: w @ XtX @ w - 2 * w @ Xty, w0, jac=lambda w: 2 * XtX @ w - 2 * Xty, method='SLSQP', bounds=[(0, 1)] * k,
                 constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1, 'jac': lambda w: np.ones(k)}, options={'ftol': 1e-14, 'maxiter': 1000})
    assert r.success, r.message
    return r.x
for lay in ['one', 'base', 'perset']:
    print(f'== {lay}')
    for t in D:
        tr = [o for o in D if o != t]
        w = fit(np.vstack([D[o]['L'][lay] for o in tr]), np.concatenate([D[o]['y'] for o in tr]))
        y = D[t]['y']; e0 = np.sqrt(((y - D[t]['q0']) ** 2).mean()); e1 = np.sqrt(((y - np.maximum(D[t]['L'][lay] @ w, 0)) ** 2).mean())
        print(f'  test {t}: fixed {e0:.5f} -> fitted {e1:.5f} ({e1 - e0:+.5f})  w ' + ' '.join(f'{x:.2f}' for x in w))
