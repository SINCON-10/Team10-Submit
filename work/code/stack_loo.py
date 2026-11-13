"""Leave-one-origin-out check of fold-fitted blend weights (nothing from Public).
Components (log1p): per model type averaged over the sets (A, E, C, H) + V (old set). Current C19 recipe = q0.
Weights (>=0, sum 1) fit on two origins, scored on the third. Only non-event, non-3127 rows are re-blended; others keep q0.
args: OUT ATT_CSV SETS"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
OUT = Path(sys.argv[1]); games = pd.read_csv(sys.argv[2], parse_dates=['date']).date; SETS = sys.argv[3].split(',')
D = {}
for o in ['o0227', 'o0331', 'o0430']:
    d = OUT / o; p = pd.read_parquet(d / 'preds.parquet')
    y, ev, V = p.y.to_numpy(), p.ev.eq(1).to_numpy(), p.V.to_numpy()
    s27 = p.station_id.eq(3127).to_numpy(); game = s27 & pd.to_datetime(p.date).isin(games).to_numpy(); keep_old = s27 & ev & ~game
    def c9(A, E, C, H):
        Em = (E + C + H) / 3; M = .8 * Em + .2 * A; B = np.where(ev, M, .8 * M + .2 * V)
        return np.where(s27, np.where(~ev, A, (A + Em) / 2), B)
    comp = {'old': (p.A.to_numpy(), p.E42.to_numpy(), pd.read_parquet(d / 'preds_base_C.parquet').C.to_numpy(), p.H.to_numpy())}
    for s in SETS:
        a, c = pd.read_parquet(d / f'preds_{s}_AEH.parquet'), pd.read_parquet(d / f'preds_{s}_C.parquet')
        comp[s] = (a.A.to_numpy(), a.E.to_numpy(), c.C.to_numpy(), a.H.to_numpy())
    S = [c9(*v) for k, v in comp.items() if k != 'old']; old = c9(*comp['old'])
    q0 = np.maximum(np.where(keep_old, old, np.where(game, np.mean(S, 0), (old + np.sum(S, 0)) / (1 + len(S)))), 0)
    X = np.column_stack([np.mean([v[i] for v in comp.values()], 0) for i in range(4)] + [V])   # A E C H V
    m = ~ev & ~s27
    D[o] = dict(y=y, q0=q0, X=X, m=m, off=p.offday.eq(1).to_numpy(), hour=p.hour.to_numpy())
NAMES = ['A', 'E', 'C', 'H', 'V']; W0 = np.array([.16, .64 / 3, .64 / 3, .64 / 3, .2])
def fit(X, y):
    n = len(y); XtX = X.T @ X / n; Xty = X.T @ y / n
    f = lambda w: w @ XtX @ w - 2 * w @ Xty; g = lambda w: 2 * XtX @ w - 2 * Xty
    r = minimize(f, W0, jac=g, method='SLSQP', bounds=[(0, 1)] * 5, constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1, 'jac': lambda w: np.ones(5)},
                 options={'ftol': 1e-14, 'maxiter': 500})
    assert r.success, r.message
    return r.x
def groups(dd, how):
    if how == 'one': return np.zeros(len(dd['y']), int)
    if how == 'off': return dd['off'].astype(int)
    if how == 'off x band': return dd['off'].astype(int) * 3 + np.select([dd['hour'] < 10, dd['hour'] < 20], [0, 1], 2)
for how in ['one', 'off', 'off x band']:
    print(f'\n== weights by {how}')
    for test in D:
        tr = [o for o in D if o != test]
        gt = groups(D[test], how); q = D[test]['q0'].copy(); ws = []
        for g in np.unique(gt):
            Xs = np.vstack([D[o]['X'][D[o]['m'] & (groups(D[o], how) == g)] for o in tr]); ys = np.concatenate([D[o]['y'][D[o]['m'] & (groups(D[o], how) == g)] for o in tr])
            w = fit(Xs, ys); ws.append(w)
            r = D[test]['m'] & (gt == g); q[r] = np.maximum(D[test]['X'][r] @ w, 0)
        y = D[test]['y']; e0 = np.sqrt(((y - D[test]['q0']) ** 2).mean()); e1 = np.sqrt(((y - q) ** 2).mean())
        wtxt = ' | '.join(' '.join(f'{n}{x:.2f}' for n, x in zip(NAMES, w)) for w in ws)
        print(f'  test {test}: C19 recipe {e0:.5f} -> fitted {e1:.5f} ({e1 - e0:+.5f})   w: {wtxt}')
# fixed-weight sanity: the c9 weights applied to set-averaged components should be close to q0 on non-event rows
o = 'o0430'; r = D[o]['m']
print('\nsanity (o0430, non-event rows): rmse q0 vs X@W0 diff', float(np.sqrt(((D[o]['q0'][r] - np.maximum(D[o]['X'][r] @ W0, 0)) ** 2).mean())))
