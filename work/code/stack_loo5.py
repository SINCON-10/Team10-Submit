"""Leave-one-origin-out: fold-fitted model-type weights for the rows C20w leaves on fixed rules. Nothing from Public.
groups: ev (event rows, not 3127) | s27game (3127 SSG game days) | s27none (3127, no event) | s27other (3127 other events)
fixed rule per group = what C19/C20w do there (C9 blend + set routing). args: OUT ATT_CSV SETS"""
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
    newm = lambda i: np.mean([comp[s][i] for s in SETS], 0)          # new sets only (game rows use new sets)
    allm = lambda i: np.mean([v[i] for v in comp.values()], 0)
    X_all = np.column_stack([allm(i) for i in range(4)] + [V]); X_new = np.column_stack([newm(i) for i in range(4)] + [V])
    X_old = np.column_stack(list(comp['old']) + [V])
    G = {'ev': (ev & ~s27, X_all), 's27game': (game, X_new), 's27none': (s27 & ~ev, X_all), 's27other': (keep_old, X_old)}
    D[o] = {g: dict(y=y[m], q0=q0[m], X=X[m]) for g, (m, X) in G.items()}
    D[o]['_sse_all'] = ((y - q0) ** 2).sum(); D[o]['_n'] = len(y)
def fit(X, y):
    k = X.shape[1]; n = len(y); XtX = X.T @ X / n; Xty = X.T @ y / n
    r = minimize(lambda w: w @ XtX @ w - 2 * w @ Xty, np.full(k, 1 / k), jac=lambda w: 2 * XtX @ w - 2 * Xty, method='SLSQP', bounds=[(0, 1)] * k,
                 constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1, 'jac': lambda w: np.ones(k)}, options={'ftol': 1e-14, 'maxiter': 1000})
    assert r.success, r.message
    return r.x
for g in ['ev', 's27game', 's27none', 's27other']:
    print(f'== {g}  (rows per origin: ' + ', '.join(str(len(D[o][g]['y'])) for o in D) + ')')
    for t in D:
        tr = [o for o in D if o != t]
        w = fit(np.vstack([D[o][g]['X'] for o in tr]), np.concatenate([D[o][g]['y'] for o in tr]))
        dd = D[t][g]; e0 = ((dd['y'] - dd['q0']) ** 2).sum(); e1 = ((dd['y'] - np.maximum(dd['X'] @ w, 0)) ** 2).sum()
        tot0 = np.sqrt(D[t]['_sse_all'] / D[t]['_n']); tot1 = np.sqrt((D[t]['_sse_all'] - e0 + e1) / D[t]['_n'])
        print(f'  test {t}: group rmse {np.sqrt(e0/len(dd["y"])):.4f} -> {np.sqrt(e1/len(dd["y"])):.4f} | all-row rmse change {tot1 - tot0:+.5f}  w ' + ' '.join(f'{x:.2f}' for x in w))
