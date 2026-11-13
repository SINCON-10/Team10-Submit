"""Leave-one-origin-out: does adding simple baselines (from the fold feature table) as blend columns help? Nothing from Public.
Columns: C20w layout (A,E,C,H mean over sets + V) + candidates built only from cutoff features:
  snT = lvl8 + yoy_delta, snA = lvl8 + yoy_delta_all, sn4 = lvl8 + lvl4 + yoy_delta_all, snD = lvl8 + lvl8_dow + yoy_delta_all
args: OUT SETS"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
OUT = Path(sys.argv[1]); SETS = sys.argv[2].split(',')
D = {}
for o in ['o0227', 'o0331', 'o0430']:
    d = OUT / o; p = pd.read_parquet(d / 'preds.parquet')
    f = pd.read_parquet(d / 'fold_future.parquet', columns=['date', 'station_id', 'hour', 'lvl8', 'lvl4', 'lvl8_dow', 'yoy_delta', 'yoy_delta_all'])
    assert (f.station_id.to_numpy() == p.station_id.to_numpy()).all() and (f.hour.to_numpy() == p.hour.to_numpy()).all()
    assert (pd.to_datetime(f.date).to_numpy() == pd.to_datetime(p.date).to_numpy()).all()
    m = ~p.ev.eq(1).to_numpy() & ~p.station_id.eq(3127).to_numpy()
    comp = [(p.A.to_numpy(), p.E42.to_numpy(), pd.read_parquet(d / 'preds_base_C.parquet').C.to_numpy(), p.H.to_numpy())]
    for s in SETS:
        a, c = pd.read_parquet(d / f'preds_{s}_AEH.parquet'), pd.read_parquet(d / f'preds_{s}_C.parquet')
        comp.append((a.A.to_numpy(), a.E.to_numpy(), c.C.to_numpy(), a.H.to_numpy()))
    yd, ya = f.yoy_delta.fillna(0).to_numpy(), f.yoy_delta_all.fillna(0).to_numpy()
    base = [np.mean([v[i] for v in comp], 0) for i in range(4)] + [p.V.to_numpy()]
    extra = {'snT': f.lvl8 + yd, 'snA': f.lvl8 + ya, 'sn4': f.lvl8 + f.lvl4.fillna(0) + ya, 'snD': f.lvl8 + f.lvl8_dow.fillna(0) + ya}
    D[o] = dict(y=p.y.to_numpy()[m], X=np.column_stack(base)[m], E={k: np.maximum(np.asarray(v, float), 0)[m] for k, v in extra.items()})
def fit(X, y):
    k = X.shape[1]; n = len(y); XtX = X.T @ X / n; Xty = X.T @ y / n
    r = minimize(lambda w: w @ XtX @ w - 2 * w @ Xty, np.full(k, 1 / k), jac=lambda w: 2 * XtX @ w - 2 * Xty, method='SLSQP', bounds=[(0, 1)] * k,
                 constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1, 'jac': lambda w: np.ones(k)}, options={'ftol': 1e-14, 'maxiter': 1000})
    assert r.success, r.message
    return r.x
for o in D:
    y = D[o]['y']; print(o, 'baseline rmse alone: ' + ' '.join(f'{k} {np.sqrt(((y - v) ** 2).mean()):.4f}' for k, v in D[o]['E'].items()))
for ex in [[], ['snT'], ['snA'], ['sn4'], ['snD'], ['snA', 'sn4', 'snD']]:
    out = []
    for t in D:
        tr = [o for o in D if o != t]
        X = lambda o: np.column_stack([D[o]['X']] + [D[o]['E'][k] for k in ex])
        w = fit(np.vstack([X(o) for o in tr]), np.concatenate([D[o]['y'] for o in tr]))
        e = np.sqrt(((D[t]['y'] - np.maximum(X(t) @ w, 0)) ** 2).mean()); out.append((e, w))
    if not ex:
        ref = [e for e, _ in out]
    print(f"{'+'.join(ex) or 'C20w layout':16s} " + ' '.join(f'{e:.5f} ({e - r0:+.5f})' for (e, _), r0 in zip(out, ref)) + '  w(last fold) ' + ' '.join(f'{x:.2f}' for x in out[-1][1]))
