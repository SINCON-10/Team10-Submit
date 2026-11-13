"""Leave-one-origin-out: small LightGBM on top of the fold-fitted linear blend (ordinary rows only). Nothing from Public.
target = y - linear blend; features = component preds relative to the blend, model spread, hour, offday, dow, horizon, blend level.
args: OUT SETS"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.optimize import minimize
OUT = Path(sys.argv[1]); SETS = sys.argv[2].split(',')
D = {}
for o in ['o0227', 'o0331', 'o0430']:
    d = OUT / o; p = pd.read_parquet(d / 'preds.parquet')
    m = ~p.ev.eq(1).to_numpy() & ~p.station_id.eq(3127).to_numpy()
    comp = [(p.A.to_numpy(), p.E42.to_numpy(), pd.read_parquet(d / 'preds_base_C.parquet').C.to_numpy(), p.H.to_numpy())]
    for s in SETS:
        a, c = pd.read_parquet(d / f'preds_{s}_AEH.parquet'), pd.read_parquet(d / f'preds_{s}_C.parquet')
        comp.append((a.A.to_numpy(), a.E.to_numpy(), c.C.to_numpy(), a.H.to_numpy()))
    X = np.column_stack([np.mean([v[i] for v in comp], 0) for i in range(4)] + [p.V.to_numpy()])
    sd = np.std(np.column_stack([v[i] for v in comp for i in range(4)]), axis=1)
    D[o] = dict(y=p.y.to_numpy()[m], X=X[m], sd=sd[m], hour=p.hour.to_numpy()[m], off=p.offday.to_numpy()[m],
                dow=pd.to_datetime(p.date).dt.dayofweek.to_numpy()[m], hz=p.horizon.to_numpy()[m])
def fitw(X, y):
    n = len(y); XtX = X.T @ X / n; Xty = X.T @ y / n; w0 = np.full(5, .2)
    r = minimize(lambda w: w @ XtX @ w - 2 * w @ Xty, w0, jac=lambda w: 2 * XtX @ w - 2 * Xty, method='SLSQP', bounds=[(0, 1)] * 5,
                 constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1, 'jac': lambda w: np.ones(5)}, options={'ftol': 1e-14})
    return r.x
def feats(dd, w):
    b = dd['X'] @ w
    return b, np.column_stack([dd['X'] - b[:, None], dd['sd'], dd['hour'], dd['off'], dd['dow'], dd['hz'], b])
for t in D:
    tr = [o for o in D if o != t]
    w = fitw(np.vstack([D[o]['X'] for o in tr]), np.concatenate([D[o]['y'] for o in tr]))
    Fs, ys = [], []
    for o in tr:
        b, F = feats(D[o], w); Fs.append(F); ys.append(D[o]['y'] - b)
    bt, Ft = feats(D[t], w); y = D[t]['y']
    e_lin = np.sqrt(((y - np.maximum(bt, 0)) ** 2).mean())
    out = []
    for nt in (100, 300):
        g = lgb.LGBMRegressor(n_estimators=nt, learning_rate=.03, num_leaves=15, min_child_samples=2000, subsample=.7, subsample_freq=1,
                              colsample_bytree=.8, reg_lambda=10, n_jobs=4, verbose=-1).fit(np.vstack(Fs), np.concatenate(ys))
        e = np.sqrt(((y - np.maximum(bt + g.predict(Ft), 0)) ** 2).mean()); out.append(f'{nt} trees {e:.5f} ({e - e_lin:+.5f})')
    print(f'test {t}: linear fitted {e_lin:.5f} | gbm on top: ' + ' | '.join(out))
