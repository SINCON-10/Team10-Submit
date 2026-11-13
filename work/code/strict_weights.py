"""Time-separated check of fold-fitted blend weights (answers the Codex review, 2026-11-13).
Weights are fitted only on labels dated BEFORE the evaluation period starts; duplicate (date, station, hour) labels keep the
prediction from the most recent cutoff. Evaluation = all rows of that origin (ordinary rows re-blended, event/3127 rows fixed).
Optional EXTRA = file_stem:column added as a 6th column. Nothing from Public. args: OUT ATT_CSV SETS [EXTRA]"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
OUT = Path(sys.argv[1]); games = pd.read_csv(sys.argv[2], parse_dates=['date']).date; SETS = sys.argv[3].split(',')
EXTRA = [e.split(':') for e in sys.argv[4:]]
START = {'o0227': '2026-02-28', 'o0331': '2026-04-01', 'o0430': '2026-05-01'}
D = {}
for o in START:
    d = OUT / o; p = pd.read_parquet(d / 'preds.parquet')
    y, ev, V = p.y.to_numpy(), p.ev.eq(1).to_numpy(), p.V.to_numpy()
    s27 = p.station_id.eq(3127).to_numpy(); game = s27 & pd.to_datetime(p.date).isin(games).to_numpy(); keep_old = s27 & ev & ~game
    def c9(A, E, C, H):
        Em = (E + C + H) / 3; M = .8 * Em + .2 * A; B = np.where(ev, M, .8 * M + .2 * V)
        return np.where(s27, np.where(~ev, A, (A + Em) / 2), B)
    comp = {'old': (p.A.to_numpy(), p.E42.to_numpy(), pd.read_parquet(d / 'preds_base_C.parquet').C.to_numpy(), p.H.to_numpy())}
    for s in SETS:
        a, c = pd.read_parquet(d / f'preds_{s}_AEH.parquet'), pd.read_parquet(d / f'preds_{s}_C.parquet')
        e = a.E.to_numpy(); f3 = d / f'preds_{s}_e3k_E.parquet'
        if s in [x for x in os.environ.get('E3KSETS', '').split(',') if x]: e = pd.read_parquet(f3).E.to_numpy()   # 3000-tree residual LightGBM
        aa = a.A.to_numpy()
        if s in [x for x in os.environ.get('A4KSETS', '').split(',') if x]:   # 4000-tree direct LightGBM (shared by att/att_ball, att_smc/att_ball_smc)
            src = {'att_ball': 'att', 'att_ball_smc': 'att_smc'}.get(s, s)
            aa = (pd.read_parquet(d / 'preds_att_acurve_A.parquet').A_4000 if (o == 'o0227' and src == 'att') else pd.read_parquet(d / f'preds_{src}_a4k_A.parquet').A).to_numpy()
        comp[s] = (aa, e, c.C.to_numpy(), a.H.to_numpy())
    S = [c9(*v) for k, v in comp.items() if k != 'old']; old = c9(*comp['old'])
    q0 = np.maximum(np.where(keep_old, old, np.where(game, np.mean(S, 0), (old + np.sum(S, 0)) / (1 + len(S)))), 0)
    cols = [np.mean([v[i] for v in comp.values()], 0) for i in range(4)] + [V]
    for stem, col in EXTRA:
        cols.append(pd.read_parquet(d / f'preds_{stem}.parquet')[col].to_numpy())
    D[o] = dict(key=p.date.astype(str) + '|' + p.station_id.astype(str) + '|' + p.hour.astype(str), date=pd.to_datetime(p.date),
                y=y, q0=q0, X=np.column_stack(cols), m=~ev & ~s27)
# overlap of label keys between origins
for t in START:
    others = set().union(*[set(D[o]['key']) for o in START if o != t]); k = D[t]['key']
    print(f'{t}: label keys also in the other origins {k.isin(others).mean():.2%}')
def fit(X, y):
    k = X.shape[1]; n = len(y); XtX = X.T @ X / n; Xty = X.T @ y / n
    r = minimize(lambda w: w @ XtX @ w - 2 * w @ Xty, np.full(k, 1 / k), jac=lambda w: 2 * XtX @ w - 2 * Xty, method='SLSQP', bounds=[(0, 1)] * k,
                 constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1, 'jac': lambda w: np.ones(k)}, options={'ftol': 1e-14, 'maxiter': 1000})
    assert r.success, r.message
    return r.x
order = list(START)                     # o0227 (oldest cutoff) ... o0430 (newest)
for t, fe in [('o0331', '2026-04-01'), ('o0430', '2026-04-01'), ('o0430', '2026-05-01')]:   # (eval origin, fit labels end before)
    st = pd.Timestamp(fe); parts = []
    for o in order:                      # rows dated before the eval start, from every origin
        r = D[o]['m'] & (D[o]['date'] < st).to_numpy()
        parts.append(pd.DataFrame({'key': D[o]['key'][r].to_numpy(), 'rank': order.index(o), 'i': np.flatnonzero(r), 'o': o}))
    f = pd.concat(parts).sort_values('rank').drop_duplicates('key', keep='last')     # most recent cutoff wins
    X = np.vstack([D[o]['X'][f.i[f.o == o].to_numpy()] for o in order]); y = np.concatenate([D[o]['y'][f.i[f.o == o].to_numpy()] for o in order])
    days = pd.to_datetime(f.key.str.split('|').str[0]); w = fit(X, y)
    q = D[t]['q0'].copy(); m = D[t]['m']; q[m] = np.maximum(D[t]['X'][m] @ w, 0)
    e0 = np.sqrt(((D[t]['y'] - D[t]['q0']) ** 2).mean()); e1 = np.sqrt(((D[t]['y'] - q) ** 2).mean())
    print(f'eval {t} ({START[t]}~06-30, {D[t]["date"].dt.date.nunique()}d): weights from labels {days.min().date()}~{days.max().date()} ({days.dt.date.nunique()}d, {len(y):,} rows) '
          f'| fixed C19 recipe {e0:.6f} -> fitted {e1:.6f} ({e1 - e0:+.6f}) | w ' + ' '.join(f'{x:.3f}' for x in w))
