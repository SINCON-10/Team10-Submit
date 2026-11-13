"""E500 vs E3000 (LGB residual) per set: single-model RMSLE on the early window 2/28-3/31 of o0227 (selection window,
labels no other origin uses), then on o0331 4/1-6/30 and o0430 5/1-6/30 (later, auxiliary), and o0227 full 123 days.
Also the fixed C19 recipe with E replaced by E3k in the sets that have it (no fitted weights). args: OUT ATT_CSV SETS"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(sys.argv[1]); games = pd.read_csv(sys.argv[2], parse_dates=['date']).date; SETS = sys.argv[3].split(',')
def rm(y, p, r=None):
    r = np.ones(len(y), bool) if r is None else r
    return np.sqrt(((y[r] - p[r]) ** 2).mean())
for o in ['o0227', 'o0331', 'o0430']:
    d = OUT / o; p = pd.read_parquet(d / 'preds.parquet')
    y, ev, V = p.y.to_numpy(), p.ev.eq(1).to_numpy(), p.V.to_numpy(); dt = pd.to_datetime(p.date)
    import os
    keep = ~p.station_id.isin([3110, 3111]).to_numpy() if os.environ.get('EX3110') else np.ones(len(p), bool)
    early = (dt < '2026-04-01').to_numpy() & keep
    s27 = p.station_id.eq(3127).to_numpy(); game = s27 & dt.isin(games).to_numpy(); keep_old = s27 & ev & ~game
    def c9(A, E, C, H):
        Em = (E + C + H) / 3; M = .8 * Em + .2 * A; B = np.where(ev, M, .8 * M + .2 * V)
        return np.where(s27, np.where(~ev, A, (A + Em) / 2), B)
    old = c9(p.A.to_numpy(), p.E42.to_numpy(), pd.read_parquet(d / 'preds_base_C.parquet').C.to_numpy(), p.H.to_numpy())
    S0, S1, got = [], [], []
    for s in SETS:
        a, c = pd.read_parquet(d / f'preds_{s}_AEH.parquet'), pd.read_parquet(d / f'preds_{s}_C.parquet')
        f = d / f'preds_{s}_e3k_E.parquet'
        e3 = pd.read_parquet(f).E.to_numpy() if f.exists() else None
        S0.append(c9(a.A.to_numpy(), a.E.to_numpy(), c.C.to_numpy(), a.H.to_numpy()))
        S1.append(c9(a.A.to_numpy(), a.E.to_numpy() if e3 is None else e3, c.C.to_numpy(), a.H.to_numpy()))
        if e3 is not None:
            got.append(s); E5 = a.E.to_numpy()
            line = f'{o} {s:13s} single E500 vs E3k:'
            if o == 'o0227': line += f' early 2/28-3/31 {rm(y, E5, early):.5f}->{rm(y, e3, early):.5f} |'
            line += f' full {rm(y, E5, keep):.5f}->{rm(y, e3, keep):.5f}'
            print(line)
    def q(S): return np.maximum(np.where(keep_old, old, np.where(game, np.mean(S, 0), (old + np.sum(S, 0)) / (1 + len(S)))), 0)
    q0, q1 = q(S0), q(S1)
    line = f'{o} C19 recipe, E->E3k in {got}:'
    if o == 'o0227': line += f' early {rm(y, q0, early):.6f}->{rm(y, q1, early):.6f} ({rm(y, q1, early) - rm(y, q0, early):+.6f}) |'
    ordn = keep & ~ev & ~s27   # rows whose inputs do not depend on the detected (ridership-based) event flags
    print(line + f' full {rm(y, q0, keep):.6f}->{rm(y, q1, keep):.6f} ({rm(y, q1, keep) - rm(y, q0, keep):+.6f}) [{dt.dt.date.nunique()}d]'
          f' | ordinary rows {rm(y, q1, ordn) - rm(y, q0, ordn):+.6f}' + (f' (early {rm(y, q1, ordn & early) - rm(y, q0, ordn & early):+.6f})' if o == 'o0227' else ''))
