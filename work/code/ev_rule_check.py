"""Fixed-rule check (no fitting) for event rows outside 3127. Nothing from Public. args: OUT ATT_CSV SETS"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(sys.argv[1]); games = pd.read_csv(sys.argv[2], parse_dates=['date']).date; SETS = sys.argv[3].split(',')
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
    W = np.array([.109, 0, .238, .481, .172]); X = np.column_stack([np.mean([v[i] for v in comp.values()], 0) for i in range(4)] + [V])
    m_ord = ~ev & ~s27; q_w = q0.copy(); q_w[m_ord] = np.maximum(X[m_ord] @ W, 0)     # C20w
    m = ev & ~s27
    alts = {'C20w (ev rows = .8Em+.2A)': q_w[m], 'H only': X[m, 3], '(C+H)/2': X[m, 2:4].mean(1), '.8H+.2A': .8 * X[m, 3] + .2 * X[m, 0],
            'C20w weights w/o V': X[m, :4] @ (W[:4] / W[:4].sum())}
    n = len(y); base_sse = ((y - q_w) ** 2).sum(); e_base = ((y[m] - q_w[m]) ** 2).sum()
    print(o, f'ev rows {m.sum()}: ' + ' | '.join(f'{k} grp {np.sqrt(((y[m] - np.maximum(v, 0)) ** 2).mean()):.4f} all {np.sqrt((base_sse - e_base + ((y[m] - np.maximum(v, 0)) ** 2).sum()) / n) - np.sqrt(base_sse / n):+.5f}' for k, v in alts.items()))
    pe = p[m].assign(r0=y[m] - q_w[m], r1=y[m] - X[m, 3])
    print('   by station (sse change H-only vs C20w):', pe.groupby('station_id').apply(lambda g: (g.r1 ** 2).sum() - (g.r0 ** 2).sum()).round(2).sort_values().to_dict())
