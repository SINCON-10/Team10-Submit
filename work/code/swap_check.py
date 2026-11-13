"""C19 recipe (fixed, no fitted weights) with model swaps inside the new sets, per fold origin:
  E -> 3000-tree residual LightGBM for sets in env E3KSETS (preds_{s}_e3k_E.parquet)
  A -> 4000-tree direct LightGBM for sets in env A4KSETS (direct model ignores the residual base: att_ball uses att's,
       att_ball_smc uses att_smc's; o0227 att comes from the curve run, column A_4000)
Reports o0227 early window (2/28-3/31, the selection window) and full 123 days, o0331/o0430 (auxiliary); all rows,
ordinary rows (not ev, not 3127) and 3127 rows; 3110/3111 excluded. args: OUT ATT_CSV SETS"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(sys.argv[1]); games = pd.read_csv(sys.argv[2], parse_dates=['date']).date; SETS = sys.argv[3].split(',')
E3 = [x for x in os.environ.get('E3KSETS', '').split(',') if x]; A4 = [x for x in os.environ.get('A4KSETS', '').split(',') if x]
SRC = {'att': 'att', 'att_ball': 'att', 'att_smc': 'att_smc', 'att_ball_smc': 'att_smc', 'attwx': 'attwx', 'attwx_smc': 'attwx_smc', 'att_d4': 'att_d4'}
def rm(y, p, r): return np.sqrt(((y[r] - p[r]) ** 2).mean())
for o in ['o0227', 'o0331', 'o0430']:
    d = OUT / o; p = pd.read_parquet(d / 'preds.parquet')
    y, ev, V = p.y.to_numpy(), p.ev.eq(1).to_numpy(), p.V.to_numpy(); dt = pd.to_datetime(p.date)
    keep = ~p.station_id.isin([3110, 3111]).to_numpy(); early = keep & (dt < '2026-04-01').to_numpy()
    s27 = p.station_id.eq(3127).to_numpy(); game = s27 & dt.isin(games).to_numpy(); keep_old = s27 & ev & ~game
    def c9(A, E, C, H):
        Em = (E + C + H) / 3; M = .8 * Em + .2 * A; B = np.where(ev, M, .8 * M + .2 * V)
        return np.where(s27, np.where(~ev, A, (A + Em) / 2), B)
    old = c9(p.A.to_numpy(), p.E42.to_numpy(), pd.read_parquet(d / 'preds_base_C.parquet').C.to_numpy(), p.H.to_numpy())
    def a4k(s):
        src = SRC[s]
        if o == 'o0227' and src == 'att': return pd.read_parquet(d / 'preds_att_acurve_A.parquet').A_4000.to_numpy()
        f = d / f'preds_{src}_a4k_A.parquet'
        return pd.read_parquet(f).A.to_numpy() if f.exists() else None
    variants = {'C19': [], '+E3k': [], '+A4k': [], '+both': []}; miss = []
    for s in SETS:
        a, c = pd.read_parquet(d / f'preds_{s}_AEH.parquet'), pd.read_parquet(d / f'preds_{s}_C.parquet')
        A0, E0, C0, H0 = a.A.to_numpy(), a.E.to_numpy(), c.C.to_numpy(), a.H.to_numpy()
        f3 = d / f'preds_{s}_e3k_E.parquet'; E1 = pd.read_parquet(f3).E.to_numpy() if (s in E3 and f3.exists()) else None
        A1 = a4k(s) if s in A4 else None
        if s in E3 and E1 is None: miss.append(f'E3k:{s}')
        if s in A4 and A1 is None: miss.append(f'A4k:{s}')
        E1 = E0 if E1 is None else E1; A1 = A0 if A1 is None else A1
        if os.environ.get('A4K_KEEP27') == '1': A1 = np.where(s27, A0, A1)
        variants['C19'].append(c9(A0, E0, C0, H0)); variants['+E3k'].append(c9(A0, E1, C0, H0))
        variants['+A4k'].append(c9(A1, E0, C0, H0)); variants['+both'].append(c9(A1, E1, C0, H0))
    oldm = old
    if os.environ.get('OLDUP') == '1' and (d / 'preds_base_a4k_A.parquet').exists() and (d / 'preds_base_e3k_E.parquet').exists():   # old set with A 4000 / E 3000 trees (3127 keeps the old A)
        oldm = c9(np.where(s27, p.A.to_numpy(), pd.read_parquet(d / 'preds_base_a4k_A.parquet').A.to_numpy()), pd.read_parquet(d / 'preds_base_e3k_E.parquet').E.to_numpy(), pd.read_parquet(d / 'preds_base_C.parquet').C.to_numpy(), p.H.to_numpy())
    elif os.environ.get('OLDUP') == '1': miss.append('old-set upgrade')
    q = {k: np.maximum(np.where(keep_old, old, np.where(game, np.mean(S, 0), ((old if k == 'C19' else oldm) + np.sum(S, 0)) / (1 + len(S)))), 0) for k, S in variants.items()}
    ordn = keep & ~ev & ~s27; r27 = keep & s27
    print(f'{o} [{dt.dt.date.nunique()}d]' + (f' missing {miss}' if miss else ''))
    for k in ['+E3k', '+A4k', '+both']:
        line = f'   {k:6s} vs C19: full {rm(y, q["C19"], keep):.6f}->{rm(y, q[k], keep):.6f} ({rm(y, q[k], keep) - rm(y, q["C19"], keep):+.6f})'
        if o == 'o0227': line += f' | early {rm(y, q[k], early) - rm(y, q["C19"], early):+.6f}'
        line += f' | ordinary {rm(y, q[k], ordn) - rm(y, q["C19"], ordn):+.6f} | 3127 {rm(y, q[k], r27) - rm(y, q["C19"], r27):+.5f}'
        print(line)
