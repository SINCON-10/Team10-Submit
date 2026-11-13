"""C23 = C19 recipe (fixed C9 blend per set, same set routing, lunar adjustment) with the residual LightGBM E (500 trees)
replaced by the 3000-tree version in the sets listed in env E3K (env WFIX: C20w-style weights on ordinary rows) (finals final_{s}_e3k_E.npy, trained with DROP_BREAK=1 and
the repaired test file, so they are also valid for 3110/3111). No fitted weights. Tree count chosen on the 2/28-3/31 window of
the 2/27 fold (e3k_check.py), recipe change checked on o0227 (123d) and o0331/o0430 (91/61d, auxiliary). Nothing from Public.
args: PV4 RAW CAND OUT NAME REF SETS CLSETS ADD C17NAME C19NAME   (as make_c20w.py; env C19ADD as there)
"""

import hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostRegressor, Pool

PV4, RAW, CAND, OUT = map(Path, sys.argv[1:5]); NAME, REF = sys.argv[5], sys.argv[6]
SETS, CLSETS = sys.argv[7].split(','), sys.argv[8].split(','); ADD = sys.argv[9].split(','); C17N = sys.argv[10]
STATIONS = [3110, 3111]
sys.path.insert(0, str(PV4))
from lunar_adjust import lunar_adjust  # noqa: E402

te = pd.read_csv(PV4 / 'test_cutoff_features.csv.gz')
test = pd.read_csv(RAW / 'test.csv', parse_dates=['date'])
train = pd.read_csv(RAW / 'train.csv', usecols=['date', 'station_id', 'hour', 'boardings'], parse_dates=['date'])
assert te.example_id.tolist() == test.example_id.tolist() and train.date.max() == pd.Timestamp('2026-06-30')
feats = [c for c in te.columns if c not in ('date', 'station_id', 'y', 'example_id')]
CATS = ['sid', 'line', 'dow', 'dt3']
for c in CATS:
    te[c] = te[c].astype(int)
cm = CatBoostRegressor(); cm.load_model(str(CAND / 'model_C7_cat1500_seasonal_resid.cbm'))
C = np.maximum((te.lvl8 + te.yoy_delta.fillna(0)).to_numpy() + cm.predict(Pool(te[feats], cat_features=CATS)), 0)
H = np.load(OUT / 'final_hgb_test.npy')


def load(name):
    s = pd.read_csv(CAND / f'submission_{name}.csv')
    assert s.example_id.tolist() == test.example_id.tolist()
    return np.log1p(s.boardings.to_numpy())


A, E, C2 = (load(n) for n in ['C0_lgb_direct', 'C1_lgb_seasonal_resid', 'C2_resid_v2_20_nonevent'])
V = (C2 - .8 * E) / .2
s27 = te.station_id.eq(3127).to_numpy()
aff = te.station_id.isin(STATIONS).to_numpy()


def c9(A, E, C, H, ev):
    Em = (E + C + H) / 3
    M = .8 * Em + .2 * A
    B = np.where(ev, M, .8 * M + .2 * V)
    return np.maximum(np.where(s27, np.where(~ev, A, (A + Em) / 2), B), 0)


ev_old = te.ev.eq(1).to_numpy()
ev_new = np.load(OUT / 'final_att_ev.npy').astype(bool)
extra = ev_new & ~ev_old
assert extra.sum() == 5 * 18 and (ev_old <= ev_new).all() and not ev_new[aff].any()
games = pd.read_csv(PV4.parent.parent / 'claude_works/external/ssg_munhak_attendance.csv', parse_dates=['date']).date
game27 = s27 & pd.to_datetime(te.date).isin(games).to_numpy()
assert pd.Series(te.date[game27]).nunique() == 35
concert27 = s27 & ev_old & ~game27

# C15 exactly as make_c12.py builds it
B_old = c9(A, E, C, H, ev_old)
news = [c9(*(np.load(OUT / f'final_{s}_{k}.npy') for k in 'AECH'), ev_new) for s in SETS]
B_all = (B_old + np.sum(news, axis=0)) / (1 + len(news))
B = np.where(concert27, B_old, np.where(game27 | extra, np.mean(news, axis=0), B_all))


def finish(B):
    out = lunar_adjust(train, test[['date', 'station_id', 'hour']].assign(p=B))
    assert (out[['date', 'station_id', 'hour']].to_numpy() == test[['date', 'station_id', 'hour']].to_numpy()).all()
    return np.maximum(out.p.to_numpy(), 0)


ref = load(REF)
assert np.abs(np.expm1(finish(B)) - np.expm1(ref)).max() < 1e-3, 'pieces do not reproduce the reference'

# 3110/3111: mean of the clean sets in the same C9 blend (no event rows there)
cl = {s: [np.load(OUT / f'final_{s}_cl_{k}.npy') for k in 'AECH'] for s in CLSETS}
cl9 = [c9(*v, ev_old if s == 'base' else ev_new) for s, v in cl.items()]
addv = {s: [np.load(OUT / f'final_{s}_{k}.npy') for k in 'AECH'] for s in ADD}
import os  # noqa: E402
ADD19 = os.environ.get('C19ADD', ','.join(ADD)).split(',')   # sets that rebuild the C19 reference (ADD may hold more)
adds = [c9(*addv[s], ev_new) for s in ADD19]
news2 = news + adds
B2 = np.where(concert27, B_old, np.where(game27 | extra, np.mean(news2, axis=0), (B_old + np.sum(news2, axis=0)) / (1 + len(news2))))
B_cl = np.mean(cl9 + adds, axis=0)
p19 = finish(np.where(aff, B_cl, B2))
C19N = sys.argv[11]
assert np.abs(np.expm1(p19) - np.expm1(load(C19N))).max() < 1e-3, 'pieces do not reproduce ' + C19N

E3K = [x for x in os.environ.get('E3K', '').split(',') if x]
assert E3K, 'set E3K (E3K=none: no swap, for reproduction checks)'
E3KF = {x.split(':')[0]: (x.split(':')[1] if ':' in x else f'{x}_e3k') for x in E3K}   # set -> final file stem (set:stem = same model trained under another TAG)
E3K = list(E3KF)


A4K = dict(x.split(':') for x in os.environ.get('A4K', '').split(',') if x)   # set:file stem of a 4000-tree direct LightGBM final


def e3(s, v):   # swap E for the 3000-tree residual LightGBM of the same set (and A for the 4000-tree direct one, env A4K)
    v = list(v)
    if s in E3K:
        v[1] = np.load(OUT / f'final_{E3KF[s]}_E.npy'); assert v[1].shape == v[0].shape and np.isfinite(v[1]).all()
    if s in A4K:
        a4 = np.load(OUT / f'final_{A4K[s]}_A.npy'); assert a4.shape == v[1].shape and np.isfinite(a4).all()
        v[0] = np.where(s27, v[0], a4) if os.environ.get('A4K_KEEP27') == '1' else a4   # A4K_KEEP27=1: station 3127 keeps the 500-tree A (3127 evidence is mixed across folds)
    return v


news3 = [c9(*e3(s, [np.load(OUT / f'final_{s}_{k}.npy') for k in 'AECH']), ev_new) for s in SETS]
adds3 = [c9(*e3(s, addv[s]), ev_new) for s in ADD19]
cl93 = [c9(*e3(s, v), ev_old if s == 'base' else ev_new) for s, v in cl.items()]
n3 = news3 + adds3
B_oldm = B_old
if os.environ.get('OLDUP') == '1':   # old set with A 4000 / E 3000 trees in the average (final_base_a4k_A / final_base_e3k_E; 3127 keeps the old A; concert rows keep the old set as is)
    a_up, e_up = np.load(OUT / 'final_base_a4k_A.npy'), np.load(OUT / 'final_base_e3k_E.npy'); assert a_up.shape == A.shape == e_up.shape and np.isfinite(a_up).all() and np.isfinite(e_up).all()
    B_oldm = c9(np.where(s27, A, a_up), e_up, C, H, ev_old)
B3 = np.where(concert27, B_old, np.where(game27 | extra, np.mean(n3, axis=0), (B_oldm + np.sum(n3, axis=0)) / (1 + len(n3))))
Bf = np.where(aff, np.mean(cl93 + adds3, axis=0), B3)
WFIX = os.environ.get('WFIX')   # optional: weights A,E,C,H,V on ordinary rows (not ev_new, not 3127), as in make_c20w.py
if WFIX:
    w = np.array([float(x) for x in WFIX.split(',')]); assert len(w) == 5 and abs(w.sum() - 1) < 1e-6 and (w >= 0).all()
    allv3 = [[A, E, C, H]] + [e3(s, [np.load(OUT / f'final_{s}_{k}.npy') for k in 'AECH']) for s in SETS] + [e3(s, addv[s]) for s in ADD19]
    clv3 = [e3(s, v) for s, v in cl.items()] + [e3(s, addv[s]) for s in ADD19]
    Xt = np.column_stack([np.mean([v[i] for v in allv3], 0) for i in range(4)] + [V])
    Xc = np.column_stack([np.mean([v[i] for v in clv3], 0) for i in range(4)] + [V]); Xt[aff] = Xc[aff]
    ordin = ~ev_new & ~s27; assert not (ordin & (concert27 | game27 | extra)).any()
    Bf = Bf.copy(); Bf[ordin] = np.maximum(Xt[ordin] @ w, 0)
    print('WFIX A E C H V on ordinary rows:', ' '.join(f'{x:.3f}' for x in w), f'({int(ordin.sum()):,} rows)')
p = finish(Bf)
used = [s for s in SETS + ADD19 + list(cl) if s in E3K]
assert E3K == ['none'] or set(E3K) <= set(used), ('E3K sets not in the recipe', set(E3K) - set(used))
d19 = p - p19
sub = pd.DataFrame({'example_id': test.example_id, 'boardings': np.expm1(p)})
assert len(sub) == 157194 and sub.example_id.is_unique and np.isfinite(sub.boardings).all() and sub.boardings.ge(0).all()
path = CAND / f'submission_{NAME}.csv'
sub.to_csv(path, index=False, float_format='%.6f', lineterminator=chr(10))
print(path.name, hashlib.sha256(path.read_bytes()).hexdigest()[:8], f'| E3K {E3K} {[v for v in E3KF.values() if not v.endswith("_e3k")]} | A4K {sorted(A4K)} | rmse vs {C19N} {np.sqrt((d19 ** 2).mean()):.4f}, '
      f'mean {d19.mean():+.4f}, max |d| {np.abs(d19).max():.3f}, concert-3127 rows max |d| {np.abs(d19[concert27]).max():.1e}')
t = te[['station_id', 'hour']].assign(d=d19, aff=aff)
print('mean log change vs C19 by hour band:', t.groupby(pd.cut(t.hour, [0, 9, 16, 19, 23])).d.mean().round(4).to_dict())
g = t.groupby('station_id').d.mean(); print('stations with the largest mean change:', g.abs().sort_values().tail(6).index.map(lambda k: f'{k} {g[k]:+.3f}').tolist())
