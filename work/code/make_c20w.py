"""C20w = C19 (or any make_c18 recipe) with fold-fitted model-type weights on ordinary rows.
Ordinary rows = not an event row (ev_new) and not 3127. There the C9 blend (A .16, E/C/H .2133 each, V .2, per set, then
set mean) is replaced by w . [A, E, C, H, V], A..H averaged over the same sets. w >= 0, sum 1, least squares on the three
fold origins (o0227/o0331/o0430, log1p). Leave-one-origin-out with this exact form improved every held-out origin
(stack_loo.py: -0.00017 / -0.00081 / -0.00008). Event rows, 3127 and the lunar adjustment are unchanged. Nothing from Public.
args: PV4 RAW CAND OUT NAME REF SETS CLSETS ADD C17NAME C19NAME   (first ten as make_c18.py; C19NAME = file it must rebuild)
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

# fold-fitted weights (same sets, same routing as compare_sets.py / stack_loo.py)
from scipy.optimize import minimize  # noqa: E402
FS = SETS + ADD; FD = OUT
XtX = np.zeros((5, 5)); Xty = np.zeros(5); n = 0
for o in ([] if os.environ.get('WFIX') else ['o0227', 'o0331', 'o0430']):
    d = FD / o; q = pd.read_parquet(d / 'preds.parquet')
    fev = q.ev.eq(1).to_numpy(); fs27 = q.station_id.eq(3127).to_numpy(); m = ~fev & ~fs27
    comp = [(q.A.to_numpy(), q.E42.to_numpy(), pd.read_parquet(d / 'preds_base_C.parquet').C.to_numpy(), q.H.to_numpy())]
    for s in FS:
        a, c = pd.read_parquet(d / f'preds_{s}_AEH.parquet'), pd.read_parquet(d / f'preds_{s}_C.parquet')
        comp.append((a.A.to_numpy(), a.E.to_numpy(), c.C.to_numpy(), a.H.to_numpy()))
    X = np.column_stack([np.mean([v[i] for v in comp], 0) for i in range(4)] + [q.V.to_numpy()])[m]; y = q.y.to_numpy()[m]
    XtX += X.T @ X; Xty += X.T @ y; n += len(y)
XtX /= max(n, 1); Xty /= max(n, 1)
if not n: XtX, Xty = np.eye(5), np.full(5, .2)
W0 = np.array([.16, .64 / 3, .64 / 3, .64 / 3, .2])
r = minimize(lambda w: w @ XtX @ w - 2 * w @ Xty, W0, jac=lambda w: 2 * XtX @ w - 2 * Xty, method='SLSQP', bounds=[(0, 1)] * 5,
             constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1, 'jac': lambda w: np.ones(5)}, options={'ftol': 1e-14, 'maxiter': 500})
assert r.success, r.message
w = r.x
if os.environ.get('WFIX'):   # fixed weights (e.g. while a new set's folds are unfinished)
    w = np.array([float(x) for x in os.environ['WFIX'].split(',')]); assert len(w) == 5 and abs(w.sum() - 1) < 1e-6
print('WFIX' if os.environ.get('WFIX') else 'fitted', end=' '); print('fold-fitted weights A E C H V:', ' '.join(f'{x:.3f}' for x in w))

# test components averaged over the same sets (3110/3111: clean sets + ADD, as in C19)
allv = [(A, E, C, H)] + [tuple(np.load(OUT / f'final_{s}_{k}.npy') for k in 'AECH') for s in SETS] + list(addv.values())
clv = list(cl.values()) + list(addv.values())
Xt = np.column_stack([np.mean([v[i] for v in allv], 0) for i in range(4)] + [V])
Xc = np.column_stack([np.mean([v[i] for v in clv], 0) for i in range(4)] + [V])
Xt[aff] = Xc[aff]
ordin = ~ev_new & ~s27
assert not (ordin & (concert27 | game27 | extra)).any()
Bw = np.where(aff, B_cl, B2).copy(); Bw[ordin] = np.maximum(Xt[ordin] @ w, 0)
EVRULE = os.environ.get('EVRULE', '')   # event rows outside 3127: 'H' = HistGB mean over sets (fold check: -0.00042/-0.00055/-0.00006)
if EVRULE:
    evr = ev_new & ~s27; assert not (evr & aff).any()
    Hnew = np.mean([np.load(OUT / f'final_{s}_H.npy') for s in SETS] + [addv[s][3] for s in ADD], 0)   # extra events: old set does not flag them
    Hev = np.where(extra, Hnew, Xt[:, 3])
    if EVRULE == 'H':
        Bw[evr] = np.maximum(Hev[evr], 0)
    elif EVRULE == 'H8A2':
        Aev = np.where(extra, np.mean([np.load(OUT / f'final_{s}_A.npy') for s in SETS] + [addv[s][0] for s in ADD], 0), Xt[:, 0])
        Bw[evr] = np.maximum(.8 * Hev[evr] + .2 * Aev[evr], 0)
    else:
        raise SystemExit('unknown EVRULE')
    print(f'EVRULE={EVRULE}: {int(evr.sum())} event rows outside 3127 ({int((evr & extra).sum())} extra-event rows use new sets only)')
p = finish(Bw)
d19 = p - p19
sub = pd.DataFrame({'example_id': test.example_id, 'boardings': np.expm1(p)})
assert len(sub) == 157194 and sub.example_id.is_unique and np.isfinite(sub.boardings).all() and sub.boardings.ge(0).all()
path = CAND / f'submission_{NAME}.csv'
sub.to_csv(path, index=False, float_format='%.6f', lineterminator=chr(10))
print(path.name, hashlib.sha256(path.read_bytes()).hexdigest()[:8], f'| sets {["old"] + FS} | rmse vs {C19N} {np.sqrt((d19 ** 2).mean()):.4f}, mean {d19.mean():+.4f}, '
      f'max |d| {np.abs(d19).max():.3f}, unchanged rows (event/3127) max |d| {np.abs(d19[~ordin]).max():.2e} outside lunar days')
