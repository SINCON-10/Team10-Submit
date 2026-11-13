"""C18 = C17 + extra model sets (ADD) averaged in, e.g. att_ball (residual base lvl8 + yoy_delta_all).
ADD sets are trained without the break rows and on the repaired test file, so they are used on every row incl. 3110/3111.
Checks: rebuilds C15 and C17 from the pieces before adding.
args: PV4 RAW CAND OUT NAME REF SETS CLSETS ADD C17NAME

(original C17 notes:)
C17 = C15 with 계양 3110 / 귤현 3111 predicted by model sets retrained without the Geomdan-opening break rows.

Why: the three new Line-1 stations opened 2025-06-28 and took a permanent share of 3110 (-0.53 log beyond the all-station
Jun->Jul change) and 3111 (-0.24). Training rows whose cutoff level window (lvl8, 8 weeks) is before the opening and whose
target is after it carry that one-off drop as a residual of -0.3..-0.55 on Jul-Oct targets, so every tree model learns
"3110/3111 fall in Jul-Oct" and repeats it for Jul-Oct 2026 (C15 3110 Jul weekday ~1,725/day vs June 2026 actual ~2,514).
Fix (cl_run.sh -> final_att.py DROP_BREAK=1 + TEST_FILE=test_fix.csv.gz): drop those rows (3110/3111 only) and use the
repaired partner-year features (fix_test_yoy.py). No ridership after 2026-06-30 is used; nothing is tuned.
Only the 3110/3111 rows take the retrained sets (same C9 blend, V kept); every other row is exactly C15.
args: PV4 RAW CAND OUT NAME REF SETS CLSETS   e.g. ... C17_c15_geomdan C15_avg6_d4 att,attwx,att_smc,attwx_smc,att_d4 base,att,att_smc
  CLSETS: sets retrained clean (files OUT/final_{s}_cl_{A,E,C,H}.npy, s=base for the old set)
codex runtime (catboost for the old CatBoost model).
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
assert np.abs(np.expm1(finish(np.where(aff, np.mean(cl9, axis=0), B))) - np.expm1(load(C17N))).max() < 1e-3, 'pieces do not reproduce C17'
adds = [c9(*(np.load(OUT / f'final_{s}_{k}.npy') for k in 'AECH'), ev_new) for s in ADD]
news2 = news + adds
B2 = np.where(concert27, B_old, np.where(game27 | extra, np.mean(news2, axis=0), (B_old + np.sum(news2, axis=0)) / (1 + len(news2))))
B_cl = np.mean(cl9 + adds, axis=0)
p = finish(np.where(aff, B_cl, B2))
c17 = load(C17N); d17 = p - c17; dref = float(np.sqrt((d17 ** 2).mean()))
sub = pd.DataFrame({'example_id': test.example_id, 'boardings': np.expm1(p)})
assert len(sub) == 157194 and sub.example_id.is_unique and np.isfinite(sub.boardings).all() and sub.boardings.ge(0).all()
path = CAND / f'submission_{NAME}.csv'
sub.to_csv(path, index=False, float_format='%.6f', lineterminator='\n')
print(path.name, hashlib.sha256(path.read_bytes()).hexdigest()[:8], f'| added {ADD} | rmse vs {C17N} {dref:.4f}, mean {d17.mean():+.4f}, max |d| {np.abs(d17).max():.3f}, game rows {np.sqrt((d17[game27] ** 2).mean()):.4f}, concert rows changed {(np.abs(d17[concert27]) > 1e-6).sum()}')

print('3110/3111 rmse vs C17', round(float(np.sqrt((d17[aff] ** 2).mean())), 4))
