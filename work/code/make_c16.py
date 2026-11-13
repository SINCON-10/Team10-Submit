"""C16 = C15 recipe (six model sets averaged in the C9 blend) with every tree model re-predicted on the repaired test
features (fix_test_yoy.py: partner-year comparison of 계양 3110 / 귤현 3111). Training data and models are unchanged.
Checks: (1) the original pieces still rebuild C9; (2) every re-predicted model equals its original on all rows outside
3110/3111; (3) the result equals the reference candidate (C15) outside 3110/3111.
V (v2 structural baseline, 20% on non-event rows) is not affected and is kept as is.
args: PV4 RAW CAND OUT TEST_FIX NAME REF SETS    e.g. ... test_fix.csv.gz C16_fix_yoy C15_avg6_d4 att,attwx,att_smc,attwx_smc,att_d4
codex runtime (catboost for the old CatBoost model).
"""
import hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostRegressor, Pool

PV4, RAW, CAND, OUT, TFIX = map(Path, sys.argv[1:6]); NAME, REF = sys.argv[6], sys.argv[7]; SETS = sys.argv[8].split(',')
sys.path.insert(0, str(PV4))
from lunar_adjust import lunar_adjust  # noqa: E402

te = pd.read_csv(PV4 / 'test_cutoff_features.csv.gz'); tf = pd.read_csv(TFIX)
test = pd.read_csv(RAW / 'test.csv', parse_dates=['date'])
train = pd.read_csv(RAW / 'train.csv', usecols=['date', 'station_id', 'hour', 'boardings'], parse_dates=['date'])
assert te.example_id.tolist() == test.example_id.tolist() == tf.example_id.tolist()
fixed = (te.yoy_delta.ne(tf.yoy_delta) & te.yoy_delta.notna()) | te.yoy_raw.ne(tf.yoy_raw) & te.yoy_raw.notna()
aff = te.station_id.isin(tf.station_id[fixed].unique()).to_numpy()
print('repaired stations', sorted(tf.station_id[fixed].unique()), 'rows', int(aff.sum()))
feats = [c for c in te.columns if c not in ('date', 'station_id', 'y', 'example_id')]
CATS = ['sid', 'line', 'dow', 'dt3']
for d in (te, tf):
    for c in CATS:
        d[c] = d[c].astype(int)
cm = CatBoostRegressor(); cm.load_model(str(CAND / 'model_C7_cat1500_seasonal_resid.cbm'))
sn = lambda d: (d.lvl8 + d.yoy_delta.fillna(0)).to_numpy()
C_o = np.maximum(sn(te) + cm.predict(Pool(te[feats], cat_features=CATS)), 0)
C_f = np.maximum(sn(tf) + cm.predict(Pool(tf[feats], cat_features=CATS)), 0)


def load(name):
    s = pd.read_csv(CAND / f'submission_{name}.csv')
    assert s.example_id.tolist() == test.example_id.tolist()
    return np.log1p(s.boardings.to_numpy())


A_o, E_o, C2 = (load(n) for n in ['C0_lgb_direct', 'C1_lgb_seasonal_resid', 'C2_resid_v2_20_nonevent'])
H_o = np.load(OUT / 'final_hgb_test.npy')
V = (C2 - .8 * E_o) / .2
A_f, E_f, H_f = (np.load(OUT / f'final_base_fx_{k}.npy') for k in 'AEH')
s27 = te.station_id.eq(3127).to_numpy()


def same(a, b, tol, what):
    dmax = float(np.abs(a[~aff] - b[~aff]).max())
    assert dmax < tol, (what, dmax)
    return dmax


print('old set outside repaired rows: max |diff| A %.1e E %.1e C %.1e H %.1e' % (
    same(A_o, A_f, 2e-6, 'A'), same(E_o, E_f, 2e-6, 'E'), same(C_o, C_f, 1e-9, 'C'), same(H_o, H_f, 1e-6, 'H')))


def c9(A, E, C, H, ev):
    Em = (E + C + H) / 3
    M = .8 * Em + .2 * A
    B = np.where(ev, M, .8 * M + .2 * V)
    return np.maximum(np.where(s27, np.where(~ev, A, (A + Em) / 2), B), 0)


ev_old = te.ev.eq(1).to_numpy()
ev_new = np.load(OUT / 'final_att_ev.npy').astype(bool)
extra = ev_new & ~ev_old
assert extra.sum() == 5 * 18 and (ev_old <= ev_new).all()
games = pd.read_csv(PV4.parent.parent / 'claude_works/external/ssg_munhak_attendance.csv', parse_dates=['date']).date
game27 = s27 & pd.to_datetime(te.date).isin(games).to_numpy()
assert pd.Series(te.date[game27]).nunique() == 35
concert27 = s27 & ev_old & ~game27


def finish(B):
    out = lunar_adjust(train, test[['date', 'station_id', 'hour']].assign(p=B))
    assert (out[['date', 'station_id', 'hour']].to_numpy() == test[['date', 'station_id', 'hour']].to_numpy()).all()
    return np.maximum(out.p.to_numpy(), 0)


assert np.abs(np.expm1(finish(c9(A_o, E_o, C_o, H_o, ev_old))) - np.expm1(load('C9_C8_3127rule'))).max() < 1e-4, 'C9 rebuild'
B_old = c9(A_f, E_f, C_f, H_f, ev_old)
news = []
for s in SETS:
    o = [np.load(OUT / f'final_{s}_{k}.npy') for k in 'AECH']
    f = [np.load(OUT / f'final_{s}_fx_{k}.npy') for k in 'AECH']
    print(f'{s}: outside repaired rows max |diff| ' + ' '.join(f'{k} {same(a, b, 1e-6, s + k):.1e}' for k, a, b in zip('AECH', o, f)))
    news.append(c9(*f, ev_new))
B_new = np.mean(news, axis=0)
B_all = (B_old + np.sum(news, axis=0)) / (1 + len(news))
B = np.where(concert27, B_old, np.where(game27 | extra, B_new, B_all))
p = finish(B)
ref = load(REF)
dref = float(np.abs(p[~aff] - ref[~aff]).max())
assert dref < 1e-5, ('differs from reference outside repaired rows', dref)
sub = pd.DataFrame({'example_id': test.example_id, 'boardings': np.expm1(p)})
assert len(sub) == 157194 and sub.example_id.is_unique and np.isfinite(sub.boardings).all() and sub.boardings.ge(0).all()
path = CAND / f'submission_{NAME}.csv'
sub.to_csv(path, index=False, float_format='%.6f', lineterminator='\n')
d = p - ref; t = test.assign(d=d, off=te.offday.to_numpy())
print(path.name, hashlib.sha256(path.read_bytes()).hexdigest()[:8], f'| equals {REF} outside repaired rows (max {dref:.1e})')
print('mean log change vs reference by station/month:')
print(t[aff].groupby([t.station_id[aff], t.date[aff].dt.strftime('%m')]).d.mean().unstack().round(3).to_string())
for s in sorted(te.station_id[aff].unique()):
    for m in ['07', '09']:
        r = (t.station_id.eq(s) & t.date.dt.strftime('%m').eq(m) & t.off.eq(0)).to_numpy()
        print(f'{s} {m} weekday daily boardings: {REF} {np.expm1(ref[r]).sum() / r.sum() * 18:.0f} -> {NAME} {np.expm1(p[r]).sum() / r.sum() * 18:.0f}')
