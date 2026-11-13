"""C12 = average of whole model sets in the C9 recipe (old / +attendance / +attendance+reference weather [/ seed variants]).
Routing (validated on 3 origins, every set uses the same C9 blend and the same event flags):
  3127 non-game event days (concerts/festivals) : old set only (no-attendance class was learned from false detections)
  3127 SSG game days and the 5 extra event days  : mean of the new sets (old set has no attendance / no flag there)
  everything else                                : mean of old + new sets
then the same Chuseok lunar_adjust.
args: PV4 RAW CAND OUT NAME SETS     (SETS: comma list of new set prefixes, e.g. att,attwx  -> final_att_A.npy, final_attwx_A.npy)
codex runtime (catboost for the old CatBoost model).
"""
import hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostRegressor, Pool

PV4, RAW, CAND, OUT = map(Path, sys.argv[1:5]); NAME = sys.argv[5]; SETS = sys.argv[6].split(',')
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
sn_te = (te.lvl8 + te.yoy_delta.fillna(0)).to_numpy()
cm = CatBoostRegressor(); cm.load_model(str(CAND / 'model_C7_cat1500_seasonal_resid.cbm'))
C = np.maximum(sn_te + cm.predict(Pool(te[feats], cat_features=CATS)), 0)
H = np.load(OUT / 'final_hgb_test.npy')


def load(name):
    s = pd.read_csv(CAND / f'submission_{name}.csv')
    assert s.example_id.tolist() == test.example_id.tolist()
    return np.log1p(s.boardings.to_numpy())


A, E, C2 = (load(n) for n in ['C0_lgb_direct', 'C1_lgb_seasonal_resid', 'C2_resid_v2_20_nonevent'])
V = (C2 - .8 * E) / .2
s27 = te.station_id.eq(3127).to_numpy()


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

B_old = c9(A, E, C, H, ev_old)
news = [c9(*(np.load(OUT / f'final_{s}_{k}.npy') for k in 'AECH'), ev_new) for s in SETS]
B_new = np.mean(news, axis=0)
B_all = (B_old + np.sum(news, axis=0)) / (1 + len(news))
B = np.where(concert27, B_old, np.where(game27 | extra, B_new, B_all))


def finish(B):
    out = lunar_adjust(train, test[['date', 'station_id', 'hour']].assign(p=B))
    assert (out[['date', 'station_id', 'hour']].to_numpy() == test[['date', 'station_id', 'hour']].to_numpy()).all()
    return np.maximum(out.p.to_numpy(), 0)


C9 = load('C9_C8_3127rule')
assert np.abs(np.expm1(finish(B_old)) - np.expm1(C9)).max() < 1e-4, 'old pieces do not reproduce C9'
p = finish(B)
sub = pd.DataFrame({'example_id': test.example_id, 'boardings': np.expm1(p)})
assert len(sub) == 157194 and sub.example_id.is_unique and np.isfinite(sub.boardings).all() and sub.boardings.ge(0).all()
path = CAND / f'submission_{NAME}.csv'
sub.to_csv(path, index=False, float_format='%.6f', lineterminator='\n')
C10 = load('C10_C9_att_events')
d9, d10 = p - C9, p - C10
print(path.name, hashlib.sha256(path.read_bytes()).hexdigest()[:8], 'sets', SETS,
      f'| rmse vs C9 {np.sqrt((d9 ** 2).mean()):.4f} vs C10 {np.sqrt((d10 ** 2).mean()):.4f} mean diff vs C10 {d10.mean():+.4f}',
      f'| game rows vs C10 {np.sqrt((d10[game27] ** 2).mean()):.4f} | concert rows changed vs C9 {(np.abs(d9[concert27]) > 1e-6).sum()}',
      f'| new-set spread (rmse between sets) {np.sqrt(((news[0] - B_old) ** 2).mean()):.4f}')
