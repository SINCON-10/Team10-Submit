"""C10 = C9 with (a) SSG attendance feature models on 3127 rows and (b) 5 extra event days (3138 x4, 3210 x1).
Rows outside 3127 and the 5 extra station-days keep the exact C9 pre-Chuseok blend; then the same lunar_adjust.
args: PV4 RAW CAND OUT [all]     (codex runtime; 'all' = use the attendance models on every row)
"""
import hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostRegressor, Pool

PV4, RAW, CAND, OUT = map(Path, sys.argv[1:5]); MODE = sys.argv[5] if len(sys.argv) > 5 else ''
ALL, C11 = MODE == 'all', MODE == 'c11'   # c11: att+weather-reference models on every row except 3127 non-game event days
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
    return np.where(s27, np.where(~ev, A, (A + Em) / 2), B)


ev_old = te.ev.eq(1).to_numpy()
ev_new = np.load(OUT / 'final_att_ev.npy').astype(bool)
extra = ev_new & ~ev_old
assert extra.sum() == 5 * 18 and (ev_old <= ev_new).all()
B_old = np.maximum(c9(A, E, C, H, ev_old), 0)
An, En, Cn, Hn = (np.load(OUT / f'final_att{"wx" if C11 else ""}_{k}.npy') for k in 'AECH')
B_new = np.maximum(c9(An, En, Cn, Hn, ev_new), 0)
# new models only on SSG game days at 3127 (attendance known) + the 5 extra days. 3127 concert/festival days keep C9:
# in the folds the no-attendance 'other event' class is mostly small false detections, so big concerts got worse there.
games = pd.read_csv(PV4.parent.parent / 'claude_works/external/ssg_munhak_attendance.csv', parse_dates=['date']).date
game27 = s27 & pd.to_datetime(te.date).isin(games).to_numpy()
assert pd.Series(te.date[game27]).nunique() == 35 and (ev_old[game27]).all()
mask = np.ones(len(te), bool) if ALL else ~(s27 & ev_new & ~game27) if C11 else (game27 | extra)


def finish(B):
    out = lunar_adjust(train, test[['date', 'station_id', 'hour']].assign(p=B))
    assert (out[['date', 'station_id', 'hour']].to_numpy() == test[['date', 'station_id', 'hour']].to_numpy()).all()
    return np.maximum(out.p.to_numpy(), 0)


C9 = load('C9_C8_3127rule')
chk = finish(B_old)
assert np.abs(np.expm1(chk) - np.expm1(C9)).max() < 1e-4, 'old pieces do not reproduce C9'
p = finish(np.where(mask, B_new, B_old))
sub = pd.DataFrame({'example_id': test.example_id, 'boardings': np.expm1(p)})
assert len(sub) == 157194 and sub.example_id.is_unique and np.isfinite(sub.boardings).all() and sub.boardings.ge(0).all()
path = CAND / ('submission_C10all_C9_att_events.csv' if ALL else 'submission_C11_C10_wx.csv' if C11 else 'submission_C10_C9_att_events.csv')
sub.to_csv(path, index=False, float_format='%.6f', lineterminator='\n')
d = p - C9; ch = np.abs(d) > 1e-6
print(path.name, hashlib.sha256(path.read_bytes()).hexdigest()[:8], f'rows changed {ch.sum():,}',
      f'rmse vs C9 {np.sqrt((d ** 2).mean()):.4f}',
      f'| 3127 game rows mean diff {d[game27].mean():+.4f} rmse {np.sqrt((d[game27] ** 2).mean()):.4f}',
      f'| 3127 non-game changed {(ch & s27 & ~game27).sum()}',
      f'| extra event rows mean diff {d[extra].mean():+.4f} max {d[extra].max():+.3f}',
      f'| other rows changed {(ch & ~mask).sum()}')
