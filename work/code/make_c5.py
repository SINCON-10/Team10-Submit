"""C5 = lunar_adjust(C4)   (teammate's Chuseok 3-day correction, default w=0.8, uses train <= 2026-06-30 only)
C6 = lunar_adjust(blend)  blend = C4 on event rows and the 3 new stations, else 0.5*C4 + 0.5*team final (fixed weight)
"""
import hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd

PV4, RAW, CAND, TEAM = map(Path, sys.argv[1:5])
sys.path.insert(0, str(PV4))
from lunar_adjust import lunar_adjust  # noqa: E402

test = pd.read_csv(RAW / 'test.csv', parse_dates=['date'])
train = pd.read_csv(RAW / 'train.csv', usecols=['date', 'station_id', 'hour', 'boardings'], parse_dates=['date'])
assert train.date.max() == pd.Timestamp('2026-06-30')
te = pd.read_csv(PV4 / 'test_cutoff_features.csv.gz', usecols=['example_id', 'station_id', 'ev'])
assert te.example_id.tolist() == test.example_id.tolist()


def load(path):
    s = pd.read_csv(path).set_index('example_id').loc[test.example_id]
    return np.log1p(s.boardings.to_numpy())


C4 = load(CAND / 'submission_C4_C3_with_direct20.csv')
T = load(TEAM)
keep = te.ev.eq(1).to_numpy() | te.station_id.isin([3107, 3108, 3109]).to_numpy()
BL = np.where(keep, C4, .5 * C4 + .5 * T)
chu = test.date.isin(pd.to_datetime(['2026-09-24', '2026-09-25', '2026-09-26'])).to_numpy()


def adjust(p):
    out = lunar_adjust(train, test[['date', 'station_id', 'hour']].assign(p=p))
    assert len(out) == len(test) and (out[['date', 'station_id', 'hour']].to_numpy() == test[['date', 'station_id', 'hour']].to_numpy()).all()
    q = out.p.to_numpy()
    assert np.allclose(q[~chu], p[~chu])           # only the 3 Chuseok days change
    return np.maximum(q, 0)


for name, p in [('C5_C4_chuseok', adjust(C4)), ('C6_C4_team50_chuseok', adjust(BL))]:
    sub = pd.DataFrame({'example_id': test.example_id, 'boardings': np.expm1(p)})
    assert len(sub) == 157194 and sub.example_id.is_unique and np.isfinite(sub.boardings).all() and sub.boardings.ge(0).all()
    path = CAND / f'submission_{name}.csv'
    sub.to_csv(path, index=False, float_format='%.6f', lineterminator='\n')
    print(f'{path.name} {hashlib.sha256(path.read_bytes()).hexdigest()[:8]}  chuseok mean change {float((p - (C4 if name.startswith("C5") else BL))[chu].mean()):+.3f}'
          f'  rmse vs C3-family C4 {float(np.sqrt(((p - C4) ** 2).mean())):.4f}  rows blended {int((~keep).sum())}')
