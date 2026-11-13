"""C27 = an existing candidate + the weekend/holiday level correction of wkfix_check.py (slopes fitted on the 2/28-3/31 window of
the 2/27 fold only; later windows of all three folds improve). p += B1*d1 + B2*d2 (log) on ordinary Sat / Sun-holiday rows:
not event rows (final_att_ev.npy), not stations 3107-3111 / 3127, not the lunar core days (is_major; lunar_adjust owns them).
d1, d2 use train.csv up to 2026-06-30 only (52-week and partner-year 2025 levels). Nothing from Public.
args: RAW PV4 CAND OUT IN_NAME OUT_NAME B1 B2 [B3 = slope of the weekday hour term] [B4 = slope per daytime rain hour (10-19h, weather_incheon.csv) on the weekend/holiday rows]"""
import hashlib, io, contextlib, sys
from pathlib import Path
import numpy as np, pandas as pd
RAW, PV4, CAND, OUT = map(Path, sys.argv[1:5]); IN, NAME = sys.argv[5], sys.argv[6]; B1, B2 = float(sys.argv[7]), float(sys.argv[8]); B3 = float(sys.argv[9]) if len(sys.argv) > 9 else 0.; B4 = float(sys.argv[10]) if len(sys.argv) > 10 else 0.
sys.argv = [sys.argv[0], str(OUT), str(PV4.parent.parent / 'claude_works/external/ssg_munhak_attendance.csv'), str(RAW), str(PV4)]
from wkfix_check import corr_terms, EXCL, P   # noqa: E402
train = pd.read_csv(RAW / 'train.csv', parse_dates=['date']); test = pd.read_csv(RAW / 'test.csv', parse_dates=['date']); meta = pd.read_csv(RAW / 'metaData.csv')
assert train.date.max() == pd.Timestamp('2026-06-30') and len(test) == 157194
sub = pd.read_csv(CAND / f'submission_{IN}.csv'); assert sub.example_id.tolist() == test.example_id.tolist()
with contextlib.redirect_stdout(io.StringIO()):
    cal = P.build_calendar(); cube = P.Cube(train, test, meta, cal)
c = cube.n_train_days - 1; assert not np.isfinite(cube.Y[c + 1:]).any()
k, d1, d2 = corr_terms(cube, cal, c, c, pd.DatetimeIndex(test.date), test.station_id.to_numpy(), test.hour.to_numpy())
ev = np.load(OUT / 'final_att_ev.npy').astype(bool); major = cal.set_index('date').loc[test.date, 'is_major'].to_numpy() == 1
m = ~ev & ~test.station_id.isin(EXCL).to_numpy() & np.isin(k, [1, 2]) & np.isfinite(d1) & ~major
adj = np.where(m, B1 * d1, 0.) + np.where(m & np.isfinite(d2), B2 * d2, 0.)
if B3 or B4:   # weekday ordinary rows: previous hour's 8-week level + 52-week gap between the two hours - own 8-week level (nfix_check.py term hr52)
    from nfix_check import terms
    k3, T = terms(cube, cal, c, c, pd.DatetimeIndex(test.date), test.station_id.to_numpy(), test.hour.to_numpy()); assert (k3 == k).all()
    mw = ~ev & ~test.station_id.isin(EXCL).to_numpy() & (k == 0) & np.isfinite(T['hr52'])
    adj = adj + np.where(mw, B3 * T['hr52'], 0.); m = m | (mw & (B3 != 0)); print(f'  weekday hr52 term: B3 {B3}, rows {int(mw.sum()):,}, adjustment sd {(B3 * T["hr52"][mw]).std():.4f}, mean {(B3 * T["hr52"][mw]).mean():+.4f}')
if B4:   # weekend/holiday ordinary rows: fewer riders per hour of daytime rain than the models give (nfix_check.py term rainmid)
    mwe = ~ev & ~test.station_id.isin(EXCL).to_numpy() & np.isin(k, [1, 2]) & np.isfinite(d1) & ~major
    adj = adj + np.where(mwe, B4 * T['rainmid'], 0.); dd_ = test[mwe].assign(h=T['rainmid'][mwe]).groupby('date').h.first()
    print(f'  weekend rain term: B4 {B4}, weekend/holiday days {len(dd_)}, with daytime rain {int((dd_ > 0).sum())}, mean rain hours on those {dd_[dd_ > 0].mean():.1f}, max {dd_.max():.0f}, largest shift {B4 * dd_.max():+.3f}')
p = np.log1p(sub.boardings.to_numpy()); pn = np.maximum(p + adj, 0)
out = pd.DataFrame({'example_id': test.example_id, 'boardings': np.expm1(pn)})
assert len(out) == 157194 and out.example_id.is_unique and np.isfinite(out.boardings).all() and out.boardings.ge(0).all()
path = CAND / f'submission_{NAME}.csv'; out.to_csv(path, index=False, float_format='%.6f', lineterminator=chr(10))
t = test.assign(a=adj, m=m, mon=test.date.dt.strftime('%m'), k=k)
print(path.name, hashlib.sha256(path.read_bytes()).hexdigest()[:8], f'| base {IN} | B1 {B1} B2 {B2} | rows changed {int(m.sum()):,} ({m.mean():.1%}), d2 missing {int((m & ~np.isfinite(d2)).sum()):,}')
print(f'  adjustment on changed rows: mean {adj[m].mean():+.4f}, sd {adj[m].std():.4f}, min {adj[m].min():+.3f}, max {adj[m].max():+.3f} | sd d1 {d1[m].std():.3f}, sd d2 {np.nanstd(d2[m]):.3f}')
print('  mean adjustment by month / day type (1 Sat, 2 Sun-holiday):'); print(t[t.m].groupby(['k', 'mon']).a.mean().unstack().round(4).to_string())
g = t[t.m].groupby('station_id').a.mean(); print('  stations with the largest mean adjustment:', g.abs().sort_values().tail(6).index.map(lambda s: f'{s} {g[s]:+.3f}').tolist())
