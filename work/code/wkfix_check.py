"""Weekend/holiday level correction, checked time-ordered (diagnostic + selection of the slopes on the 2/28-3/31 window of o0227).
Idea: for Sat / Sun-holiday rows the 8-week level (lvl8) is a mean of ~8 days and the partner-year seasonal change (yoy_delta) a mean
of 2-3 days, so both are noisy. Long-run alternatives from data up to the origin only:
  d1 = (weekday 8-week level + 52-week [weekend - weekday] gap) - weekend 8-week level
  d2 = partner-year seasonal change of weekdays - partner-year seasonal change of the row's own day type (same +-7-day window)
Correction: p += b1*d1 + b2*d2 on ordinary weekend/holiday rows (not event, not 3107-3111, not 3127).
Recipe = C25b-style (E3k all sets, A4k where the fold file exists, 3127 keeps the 500-tree A). args: OUT ATT_CSV RAW PV4"""
import sys, io, contextlib
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(sys.argv[1]); games = pd.read_csv(sys.argv[2], parse_dates=['date']).date; RAW = Path(sys.argv[3]); sys.path.insert(0, sys.argv[4]); sys.dont_write_bytecode = True
import preprocess as P
train = pd.read_csv(RAW / 'train.csv', parse_dates=['date']); test = pd.read_csv(RAW / 'test.csv', parse_dates=['date']); meta = pd.read_csv(RAW / 'metaData.csv')
SETS = 'att,attwx,att_smc,attwx_smc,att_d4,att_ball,att_ball_smc'.split(',')
SRC = {'att': 'att', 'att_ball': 'att', 'att_smc': 'att_smc', 'att_ball_smc': 'att_smc', 'attwx': 'attwx', 'attwx_smc': 'attwx_smc', 'att_d4': 'att_d4'}
EXCL = [3107, 3108, 3109, 3110, 3111, 3127]


def corr_terms(cube, cal, c, a_end, dates, sid, hour):
    """d1, d2 for rows (dates, sid, hour); uses cube data up to index a_end only (c = origin index)."""
    L8, _ = cube.level(c, 8); L52, _ = cube.level(c, 52)
    k = cal.set_index('date').loc[dates, 'dt3'].to_numpy(); si = pd.Series(sid).map(cube.sidx).to_numpy(); hi = np.asarray(hour) - 6
    d1 = (L8[0][si, hi] + (L52[k, si, hi] - L52[0][si, hi])) - L8[k, si, hi]
    d2 = np.full(len(k), np.nan); di = np.asarray((pd.DatetimeIndex(dates) - P.START).days)
    cp = c - 364
    if cp - 55 >= 0:
        L8p, _ = cube.level(cp, 8)
        for d in np.unique(di):
            dp = d - 364
            if dp < 0 or dp + 7 > a_end: continue
            r = di == d; kk = int(k[r][0])
            if kk == 0: continue
            own = cube.window_mean(dp, 7, kk, hi_limit=a_end) - L8p[kk]; wk = cube.window_mean(dp, 7, 0, hi_limit=a_end) - L8p[0]
            d2[r] = (wk - own)[si[r], hi[r]]
    return k, d1, d2


if __name__ == '__main__':
    res = {}
    for o, od in [('o0227', '2026-02-27'), ('o0331', '2026-03-31'), ('o0430', '2026-04-30')]:
        with contextlib.redirect_stdout(io.StringIO()):
            cal = P.build_calendar(); cube = P.Cube(train[train.date <= od], test, meta, cal)
        c = (pd.Timestamp(od) - P.START).days
        d = OUT / o; p = pd.read_parquet(d / 'preds.parquet')
        y, ev, V = p.y.to_numpy(), p.ev.eq(1).to_numpy(), p.V.to_numpy(); dt = pd.to_datetime(p.date)
        s27 = p.station_id.eq(3127).to_numpy(); game = s27 & dt.isin(games).to_numpy(); keep_old = s27 & ev & ~game
        def c9(A, E, C, H):
            Em = (E + C + H) / 3; M = .8 * Em + .2 * A; B = np.where(ev, M, .8 * M + .2 * V)
            return np.where(s27, np.where(~ev, A, (A + Em) / 2), B)
        old = c9(p.A.to_numpy(), p.E42.to_numpy(), pd.read_parquet(d / 'preds_base_C.parquet').C.to_numpy(), p.H.to_numpy())
        Sx = []
        for s in SETS:
            a, cc = pd.read_parquet(d / f'preds_{s}_AEH.parquet'), pd.read_parquet(d / f'preds_{s}_C.parquet'); A, E = a.A.to_numpy(), a.E.to_numpy()
            f = d / f'preds_{s}_e3k_E.parquet'
            if f.exists(): E = pd.read_parquet(f).E.to_numpy()
            f = d / f'preds_{SRC[s]}_a4k_A.parquet'
            A1 = pd.read_parquet(d / 'preds_att_acurve_A.parquet').A_4000.to_numpy() if (o == 'o0227' and SRC[s] == 'att') else (pd.read_parquet(f).A.to_numpy() if f.exists() else None)
            if A1 is not None: A = np.where(s27, A, A1)
            Sx.append(c9(A, E, cc.C.to_numpy(), a.H.to_numpy()))
        q = np.maximum(np.where(keep_old, old, np.where(game, np.mean(Sx, 0), (old + np.sum(Sx, 0)) / (1 + len(Sx)))), 0)
        k, d1, d2 = corr_terms(cube, cal, c, c, dt, p.station_id.to_numpy(), p.hour.to_numpy())
        m = ~ev & ~p.station_id.isin(EXCL).to_numpy() & np.isin(k, [1, 2]) & np.isfinite(d1)
        res[o] = dict(r=y - q, d1=np.where(m, d1, 0.), d2=np.where(m & np.isfinite(d2), d2, 0.), m=m, early=(dt < '2026-04-01').to_numpy(), keep=~p.station_id.isin([3110, 3111]).to_numpy(), k=k)
        print(f'{o}: weekend/holiday ordinary rows {m.sum():,} | sd d1 {d1[m].std():.3f} | d2 available {np.isfinite(d2[m]).mean():.0%}, sd {np.nanstd(d2[m]):.3f}, corr(d1,d2) {np.corrcoef(d1[m & np.isfinite(d2)], d2[m & np.isfinite(d2)])[0, 1]:+.2f}')
    R = res['o0227']; s = R['m'] & R['early']
    X = np.column_stack([R['d1'][s], R['d2'][s]]); b12 = np.linalg.lstsq(X, R['r'][s], rcond=None)[0]
    b1 = float((R['d1'][s] * R['r'][s]).sum() / (R['d1'][s] ** 2).sum()); b2 = float((R['d2'][s] * R['r'][s]).sum() / (R['d2'][s] ** 2).sum())
    print(f'slopes fitted on 2/28-3/31 of o0227 only ({s.sum():,} rows): d1 alone {b1:+.3f} | d2 alone {b2:+.3f} | joint d1 {b12[0]:+.3f}, d2 {b12[1]:+.3f}')
    for o, R in res.items():
        ev_rows = R['keep'] & ~R['early'] if o == 'o0227' else R['keep']
        base = np.sqrt((R['r'][ev_rows] ** 2).mean()); out = [f'{o} {"4/1-6/30" if o == "o0227" else "all days"}: base {base:.6f}']
        for lab, (a1, a2) in {'d1': (b1, 0), 'd2': (0, b2), 'joint': tuple(b12), 'd1=.15': (.15, 0), 'd1=.2': (.2, 0)}.items():
            out.append(f'{lab} {np.sqrt(((R["r"] - a1 * R["d1"] - a2 * R["d2"])[ev_rows] ** 2).mean()) - base:+.6f}')
        print(' | '.join(out))
