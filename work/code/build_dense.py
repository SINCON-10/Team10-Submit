"""Training table with extra cutoffs inside each month (default the 7th, 14th and 21st) on top of the month ends.
Same rows as preprocess.make_training_set (2025-01-15 + month ends, each followed by up to 123 days <= ORIGIN),
plus the same build_rows call from each extra cutoff. Passengers after ORIGIN are removed by cube_at (asserted inside).
DAYS=14 reproduces build_smc.py (twice a month).
Check (memory-light): row count of the month-end part equals REF, and two month-end cutoffs match REF row by row.
args: BUNDLE DATA ORIGIN REF OUTFILE [DAYS]
"""
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

BUNDLE, DATA = map(Path, sys.argv[1:3]); ORIGIN = pd.Timestamp(sys.argv[3]); REF, OUTF = Path(sys.argv[4]), Path(sys.argv[5])
DAYS = [int(x) for x in (sys.argv[6] if len(sys.argv) > 6 else '7,14,21').split(',')]
t0 = time.time(); KEYS = ['date', 'station_id', 'hour']; ONE = pd.Timedelta(days=1)
sys.path.insert(0, str(BUNDLE))
from data_v4 import DataEngine  # noqa: E402

eng = DataEngine(DATA, BUNDLE); P = eng.prep
cube, history = eng.cube_at(ORIGIN)
base = [pd.Timestamp('2025-01-15')] + [c for c in pd.date_range('2025-03-01', ORIGIN, freq='MS') - ONE if c < ORIGIN]
extra = [m + pd.Timedelta(days=d - 1) for m in pd.date_range('2025-03-01', ORIGIN, freq='MS') for d in DAYS]
extra = [c for c in extra if c < ORIGIN and c not in set(base)]
cuts = sorted(set(base) | set(extra))
parts = []
with warnings.catch_warnings():
    warnings.simplefilter('ignore', RuntimeWarning)
    for cu in cuts:
        t1 = min(cu + pd.Timedelta(days=P.HORIZON), ORIGIN)
        parts.append(P.build_rows(cube, cu, cu + ONE, t1, require_target=True, avail_end=ORIGIN))
    tr = P.cutoff_table(pd.concat(parts, ignore_index=True), cube)
del parts
assert tr.date.max() <= ORIGIN and (tr.horizon > 0).all() and np.isfinite(tr[['y', 'lvl8']]).all().all()
org = tr.date - pd.to_timedelta(tr.horizon, unit='D')
ref = pd.read_parquet(REF) if REF.suffix == '.parquet' else pd.read_csv(REF, parse_dates=['date'])
ref['date'] = pd.to_datetime(ref.date)
ro = ref.date - pd.to_timedelta(ref.horizon, unit='D')
assert set(ro.unique()) <= set(org.unique())
assert int(org.isin(set(ro.unique())).sum()) == len(ref)
k = KEYS + ['horizon']
for c in sorted(ro.unique())[-2:]:
    a = tr[org.eq(c)].sort_values(k).reset_index(drop=True); b = ref[ro.eq(c)].sort_values(k).reset_index(drop=True)
    cols = [x for x in b.columns if x in a.columns and x != 'date']
    assert len(a) == len(b) and len(cols) >= 55
    bad = [x for x in cols if not np.allclose(pd.to_numeric(a[x], errors='coerce').to_numpy(float),
                                              pd.to_numeric(b[x], errors='coerce').to_numpy(float), equal_nan=True, atol=1e-5)]
    assert not bad, (c, bad)
del ref
tr.to_parquet(OUTF, index=False)
print(f'{ORIGIN.date()} days {DAYS}: train {len(tr):,} rows, cutoffs {org.nunique()}, month-end part matches reference, {time.time()-t0:.0f}s', flush=True)
