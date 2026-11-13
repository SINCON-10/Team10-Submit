"""Training table with twice-a-month cutoffs (month end + 14th) instead of month end only.
Same DataEngine / preprocess.make_training_set, only cut_freq='SMS'. Passengers after ORIGIN are removed by cube_at (asserted inside).
Check: the month-end rows of the new table must equal the existing table (REF = fold_train.parquet, or train_cutoff_features.csv.gz).
args: BUNDLE DATA ORIGIN REF OUTFILE
"""
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

BUNDLE, DATA = map(Path, sys.argv[1:3]); ORIGIN = pd.Timestamp(sys.argv[3]); REF, OUTF = Path(sys.argv[4]), Path(sys.argv[5])
t0 = time.time(); KEYS = ['date', 'station_id', 'hour']
sys.path.insert(0, str(BUNDLE))
from data_v4 import DataEngine  # noqa: E402

eng = DataEngine(DATA, BUNDLE)
cube, history = eng.cube_at(ORIGIN)
with warnings.catch_warnings():
    warnings.simplefilter('ignore', RuntimeWarning)
    tr = eng.prep.cutoff_table(eng.prep.make_training_set(cube, ORIGIN, cut_freq='SMS'), cube)
assert tr.date.max() <= ORIGIN and (tr.horizon > 0).all() and np.isfinite(tr[['y', 'lvl8']]).all().all()
org = tr.date - pd.to_timedelta(tr.horizon, unit='D')
assert not tr.assign(o=org).duplicated(['o'] + KEYS).any()
ref = pd.read_parquet(REF) if REF.suffix == '.parquet' else pd.read_csv(REF, parse_dates=['date'])
ref['date'] = pd.to_datetime(ref.date)
ro = ref.date - pd.to_timedelta(ref.horizon, unit='D')
sub = tr[org.isin(set(ro.unique()))].reset_index(drop=True)
assert len(sub) == len(ref), (len(sub), len(ref))
k = ['date', 'station_id', 'horizon']
a = sub.sort_values(k + ['hour']).reset_index(drop=True); b = ref.sort_values(k + ['hour']).reset_index(drop=True)
cols = [c for c in b.columns if c in a.columns]
assert len(cols) >= 56, cols
bad = [c for c in cols if not np.allclose(pd.to_numeric(a[c], errors='coerce').to_numpy(float),
                                          pd.to_numeric(b[c], errors='coerce').to_numpy(float), equal_nan=True, atol=1e-5)
       and not (a[c].astype(str) == b[c].astype(str)).all()]
assert not bad, bad
tr.to_parquet(OUTF, index=False)
print(f'{ORIGIN.date()} SMS train {len(tr):,} rows ({len(tr)/len(ref):.2f}x), cutoffs {org.nunique()} (was {ro.nunique()}), '
      f'month-end rows identical to reference, {time.time()-t0:.0f}s', flush=True)
