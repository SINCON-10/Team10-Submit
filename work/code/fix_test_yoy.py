"""Test features with the partner-year (2025) comparison repaired for stations hit by the 2025-06-28 Geomdan opening.

Why: the three new Line-1 stations (3107-3109) opened 2025-06-28 and took riders from the neighbouring stations 계양(3110) and
귤현(3111): their weekday level fell -0.55 / -0.25 log from June to July 2025 beyond the all-station seasonal change (-0.05),
and 2026 levels stayed at the new level. yoy_delta / yoy_raw compare the 2025 target-season window (Jul-Oct, after the
opening) with the 2025 cutoff window (May-Jun, before it), so for the test period they carry that one-off drop again
(3110 at 08h: yoy_delta -0.90 vs all-station -0.13). Training never contains a partner window in Jul-Oct 2025, and the
validation origins (Feb-Apr 2026) are not affected, so this cannot show up in validation.
Fix: for those two stations only, use the team's own all-station versions (yoy_delta_all, yoy_raw_all) of the same features.
No ridership after 2026-06-30 is used.
args: PV4 OUTFILE
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

PV4, OUT = Path(sys.argv[1]), Path(sys.argv[2])
STATIONS = [3110, 3111]
te = pd.read_csv(PV4 / 'test_cutoff_features.csv.gz')
orig = te.copy()
m = te.station_id.isin(STATIONS)
te.loc[m, 'yoy_delta'] = te.loc[m, 'yoy_delta_all']
te.loc[m, 'yoy_raw'] = te.loc[m, 'yoy_raw_all']
changed = [c for c in te.columns if not te[c].equals(orig[c])]
assert set(changed) <= {'yoy_delta', 'yoy_raw'}, changed
assert te.loc[~m].equals(orig.loc[~m]) and m.sum() == 2 * 123 * 18
te.to_csv(OUT, index=False, compression='gzip')
d = (te.yoy_delta - orig.yoy_delta)[m]
print(f'{OUT.name}: {int(m.sum())} rows of {STATIONS} changed (yoy_delta, yoy_raw); mean change in yoy_delta '
      + ', '.join(f'{s}: {d[te.station_id[m].eq(s)].mean():+.3f}' for s in STATIONS))
