"""Final file checks for candidate submissions: 157,194 rows, example_id identical to test.csv (same order, no duplicates),
no NaN/inf/negative boardings, header exactly example_id,boardings; full SHA256. args: RAW_TEST CSV [CSV ...]"""
import hashlib, sys
import numpy as np, pandas as pd
test = pd.read_csv(sys.argv[1], usecols=['example_id'])
for f in sys.argv[2:]:
    b = open(f, 'rb').read(); s = pd.read_csv(f)
    ok = {'rows': len(s) == 157194, 'cols': list(s.columns) == ['example_id', 'boardings'],
          'ids': s.example_id.tolist() == test.example_id.tolist(), 'unique': s.example_id.is_unique,
          'finite': bool(np.isfinite(s.boardings).all()), 'nonneg': bool(s.boardings.ge(0).all())}
    print(f"{'PASS' if all(ok.values()) else 'FAIL'} {f.split('/')[-1]} sha256 {hashlib.sha256(b).hexdigest()} "
          f"| rows {len(s):,} mean {s.boardings.mean():.3f}" + ('' if all(ok.values()) else f' | {ok}'))
