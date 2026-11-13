"""Tree-count curves on the 2/27 fold (att set): single-model RMSLE after k trees on the selection window 2/28-3/31 and on the
full 123 days (3110/3111 excluded: curve runs drop the break rows). Choose k on the selection window only. args: OUT"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
d = Path(sys.argv[1]) / 'o0227'; p = pd.read_parquet(d / 'preds.parquet'); y = p.y.to_numpy(); dt = pd.to_datetime(p.date)
keep = ~p.station_id.isin([3110, 3111]).to_numpy(); early = keep & (dt < '2026-04-01').to_numpy(); late = keep & ~early
for f, m in [('preds_att_ecurve_E.parquet', 'E'), ('preds_att_acurve_A.parquet', 'A'), ('preds_att_hcurve_H.parquet', 'H')]:
    if not (d / f).exists():
        print(f, 'not yet'); continue
    c = pd.read_parquet(d / f); assert (c.station_id.to_numpy() == p.station_id.to_numpy()).all()
    ks = sorted(int(x.split('_')[1]) for x in c.columns if x.startswith(m + '_'))
    rows = []
    for k in ks:
        v = c[f'{m}_{k}'].to_numpy(); rows.append((k, *(np.sqrt(((y[r] - v[r]) ** 2).mean()) for r in (early, late, keep))))
    t = pd.DataFrame(rows, columns=['k', 'S 2/28-3/31', '4/1-6/30 (same fold)', 'full 123d']).set_index('k').round(5)
    print(f'{m} curve (last col {m} = {c[m].shape[0]} rows):'); print(t.to_string()); print('  best k on the selection window:', t['S 2/28-3/31'].idxmin())
