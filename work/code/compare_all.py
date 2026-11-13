"""Score every candidate recipe on the three origins (2/27, 3/31, 4/30); change vs C4 per origin."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(sys.argv[1])
rows = []
for o in ['o0227', 'o0331', 'o0430']:
    p = pd.read_parquet(OUT / o / 'preds.parquet')
    C = (pd.read_parquet(OUT / 'fold_cat_preds.parquet').cat1500 if o == 'o0227' else pd.read_parquet(OUT / o / 'cat.parquet').C).to_numpy()
    assert len(C) == len(p)
    y, ev, A, V = p.y.to_numpy(), p.ev.eq(1).to_numpy(), p.A.to_numpy(), p.V.to_numpy()
    s27 = p.station_id.eq(3127).to_numpy(); E, H = p.E42.to_numpy(), p.H.to_numpy()

    def rec(Em, rule27=False):
        M = .8 * Em + .2 * A
        B = np.where(ev, M, .8 * M + .2 * V)
        B27 = np.where(~ev, A, (A + Em) / 2) if rule27 else (A + Em) / 2
        return np.where(s27, B27, B)

    mixes = {'C4 (LGB)': E, 'LGB+HGB': (E + H) / 2, 'C7 LGB+Cat': (E + C) / 2, 'C8 LGB+Cat+HGB': (E + C + H) / 3}
    for k, Em in mixes.items():
        for rule in (False, True):
            q = rec(Em, rule)
            rows.append(dict(origin=o, recipe=k + (' | 3127 non-event=A' if rule else ''), rmse=float(np.sqrt(((y - q) ** 2).mean()))))
t = pd.DataFrame(rows)
t['vs_C4'] = t.rmse - t.origin.map(t[t.recipe.eq('C4 (LGB)')].set_index('origin').rmse)
w = t.pivot(index='recipe', columns='origin', values='vs_C4')
w['mean'] = w.mean(axis=1); w['worst'] = w[['o0227', 'o0331', 'o0430']].max(axis=1)
print(w.sort_values('mean').round(5).to_string())
print(t[t.recipe.eq('C4 (LGB)')].round(5).to_string(index=False))
