import numpy as np, pandas as pd
S = __import__('os').environ['S'];   # package: path from paths.sh; CAND = f'{S}/claude_works/candidates'
te = pd.read_csv(f'{S}/data/processed_v4/test_cutoff_features.csv.gz')
print([c for c in te.columns][:80])
new = [3107, 3108, 3109]
t = te[te.station_id.isin(new)].copy(); t['m'] = t.date.str[:7]
cols = [c for c in te.columns if c.startswith(('lvl', 'yoy', 'p_', 'part'))][:30]
print('feature means by station/month:'); print(t.groupby(['station_id', 'm'])[cols].mean().round(3).T.to_string())
test = pd.read_csv(f'{S}/data/raw/test.csv', parse_dates=['date'])
for name in ['C19_add_ball_smc', 'C20w_c19_foldweights', 'C22f_c21b_evH']:
    s = pd.read_csv(f'{CAND}/submission_{name}.csv'); x = test.assign(b=s.boardings.to_numpy())
    x = x[x.station_id.isin(new) & (x.date.dt.dayofweek < 5)]
    print(name, 'weekday mean daily boardings:'); print(x.groupby(['station_id', x.date.dt.to_period('M')]).b.sum().div(x.groupby(['station_id', x.date.dt.to_period('M')]).date.nunique()).unstack(0).round(0).to_string())
