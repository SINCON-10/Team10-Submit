import sys, numpy as np, pandas as pd
RAW = __import__('os').environ['RAW']   # package: path from paths.sh
tr = pd.read_csv(f'{RAW}/train.csv', usecols=['date', 'station_id', 'hour', 'boardings'], parse_dates=['date'])
first = tr[tr.boardings > 0].groupby('station_id').date.min()
print('stations whose first ridership is after 2025-03-01:'); print(first[first > '2025-03-01'].to_string())
new = first[first > '2025-03-01'].index.tolist()
d = tr[tr.station_id.isin(new)].groupby(['station_id', 'date']).boardings.sum().reset_index()
d['wd'] = d.date.dt.dayofweek < 5
m = d[d.wd].groupby(['station_id', d.date.dt.to_period('M')]).boardings.mean().unstack(0).round(0)
print('weekday mean daily boardings by month:'); print(m.to_string())
