"""Where does the fold error of the current recipe come from? (diagnostic only, nothing tuned)
args: OUT ATT_CSV RECIPE   e.g. ../out att.csv att,attwx,att_smc,attwx_smc,att_d4,att_ball"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(sys.argv[1]); games = pd.read_csv(sys.argv[2], parse_dates=['date']).date; R = sys.argv[3].split(',')
KEYS = ['date', 'station_id', 'hour']
allr = []
for o in ['o0227', 'o0331', 'o0430']:
    d = OUT / o; p = pd.read_parquet(d / 'preds.parquet')
    y, ev, V = p.y.to_numpy(), p.ev.eq(1).to_numpy(), p.V.to_numpy()
    s27 = p.station_id.eq(3127).to_numpy()
    game = s27 & pd.to_datetime(p.date).isin(games).to_numpy(); keep_old = s27 & ev & ~game
    def c9(A, E, C, H):
        Em = (np.asarray(E) + np.asarray(C) + np.asarray(H)) / 3; M = .8 * Em + .2 * np.asarray(A)
        B = np.where(ev, M, .8 * M + .2 * V); return np.where(s27, np.where(~ev, A, (np.asarray(A) + Em) / 2), B)
    def load(n):
        a, c = pd.read_parquet(d / f'preds_{n}_AEH.parquet'), pd.read_parquet(d / f'preds_{n}_C.parquet')
        return c9(a.A, a.E, c.C, a.H)
    old = c9(p.A, p.E42, pd.read_parquet(d / 'preds_base_C.parquet').C, p.H)
    S = [load(n) for n in R]
    q = np.maximum(np.where(keep_old, old, np.where(game, np.mean(S, 0), (old + np.sum(S, 0)) / (1 + len(S)))), 0)
    allr.append(p[KEYS + ['y', 'ev', 'offday', 'horizon']].assign(o=o, q=q, r=y - q))
t = pd.concat(allr, ignore_index=True)
t['dow'] = pd.to_datetime(t.date).dt.dayofweek; t['ylev'] = pd.cut(np.expm1(t.y), [-1, 10, 30, 100, 300, 1000, 1e9])
t['hb'] = pd.cut(t.horizon, [0, 14, 30, 60, 90, 130])
for o, g in t.groupby('o'):
    print(o, f'rmse {np.sqrt((g.r**2).mean()):.4f} mean bias(y-q) {g.r.mean():+.4f}')
def tab(key, top=None):
    g = t.groupby(key, observed=True).r
    out = pd.DataFrame({'n': g.size(), 'rmse': np.sqrt(g.apply(lambda x: (x**2).mean())), 'bias': g.mean(),
                        'sse_share': g.apply(lambda x: (x**2).sum()) / (t.r**2).sum()})
    out = out.sort_values('sse_share', ascending=False) if top else out
    print(f'\n== by {key}'); print((out.head(top) if top else out).round(4).to_string())
tab('hour'); tab('offday'); tab('ylev'); tab('hb'); tab('station_id', 12)
# bias per origin x offday, and per origin x month
t['m'] = pd.to_datetime(t.date).dt.month
print('\nbias by origin x month'); print(t.pivot_table(index='o', columns='m', values='r', aggfunc='mean').round(4).to_string())
print('\nbias by origin x offday'); print(t.pivot_table(index='o', columns='offday', values='r', aggfunc='mean').round(4).to_string())
# station-day level vs within-day shape: decompose residual into station-day mean + hourly remainder
sd = t.groupby(['o', 'station_id', 'date']).r.transform('mean')
print(f'\nSSE share: station-day level part {((sd**2).sum() / (t.r**2).sum()):.3f}, within-day shape part {(((t.r - sd)**2).sum() / (t.r**2).sum()):.3f}')
dl = t.groupby(['o', 'date']).r.transform('mean')
print(f'SSE share of network-day level (all stations same day) {((dl**2).sum() / (t.r**2).sum()):.3f}')

# ---- 3127 detail
g27 = t[t.station_id.eq(3127)].copy()
g27['game'] = pd.to_datetime(g27.date).isin(games)
g27['kind'] = np.where(g27.game, 'game', np.where(g27.ev.eq(1), 'other_ev', 'none'))
print('\n3127 by kind x origin: n, rmse, bias, share of ALL-station SSE')
tot = (t.r ** 2).sum()
for (k, o), g in g27.groupby(['kind', 'o']):
    print(f'  {k:8s} {o} n {len(g):5d} rmse {np.sqrt((g.r**2).mean()):.3f} bias {g.r.mean():+.3f} share {(g.r**2).sum()/tot:.4f}')
print('3127 none-days by hour (all origins): rmse / bias')
h = g27[g27.kind.eq('none')].groupby('hour').r.agg(lambda x: np.sqrt((x**2).mean())).round(3)
b = g27[g27.kind.eq('none')].groupby('hour').r.mean().round(3)
print(pd.DataFrame({'rmse': h, 'bias': b}).T.to_string())
print('3127 game-days by hour: rmse / bias')
h = g27[g27.kind.eq('game')].groupby('hour').r.agg(lambda x: np.sqrt((x**2).mean())).round(3)
b = g27[g27.kind.eq('game')].groupby('hour').r.mean().round(3)
print(pd.DataFrame({'rmse': h, 'bias': b}).T.to_string())
