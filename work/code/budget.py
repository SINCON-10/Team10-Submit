"""Error budget on the Mar-Jun 2026 fold: where is the squared error, and what would removing each part buy?"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

df = pd.read_parquet(Path(sys.argv[1]) / 'preds.parquet')
P = sys.argv[2] if len(sys.argv) > 2 else 'pred_gbm_orev'
df['e'] = df.y - df[P]
df['new3'] = df.station_id.isin([3107, 3108, 3109])
df['month'] = df.date.dt.month
df['hz'] = ((df.date - pd.Timestamp('2026-02-27')).dt.days - 1) // 30
mse = float((df.e ** 2).mean()); base = np.sqrt(mse)
print(f'model {P}: RMSLE {base:.5f}  (mean bias {df.e.mean():+.4f})')


def without(mask=None, sub=None):
    e = df.e.copy()
    if sub is not None:
        e = e - df.groupby(sub).e.transform('mean')
    if mask is not None:
        e = e.where(~mask, 0.)
    return np.sqrt((e ** 2).mean())


print('\n[A] 오라클: 이 부분을 완벽히 맞히면 RMSLE (개선폭)')
rows = [
    ('전체 평균 편향 제거', without(sub=np.zeros(len(df)))),
    ('역별 편향(123일 평균) 제거', without(sub='station_id')),
    ('역×시간대 편향 제거', without(sub=['station_id', 'hour'])),
    ('날짜별 편향(전 역 공통) 제거', without(sub='date')),
    ('날짜×시간대 편향(전 역 공통) 제거', without(sub=['date', 'hour'])),
    ('역×월 편향 제거', without(sub=['station_id', 'month'])),
    ('역-일 편향(하루 평균) 제거', without(sub=['station_id', 'date'])),
    ('행사 역-일 완벽', without(mask=df.diagnostic_event.eq(1))),
    ('공휴일 완벽', without(mask=df.is_hol.eq(1))),
    ('비근무일(주말+휴일) 완벽', without(mask=df.offday.eq(1))),
    ('신규 3역 완벽', without(mask=df.new3)),
    ('06시·23시 완벽', without(mask=df.hour.isin([6, 23]))),
]
for name, v in rows:
    print(f'  {name:28s} {v:.5f}  ({v-base:+.5f})')

print('\n[B] 제곱오차 점유율 (셀 비율 → 오차 비율, 그룹 RMSLE)')
groups = [('행사 역-일', df.diagnostic_event.eq(1)), ('공휴일', df.is_hol.eq(1)), ('주말', df.dow.ge(5) & df.is_hol.eq(0)),
          ('신규 3역', df.new3), ('06시', df.hour.eq(6)), ('23시', df.hour.eq(23))]
groups += [(f'{m}월', df.month.eq(m)) for m in sorted(df.month.unique())]
groups += [(f'horizon {30*h}-{30*h+29}일', df.hz.eq(h)) for h in sorted(df.hz.unique())]
for name, m in groups:
    share = float((df.e[m] ** 2).sum() / (df.e ** 2).sum())
    print(f'  {name:18s} 셀 {m.mean():6.1%} → 오차 {share:6.1%}   RMSLE {np.sqrt((df.e[m]**2).mean()):.4f}  편향 {df.e[m].mean():+.4f}')

print('\n[C] 오차 큰 역 10 (점유율, RMSLE, 123일 평균 편향)')
st = df.groupby('station_id').e.agg(sse=lambda s: (s ** 2).sum(), rmsle=lambda s: np.sqrt((s ** 2).mean()), bias='mean')
st['share'] = st.sse / st.sse.sum()
print(st.sort_values('sse', ascending=False).head(10)[['share', 'rmsle', 'bias']].round(4).to_string())

print('\n[D] 오차 큰 날짜 10 (전 역 평균 편향)')
dd = df.groupby('date').e.agg(sse=lambda s: (s ** 2).sum(), bias='mean', rmsle=lambda s: np.sqrt((s ** 2).mean()))
dd['share'] = dd.sse / dd.sse.sum()
dd['dow'] = [('월화수목금토일')[d.dayofweek] for d in dd.index]
print(dd.sort_values('sse', ascending=False).head(10)[['dow', 'share', 'rmsle', 'bias']].round(4).to_string())

print('\n[E] 월별 전 역 편향 추이 (주 단위)')
wk = df.groupby(df.date.dt.to_period('W')).e.mean()
print(' '.join(f'{str(k)[5:10]}:{v:+.3f}' for k, v in wk.items()))

if {'pred_gbm_orev', 'base_v2'} <= set(df):
    print('\n[F] LightGBM + v2 구조 블렌드 (log 공간)')
    for w in [0, .2, .3, .4, .5]:
        p = (1 - w) * df.pred_gbm_orev + w * df.base_v2
        print(f'  v2 {w:.1f}: {np.sqrt(((df.y - p) ** 2).mean()):.5f}')
    print(f'  오차 상관 {np.corrcoef(df.y-df.pred_gbm_orev, df.y-df.base_v2)[0,1]:.3f}')
