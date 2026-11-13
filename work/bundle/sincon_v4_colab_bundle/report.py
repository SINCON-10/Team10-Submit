"""Small standalone HTML report; no large chart/render dependencies required."""
from pathlib import Path
import html
import json

import numpy as np
import pandas as pd

from common import read_json


def curve_svg(frame):
    frame = frame.dropna(subset=['step', 'validation_rmsle'])
    if not len(frame):
        return ''
    x = frame.step.to_numpy(dtype=float)
    y = frame.validation_rmsle.to_numpy(dtype=float)
    px = 45+(x-x.min())/max(float(np.ptp(x)), 1)*600
    py = 165-(y-y.min())/max(float(np.ptp(y)), .0001)*130
    points = ' '.join(f'{a:.1f},{b:.1f}' for a, b in zip(px, py))
    j = int(np.argmin(y))
    return f'''<svg viewBox="0 0 700 205" role="img" aria-label="검증 RMSLE 학습 곡선">
      <path d="M45 25 V165 H655" fill="none" stroke="#94a3b8"/>
      <polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2"/>
      <circle cx="{px[j]:.1f}" cy="{py[j]:.1f}" r="4" fill="#ef4444"/>
      <text x="45" y="195">학습량 {x.min():g} → {x.max():g}</text>
      <text x="350" y="195">최저 RMSLE {y[j]:.5f}, 학습량 {x[j]:g}</text></svg>'''


def write_report(output):
    output = Path(output)
    sections = []
    for split in ['calibration', 'evaluation']:
        path = output/split/'scores.csv'
        if not path.is_file():
            continue
        score = pd.read_csv(path)
        rank = score.loc[score.group.eq('전체'), ['model', 'rmsle', 'rows']].sort_values('rmsle')
        group = score.pivot(index='model', columns='group', values='rmsle').reindex(rank.model)
        sections.append(f'<h2>{split} · 123일</h2>'+rank.to_html(index=False, float_format=lambda x:f'{x:.6f}')+
                        '<details><summary>행사·공휴일·신규 역·월별 점수</summary>'+group.to_html(float_format=lambda x:f'{x:.6f}')+'</details>')
    selections = []
    for path in output.glob('search/*/*/selection.json'):
        value = read_json(path)
        selections.append(dict(period=path.parents[1].name, model=path.parent.name,
            parameters=json.dumps(value['parameters'], ensure_ascii=False), amount=value['amount'],
            rmsle=value['rmsle'], hit_cap=value['recipe']['hit_cap']))
    if selections:
        table = pd.DataFrame(selections)
        table.to_csv(output/'selected_settings.csv', index=False, encoding='utf-8-sig')
        sections.append('<h2>선택된 파라미터와 학습량</h2>'+table.to_html(index=False))
    for path in sorted(output.glob('search/*/*/best_model/learning_curve.csv')):
        sections.append('<details><summary>'+html.escape(str(path.relative_to(output)))+'</summary>'+
                        curve_svg(pd.read_csv(path))+'</details>')
    if (output/'ensemble_search/period.json').is_file():
        sections.append('<details><summary>앙상블 내부 학습·점검 범위 (최종 성능 아님)</summary><pre>'+
                        html.escape((output/'ensemble_search/period.json').read_text(encoding='utf-8'))+'</pre></details>')
    for name in ['BEST_CONFIGURATION.json', 'BEST_MODEL.json', 'leakage_checks.json', 'submission_checks.json', 'replay_check.json']:
        if (output/name).is_file():
            sections.append(f'<details><summary>{name}</summary><pre>'+html.escape((output/name).read_text(encoding='utf-8'))+'</pre></details>')
    text = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>SINCON v4 실험 결과</title>
    <style>body{font:15px/1.7 system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 22px;color:#172033;background:#f8fafc}
    h1,h2{color:#0f172a}table{border-collapse:collapse;width:100%;font-size:13px;background:white;margin:16px 0}
    th,td{padding:8px;border-bottom:1px solid #dbe3ef;text-align:left}th{background:#e8eef8}details{background:white;padding:12px;margin:12px 0;overflow:auto}
    pre{white-space:pre-wrap;overflow-wrap:anywhere}svg{max-width:760px;width:100%;font:13px sans-serif}.note{padding:16px;background:#e8eef8;border-radius:10px}</style>
    <h1>SINCON v4 모델 실험실</h1><p class="note">최종 평가는 2026-02-28~06-30의 123일 RMSLE입니다. 작을수록 좋습니다.
    첫 모델부터 2025-10-28~2026-02-27 중간검증으로 파라미터·학습량을 선택합니다.
    개통 전 검증과 초기 200회 고정 단계는 없습니다. 앙상블은 같은 중간검증 예측의 앞 82일로 학습하고
    뒤 41일로 설정·학습량을 고른 뒤 중간검증 전체로 재학습합니다. 중간검증은 설정 선택에 재사용되는 개발 자료입니다.
    최종 123일 점수로 고정된 후보 모델을 비교해 최고 모델을 저장합니다. 이 정답으로 가중치나 학습량을 다시 맞추지 않습니다.</p>
    <p>표 모델·FT-Transformer는 공통 55개 v4 특징을 사용합니다. 구조 모듈은 구조 기준선을 추가하고,
    N-HiTS/TFT/TiDE는 같은 v4 원천의 일별 역-시간 시계열과 달력·날씨·정적 특징을 사용합니다.
    과거 행사 정답 기반 표시는 진단 집계에만 사용합니다. 과거 일정표가 없어 해당 입력은 알려지지 않은 상태입니다.</p>
    <p>상한 도달은 수렴을 보장하지 않습니다. 학습 곡선이 계속 내려가면 상한을 늘리는 새 실험을 만드세요.
    시드 42를 고정하지만 서로 다른 GPU·패키지 환경에서 비트 단위 일치를 보장하지 않습니다.</p>'''
    path = output/'report.html'
    path.write_text(text+''.join(sections)+'</html>', encoding='utf-8')
    return path
