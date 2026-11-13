# 자료 출처와 생성 기록

이 문서는 사용한 자료를 모두 적습니다. 순서는 제공 데이터 → 외부 데이터 → 파생 데이터 → 캐시 → 수동 설정값입니다. 모든 파일의 SHA-256은 `provenance/MANIFEST.sha256`에 있습니다.

## 1. 대회 제공 데이터 (변형 없음, 패키지에 넣지 않음)

| 파일 | SHA-256 | 사용 |
|---|---|---|
| `train.csv` | `1b3b21e8f1fe22a44c0ff4adba315784a5c21caea25098a15869c4ed370822bc` | 사용 (2025-01-01 ~ 2026-06-30 승차) |
| `test.csv` | `d257fed268db7919fd8fbc45b6c78dfbdb09b6bb40d4ae1d3f2a666c6e69fefa` | 사용 (예측 대상 행, example_id 순서) |
| `metaData.csv` | `9870dc6c9cbe7e6fc810bbe589dd9e06f309bb173a268a1bd63478f430785380` | 사용 (역 노선 정보) |
| `sample_submission.csv` | `8c5e10d4201460ddff2d78bf10a229ec8b604e86ae2f18bcd96969093981c93d` | 해당 없음 (형식 참고만) |

`data/raw/`에 넣으면 `run_quick.sh`가 해시를 확인합니다.

## 2. 외부 데이터

| 자료 | 파일 | 출처 | 조회일 | 라이선스 | 기간과 내용 | 쓰는 곳 |
|---|---|---|---|---|---|---|
| 날씨 실측 | `data/processed_v4/weather_incheon.csv` | Open-Meteo Historical Weather API (ERA5 재분석), 인천 37.46N 126.70E | 2026-11-12 | CC BY 4.0 | 2025-01-01 ~ 2026-10-31 매시 6~23시: 기온, 체감온도, 강수, 적설, 풍속 | 팀 전처리 `w_*` 특징, `wxfeat.py` 기준일 날씨 특징 |
| SSG 홈경기 관중 수 | `claude_works/external/ssg_munhak_attendance.csv` | KBO 공식 홈페이지 기록실 '일자별 관중 현황' (정규시즌, 구장 문학) | 2026-11-12 | 별도 표기 없음 (공개 통계를 날짜별로 옮겨 적음) | 2025년 73경기, 2026년 71경기 (날짜, 홈, 원정, 관중) | 특징 `att` (3127 경기일만) |
| 공휴일 | `preprocess.py` 안 표 | 관공서의 공휴일에 관한 규정, 2026 제헌절·노동절 공휴일 지정 (인사혁신처 보도 2026-06-28) | 2026-11-11 | 공공 정보 | 2025~2026 공휴일, 대체공휴일 | 달력 특징 |
| SSG 홈경기 일정·종료 시각 | `preprocess.py` 안 표 | KBO·SSG 공식 일정, 나무위키 'SSG 랜더스/2026년' (CC BY-NC-SA 2.0 KR), 기사 (8/4 경기, 8/5~6 폭염 취소) | 2026-11-11 | 출처별 | 2026 시험 기간 홈경기, 귀가 피크 시각 | 행사 특징 `ev`, `ev_rel` |
| 공연·축제·축구 일정 | `preprocess.py`의 `OTHER_EVENTS_2026` | 인천시(펜타포트), NOL티켓(송도맥주축제, GHOSTIVAL), Weverse(PLAVE), 부천FC 1995 공식 일정 등 | 2026-11-11 | 출처별 | 시험 기간 행사와 피크 시각 | 행사 특징 |
| 팀원 모델 예측 | `claude_works/inputs/team_final_20261112_140730_submission.csv` | 팀 내부 (팀원 LightGBM·CatBoost 최종 예측) | 2026-11-12 | 팀 내부 | 시험 기간 예측 157,194행 | `make_c5.py`의 부산물 C6에만 쓰임. **C29에는 해당 없음** |

- 2026-07-01 이후의 날씨 실측, 경기 일정·종료 시각·취소, 관중 수는 승차 인원이 아닌 정보입니다. 대회 규칙 3절(수정)과 운영진 답변으로 사용이 허용됐습니다.
- 날씨 조회 스크립트(`fetch_weather.py`)는 팀 전처리 담당이 실행했고, 이 패키지에는 없습니다. 조회 결과 파일은 그대로 넣었습니다.
- 공개 데이터에서 평가 정답 또는 그에 준하는 자료(2026-07-01 이후 승차 인원)를 열람·확보·사용한 사실은 없습니다.

## 3. 파생 데이터 (학습 입력까지의 생성 경로)

| 파일 | 만드는 코드 | 입력 | 설명 |
|---|---|---|---|
| `data/processed_v4/train_cutoff_features.csv.gz` | `preprocess.py --stacked` (팀) | train, metaData, 날씨, 일정 | 월말 기준일마다 다음 123일을 쌓은 학습표, 2,391,876행 × 59열 |
| `data/processed_v4/test_cutoff_features.csv.gz` | 같음 | 같음 + test | 2026-06-30 기준 시험 특징, 157,194행 × 58열 |
| `data/processed_v4/train_processed.csv.gz`, `test_processed.csv.gz` | 같음 | 같음 | 행마다 달력·행사·이상치 표시를 붙인 탐색용 표. `wxfeat.py`가 날짜별 `dt3`, `normal_day`만 읽음 |
| `work/out/final_train_smc.parquet` | `build_smc.py` | processed_v4, 팀 코드 묶음 DataEngine | 기준일 월 2회(월말·14일) 학습표, 457만 행. 월말 부분은 팀 표와 같음을 확인 |
| `work/out/final_train_d4.parquet` | `build_dense.py ... 7,14,21` | 같음 | 기준일 월 4회 학습표, 약 900만 행 |
| `work/out/test_fix.csv.gz` | `fix_test_yoy.py` | test_cutoff_features | 계양·귤현만 `yoy_delta`·`yoy_raw`를 전체 역 버전으로 바꾼 시험 특징 |
| 특징 `att` | `final_att.py` (`add_att`) | 관중 수 CSV | 3127 경기일 관중/1000 (더블헤더는 최댓값), 나머지 빈 값 |
| 특징 `wx_*` | `wxfeat.py` | 날씨 CSV, train_processed의 달력 | 기준일 날씨 4개 (`provenance/FEATURES.md`) |
| 검증용 학습표 (`o0227` 등의 fold_train, fold_future, train_smc, train_d4) | `build_fold.py`, `fold_multi.py`, `build_smc.py`, `build_dense.py` | 기준일 이후 승차를 지운 자료 | 용량 때문에 패키지에서 뺌 (약 700MB, 재생성 가능). 검증 예측은 `records/validation_preds/`에 있음 |

## 4. 캐시와 모델 산출물

세부 목록은 `provenance/ARTIFACTS.md`에 있습니다.

| 위치 | 내용 | 다시 만드는 법 |
|---|---|---|
| `work/out/final_*.npy` (66개) | 최종 학습 모델의 시험 기간 예측 (log1p, test.csv 행 순서) | `run_full.sh` 1~7단계 |
| `claude_works/candidates/model_C0_lgb_direct.txt`, `model_C1_lgb_seasonal_resid.txt` | 예전 세트 LightGBM 모델 | `final_lgb.py` |
| `claude_works/candidates/model_C7_cat1500_seasonal_resid.cbm` | 예전 세트 CatBoost 모델 (조합 스크립트가 불러 예측) | `final_cat.py` |
| `claude_works/candidates/submission_C*.csv` (15개) | C29까지의 중간 후보. 다음 단계가 조각에서 다시 만들어 같은지 확인하는 기준 | `run_full.sh` 각 단계 |
| `work/bundle/sincon_v4_colab_bundle/` | 팀 v4 코드 (DataEngine, v2 구조 기준선). 안의 `data/`는 processed_v4 파일의 사본 | 팀 제공 (`codex_works/colab_model_lab_v4`에서 만든 묶음) |

사전학습 모델, 임베딩, 외부 초기 가중치는 쓰지 않았습니다(해당 없음).

## 5. 수동 설정값과 근거

| 값 | 위치 | 근거 |
|---|---|---|
| 시험 기간 행사 목록 (SSG, 공연, 축제, 부천FC) | `preprocess.py` `OTHER_EVENTS_2026` (팀) | 공식 일정. 피크 시각은 2025년 같은 행사의 실제 피크 |
| 추가 행사 5건: 3138 2026-08-01 23시, 08-22·08-23·08-29 22시 / 3210 07-11 22시 | `final_att.py` `EXTRA` | 2025년 같은 행사(펜타포트, 송도맥주축제, 흠뻑쇼) 때 그 역에서 탐지된 피크 |
| 계양·귤현 학습행 제외 규칙: 날짜 ≥ 2025-06-28이고 기준일−56일 < 2025-06-28 | `final_att.py`, `fold_att.py` (`DROP_BREAK=1`) | 검단 연장 3역 개통일. 8주 수준 창이 개통 전후에 걸친 행 |
| 세트 안 비중 0.8 / 0.2, E·C·H 1/3, V 0.2 (행사 행 0) | `make_c8.py`, `make_c23.py`의 `c9` | 설명서 6.1절 (검증) |
| 3127 규칙, 행 배정 | `make_c23.py` | 설명서 6.1~6.2절 (검증) |
| 트리 수 A 4000, E 3000 (3127은 A 500 유지) | `run_full.sh` 7단계, `A4K_KEEP27=1` | 설명서 4.2절 (검증 곡선과 세트 평균 비교) |
| 추석 보정 w=0.8 | `lunar_adjust.py` 기본값 (팀) | 팀원 검증 (2025 추석 3일 0.46 → 0.31) |
| 주말·공휴일 보정 B1 0.285, B2 0.190 / 평일 시간대 B3 0.105 | `make_c27.py` 인자 | 설명서 6.4절 (o0227 검증 예측에 최소제곱) |
