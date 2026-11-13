# SINCON 2026 — 팀 쉽조~ 최종 제출 자료

인천 지하철 71개 역의 시간대별 승차 인원(2026-07-01 ~ 10-31, 157,194행)을 예측했습니다. 평가는 RMSLE입니다.

| 항목 | 값 |
|---|---|
| 최종 예측 파일 | `predictions/submission_C29_c25c_wkfix_hr.csv` (리더보드에 제출한 원본 그대로) |
| SHA-256 | `285f61657b8301f4c87caf0adcd360941d3db8e3320552734cf30f20fbd85d72` |
| 행 수 / 열 | 157,194행 / `example_id`, `boardings` |
| Public RMSLE | 0.149556 |
| 최종 선택 | C29 1개 |

이 문서는 실행 방법과 자료 위치를 설명합니다. 모델 설계와 결정 근거는 [`records/METHOD.md`](records/METHOD.md)(설명서)에 있습니다.

---

## 1. 파이프라인 한눈에 보기

```
제공 데이터 train / test / metaData
 + 날씨 실측 (Open-Meteo ERA5)  + 행사·공휴일 일정  + SSG 홈경기 관중 수
   │
   ▼  팀 전처리 preprocess.py
data/processed_v4   기준일(cutoff) 특징 55개, 월말 기준일마다 다음 123일을 쌓은 학습표 239만 행
   │
   ▼  학습표 확장·수정 (build_smc.py, build_dense.py, fix_test_yoy.py)
   ▼  모델 세트 8개 × 모델 4종 (A, E, C, H) + 구조 기준선 V
   ▼  세트 안 고정 조합 → 세트 평균 → 행 배정 (행사·3127·계양/귤현) → 추석 보정     = C25c
   ▼  주말·공휴일 수준 보정 + 평일 시간대 보정 (make_c27.py)                     = C29 (최종)
```

## 2. 폴더 구성

대회 안내의 권장 구성과 이 패키지의 위치를 대응시켰습니다. 코드가 상대 경로로 서로를 찾기 때문에 작업 때 쓰던 구조를 그대로 유지했습니다.

| 권장 위치 | 이 패키지 | 내용 |
|---|---|---|
| `README.md` | `README.md` | 이 문서 (실행 방법, 자료 위치, 자체 점검) |
| `code/`, `configs/` | `work/code/`, `work/bundle/`, `paths.sh`, `run_quick.sh`, `run_full.sh`, `requirements-*.txt` | 전체 코드, 실행 순서, 의존성 고정 파일 |
| `data/`, `provenance/` | `data/processed_v4/`, `claude_works/external/`, `claude_works/inputs/`, `provenance/` | 실제 사용 자료, 출처 작성표, 특징 사전, 해시 |
| `models/`, `artifacts/` | `work/out/`, `claude_works/candidates/`, `provenance/ARTIFACTS.md` | 최종 모델의 시험 기간 예측, 학습표, 저장된 모델, 중간 후보, 산출물 작성표 |
| `predictions/` | `predictions/` | 평가 대상 최종 예측 원본 |
| `records/` | `records/` | 설명서(METHOD.md), 후보별 실험 기록, 학습 로그, 검증 예측, 재현 확인 로그, 이전 코드 백업 |

세부 위치

```
paths.sh                  모든 경로와 파이썬 실행 파일을 정하는 설정 (다른 스크립트가 source 함)
run_quick.sh              빠른 재현: 학습 없이 저장된 모델 예측으로 C25c → C29를 다시 만들고 제출 파일과 비교
run_full.sh               전체 재현: 제공 데이터부터 모든 모델을 다시 학습해 C29를 만듦 (8단계)
requirements-lgb.txt      환경 1 (Python 3.13.14): LightGBM, scikit-learn, pandas
requirements-cat.txt      환경 2 (Python 3.12.14): CatBoost
data/raw/                 대회 제공 train.csv, test.csv, metaData.csv를 넣는 곳 (안내 README만 있음)
data/processed_v4/        팀 전처리 결과와 코드 (preprocess.py, lunar_adjust.py, weather_incheon.csv, 설명서 docx)
claude_works/external/    ssg_munhak_attendance.csv (문학 SSG 홈경기 관중 수)
claude_works/inputs/      팀원 모델 예측 1개 (make_c5.py 인자. C29에는 쓰이지 않는 부산물 C6에만 쓰임)
claude_works/candidates/  C29까지 이어지는 중간 후보 CSV 15개, 예전 세트 모델(model_C0/C1 .txt, model_C7 .cbm)
work/code/                파이프라인 스크립트 전체 (분석·점검용 스크립트 포함, 4절 표 참고)
work/bundle/              팀 v4 코드 묶음 (DataEngine, v2 구조 기준선 vendor/sincon_v2.py)
work/out/                 최종 학습(기준일 2026-06-30) 모델의 시험 기간 예측 final_*.npy 66개, 학습표 2개, 수정 시험 특징
predictions/              최종 제출 파일
provenance/               DATA_SOURCES.md(자료 출처), FEATURES.md(특징 사전), ARTIFACTS.md(산출물 작성표), MANIFEST.sha256
records/                  METHOD.md(설명서), candidates_history_README.md(후보별 기록), PUBLIC_SCORES.md, logs/, validation_preds/, repro_check/, code_backups/
```

## 3. 실행 방법

### 3.1 환경 준비

| 항목 | 작성자 환경 |
|---|---|
| OS | Windows 11 Pro 10.0.26200 (x64), 셸은 Git Bash. 스크립트는 bash라 Linux·Colab에서도 같은 명령으로 돕니다 |
| CPU / 메모리 | 논리 32스레드 / 32GB. 큰 작업(att_d4)은 한 개에 약 8~10GB를 씁니다 |
| GPU | NVIDIA RTX 4070 Laptop (드라이버 610.62). CatBoost GPU는 `att_ball`, `att_ball_smc` 두 세트의 C 모델에만 씀 |
| 환경 1 `PY_LGB` | Python 3.13.14 + `requirements-lgb.txt` (numpy 2.4.2, pandas 3.0.3, lightgbm 4.6.0, scikit-learn 1.9.0, scipy 1.17.1, pyarrow 24.0.0) |
| 환경 2 `PY_CAT` | Python 3.12.14 + `requirements-cat.txt` (numpy 2.3.5, pandas 3.0.1, catboost 1.2.10, scipy 1.18.1, pyarrow 26.0.1) |

두 환경은 작업 중 CatBoost가 설치된 실행기가 따로 있어서 나뉜 것입니다. 모든 패키지가 있는 환경 하나(예: Colab)로 돌려도 됩니다. 이때는 `PY_LGB`와 `PY_CAT`을 같은 `python`으로 두면 됩니다. 버전이 다르면 소수점 아래 아주 작은 차이가 날 수 있습니다(4.4절).

```bash
# 환경 1
python3.13 -m venv .venv_lgb && .venv_lgb/Scripts/pip install -r requirements-lgb.txt     # Linux: .venv_lgb/bin/pip
# 환경 2
python3.12 -m venv .venv_cat && .venv_cat/Scripts/pip install -r requirements-cat.txt
export PY_LGB=$PWD/.venv_lgb/Scripts/python PY_CAT=$PWD/.venv_cat/Scripts/python
# 대회 제공 데이터를 data/raw/ 에 복사 (train.csv, test.csv, metaData.csv)
```

### 3.2 빠른 재현 (학습 없음, 약 1분)

```bash
bash run_quick.sh
```

저장된 모델 예측(`work/out/final_*.npy`)으로 C25c와 C29를 다시 만들고 제출 파일과 비교합니다.

- 단계 [0]: 제공 데이터의 SHA-256을 확인합니다.
- 단계 [1]: `make_c23.py`가 중간 후보 C15·C17·C19를 조각에서 다시 만들어 동봉 파일과 같은지 확인한 다음, C25c를 만듭니다. 다르면 중단합니다.
- 단계 [2]: `make_c27.py`가 C29를 만듭니다.
- 단계 [3]: 형식 검사(`check_final.py`)를 하고 제출 파일과 비교합니다.

작성자 환경에서의 결과(`records/repro_check/run_quick.log`): `rebuilt C29 sha256 285f61657b8301f4 | submitted 285f61657b8301f4 | max |diff| 0.000000 boardings -> IDENTICAL` ZIP을 새 폴더에 푼 복사본에서도 같은 결과였습니다(`records/repro_check/run_quick_from_zip.log`).

### 3.3 전체 재현 (모든 모델 다시 학습)

```bash
bash run_full.sh          # 중간부터 다시: bash run_full.sh 5
```

출력은 `work/out_full/`과 `claude_works/candidates_full/`에만 씁니다. 동봉 파일은 바뀌지 않습니다. 각 작업의 로그는 `work/out_full/logs/`에 남습니다.

| 단계 | 내용 | 만드는 것 | 작성자 환경 시간 (8스레드, 한 번에 하나) |
|---|---|---|---|
| 1 | 예전 세트: LightGBM 직접(A)·잔차(E) 500회와 v2 구조 기준선(V), CatBoost(C) 1500회, HistGB(H) 1500회 | C0~C5, C7, C8, C9, `final_hgb_test.npy` | 약 50분 |
| 2 | 기준일 월 2회·4회 학습표, 계양·귤현 시험 특징 수정 | `final_train_smc.parquet`, `final_train_d4.parquet`, `test_fix.csv.gz` | 약 20분 |
| 3 | att 세트(+관중 수)와 시험 기간 행사 표시 | `final_att_*.npy`, `final_att_ev.npy`, C10 | 약 45분 |
| 4 | attwx·att_smc·attwx_smc·att_d4 세트 | `final_{세트}_*.npy`, C15 | 약 4시간 |
| 5 | 계양·귤현용 세트 6개 (검단 개통 전후 학습행 제외) | `final_{세트}_cl_*.npy`, C17 | 약 4.5시간 |
| 6 | 전체 역 기준선 세트 att_ball·att_ball_smc (C는 GPU) | `final_att_ball*_*.npy`, C19 | 약 50분 |
| 7 | 잔차 LightGBM 3000회(7세트), 직접 LightGBM 4000회(5개 파일이 7세트에 쓰임) | `final_*_e3k*_E.npy`, `final_*_a4k_A.npy` | 약 3.5시간 |
| 8 | 조합과 보정 | C25c, C29, 제출 파일과 비교 | 약 2분 |

전체는 한 번에 하나씩 돌리면 약 14시간입니다. 4~7단계는 세트끼리 서로 기다리지 않아서, 메모리가 허락하면 나눠서 동시에 돌릴 수 있습니다. GPU가 없으면 6단계의 `CAT_GPU=1`을 빼고 CPU로 돌립니다. 이 경우 두 세트의 C 모델 값이 달라지므로 결과도 조금 달라집니다.

### 3.4 (선택) 전처리 다시 만들기

`data/processed_v4`는 팀 전처리 코드의 결과물입니다. 다시 만들려면 아래처럼 실행합니다.

```bash
cp data/processed_v4/weather_incheon.csv data/raw/      # 전처리는 data_dir에서 날씨 파일을 찾음
"$PY_LGB" data/processed_v4/preprocess.py --data_dir data/raw --out_dir data/processed_regen --stacked
gzip data/processed_regen/*.csv                          # 동봉본은 .csv.gz
```

### 3.5 실행 양식 (대회 안내 2절)

| 단계 | 작성 정보 |
|---|---|
| 환경 준비 | 3.1절. Windows 11 x64, Python 3.13.14 / 3.12.14, `requirements-lgb.txt`, `requirements-cat.txt`, CPU 32스레드, GPU RTX 4070(CatBoost 두 세트만) |
| 학습 및 선택 | 명령은 `run_full.sh` 1~7단계. 허용 초기 모델: 없음(사전학습 모델 미사용, 모두 처음부터 학습). 검증 분할: 기준일 2026-02-27(주), 03-31, 04-30(보조)의 시간순 검증(설명서 2절). 시드는 모두 42(LightGBM `seed`, HistGB `random_state`, CatBoost `random_seed`). 생성 모델: `work/out/final_*.npy`(시험 기간 예측), `claude_works/candidates/model_*`. 예상 시간: 3.3절 표 |
| 최종 추론 | `run_full.sh` 8단계 또는 `run_quick.sh`. 입력은 `work/out/final_*.npy`, `claude_works/candidates/`, `data/processed_v4`, `data/raw`. 후처리 설정은 추석 보정 `lunar_adjust.py`(기본 w=0.8), 주말·공휴일 보정 B1 0.285 / B2 0.190 / 평일 시간대 B3 0.105 |
| 재현 변동 | 4.4절 |

## 4. 최종 예측과 코드·모델·설정의 관계

### 4.1 C29를 만드는 명령 (run_quick.sh와 같음)

```bash
cd work/code
A4K_KEEP27=1 C19ADD=att_ball,att_ball_smc \
E3K=att,att_ball,att_smc,att_ball_smc,attwx,attwx_smc,att_d4:att_d4_e3kp \
A4K=att:att_a4k,att_ball:att_a4k,att_smc:att_smc_a4k,att_ball_smc:att_smc_a4k,attwx:attwx_a4k,attwx_smc:attwx_smc_a4k,att_d4:att_d4_a4k \
$PY_CAT make_c23.py $D $RAW $CAND $W/out C25c_c19_e3k7_a4k7_keep27 C15_avg6_d4 $SETS $CLSETS att_ball,att_ball_smc C17_geomdan_6sets C19_add_ball_smc
$PY_LGB make_c27.py $RAW $D $CAND $W/out C25c_c19_e3k7_a4k7_keep27 C29_c25c_wkfix_hr 0.285 0.190 0.105
```

### 4.2 후보 사슬

`C29` ← C25c + 주말·공휴일 보정 (`make_c27.py`)
`C25c` ← C19 + 7세트의 E를 3000회, A를 4000회로 교체 (`make_c23.py`, 3127역은 500회 A 유지)
`C19` ← C17 + 전체 역 기준선 세트 2개 (`make_c18.py`)
`C17` ← C15 + 계양·귤현을 다시 학습한 세트로 예측 (`make_c17.py`)
`C15` ← 예전 세트 + 새 세트 5개 평균 (`make_c12.py`, C9·C10을 조각에서 다시 만들어 확인)
`C10` ← C9 + 3127 경기일 관중 모델, 추가 행사 5건 (`make_c10.py`)
`C9` ← C8 + 3127 규칙 (`make_c8.py rule27`)
`C8`, `C7`, `C5`, `C4`, `C0`~`C3` ← 예전 세트 (`make_c8.py`, `final_cat.py`, `make_c5.py`, `make_c4.py`, `final_lgb.py`)

각 `make_*.py`는 이전 후보를 조각에서 다시 만들어 동봉 CSV와 같은지 확인(assert)합니다. 그래서 사슬 중간이 어긋나면 바로 멈춥니다. 단계별 설명과 검증 수치는 `records/candidates_history_README.md`에 있습니다.

### 4.3 코드 목록

| 구분 | 파일 |
|---|---|
| 전처리 (팀) | `data/processed_v4/preprocess.py`, `data/processed_v4/lunar_adjust.py`(추석 보정), `work/bundle/sincon_v4_colab_bundle/`(DataEngine, v2 구조 기준선) |
| 학습표·특징 | `build_smc.py`, `build_dense.py`, `fix_test_yoy.py`, `wxfeat.py`, `build_fold.py`(검증 구간 생성) |
| 최종 학습 | `final_lgb.py`, `final_cat.py`, `final_hgb.py`, `final_att.py`, `cl_run.sh` |
| 조합·후처리 | `make_c4.py`, `make_c5.py`, `make_c8.py`, `make_c10.py`, `make_c12.py`, `make_c17.py`, `make_c18.py`, `make_c23.py`, `make_c27.py`, `wkfix_check.py`, `nfix_check.py`(C29 보정 항 정의), `check_final.py` |
| 검증 | `fold_multi.py`, `fold_multi_cat.py`, `fold_att.py`, `fold_att_curve.py`, `fold_att_leaves.py`, `fold_lgb.py`, `fold_cat.py`, `fold_hgb.py`, `fold_exp2.py`, `compare_*.py`, `swap_check.py`, `e3k_check.py`, `c3k6_check.py`, `curve_eval.py`, `stack_*.py`, `strict_weights.py`, `blend_check.py`, `ev_rule_check.py`, `late_bias.py`, `lunar_eval.py` |
| 분석 (예측에 영향 없음) | `budget.py`, `err_buckets.py`, `new_station_look*.py`, `make_c16.py`, `make_c20w.py`, `final_att_v2.py`, `make_c5.py`의 C6 부분 |

- 작업 때 쓰던 코드와 같습니다. 다만 두 군데만 작성자 PC의 절대 경로를 패키지 상대 경로로 바꿨습니다: `wxfeat.py` 13행, `cl_run.sh` 4~5행과 마지막 줄의 실행기 지정.
- 분석용 `lunar_eval.py`, `new_station_look*.py`는 작성자 경로를 `paths.sh`의 `S`·`RAW`로 바꿨습니다(`source paths.sh` 후 실행). 기록 로그의 작성자 폴더 경로도 `$W/`로 줄였습니다.
- 이전 버전 코드는 `records/code_backups/`에 있습니다.

### 4.4 재현 변동

- **LightGBM:** `deterministic=True`, `num_threads=8`, `seed=42`로 고정했습니다. 같은 버전이면 같은 결과가 나옵니다.
- **HistGB:** `random_state=42`, `early_stopping=False`입니다.
- **CatBoost:**
  - CPU는 `thread_count=8`, `random_seed=42`입니다.
  - GPU(두 세트)는 GPU 합산 순서 때문에 실행마다 아주 작은 차이가 날 수 있습니다.
- **작성자 환경에서 다시 학습한 확인 결과:** `records/repro_check/spot_check.log`
  - 확인한 작업: 예전 세트 LightGBM·v2(C0~C3), HistGB, 4000회 LightGBM, CatBoost CPU와 GPU 각 1개, 행사 표시, 시험 특징 수정
  - CPU 작업 9개는 동봉 파일과 값이 모두 같았습니다.
  - CatBoost GPU(att_ball C)만 달랐습니다. 모델 출력 차이는 RMSE 0.0004, 최대 0.031(log1p)입니다. C29에서 이 모델 비중이 약 1/40이라 최종 차이는 RMSE 약 1e-5로 추정합니다.
  - 자세한 표는 [설명서 7절](records/METHOD.md#7-재현성-확인)에 있습니다.
- **조합 단계:** 각 `make_*.py`의 확인 기준은 승차 0.001명입니다. 제출 파일은 소수점 6자리로 저장됩니다.
- **버전이 다를 때:** 다른 버전의 numpy·pandas·LightGBM·CatBoost를 쓰면 트리 분할이 미세하게 달라질 수 있습니다. 그 경우 C29와 RMSLE 차이는 1e-4 수준 이하로 예상합니다(검증 안 함).

## 5. 규칙 준수와 자체 점검

- **2026-07-01 이후 실제 승차 인원:** 학습·특징·검증·후처리 어디에도 쓰지 않았습니다.
  - 모든 입력은 `train.csv`(2026-06-30까지)와 승차 인원이 아닌 외부 정보입니다.
  - 학습 특징은 팀 전처리가 기준일 이전 자료로만 만든 표입니다.
  - 조합 스크립트(`make_c5/8/10/12/17/18/23/27.py`, `final_cat.py`)는 `train.csv`의 마지막 날이 2026-06-30인지 확인(assert)합니다. `make_c27.py`는 6/30 이후 승차 값이 비어 있는지도 확인합니다.
- **7/1 이후 정보 중 사용한 것:**
  - 날씨 실측 (Open-Meteo ERA5)
  - SSG 홈경기 일정·종료 시각·관중 수
  - 공연·축제 일정
  - 대회 규칙 3절(수정)과 운영진 답변으로 허용된 정보입니다.
- **시험 정답·평가 자료:** 공개 데이터를 포함해 어디서도 열람·확보·사용하지 않았습니다.
- **Public 점수:** 이미 만든 후보 중 하나를 고를 때만 봤습니다.
  - 가중치·보정 계수·트리 수는 모두 검증 구간에서 정했습니다.
  - 점수 기록은 `records/PUBLIC_SCORES.md`에 있습니다.
- **수동 설정값:** 근거는 설명서 6절에 있습니다.

대회 안내 4절 점검표

- [x] 최종 예측 파일과 실제 사용한 코드, 모델, 설정의 관계를 명시함 (4절, `provenance/ARTIFACTS.md`)
- [x] 데이터와 모델 작성표에 사용 항목을 모두 기록하고, 미사용 항목은 해당 없음으로 표시함 (`provenance/DATA_SOURCES.md`, `provenance/ARTIFACTS.md`)
- [x] 제공 자료부터 최종 예측까지 필요한 경로, 명령, 의존성, 시드를 설명함 (3절)
- [x] 평가 정답 또는 관련 자료를 열람·확보·사용한 사실: 없음

## 6. GitHub 저장소로 올릴 때

GitHub은 100MB를 넘는 파일을 받지 않습니다. 그래서 `.gitignore`에서 아래 파일을 뺐습니다.

- 대회 제공 데이터 (`data/raw/*.csv`)
- 100MB가 넘는 파일 2개: `data/processed_v4/train_cutoff_features.csv.gz` 129MB, `work/out/final_train_d4.parquet` 132MB
- 실행 중 생기는 폴더

빠진 두 큰 파일은 Git LFS로 올리거나 ZIP에서 따로 받아 같은 위치에 두면 됩니다. `final_train_d4.parquet`는 `run_full.sh` 2단계로도 다시 만들 수 있습니다.

- 빠른 재현(`run_quick.sh`)에는 이 두 파일이 필요 없습니다. 저장소만 받아도 원본 CSV 3개를 `data/raw/`에 넣으면 바로 확인할 수 있습니다.
- `.gitattributes`(`* -text`)는 Windows에서 받을 때 줄바꿈이 CRLF로 바뀌는 것을 막습니다. 바뀌면 bash 스크립트가 깨지고 해시가 어긋납니다.
- 나머지 파일은 모두 100MB 미만입니다. `work/out/final_train_smc.parquet`(69MB)는 올라가지만 GitHub이 50MB 초과 경고를 띄웁니다.
- 두 파일을 뺀 저장소 크기는 약 560MB입니다. 검증 예측 `records/validation_preds/`(279MB)를 빼려면 `.gitignore` 마지막 줄의 `#`을 지우면 됩니다.
- 받은 뒤에는 `provenance/MANIFEST.sha256`로 파일이 온전한지 확인할 수 있습니다: `sha256sum -c --ignore-missing provenance/MANIFEST.sha256` (`--ignore-missing`은 Git에서 뺀 두 파일을 건너뜀).
