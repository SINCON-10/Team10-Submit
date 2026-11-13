# 산출물 작성표

- **값:** `work/out/final_*.npy`는 최종 학습(기준일 2026-06-30) 모델이 낸 시험 기간 예측입니다. log1p 값이고, test.csv 행 순서로 157,194개입니다.
- **모델 파일 대신 쓰는 이유:** 조합 단계(`make_c23.py`, `make_c27.py`)는 이 예측만 읽습니다. 모델 파일을 따로 남기지 않은 모델은 `run_full.sh`의 같은 단계로 다시 학습하면 같은 예측이 나옵니다(GPU CatBoost는 근사, README 4.4).
- **단계 번호:** `run_full.sh`의 단계입니다.
- **패키지에 넣은 범위:** C29에 쓰인 것만 넣었습니다. 작업 중 만든 다른 산출물(ballwx, ball_d4, dcat, 예전 세트 4000/3000회 등)은 C29에 쓰이지 않아(해당 없음) 넣지 않았습니다.

| 파일 | 세트 | 모델 | 단계와 설정 | 쓰는 곳 | 평균(log1p) | SHA-256 앞 12자리 |
|---|---|---|---|---|---|---|
| final_att_A.npy | att | LightGBM 직접 500회 | 3: 기본 | C29 3127 행 + C15·C17·C19 재구성 확인 | 5.3788 | 9a5d0b9eda9c |
| final_att_a4k_A.npy | att, att_ball | LightGBM 직접 4000회 | 7: `LGB_ROUNDS=4000` | C29 (3127 제외 행) | 5.3832 | 56a03a46cdb2 |
| final_att_ball_A.npy | att_ball | LightGBM 직접 500회 | 6: BASE_ALL=1 DROP_BREAK=1 TEST_FILE=test_fix TAG=_ball (C: CAT_GPU=1) | C29 3127 행 + C19 재구성 확인 | 5.3838 | eece7b84f591 |
| final_att_ball_C.npy | att_ball | CatBoost 잔차 1500회 | 6: BASE_ALL=1 DROP_BREAK=1 TEST_FILE=test_fix TAG=_ball (C: CAT_GPU=1) | C29 | 5.3545 | d8d6eca0e5da |
| final_att_ball_E.npy | att_ball | LightGBM 잔차 500회 | 6: BASE_ALL=1 DROP_BREAK=1 TEST_FILE=test_fix TAG=_ball (C: CAT_GPU=1) | C19 재구성 확인만 (C29는 3000회 E 사용) | 5.3680 | 150fffa7a6a6 |
| final_att_ball_e3k_E.npy | att_ball | LightGBM 잔차 3000회 | 7: `LGB_ROUNDS=3000` | C29 | 5.3712 | 0da6810be340 |
| final_att_ball_H.npy | att_ball | HistGB 잔차 1500회 | 6: BASE_ALL=1 DROP_BREAK=1 TEST_FILE=test_fix TAG=_ball (C: CAT_GPU=1) | C29 | 5.3705 | 2b267d6f4eee |
| final_att_ball_smc_A.npy | att_ball_smc | LightGBM 직접 500회 | 6: BASE_ALL=1 DROP_BREAK=1 TEST_FILE=test_fix TRAIN_FILE=final_train_smc TAG=_ball_smc (C: CAT_GPU=1) | C29 3127 행 + C19 재구성 확인 | 5.3823 | 67eaaf9b63a5 |
| final_att_ball_smc_C.npy | att_ball_smc | CatBoost 잔차 1500회 | 6: BASE_ALL=1 DROP_BREAK=1 TEST_FILE=test_fix TRAIN_FILE=final_train_smc TAG=_ball_smc (C: CAT_GPU=1) | C29 | 5.3540 | 6bfce00bb010 |
| final_att_ball_smc_E.npy | att_ball_smc | LightGBM 잔차 500회 | 6: BASE_ALL=1 DROP_BREAK=1 TEST_FILE=test_fix TRAIN_FILE=final_train_smc TAG=_ball_smc (C: CAT_GPU=1) | C19 재구성 확인만 (C29는 3000회 E 사용) | 5.3674 | 2e5fd8ce0184 |
| final_att_ball_smc_e3k_E.npy | att_ball_smc | LightGBM 잔차 3000회 | 7: `LGB_ROUNDS=3000` | C29 | 5.3680 | 70a946ca1dd2 |
| final_att_ball_smc_H.npy | att_ball_smc | HistGB 잔차 1500회 | 6: BASE_ALL=1 DROP_BREAK=1 TEST_FILE=test_fix TRAIN_FILE=final_train_smc TAG=_ball_smc (C: CAT_GPU=1) | C29 | 5.3707 | 0dcb7497dee2 |
| final_att_C.npy | att | CatBoost 잔차 1500회 | 3: 기본 | C29 | 5.3508 | 144a6001c31f |
| final_att_cl_A.npy | att (계양·귤현용) | LightGBM 직접 500회 | 5: `DROP_BREAK=1 TEST_FILE=test_fix` | C29 (3110·3111 행) | 5.3838 | eece7b84f591 |
| final_att_cl_C.npy | att (계양·귤현용) | CatBoost 잔차 1500회 | 5: `DROP_BREAK=1 TEST_FILE=test_fix` | C29 (3110·3111 행) | 5.3642 | 581c0c813124 |
| final_att_cl_E.npy | att (계양·귤현용) | LightGBM 잔차 500회 | 5: `DROP_BREAK=1 TEST_FILE=test_fix` | C29 (3110·3111 행) | 5.3829 | cfedbbc90ab2 |
| final_att_cl_H.npy | att (계양·귤현용) | HistGB 잔차 1500회 | 5: `DROP_BREAK=1 TEST_FILE=test_fix` | C29 (3110·3111 행) | 5.3813 | 205ad7411dca |
| final_att_d4_A.npy | att_d4 | LightGBM 직접 500회 | 4: TRAIN_FILE=final_train_d4 TAG=_d4 | C29 3127 행 + C15·C17·C19 재구성 확인 | 5.3751 | fd1267d5401a |
| final_att_d4_a4k_A.npy | att_d4 | LightGBM 직접 4000회 | 7: `LGB_ROUNDS=4000` | C29 (3127 제외 행) | 5.3793 | 22babb4661a9 |
| final_att_d4_C.npy | att_d4 | CatBoost 잔차 1500회 | 4: TRAIN_FILE=final_train_d4 TAG=_d4 | C29 | 5.3483 | ee03d9bffe69 |
| final_att_d4_cl_A.npy | att_d4 (계양·귤현용) | LightGBM 직접 500회 | 5: `DROP_BREAK=1 TEST_FILE=test_fix` | C29 (3110·3111 행) | 5.3796 | 643d02da7bee |
| final_att_d4_cl_C.npy | att_d4 (계양·귤현용) | CatBoost 잔차 1500회 | 5: `DROP_BREAK=1 TEST_FILE=test_fix` | C29 (3110·3111 행) | 5.3602 | e513a5156a81 |
| final_att_d4_cl_E.npy | att_d4 (계양·귤현용) | LightGBM 잔차 500회 | 5: `DROP_BREAK=1 TEST_FILE=test_fix` | C29 (3110·3111 행) | 5.3701 | 6fdec5a4b34a |
| final_att_d4_cl_H.npy | att_d4 (계양·귤현용) | HistGB 잔차 1500회 | 5: `DROP_BREAK=1 TEST_FILE=test_fix` | C29 (3110·3111 행) | 5.3728 | 1c21e6143a57 |
| final_att_d4_E.npy | att_d4 | LightGBM 잔차 500회 | 4: TRAIN_FILE=final_train_d4 TAG=_d4 | C15·C17·C19 재구성 확인만 (C29는 3000회 E 사용) | 5.3649 | ad74a87557bd |
| final_att_d4_e3kp_E.npy | att_d4 | LightGBM 잔차 3000회 | 7: `LGB_ROUNDS=3000` | C29 | 5.3719 | 1d03030e61be |
| final_att_d4_H.npy | att_d4 | HistGB 잔차 1500회 | 4: TRAIN_FILE=final_train_d4 TAG=_d4 | C29 | 5.3641 | 29c2be5e6e50 |
| final_att_E.npy | att | LightGBM 잔차 500회 | 3: 기본 | C15·C17·C19 재구성 확인만 (C29는 3000회 E 사용) | 5.3725 | 25aea1e82c5a |
| final_att_e3k_E.npy | att | LightGBM 잔차 3000회 | 7: `LGB_ROUNDS=3000` | C29 | 5.3805 | fb90507bd284 |
| final_att_ev.npy | — | 시험 기간 행사 표시 (추가 5건 포함) | 3: `final_att.py ... -` | C29 (행 배정) | 0.0082 | 72d93dc71cf9 |
| final_att_H.npy | att | HistGB 잔차 1500회 | 3: 기본 | C29 | 5.3653 | 8b6be94b338c |
| final_att_smc_A.npy | att_smc | LightGBM 직접 500회 | 4: TRAIN_FILE=final_train_smc TAG=_smc | C29 3127 행 + C15·C17·C19 재구성 확인 | 5.3768 | 0af40f3bd57a |
| final_att_smc_a4k_A.npy | att_smc, att_ball_smc | LightGBM 직접 4000회 | 7: `LGB_ROUNDS=4000` | C29 (3127 제외 행) | 5.3765 | d9d5d9ab1829 |
| final_att_smc_C.npy | att_smc | CatBoost 잔차 1500회 | 4: TRAIN_FILE=final_train_smc TAG=_smc | C29 | 5.3507 | fc6f77c78ad1 |
| final_att_smc_cl_A.npy | att_smc (계양·귤현용) | LightGBM 직접 500회 | 5: `DROP_BREAK=1 TEST_FILE=test_fix` | C29 (3110·3111 행) | 5.3823 | 67eaaf9b63a5 |
| final_att_smc_cl_C.npy | att_smc (계양·귤현용) | CatBoost 잔차 1500회 | 5: `DROP_BREAK=1 TEST_FILE=test_fix` | C29 (3110·3111 행) | 5.3620 | 8048a251035c |
| final_att_smc_cl_E.npy | att_smc (계양·귤현용) | LightGBM 잔차 500회 | 5: `DROP_BREAK=1 TEST_FILE=test_fix` | C29 (3110·3111 행) | 5.3782 | e83db82b1522 |
| final_att_smc_cl_H.npy | att_smc (계양·귤현용) | HistGB 잔차 1500회 | 5: `DROP_BREAK=1 TEST_FILE=test_fix` | C29 (3110·3111 행) | 5.3808 | eeb893eafbc0 |
| final_att_smc_E.npy | att_smc | LightGBM 잔차 500회 | 4: TRAIN_FILE=final_train_smc TAG=_smc | C15·C17·C19 재구성 확인만 (C29는 3000회 E 사용) | 5.3649 | f919baf7afe1 |
| final_att_smc_e3k_E.npy | att_smc | LightGBM 잔차 3000회 | 7: `LGB_ROUNDS=3000` | C29 | 5.3788 | 1c35e1ed0b70 |
| final_att_smc_H.npy | att_smc | HistGB 잔차 1500회 | 4: TRAIN_FILE=final_train_smc TAG=_smc | C29 | 5.3714 | fcbd5a0af09e |
| final_attwx_A.npy | attwx | LightGBM 직접 500회 | 4: WX=1 | C29 3127 행 + C15·C17·C19 재구성 확인 | 5.3850 | 610739470b1f |
| final_attwx_a4k_A.npy | attwx | LightGBM 직접 4000회 | 7: `LGB_ROUNDS=4000` | C29 (3127 제외 행) | 5.3877 | bcff632e0aa3 |
| final_attwx_C.npy | attwx | CatBoost 잔차 1500회 | 4: WX=1 | C29 | 5.3657 | d1432edff40e |
| final_attwx_cl_A.npy | attwx (계양·귤현용) | LightGBM 직접 500회 | 5: `DROP_BREAK=1 TEST_FILE=test_fix` | C29 (3110·3111 행) | 5.3897 | 6acf64228c95 |
| final_attwx_cl_C.npy | attwx (계양·귤현용) | CatBoost 잔차 1500회 | 5: `DROP_BREAK=1 TEST_FILE=test_fix` | C29 (3110·3111 행) | 5.3789 | 140effd6282c |
| final_attwx_cl_E.npy | attwx (계양·귤현용) | LightGBM 잔차 500회 | 5: `DROP_BREAK=1 TEST_FILE=test_fix` | C29 (3110·3111 행) | 5.3954 | c773c99244d7 |
| final_attwx_cl_H.npy | attwx (계양·귤현용) | HistGB 잔차 1500회 | 5: `DROP_BREAK=1 TEST_FILE=test_fix` | C29 (3110·3111 행) | 5.3928 | daa7f8d5d937 |
| final_attwx_E.npy | attwx | LightGBM 잔차 500회 | 4: WX=1 | C15·C17·C19 재구성 확인만 (C29는 3000회 E 사용) | 5.3840 | c08396510166 |
| final_attwx_e3k_E.npy | attwx | LightGBM 잔차 3000회 | 7: `LGB_ROUNDS=3000` | C29 | 5.3941 | 935305a9ad62 |
| final_attwx_H.npy | attwx | HistGB 잔차 1500회 | 4: WX=1 | C29 | 5.3782 | d8611de1972a |
| final_attwx_smc_A.npy | attwx_smc | LightGBM 직접 500회 | 4: WX=1 TRAIN_FILE=final_train_smc TAG=_smc | C29 3127 행 + C15·C17·C19 재구성 확인 | 5.3863 | d1b702f40c3d |
| final_attwx_smc_a4k_A.npy | attwx_smc | LightGBM 직접 4000회 | 7: `LGB_ROUNDS=4000` | C29 (3127 제외 행) | 5.3858 | cc1534c46eaa |
| final_attwx_smc_C.npy | attwx_smc | CatBoost 잔차 1500회 | 4: WX=1 TRAIN_FILE=final_train_smc TAG=_smc | C29 | 5.3573 | 6df2697f8ef3 |
| final_attwx_smc_cl_A.npy | attwx_smc (계양·귤현용) | LightGBM 직접 500회 | 5: `DROP_BREAK=1 TEST_FILE=test_fix` | C29 (3110·3111 행) | 5.3887 | c57c5332d5b9 |
| final_attwx_smc_cl_C.npy | attwx_smc (계양·귤현용) | CatBoost 잔차 1500회 | 5: `DROP_BREAK=1 TEST_FILE=test_fix` | C29 (3110·3111 행) | 5.3772 | b3c5db8dadd4 |
| final_attwx_smc_cl_E.npy | attwx_smc (계양·귤현용) | LightGBM 잔차 500회 | 5: `DROP_BREAK=1 TEST_FILE=test_fix` | C29 (3110·3111 행) | 5.3834 | cd3c12f0f41c |
| final_attwx_smc_cl_H.npy | attwx_smc (계양·귤현용) | HistGB 잔차 1500회 | 5: `DROP_BREAK=1 TEST_FILE=test_fix` | C29 (3110·3111 행) | 5.3963 | 406a71f51b55 |
| final_attwx_smc_E.npy | attwx_smc | LightGBM 잔차 500회 | 4: WX=1 TRAIN_FILE=final_train_smc TAG=_smc | C15·C17·C19 재구성 확인만 (C29는 3000회 E 사용) | 5.3725 | 5928500e983d |
| final_attwx_smc_e3k_E.npy | attwx_smc | LightGBM 잔차 3000회 | 7: `LGB_ROUNDS=3000` | C29 | 5.3868 | 289eb9ba9027 |
| final_attwx_smc_H.npy | attwx_smc | HistGB 잔차 1500회 | 4: WX=1 TRAIN_FILE=final_train_smc TAG=_smc | C29 | 5.3815 | d7328acca4d8 |
| final_base_cl_A.npy | base (계양·귤현용) | LightGBM 직접 500회 | 5: `DROP_BREAK=1 TEST_FILE=test_fix` | C29 (3110·3111 행) | 5.3838 | bd274353cee2 |
| final_base_cl_C.npy | base (계양·귤현용) | CatBoost 잔차 1500회 | 5: `DROP_BREAK=1 TEST_FILE=test_fix` | C29 (3110·3111 행) | 5.3533 | 9a6c8156d840 |
| final_base_cl_E.npy | base (계양·귤현용) | LightGBM 잔차 500회 | 5: `DROP_BREAK=1 TEST_FILE=test_fix` | C29 (3110·3111 행) | 5.3814 | 5451c1cd9a6e |
| final_base_cl_H.npy | base (계양·귤현용) | HistGB 잔차 1500회 | 5: `DROP_BREAK=1 TEST_FILE=test_fix` | C29 (3110·3111 행) | 5.3751 | 135fa205ecb4 |
| final_hgb_test.npy | 예전 세트 | HistGB 잔차 1500회 | 1: `final_hgb.py` | C29 (예전 세트 H) | 5.3693 | 33795aec0992 |

## 저장된 모델과 중간 후보 (`claude_works/candidates/`)

| 파일 | 만드는 코드 | 쓰는 곳 |
|---|---|---|
| `model_C0_lgb_direct.txt`, `model_C1_lgb_seasonal_resid.txt` | `final_lgb.py` | 예전 세트 A, E (예측은 C0·C1 CSV로 저장되어 조합에 쓰임) |
| `model_C7_cat1500_seasonal_resid.cbm` | `final_cat.py` | 예전 세트 C (조합 스크립트가 불러 시험 특징으로 예측) |
| `submission_C0` ~ `C3` | `final_lgb.py` | 예전 세트 A(C0), E(C1), V(=(C2−0.8·C1)/0.2), 확인 기준 |
| `submission_C4`, `C5`, `C7`, `C8`, `C9`, `C10` | `make_c4.py`, `make_c5.py`, `final_cat.py`, `make_c8.py`, `make_c10.py` | 다음 단계의 확인 기준 (C9·C10은 `make_c12.py`가 다시 만들어 비교) |
| `submission_C15_avg6_d4`, `C17_geomdan_6sets`, `C19_add_ball_smc` | `make_c12.py`, `make_c17.py`, `make_c18.py` | `make_c23.py`가 조각에서 다시 만들어 비교 |
| `submission_C25c_c19_e3k7_a4k7_keep27` | `make_c23.py` | `make_c27.py`의 입력 |
| `submission_C29_c25c_wkfix_hr` | `make_c27.py` | 최종 (= `predictions/`) |
