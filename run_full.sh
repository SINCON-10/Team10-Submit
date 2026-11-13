#!/bin/bash
# Full reproduction: retrain every model of the C29 chain from the provided data (data/raw) and the team data (data/processed_v4),
# then build C29 and compare it with predictions/submission_C29_c25c_wkfix_hr.csv.
# Writes only into work/out_full and claude_works/candidates_full (the shipped files in work/out and claude_works/candidates stay as they are).
# usage:  bash run_full.sh [first_stage]     e.g. "bash run_full.sh 5" resumes from stage 5
# time (Windows 11, 32 threads, RTX 4070, one job at a time): about 14 hours. Stages 4-7 are independent per set and can run in parallel.
set -euo pipefail
cd "$(dirname "$0")"; source paths.sh
FIRST=${1:-1}
OUT="$W/out_full"; C="$S/claude_works/candidates_full"; mkdir -p "$OUT" "$C" "$OUT/logs"
# the blend scripts read the saved old-set CatBoost model and reference candidates from C: they are all rebuilt here in stage 1-6
cd "$W/code"
T="$OUT/test_fix.csv.gz"; SMC="$OUT/final_train_smc.parquet"; D4="$OUT/final_train_d4.parquet"
lgb() { local log=$1; shift; echo "  $(date +%H:%M) $log"; env "$@" > "$OUT/logs/$log.log" 2>&1; tail -n 1 "$OUT/logs/$log.log"; }
stage() { [ "$1" -ge "$FIRST" ]; }

if stage 1; then echo "[1/8] old set: LightGBM direct (A) / residual (E) + v2 structural (V) -> C0..C3, C4, C5, CatBoost (C) -> C7, HistGB (H) -> C8, C9"
  lgb final_lgb   "$PY_LGB" final_lgb.py "$B" "$D" "$RAW" "$C"
  lgb make_c4     "$PY_LGB" make_c4.py "$D" "$RAW" "$C"
  lgb make_c5     "$PY_LGB" make_c5.py "$D" "$RAW" "$C" "$S/claude_works/inputs/team_final_20261112_140730_submission.csv"   # team file is used only by the C6 by-product
  lgb final_cat   "$PY_CAT" final_cat.py "$D" "$RAW" "$C" 1500 0.5
  lgb final_hgb   OMP_NUM_THREADS=12 "$PY_LGB" final_hgb.py "$D" "$OUT/final_hgb_test.npy"
  lgb make_c8     "$PY_CAT" make_c8.py "$D" "$RAW" "$C" "$OUT/final_hgb_test.npy"
  lgb make_c9     "$PY_CAT" make_c8.py "$D" "$RAW" "$C" "$OUT/final_hgb_test.npy" rule27
fi
if stage 2; then echo "[2/8] training tables with 2 and 4 cutoffs a month, repaired test features for 3110/3111"
  lgb build_smc   "$PY_LGB" build_smc.py "$B" "$D" 2026-06-30 "$D/train_cutoff_features.csv.gz" "$SMC"
  lgb build_d4    "$PY_LGB" build_dense.py "$B" "$D" 2026-06-30 "$D/train_cutoff_features.csv.gz" "$D4" 7,14,21
  lgb fix_test    "$PY_LGB" fix_test_yoy.py "$D" "$T"
fi
if stage 3; then echo "[3/8] att set (+ SSG attendance) and test event flags -> C10"
  lgb final_ev    "$PY_LGB" final_att.py "$D" "$ATT" "$OUT" -
  lgb att_AEH     "$PY_LGB" final_att.py "$D" "$ATT" "$OUT" AEH
  lgb att_C       "$PY_CAT" final_att.py "$D" "$ATT" "$OUT" C
  lgb make_c10    "$PY_CAT" make_c10.py "$D" "$RAW" "$C" "$OUT"
fi
if stage 4; then echo "[4/8] sets attwx / att_smc / attwx_smc / att_d4 -> C15"
  lgb attwx_AEH     WX=1 "$PY_LGB" final_att.py "$D" "$ATT" "$OUT" AEH
  lgb attwx_C       WX=1 "$PY_CAT" final_att.py "$D" "$ATT" "$OUT" C
  lgb att_smc_AEH   TRAIN_FILE="$SMC" TAG=_smc "$PY_LGB" final_att.py "$D" "$ATT" "$OUT" AEH
  lgb att_smc_C     TRAIN_FILE="$SMC" TAG=_smc "$PY_CAT" final_att.py "$D" "$ATT" "$OUT" C
  lgb attwx_smc_AEH WX=1 TRAIN_FILE="$SMC" TAG=_smc "$PY_LGB" final_att.py "$D" "$ATT" "$OUT" AEH
  lgb attwx_smc_C   WX=1 TRAIN_FILE="$SMC" TAG=_smc "$PY_CAT" final_att.py "$D" "$ATT" "$OUT" C
  lgb att_d4_AEH    TRAIN_FILE="$D4" TAG=_d4 "$PY_LGB" final_att.py "$D" "$ATT" "$OUT" AEH
  lgb att_d4_C      TRAIN_FILE="$D4" TAG=_d4 "$PY_CAT" final_att.py "$D" "$ATT" "$OUT" C
  lgb make_c15      "$PY_CAT" make_c12.py "$D" "$RAW" "$C" "$OUT" C15_avg6_d4 "$SETS"
fi
if stage 5; then echo "[5/8] 3110/3111 sets retrained without the Geomdan-opening break rows (DROP_BREAK=1, repaired test features) -> C17"
  CL="DROP_BREAK=1 TEST_FILE=$T OMP_NUM_THREADS=8"
  for spec in "base|NOATT=1|_cl" "att||_cl" "attwx|WX=1|_cl" "att_smc|TRAIN_FILE=$SMC|_smc_cl" "attwx_smc|WX=1 TRAIN_FILE=$SMC|_smc_cl" "att_d4|TRAIN_FILE=$D4|_d4_cl"; do
    IFS='|' read -r s ev tag <<< "$spec"
    lgb cl_${s}_AEH $CL $ev TAG=$tag "$PY_LGB" final_att.py "$D" "$ATT" "$OUT" AEH
    lgb cl_${s}_C   $CL $ev TAG=$tag "$PY_CAT" final_att.py "$D" "$ATT" "$OUT" C
  done
  lgb make_c17 "$PY_CAT" make_c17.py "$D" "$RAW" "$C" "$OUT" C17_geomdan_6sets C15_avg6_d4 "$SETS" "$CLSETS"
fi
if stage 6; then echo "[6/8] all-station residual base sets att_ball / att_ball_smc (CatBoost on the GPU) -> C19"
  BA="BASE_ALL=1 DROP_BREAK=1 TEST_FILE=$T"
  lgb ball_AEH      $BA TAG=_ball OMP_NUM_THREADS=12 "$PY_LGB" final_att.py "$D" "$ATT" "$OUT" AEH
  lgb ball_C        $BA TAG=_ball CAT_GPU=1 "$PY_CAT" final_att.py "$D" "$ATT" "$OUT" C
  lgb ball_smc_AEH  $BA TRAIN_FILE="$SMC" TAG=_ball_smc OMP_NUM_THREADS=12 "$PY_LGB" final_att.py "$D" "$ATT" "$OUT" AEH
  lgb ball_smc_C    $BA TRAIN_FILE="$SMC" TAG=_ball_smc CAT_GPU=1 "$PY_CAT" final_att.py "$D" "$ATT" "$OUT" C
  lgb make_c19      "$PY_CAT" make_c18.py "$D" "$RAW" "$C" "$OUT" C19_add_ball_smc C15_avg6_d4 "$SETS" "$CLSETS" att_ball,att_ball_smc C17_geomdan_6sets
fi
if stage 7; then echo "[7/8] longer LightGBM: residual E 3000 trees (7 sets), direct A 4000 trees (5 files serve 7 sets)"
  E3="LGB_ROUNDS=3000 DROP_BREAK=1 TEST_FILE=$T OMP_NUM_THREADS=8"; A4="LGB_ROUNDS=4000 DROP_BREAK=1 TEST_FILE=$T OMP_NUM_THREADS=8"
  for spec in "att||_e3k" "att_ball|BASE_ALL=1|_ball_e3k" "att_smc|TRAIN_FILE=$SMC|_smc_e3k" "att_ball_smc|BASE_ALL=1 TRAIN_FILE=$SMC|_ball_smc_e3k" \
              "attwx|WX=1|_e3k" "attwx_smc|WX=1 TRAIN_FILE=$SMC|_smc_e3k" "att_d4|TRAIN_FILE=$D4|_d4_e3kp"; do
    IFS='|' read -r s ev tag <<< "$spec"; lgb e3k_$s $E3 $ev TAG=$tag "$PY_LGB" final_att.py "$D" "$ATT" "$OUT" E
  done
  for spec in "att||_a4k" "att_smc|TRAIN_FILE=$SMC|_smc_a4k" "attwx|WX=1|_a4k" "attwx_smc|WX=1 TRAIN_FILE=$SMC|_smc_a4k" "att_d4|TRAIN_FILE=$D4|_d4_a4k"; do
    IFS='|' read -r s ev tag <<< "$spec"; lgb a4k_$s $A4 $ev TAG=$tag "$PY_LGB" final_att.py "$D" "$ATT" "$OUT" A
  done
fi
if stage 8; then echo "[8/8] C25c (blend) -> C29 (weekend/holiday correction) and comparison with the submitted file"
  lgb make_c25c A4K_KEEP27=1 C19ADD=att_ball,att_ball_smc E3K=att,att_ball,att_smc,att_ball_smc,attwx,attwx_smc,att_d4:att_d4_e3kp \
      A4K=att:att_a4k,att_ball:att_a4k,att_smc:att_smc_a4k,att_ball_smc:att_smc_a4k,attwx:attwx_a4k,attwx_smc:attwx_smc_a4k,att_d4:att_d4_a4k \
      "$PY_CAT" make_c23.py "$D" "$RAW" "$C" "$OUT" C25c_c19_e3k7_a4k7_keep27 C15_avg6_d4 "$SETS" "$CLSETS" att_ball,att_ball_smc C17_geomdan_6sets C19_add_ball_smc
  lgb make_c29 "$PY_LGB" make_c27.py "$RAW" "$D" "$C" "$OUT" C25c_c19_e3k7_a4k7_keep27 C29_c25c_wkfix_hr 0.285 0.190 0.105
  "$PY_LGB" check_final.py "$RAW/test.csv" "$C/submission_C29_c25c_wkfix_hr.csv"
  C="$C" "$PY_LGB" - <<'PYEOF'
import os
import numpy as np, pandas as pd
a = pd.read_csv(os.path.join(os.environ['C'], 'submission_C29_c25c_wkfix_hr.csv')).boardings.to_numpy()
b = pd.read_csv(os.path.join(os.environ['S'], 'predictions', 'submission_C29_c25c_wkfix_hr.csv')).boardings.to_numpy()
la, lb = np.log1p(a), np.log1p(b)
print(f'retrained C29 vs submitted: max |diff| {np.abs(a - b).max():.4f} boardings, rmse in log1p {np.sqrt(((la - lb) ** 2).mean()):.6f}')
PYEOF
fi
echo "done $(date +%H:%M)"
