#!/bin/bash
# Retraining spot check (2026-11-13): rerun one job of each kind with the commands of run_full.sh and compare the result with the
# shipped artifact (work/out, claude_works/candidates). Output goes to work/out_spot (not shipped). Log: records/repro_check/spot_check.log
set -uo pipefail
cd "$(dirname "$0")/../.."; source paths.sh
O="$W/out_spot"; mkdir -p "$O/cand"; cd "$W/code"; T="$W/out/test_fix.csv.gz"
cmp_npy() { "$PY_LGB" -c "import numpy as np,sys; a=np.load(sys.argv[1]); b=np.load(sys.argv[2]); d=float(np.abs(a.astype(float)-b.astype(float)).max()); print(sys.argv[3], 'IDENTICAL' if np.array_equal(a,b) else f'max |diff| {d:.3e} (log1p)')" "$1" "$2" "$3"; }
cmp_file() { "$PY_LGB" -c "import hashlib,sys; h=[hashlib.sha256(open(p,'rb').read()).hexdigest() for p in sys.argv[1:3]]; print(sys.argv[3], 'IDENTICAL sha256' if h[0]==h[1] else 'sha differs '+h[0][:12]+' '+h[1][:12])" "$1" "$2" "$3"; }
(
  "$PY_LGB" final_lgb.py "$B" "$D" "$RAW" "$O/cand" > "$O/final_lgb.log" 2>&1
  for n in C0_lgb_direct C1_lgb_seasonal_resid C2_resid_v2_20_nonevent C3_C2_3127_avg_direct; do cmp_file "$O/cand/submission_$n.csv" "$CAND/submission_$n.csv" "final_lgb.py $n"; done
  "$PY_LGB" fix_test_yoy.py "$D" "$O/test_fix.csv.gz" > "$O/fix.log" 2>&1
  "$PY_LGB" -c "import pandas as pd,sys; a=pd.read_csv(sys.argv[1]); b=pd.read_csv(sys.argv[2]); print('fix_test_yoy.py test_fix.csv.gz', 'IDENTICAL content' if a.equals(b) else 'content differs')" "$O/test_fix.csv.gz" "$T"
  "$PY_LGB" final_att.py "$D" "$ATT" "$O" - > "$O/ev.log" 2>&1; cmp_npy "$O/final_att_ev.npy" "$W/out/final_att_ev.npy" "final_att.py - final_att_ev.npy"
) &
(
  OMP_NUM_THREADS=12 "$PY_LGB" final_hgb.py "$D" "$O/final_hgb_test.npy" > "$O/hgb.log" 2>&1; cmp_npy "$O/final_hgb_test.npy" "$W/out/final_hgb_test.npy" "final_hgb.py H (old set)"
) &
(
  env LGB_ROUNDS=4000 DROP_BREAK=1 TEST_FILE="$T" OMP_NUM_THREADS=8 TAG=_a4k "$PY_LGB" final_att.py "$D" "$ATT" "$O" A > "$O/a4k.log" 2>&1
  cmp_npy "$O/final_att_a4k_A.npy" "$W/out/final_att_a4k_A.npy" "final_att.py A 4000 trees (att)"
) &
(
  env BASE_ALL=1 DROP_BREAK=1 TEST_FILE="$T" TAG=_ball CAT_GPU=1 "$PY_CAT" final_att.py "$D" "$ATT" "$O" C > "$O/ball_C.log" 2>&1
  cmp_npy "$O/final_att_ball_C.npy" "$W/out/final_att_ball_C.npy" "final_att.py C CatBoost GPU (att_ball)"
  "$PY_CAT" final_att.py "$D" "$ATT" "$O" C > "$O/att_C.log" 2>&1
  cmp_npy "$O/final_att_C.npy" "$W/out/final_att_C.npy" "final_att.py C CatBoost CPU (att)"
) &
wait; echo "spot check done $(date +%H:%M)"
