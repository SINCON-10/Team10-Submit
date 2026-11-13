#!/bin/bash
# usage: cl_run.sh SET MODELS  -- retrain one model set without the 3110/3111 break rows, predict on the repaired test file
# writes out/final_{SET}_cl_{k}.npy (base -> final_base_cl_k) and out/cl_{SET}_{MODELS}.log
# package: paths come from paths.sh (was the author's local paths)
source "$(dirname "$0")/../../paths.sh"
ATT=$S/claude_works/external/ssg_munhak_attendance.csv; T=$W/out/test_fix.csv.gz
SET=$1; M=$2; ENVS="DROP_BREAK=1 TEST_FILE=$T OMP_NUM_THREADS=8"
case $SET in
  base) ENVS="$ENVS NOATT=1 TAG=_cl";;
  att) ENVS="$ENVS TAG=_cl";;
  attwx) ENVS="$ENVS WX=1 TAG=_cl";;
  att_smc) ENVS="$ENVS TRAIN_FILE=$W/out/final_train_smc.parquet TAG=_smc_cl";;
  attwx_smc) ENVS="$ENVS WX=1 TRAIN_FILE=$W/out/final_train_smc.parquet TAG=_smc_cl";;
  att_d4) ENVS="$ENVS TRAIN_FILE=$W/out/final_train_d4.parquet TAG=_d4_cl";;
  *) echo "unknown set $SET"; exit 1;;
esac
if [ "$M" = C ]; then PY="$PY_CAT"; else PY="$PY_LGB"; fi   # package: interpreters from paths.sh
cd $W/code && env $ENVS "$PY" final_att.py $D $ATT $W/out $M > $W/out/cl_${SET}_${M}.log 2>&1
echo "$SET $M exit $? $(date +%H:%M) | $(tail -n 1 $W/out/cl_${SET}_${M}.log)"
