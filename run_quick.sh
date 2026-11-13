#!/bin/bash
# Quick reproduction (no training, about 5 minutes): rebuild C25c and C29 from the shipped model outputs (work/out/final_*.npy)
# and compare with the submitted file predictions/submission_C29_c25c_wkfix_hr.csv.
# usage:  bash run_quick.sh        (set PY_LGB / PY_CAT first if the two environments are separate, see README 4)
set -euo pipefail
cd "$(dirname "$0")"; source paths.sh

echo "[0/3] provided data (data/raw) sha256"
"$PY_LGB" - <<'PYEOF'
import hashlib, os, sys
want = {'train.csv': '1b3b21e8f1fe22a44c0ff4adba315784a5c21caea25098a15869c4ed370822bc',
        'test.csv': 'd257fed268db7919fd8fbc45b6c78dfbdb09b6bb40d4ae1d3f2a666c6e69fefa',
        'metaData.csv': '9870dc6c9cbe7e6fc810bbe589dd9e06f309bb173a268a1bd63478f430785380'}
bad = 0
for f, h in want.items():
    p = os.path.join(os.environ['RAW'], f)
    got = hashlib.sha256(open(p, 'rb').read()).hexdigest() if os.path.exists(p) else 'missing'
    print(f'  {f:13s}', 'OK' if got == h else f'DIFFERENT ({got[:12]})'); bad += got != h
sys.exit(1 if bad else 0)
PYEOF

cd "$W/code"
echo "[1/3] C25c: rebuild C15 / C17 / C19 from the pieces (asserted inside), then 3000-tree E in 7 sets and 4000-tree A in 7 sets"
A4K_KEEP27=1 C19ADD=att_ball,att_ball_smc \
E3K=att,att_ball,att_smc,att_ball_smc,attwx,attwx_smc,att_d4:att_d4_e3kp \
A4K=att:att_a4k,att_ball:att_a4k,att_smc:att_smc_a4k,att_ball_smc:att_smc_a4k,attwx:attwx_a4k,attwx_smc:attwx_smc_a4k,att_d4:att_d4_a4k \
"$PY_CAT" make_c23.py "$D" "$RAW" "$CAND" "$W/out" RQ_C25c C15_avg6_d4 "$SETS" "$CLSETS" att_ball,att_ball_smc C17_geomdan_6sets C19_add_ball_smc | head -n 1

echo "[2/3] C29 = C25c + weekend/holiday level correction (B1 .285, B2 .190) + weekday hour term (B3 .105)"
"$PY_LGB" make_c27.py "$RAW" "$D" "$CAND" "$W/out" RQ_C25c RQ_C29 0.285 0.190 0.105 | head -n 1

echo "[3/3] format check and comparison with the submitted file"
"$PY_LGB" check_final.py "$RAW/test.csv" "$CAND/submission_RQ_C29.csv"
"$PY_LGB" - <<'PYEOF'
import hashlib, os
import numpy as np, pandas as pd
a = os.path.join(os.environ['CAND'], 'submission_RQ_C29.csv'); b = os.path.join(os.environ['S'], 'predictions', 'submission_C29_c25c_wkfix_hr.csv')
ha, hb = (hashlib.sha256(open(p, 'rb').read()).hexdigest() for p in (a, b))
d = float(np.abs(pd.read_csv(a).boardings.to_numpy() - pd.read_csv(b).boardings.to_numpy()).max())
print(f'rebuilt C29 sha256 {ha[:16]} | submitted {hb[:16]} | max |diff| {d:.6f} boardings ->',
      'IDENTICAL' if ha == hb else ('OK (float formatting only)' if d < 1e-3 else 'DIFFERENT (check package versions)'))
PYEOF
