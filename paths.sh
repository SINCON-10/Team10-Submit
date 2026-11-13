#!/bin/bash
# usage:  source paths.sh      (every path is relative to this package; works in Git Bash on Windows, Linux, Colab)
R="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export S="$R"                                      # package root (data/, claude_works/)
export W="$R/work"                                 # work/code = scripts, work/out = model outputs and training tables
export D="$R/data/processed_v4"; export PV4="$D"   # team preprocessed data v4 (cutoff features, preprocess.py, lunar_adjust.py, weather)
export RAW="$R/data/raw"                           # competition files train.csv / test.csv / metaData.csv (put them here)
export CAND="$R/claude_works/candidates"           # candidate CSVs of the C29 chain and the saved old-set models
export ATT="$R/claude_works/external/ssg_munhak_attendance.csv"
export B="$W/bundle/sincon_v4_colab_bundle"        # team v4 code bundle (DataEngine, v2 structural baseline)
# Two interpreters were used (see README 4). One environment that has every package also works.
export PY_LGB="${PY_LGB:-python}"                  # env 1: LightGBM / scikit-learn / pandas (requirements-lgb.txt)
export PY_CAT="${PY_CAT:-python}"                  # env 2: CatBoost (requirements-cat.txt)
export SETS=att,attwx,att_smc,attwx_smc,att_d4     # new model sets averaged since C15
export CLSETS=base,att,attwx,att_smc,attwx_smc,att_d4   # sets retrained without the 3110/3111 break rows (C17)
export PYTHONIOENCODING=utf-8
