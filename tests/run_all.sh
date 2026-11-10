#!/bin/bash
# Runs the checks of tests/ and prints PASSED / FAILED / SKIPPED per check. Needs the released input files in data/
# (python scripts/data/fetch_data.py) and, for the GPU stages, one CUDA GPU with about 3 GB free.
#   GPU=0 PY=python STAGES="presets data kernel exact dt balance smoke" bash tests/run_all.sh
# Stages: presets (CPU) presets and run tables; data (CPU) input files of us, aus and gb; kernel (GPU) gradient of the
# production-store kernel; exact (GPU) the uniform-input store kernels and their hand-written adjoints at dt = 1 and 1/24
# (tests/check_exact.py); dt (GPU) step length, rate-form unit hydrographs, hourly water balance and step invariance on gb (tests/check_dt.py);
# balance (GPU) full-system water balance of CaRT+ on us and aus; smoke (GPU, ~1 min) 50 training
# iterations of CaRT+ on us, the forward run of the trained network and the water balance of that run (written to $TMP_RUNS).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}; GPU=${GPU:-0}
STAGES=${STAGES:-"presets data kernel exact dt balance smoke"}
TMP_RUNS=${TMP_RUNS:-$(mktemp -d)}
export PYTHONPATH=$PWD/src PYTHONDONTWRITEBYTECODE=1
has() { [[ " $STAGES " == *" $1 "* ]]; }
fail=0
run() { echo "## $*"; "$@" | grep -v "^SUMMARY"; [ "${PIPESTATUS[0]}" -eq 0 ] || fail=1; }
has presets && run $PY tests/check_presets.py
if has data; then for d in us aus gb; do run $PY tests/check_data.py $d; done; fi
has kernel && run $PY tests/check_kernel_grads.py --gpu $GPU
if has exact; then for dt in 1.0 0.041666666666666664; do run $PY tests/check_exact.py --gpu $GPU --dt $dt; done; fi
has dt && run $PY tests/check_dt.py --gpu $GPU
if has balance; then
  run $PY tests/check_balance.py --dataset us --gpu $GPU
  run $PY tests/check_balance.py --dataset aus --gpu $GPU
fi
if has smoke; then
  run $PY -m cart.train --model cartplus --dataset us --seed 0 --iters 50 --eval_every 25 --gpu $GPU --out $TMP_RUNS/us_cartplus_s0
  run $PY -m cart.predict --run $TMP_RUNS/us_cartplus_s0 --gpu $GPU
  run $PY tests/check_balance.py --run $TMP_RUNS/us_cartplus_s0 --gpu $GPU
fi
echo "## done: $([ $fail -eq 0 ] && echo 'all checks passed' || echo 'some checks FAILED')"
exit $fail
