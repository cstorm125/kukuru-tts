#!/bin/bash
# Thai Kokoro fine-tune: Stage 1 → Stage 2 on ONE GPU (see docs/THAI_RUNBOOK.md
# for why single-GPU: DDP converges on paper but sounds worse — InstanceNorm
# micro-batch statistics; and lr > 1e-4 destabilizes the GAN).
#
# Usage: GPU=4 bash scripts/train_thai.sh [stage1|stage2|all]
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
STAGE="${1:-all}"
CONFIG=configs/thai_stage1.yml
PY=.venv/bin/python3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_stage() {
  local script=$1 log=$2
  echo "=== $script on GPU $GPU (log: $log) ==="
  (cd StyleTTS2 && CUDA_VISIBLE_DEVICES="$GPU" ../$PY "$script" \
      --config_path "../$CONFIG") > "$log" 2>&1 &
  local pid=$!
  # watchdog: die on NaN or 30-min stall
  while kill -0 $pid 2>/dev/null; do
    sleep 300
    if grep -qiE "mel loss: *nan|loss: *-?inf" "$log"; then
      echo "NaN detected — killing $script"; kill $pid; exit 2
    fi
    if [ -n "$(find "$log" -mmin +30)" ]; then
      echo "log stalled 30min — killing $script"; kill $pid; exit 3
    fi
    grep -E "Epoch \[|Validation" "$log" | tail -1 || true
  done
  wait $pid || { echo "$script exited nonzero — check $log"; exit 1; }
}

if [[ "$STAGE" == "stage1" || "$STAGE" == "all" ]]; then
  run_stage train_first.py train_stage1.log
fi
if [[ "$STAGE" == "stage2" || "$STAGE" == "all" ]]; then
  # Stage 2 note: batch 16 fits until the adversarial phase (joint_epoch)
  # begins, then OOMs on 80GB. Either set batch_size 8 in the config from
  # the start, or resume from the crash with thai_stage2_resume_example.yml.
  run_stage train_second.py train_stage2.log
fi
echo "done — best stage-1 ckpt: StyleTTS2/logs/thai/first_stage.pth"
echo "       stage-2 ckpts:     StyleTTS2/logs/thai/epoch_2nd_*.pth"
