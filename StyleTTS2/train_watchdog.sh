#!/bin/bash
# Watchdog for a StyleTTS2 training run. Exits (re-invoking the agent) when:
#   - NaN appears in a loss line
#   - the log stalls (no new loss line for 30 min)
#   - the heartbeat interval elapses (periodic progress check-in)
# Usage: train_watchdog.sh <logfile> [heartbeat_seconds]
LOG="$1"
HEARTBEAT="${2:-7200}"
START=$(date +%s)

while true; do
  sleep 300
  if grep -qiE "mel loss: *nan|loss: *-?inf" "$LOG"; then
    echo "WATCHDOG: NaN/Inf detected in losses"
    grep -iE "nan|inf" "$LOG" | tail -3
    exit 2
  fi
  if [ -n "$(find "$LOG" -mmin +30)" ]; then
    echo "WATCHDOG: log stalled — no output for 30+ minutes"
    tail -5 "$LOG"
    exit 3
  fi
  NOW=$(date +%s)
  if [ $((NOW - START)) -ge "$HEARTBEAT" ]; then
    echo "WATCHDOG: heartbeat — training still progressing"
    grep -E "Epoch \[" "$LOG" | tail -2
    exit 0
  fi
done
