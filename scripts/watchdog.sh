#!/bin/bash
# Overnight watchdog - 3-arm edition (Modal Starter plan caps concurrent GPU jobs at 3).
#
# Runs exactly three arms: pooled, mot, baseline. If any goes stale (its streamed log stops
# advancing => the detached job died), relaunch it - train() auto-resumes from the newest
# checkpoint on the volume, so a relaunch loses at most ~1000 steps. Arms that reach "DONE"
# are left finished. No queuing: routed and sota are intentionally NOT run tonight (their
# checkpoints are safe on the volume for a later session).
#
# Deliberately never launches a 4th arm - exceeding the 3-slot cap is what killed all five
# earlier tonight.

cd /Users/vk55.karthik/domain
ARMS="pooled mot baseline"
STALE_S=420      # 7 min without a log write => presumed dead
CYCLE_S=300      # check every 5 min
LOG=/tmp/watchdog.log

echo "$(date '+%F %T') watchdog (3-arm) started (pid $$): $ARMS" >> "$LOG"
while true; do
  for arm in $ARMS; do
    f="/tmp/relaunch_${arm}.log"
    if [ -f "$f" ] && grep -q "DONE ${arm}:" "$f"; then
      continue  # finished all 150k steps
    fi
    now=$(date +%s)
    mt=0; [ -f "$f" ] && mt=$(stat -f %m "$f" 2>/dev/null || echo 0)
    if [ $((now - mt)) -gt $STALE_S ]; then
      echo "$(date '+%F %T') ${arm} stale ($((now-mt))s) -> relaunch" >> "$LOG"
      nohup python3 -m modal run --detach stage2_modal.py --step train --arm "$arm" --steps 150000 \
        > "$f" 2>&1 &
      sleep 25
    fi
  done
  sleep $CYCLE_S
done
