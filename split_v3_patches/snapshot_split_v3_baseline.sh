#!/bin/bash
# snapshot_split_v3_baseline.sh — polls for baseline summary, then hard-link
# freezes the image cache + result artifacts. Intended to run in a workspace
# tmux (pv2_split_v3_snapshot) alongside the eval tmux.
#
# When split_v3_baseline_summary.json appears:
#   cache/split_v3               -> cache/split_v3_baseline_frozen         (hard-link, ~0 disk cost)
#   results/split_v3_baseline    -> results/split_v3_baseline_frozen        (hard-link)
#   baseline_cache.npz sha256    -> logged
#
# The frozen dirs are the canonical reference bytes for future FT evals to
# reuse via --recompute off. A wipe of cache/split_v3 will NOT touch the
# frozen copy (hard links keep inode alive until every path is unlinked).

set -euo pipefail

CACHE_SRC=/home/ray/default/paradigm_v2/cache/split_v3
CACHE_DST=/home/ray/default/paradigm_v2/cache/split_v3_baseline_frozen
RES_SRC=/home/ray/default/paradigm_v2/results/split_v3_baseline
RES_DST=/home/ray/default/paradigm_v2/results/split_v3_baseline_frozen
SUMMARY=$RES_SRC/split_v3_baseline_summary.json
LOG=/home/ray/default/logs/snapshot_split_v3_baseline.log

echo "[$(date -u +%FT%TZ)] snapshot watcher started; waiting for $SUMMARY" | tee -a "$LOG"

# Poll every 60s. Cheap; the eval takes hours.
while [ ! -s "$SUMMARY" ]; do
    sleep 60
done

echo "[$(date -u +%FT%TZ)] summary detected; taking snapshot" | tee -a "$LOG"

# 1. Hard-link the image cache.
if [ -d "$CACHE_DST" ]; then
    echo "[$(date -u +%FT%TZ)] WARN: $CACHE_DST already exists; refusing to overwrite" | tee -a "$LOG"
else
    cp -al "$CACHE_SRC" "$CACHE_DST"
    n_cache=$(find "$CACHE_DST" -type f | wc -l)
    du_cache=$(du -sh "$CACHE_DST" | awk '{print $1}')
    echo "[$(date -u +%FT%TZ)] cache snapshotted: $CACHE_DST  files=$n_cache  size=$du_cache" | tee -a "$LOG"
fi

# 2. Hard-link the result artifacts (baseline_cache.npz + metrics + summary).
if [ -d "$RES_DST" ]; then
    echo "[$(date -u +%FT%TZ)] WARN: $RES_DST already exists; refusing to overwrite" | tee -a "$LOG"
else
    cp -al "$RES_SRC" "$RES_DST"
    echo "[$(date -u +%FT%TZ)] results snapshotted: $RES_DST" | tee -a "$LOG"
fi

# 3. Record the baseline_cache.npz sha256 into the frozen dir so future FT
#    runs can verify byte-identity before trusting the cached embeddings.
NPZ=$RES_DST/baseline_cache.npz
if [ -s "$NPZ" ]; then
    SHA=$(sha256sum "$NPZ" | awk '{print $1}')
    echo "$SHA  baseline_cache.npz  frozen_at=$(date -u +%FT%TZ)" > "$RES_DST/BASELINE_SHA256.txt"
    echo "[$(date -u +%FT%TZ)] baseline_cache.npz sha256=$SHA" | tee -a "$LOG"
else
    echo "[$(date -u +%FT%TZ)] WARN: $NPZ missing after snapshot" | tee -a "$LOG"
fi

echo "[$(date -u +%FT%TZ)] snapshot done" | tee -a "$LOG"
