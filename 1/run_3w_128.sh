#!/usr/bin/env bash
set -euo pipefail

PARTS=(corpus/split_shards_3w/es.3w.part_*)

mkdir -p corpus/3w_llm/text
mkdir -p corpus/3w_llm/jsonl
mkdir -p corpus/3w_llm/checkpoints
mkdir -p logs_3w

idx=0
for src in "${PARTS[@]}"; do
  id=$(printf "%03d" "$idx")

  python filter_3w_worker.py \
    --source "$src" \
    --out-text "corpus/3w_llm/text/es3.${id}.clean.txt" \
    --out-meta "corpus/3w_llm/jsonl/es3.${id}.meta.jsonl" \
    --checkpoint "corpus/3w_llm/checkpoints/es3.${id}.ckpt.json" \
    > "logs_3w/worker_${id}.log" 2>&1 &

  idx=$((idx + 1))
done

wait
echo "All 3w workers finished."
