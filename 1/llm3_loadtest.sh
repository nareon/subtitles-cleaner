# head -n 160000 ../corpus/split/es.3w.soft.txt > corpus/split/es.3w.sample160k.txt
for i in $(seq 0 127); do
  python llm3_loadtest.py \
    --id "$i" \
    --limit 10000 \
    --source corpus/split/es.3w.sample160k.txt \
    > logs/worker_${i}.log 2>&1 &
done

wait