#!/usr/bin/env python
import time
import json
import requests
from pathlib import Path
from tqdm import tqdm
import argparse

BASE_URL = "http://localhost:8000/v1"
API_KEY  = "dummy-key"
MODEL    = "models/qwen25-15b"

SESSION = requests.Session()

SYSTEM_PROMPT = """
Eres lingüista nativo de español. Evalúa y limpia frases cortas en español.
Responde sólo con JSON como:
[{"id":0,"keep":true,"clean":"frase limpia"}, ...]
""".strip()

BATCH_SIZE = 32
MAX_CHARS_PER_BATCH = 900
MAX_CHARS_PER_LINE = 80

def safe_parse_json(s: str):
    s = s.strip()
    i = s.find("[")
    if i > 0:
        s = s[i:]
    return json.loads(s)

def build_user_content(batch):
    lines = [
        "Evalúa las siguientes frases.",
        "Devuelve JSON con objetos {\"id\":N,\"keep\":true/false,\"clean\":\"...\"}.",
        "",
        "Frases:",
    ]
    for obj in batch:
        lines.append(f"{obj['id']}: {obj['text']}")
    return "\n".join(lines)

def call_llm(batch):
    """
    Отправить один батч в LLM.
    Для лоад-теста:
      - игнорируем ошибки JSON-парсинга,
      - на любые исключения пишем в лог и считаем batch проваленным.
    """
    user_content = build_user_content(batch)
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        "temperature": 0.0,
    }

    try:
        resp = SESSION.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()

        data = resp.json()
        raw = data["choices"][0]["message"]["content"]

        # Для лоад-теста парсим только "для вида" и не падаем при ошибках
        try:
            _ = safe_parse_json(raw)
        except Exception:
            # Модель могла выдать невалидный JSON — для теста просто игнорируем
            pass

        return len(batch)

    except Exception as e:
        # Любая ошибка запроса/JSON — логируем и считаем, что этот батч не обработан
        print("[worker error]", repr(e))
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, default=0, help="worker id (для логов)")
    ap.add_argument("--limit", type=int, default=10000,
                    help="сколько строк обработать этим воркером")
    ap.add_argument("--source", type=str,
                    default="corpus/split/es.3w.soft.txt",
                    help="файл с 3-словными фразами")
    args = ap.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"no source file: {source}")

    total_done = 0
    batch = []
    batch_chars = 0

    t0 = time.time()

    with source.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(tqdm(f, desc=f"worker {args.id}")):
            if total_done >= args.limit:
                break

            phrase = line.rstrip("\n")
            if not phrase:
                continue
            if len(phrase) > MAX_CHARS_PER_LINE:
                phrase = phrase[:MAX_CHARS_PER_LINE]

            local_id = len(batch)
            batch.append({"id": local_id, "text": phrase})
            batch_chars += len(phrase)

            if len(batch) >= BATCH_SIZE or batch_chars >= MAX_CHARS_PER_BATCH:
                total_done += call_llm(batch)
                batch.clear()
                batch_chars = 0

        if batch:
            total_done += call_llm(batch)

    dt = time.time() - t0
    print(
        f"[worker {args.id}] done {total_done} phrases "
        f"in {dt:.2f}s ({total_done/dt:.2f} phrases/s)"
    )

if __name__ == "__main__":
    main()
