#!/usr/bin/env python
import argparse
import json
import os
import time
import re
import requests
from pathlib import Path
from tqdm import tqdm

# ==========================
#   АРГУМЕНТЫ
# ==========================

parser = argparse.ArgumentParser()
parser.add_argument(
    "--source",
    required=True,
    help="входной файл с 3-словными фразами (одна фраза в строке)",
)
parser.add_argument(
    "--out-text",
    required=True,
    help="выходной текстовый корпус (чистые фразы, по одной в строке)",
)
parser.add_argument(
    "--out-meta",
    required=True,
    help="выходной jsonl с метаданными",
)
parser.add_argument(
    "--checkpoint",
    required=True,
    help="файл чекпоинта (номер последней обработанной строки исходника)",
)
args = parser.parse_args()

SOURCE_CORPUS = Path(args.source)
OUTPUT_TEXT   = Path(args.out_text)
OUTPUT_META   = Path(args.out_meta)
CHECKPOINT    = Path(args.checkpoint)

# ==========================
#   ПАРАМЕТРЫ
# ==========================

# 3-словные фразы короткие, но чтобы точно не выбивать 1024 токена:
BATCH_SIZE = 32              # строк на запрос
MAX_CHARS_PER_BATCH = 900    # суммарная длина строк
MAX_CHARS_PER_LINE  = 80     # обрезать аномально длинное

BASE_URL = "http://localhost:8000/v1"
API_KEY  = "dummy-key"
MODEL    = "models/qwen25-15b"   # как в vllm serve

SESSION = requests.Session()

# ==========================
#   ЧЕКПОИНТ
# ==========================

def load_checkpoint() -> int:
    if CHECKPOINT.exists():
        try:
            data = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
            return int(data.get("last_line", -1))
        except Exception:
            return -1
    return -1


def save_checkpoint(last_line: int) -> None:
    tmp = CHECKPOINT.with_suffix(".tmp")
    tmp.write_text(json.dumps({"last_line": last_line}), encoding="utf-8")
    tmp.replace(CHECKPOINT)

# ==========================
#   LLM
# ==========================

SYSTEM_PROMPT = """
Eres lingüista nativo de español y especialista en enseñanza de español.

Tarea:
- Recibes frases cortas originales en español (unas 3 palabras).
- Para cada frase decides si sirve para tarjetas de estudio de español para extranjeros.
- Si sirve (keep = true), devuelves también una versión limpia.

BUENA FRASE (keep = true):
- 2–4 palabras tras limpiar.
- Oración o réplica casi completa.
- Útil en situaciones cotidianas.
- Español moderno y natural.

MALA FRASE (keep = false):
- Fragmento sin sentido claro.
- Demasiados nombres propios, lugares, títulos, marcas.
- Fechas, horas, números concretos.
- Ruido de subtítulos (aplausos, risas, indicaciones técnicas).
- No está en español o es muy rara.

Si keep = true, campo "clean":
- sólo palabras en español en minúsculas,
- sin signos de puntuación, números ni símbolos,
- sin nombres propios evidentes,
- palabras separadas por un solo espacio.

FORMATO DE RESPUESTA:
Responde SOLO con JSON:
[
  {"id": 0, "keep": true,  "clean": "frase limpia en minusculas"},
  {"id": 1, "keep": false, "clean": ""}
]
""".strip()


def safe_parse_json(s: str):
    s = s.strip()
    i = s.find("[")
    if i > 0:
        s = s[i:]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    last_brace = s.rfind("}")
    if last_brace != -1:
        s2 = s[: last_brace + 1]
        if not s2.strip().endswith("]"):
            s2 += "]"
        try:
            return json.loads(s2)
        except json.JSONDecodeError:
            pass

    objs = re.findall(r"\{[^{}]*\}", s, flags=re.DOTALL)
    if objs:
        candidate = "[" + ",".join(objs) + "]"
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError("JSON parse error from LLM, first 500 chars:\n" + s[:500])


def build_user_content(batch):
    """
    batch: [{"id": int, "text": str}, ...]
    Формируем компактный текст вместо JSON:
    'Frases:\n0: ...\n1: ...'
    """
    lines = [
        "Evalúa las siguientes frases en español.",
        "Para cada línea devuelve JSON con objetos {\"id\": N, \"keep\": true/false, \"clean\": \"...\"}.",
        "",
        "Frases:",
    ]
    for obj in batch:
        lines.append(f"{obj['id']}: {obj['text']}")
    return "\n".join(lines)


def call_llm(batch):
    """
    batch: [{"id": int, "text": str}, ...]
    -> dict[id] = {"keep": bool, "clean": str}
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

    while True:
        try:
            resp = SESSION.post(
                f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json=payload,
                timeout=120,
            )
        except Exception as e:
            print("[LLM ERROR] request failed, retrying:", e)
            time.sleep(5)
            continue

        if resp.status_code == 200:
            break

        print("[LLM ERROR] HTTP", resp.status_code, resp.text[:200])
        time.sleep(5)

    data = resp.json()
    raw_content = data["choices"][0]["message"]["content"].strip()

    try:
        arr = safe_parse_json(raw_content)
    except Exception as e:
        print("[JSON ERROR] batch-level error:", e)
        print("[JSON ERROR] falling back to per-item evaluation…")

        results = {}
        for obj in batch:
            single_content = build_user_content([obj])
            single_payload = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": single_content},
                ],
                "temperature": 0.0,
            }

            ok = False
            for attempt in range(3):
                try:
                    r = SESSION.post(
                        f"{BASE_URL}/chat/completions",
                        headers={"Authorization": f"Bearer {API_KEY}"},
                        json=single_payload,
                        timeout=120,
                    )
                    if r.status_code != 200:
                        time.sleep(2)
                        continue

                    c = r.json()["choices"][0]["message"]["content"].strip()
                    parsed = safe_parse_json(c)
                    item = parsed[0]
                    results[obj["id"]] = {
                        "keep": bool(item.get("keep", False)),
                        "clean": (item.get("clean") or "").strip(),
                    }
                    ok = True
                    break
                except Exception:
                    time.sleep(1)

            if not ok:
                results[obj["id"]] = {"keep": False, "clean": ""}

        return results

    result = {}
    for item in arr:
        idx = int(item["id"])
        result[idx] = {
            "keep": bool(item.get("keep", False)),
            "clean": (item.get("clean") or "").strip(),
        }
    return result

# ==========================
#   MAIN
# ==========================

def main():
    OUTPUT_TEXT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_META.parent.mkdir(parents=True, exist_ok=True)

    resume_from = load_checkpoint()
    print("Resume from source line:", resume_from)

    mode = "a" if resume_from >= 0 and OUTPUT_TEXT.exists() else "w"

    with SOURCE_CORPUS.open("r", encoding="utf-8") as inp, \
            OUTPUT_TEXT.open(mode, encoding="utf-8") as out_txt, \
            OUTPUT_META.open(mode, encoding="utf-8") as out_meta:

        batch_records = []
        batch_for_llm = []
        batch_chars = 0
        last_line_no = resume_from
        batch_counter = 0

        for line_no, line in enumerate(tqdm(inp, desc=f"scanning {SOURCE_CORPUS.name}")):
            if line_no <= resume_from:
                continue

            phrase = line.rstrip("\n")
            if not phrase:
                continue

            if len(phrase) > MAX_CHARS_PER_LINE:
                phrase = phrase[:MAX_CHARS_PER_LINE]

            local_id = len(batch_for_llm)
            batch_records.append((line_no, phrase))
            batch_for_llm.append({"id": local_id, "text": phrase})
            batch_chars += len(phrase)
            last_line_no = line_no

            if len(batch_for_llm) >= BATCH_SIZE or batch_chars >= MAX_CHARS_PER_BATCH:
                try:
                    decisions = call_llm(batch_for_llm)
                except Exception as e:
                    print("[JSON ERROR] batch failed, skipping batch:", str(e)[:200])
                    decisions = {
                        obj["id"]: {"keep": False, "clean": ""}
                        for obj in batch_for_llm
                    }

                kept = sum(1 for d in decisions.values() if d["keep"])
                print(
                    f"[info] batch #{line_no // max(1, BATCH_SIZE)}: "
                    f"kept {kept} of {len(batch_for_llm)}"
                )

                for idx_in_batch, (ln, orig_phrase) in enumerate(batch_records):
                    dec = decisions.get(idx_in_batch)
                    if dec and dec["keep"]:
                        clean = dec.get("clean", "").strip()
                        if not clean:
                            continue

                        out_txt.write(clean + "\n")
                        meta = {
                            "line_no": ln,
                            "orig": orig_phrase,
                            "clean": clean,
                            "llm_keep": True,
                        }
                        out_meta.write(json.dumps(meta, ensure_ascii=False) + "\n")

                batch_counter += 1
                if batch_counter % 50 == 0:
                    out_txt.flush()
                    out_meta.flush()
                    os.fsync(out_txt.fileno())
                    os.fsync(out_meta.fileno())

                if last_line_no >= 0:
                    save_checkpoint(last_line_no)

                batch_records.clear()
                batch_for_llm.clear()
                batch_chars = 0

        # хвостовый батч
        if batch_for_llm:
            try:
                decisions = call_llm(batch_for_llm)
            except Exception as e:
                print("[JSON ERROR] tail batch failed, skipping:", str(e)[:200])
                decisions = {
                    obj["id"]: {"keep": False, "clean": ""}
                    for obj in batch_for_llm
                }

            for idx_in_batch, (ln, orig_phrase) in enumerate(batch_records):
                dec = decisions.get(idx_in_batch)
                if dec and dec["keep"]:
                    clean = dec.get("clean", "").strip()
                    if not clean:
                        continue

                    out_txt.write(clean + "\n")
                    meta = {
                        "line_no": ln,
                        "orig": orig_phrase,
                        "clean": clean,
                        "llm_keep": True,
                    }
                    out_meta.write(json.dumps(meta, ensure_ascii=False) + "\n")

        out_txt.flush()
        out_meta.flush()
        os.fsync(out_txt.fileno())
        os.fsync(out_meta.fileno())

        if last_line_no >= 0:
            save_checkpoint(last_line_no)

    print("Clean 3-word shard written to:", OUTPUT_TEXT.resolve())
    print("Meta index written to:", OUTPUT_META.resolve())


if __name__ == "__main__":
    main()
