#!/usr/bin/env python
import json
import os
import time
import re
import requests
from pathlib import Path
from tqdm import tqdm

# ==========================
#   ПУТИ
# ==========================

SOURCE_CORPUS = Path("corpus/split/es.3w.soft.txt")

OUTPUT_TEXT         = Path("corpus/2.es3.llm_clean.txt")
OUTPUT_META_JSONL   = Path("corpus/jsonl/2.es3_llm_filtered.jsonl")
OUTPUT_META_PARQUET = Path("corpus/jsonl/2.es3_llm_filtered.parquet")
CHECKPOINT          = Path("corpus/2.es3.llm_filter_checkpoint.json")

# опциональный лог сырых ответов модели (для отладки)
DEBUG_RAW = Path("corpus/2.es3.llm_debug_raw.txt")

# ==========================
#   ПАРАМЕТРЫ
# ==========================

BATCH_SIZE = 32
MAX_CHARS_PER_BATCH = 900
MAX_CHARS_PER_LINE = 80

BASE_URL = "http://localhost:8000/v1"
API_KEY = "dummy-key"
MODEL = "models/qwen25-15b"

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
Eres lingüista nativo de español y especialista en enseñanza de español como lengua extranjera.

Objetivo general:
- Seleccionar frases MUY cortas en español (unas 3 palabras) que sirvan para tarjetas de estudio.
- Es mejor ser un poco PERMISIVO: si la frase es correcta, neutra y entendible, normalmente se acepta.

Recibes del usuario:
- Una lista numerada de frases originales en español.
- Cada línea tiene el formato: "ID: texto original".

Qué es una BUENA FRASE (keep = true):
- Frase corta y clara, que un estudiante podría encontrar en muchos contextos.
- Español correcto y natural (registro neutro o coloquial suave).
- Puede contener verbos, expresiones frecuentes, combinaciones útiles de sustantivo + adjetivo, etc.
- Puede contener un nombre propio si la frase sigue siendo útil (por ejemplo "estoy en madrid").
- No es necesario que sea una oración completa con sujeto + verbo; una expresión útil basta.

Qué es una MALA FRASE (keep = false):
- Fragmento roto sin sentido claro.
- Sólo lista de nombres, códigos o palabras sueltas sin relación.
- Mucho ruido de subtítulos, marcas de tiempo, descripciones técnicas, etc.
- No está en español.

Versión limpia cuando la frase se acepta:
- Mantén el MISMO significado básico de la frase original.
- NO inventes información nueva.
- NO traduzcas a otra lengua.
- La versión limpia debe cumplir TODAS estas reglas:
  - sólo letras del alfabeto español en minúsculas (a–z, áéíóúüñ, ç),
  - palabras separadas por UN solo espacio,
  - sin signos de puntuación (¿?, ¡!, . , ; : " ' - …),
  - sin números ni códigos.
- Si después de limpiar queda algo extraño (0 o 1 palabra sin sentido), mejor descartar la frase.

Formato de SALIDA (MUY IMPORTANTE):

- Debes devolver EXACTAMENTE una línea de salida por cada línea de entrada.
- Para cada frase de entrada con ID N:

  a) Si la frase sirve:
     N<TAB>frase_limpia_en_minusculas

     Ejemplo:
       3\tno pasa nada

  b) Si la frase NO sirve o no estás seguro:
     N<TAB>-

     Ejemplo:
       4\t-

- Usa el carácter TAB (una tabulación) entre el ID y el texto.
- Si por alguna razón no puedes escribir un TAB, usa varios espacios entre el ID y el texto.
- No añadas ningún otro texto, encabezados, listas ni comentarios.
""".strip()


def build_user_content(batch):
    """
    batch: [{"id": int, "text": str}, ...]
    Формируем нумерованный список для модели.
    """
    lines = ["Frases numeradas en español:"]
    for obj in batch:
        lines.append(f"{obj['id']}: {obj['text']}")
    return "\n".join(lines)


def parse_plain_output(raw_content, batch_size):
    """
    Разбор вывода вида:
      '0<TAB>frase limpia\n1<TAB>-\n...'
    Допускаем TAB или >=1 пробелов как разделитель.
    Возвращает dict[id] = {"keep": bool, "clean": str}
    """
    results = {}
    lines = raw_content.strip().splitlines()

    for line in lines:
        if not line.strip():
            continue

        # допускаем TAB или >=1 пробела
        m = re.match(r"^\s*(\d+)\s+(.+?)\s*$", line)
        if not m:
            continue

        idx = int(m.group(1))
        if idx < 0 or idx >= batch_size:
            continue

        text = m.group(2).strip()
        if text == "-" or text == "":
            results[idx] = {"keep": False, "clean": ""}
        else:
            results[idx] = {"keep": True, "clean": text}

    return results


_debug_written = False  # чтобы писать сырой ответ только один раз


def call_llm(batch):
    """
    batch: [{"id": int, "text": str}, ...]
    -> dict[id] = {"keep": bool, "clean": str}
    """
    global _debug_written

    user_content = build_user_content(batch)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
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

    # Сохраняем один сырой ответ для диагностики формата
    if not _debug_written:
        try:
            DEBUG_RAW.write_text(raw_content, encoding="utf-8")
            _debug_written = True
        except Exception:
            pass

    decisions = parse_plain_output(raw_content, batch_size=len(batch))

    # страховка: если модель не вернула что-то — считаем, что фразу нужно выкинуть
    for obj in batch:
        if obj["id"] not in decisions:
            decisions[obj["id"]] = {"keep": False, "clean": ""}

    return decisions


# ==========================
#   MAIN
# ==========================

def main():
    OUTPUT_TEXT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_META_JSONL.parent.mkdir(parents=True, exist_ok=True)

    resume_from = load_checkpoint()
    print("Resume from source line:", resume_from)

    mode = "a" if resume_from >= 0 and OUTPUT_TEXT.exists() else "w"

    meta_rows = []  # для дальнейшей записи в Parquet

    with SOURCE_CORPUS.open("r", encoding="utf-8") as inp, \
            OUTPUT_TEXT.open(mode, encoding="utf-8") as out_txt, \
            OUTPUT_META_JSONL.open(mode, encoding="utf-8") as out_meta:

        batch_records = []
        batch_for_llm = []
        batch_chars = 0
        last_line_no = resume_from
        batch_counter = 0

        for line_no, line in enumerate(tqdm(inp, desc="scanning 3w corpus")):
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
                    print("[PLAIN ERROR] batch failed, skipping batch:", str(e)[:200])
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
                            "id": ln,
                            "orig": orig_phrase,
                            "clean": clean,
                        }
                        out_meta.write(json.dumps(meta, ensure_ascii=False) + "\n")
                        meta_rows.append(meta)

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
                print("[PLAIN ERROR] tail batch failed, skipping:", str(e)[:200])
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
                        "id": ln,
                        "orig": orig_phrase,
                        "clean": clean,
                    }
                    out_meta.write(json.dumps(meta, ensure_ascii=False) + "\n")
                    meta_rows.append(meta)

        out_txt.flush()
        out_meta.flush()
        os.fsync(out_txt.fileno())
        os.fsync(out_meta.fileno())

        if last_line_no >= 0:
            save_checkpoint(last_line_no)

    # ==========================
    #   запись Parquet
    # ==========================
    if meta_rows:
        try:
            import pandas as pd

            df_new = pd.DataFrame(meta_rows)

            if OUTPUT_META_PARQUET.exists() and resume_from >= 0:
                # простая стратегия: читаем старый parquet, конкатенируем, перезаписываем
                df_old = pd.read_parquet(OUTPUT_META_PARQUET)
                df = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df = df_new

            OUTPUT_META_PARQUET.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(OUTPUT_META_PARQUET, index=False)
            print("Parquet written to:", OUTPUT_META_PARQUET.resolve())
        except Exception as e:
            print("[PARQUET ERROR]", e)

    print("Clean 3-word corpus written to:", OUTPUT_TEXT.resolve())
    print("Meta JSONL written to:", OUTPUT_META_JSONL.resolve())


if __name__ == "__main__":
    main()
