#!/usr/bin/env python3
"""Batch creation of cleaned Spanish flashcards with scoring.

The script reads three-word Spanish phrases, cleans them via the configured
LLM, and writes each result as a JSON line containing the original text,
the cleaned version, and a usefulness score in [0, 1].
"""
from __future__ import annotations

import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List

import requests

# ==========================
#   CONSTANTS
# ==========================
INPUT_PATH = Path("/home/ol/_p/subtitles-cleaner/1/corpus/split/es.3w.sample160k.txt")
OUTPUT_PATH = Path("/home/ol/_p/subtitles-cleaner/flashcards_es/es_flashcards.jsonl")

# LLM endpoint configuration (OpenAI-compatible)
BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "models/qwen25-15b"
API_KEY = "dummy-key"

# Processing parameters
BATCH_SIZE = 32
MAX_WORKERS = 128
MAX_RETRIES = 3
REQUEST_TIMEOUT = 90

PROMPT = """
Clasificar la frase:
- false — mala (contiene nombre propio, apelativo, número en cifras, abreviatura, palabra extranjera, jerga o símbolo especial)
- true — buena (frase neutra sin esos elementos)

Ejemplos:
{"phrase":"Juan canta bien","isGood":false,"reason":"contiene nombre propio"}
{"phrase":"María cocina sopa","isGood":false,"reason":"contiene nombre propio"}
{"phrase":"Pedro juega fútbol","isGood":false,"reason":"contiene nombre propio"}
{"phrase":"Ana lee libros","isGood":false,"reason":"contiene nombre propio"}
{"phrase":"Luis corre rápido","isGood":false,"reason":"contiene nombre propio"}
{"phrase":"Tengo 25 años","isGood":false,"reason":"contiene número en cifras"}
{"phrase":"En 1999 pasó","isGood":false,"reason":"contiene número en cifras"}
{"phrase":"Compré 300 manzanas","isGood":false,"reason":"contiene número en cifras"}
{"phrase":"ONU aprobó resolución","isGood":false,"reason":"contiene abreviatura"}
{"phrase":"EE.UU. firmó tratado","isGood":false,"reason":"contiene abreviatura"}
{"phrase":"Uso wifi gratis","isGood":false,"reason":"contiene palabra extranjera"}
{"phrase":"Estoy online siempre","isGood":false,"reason":"contiene palabra extranjera"}
{"phrase":"Esto es guay","isGood":false,"reason":"contiene jerga"}
{"phrase":"Qué rollo tío","isGood":false,"reason":"contiene jerga"}
{"phrase":"#vida es dura","isGood":false,"reason":"contiene símbolo especial"}
{"phrase":"Sígueme en @red","isGood":false,"reason":"contiene símbolo especial"}
{"phrase":"Visita www.ejemplo","isGood":false,"reason":"contiene símbolo especial"}
{"phrase":"Nike vende zapatillas","isGood":false,"reason":"contiene marca comercial"}
{"phrase":"Coca-Cola refresca bien","isGood":false,"reason":"contiene marca comercial"}
{"phrase":"Barcelona tiene playa","isGood":false,"reason":"contiene nombre propio"}
{"phrase":"Madrid ganó partido","isGood":false,"reason":"contiene nombre propio"}
{"phrase":"Cristina toca piano","isGood":false,"reason":"contiene nombre propio"}
{"phrase":"Pablo escribe cartas","isGood":false,"reason":"contiene nombre propio"}
{"phrase":"Sofía canta canciones","isGood":false,"reason":"contiene nombre propio"}

{"phrase":"El sol brilla","isGood":true,"reason":""}
{"phrase":"El agua hierve","isGood":true,"reason":""}
{"phrase":"La sopa enfría","isGood":true,"reason":""}
{"phrase":"El perro ladra","isGood":true,"reason":""}
{"phrase":"La casa arde","isGood":true,"reason":""}
{"phrase":"El viento sopla","isGood":true,"reason":""}
{"phrase":"La flor crece","isGood":true,"reason":""}
{"phrase":"El gato duerme","isGood":true,"reason":""}
{"phrase":"La puerta cierra","isGood":true,"reason":""}
{"phrase":"El tren avanza","isGood":true,"reason":""}
{"phrase":"La música suena","isGood":true,"reason":""}
{"phrase":"El campo verdea","isGood":true,"reason":""}
{"phrase":"La lluvia cae","isGood":true,"reason":""}
{"phrase":"El fuego quema","isGood":true,"reason":""}
{"phrase":"El aire refresca","isGood":true,"reason":""}
{"phrase":"La arena calienta","isGood":true,"reason":""}
{"phrase":"El pan cruje","isGood":true,"reason":""}
{"phrase":"La leche enfría","isGood":true,"reason":""}
{"phrase":"El día termina","isGood":true,"reason":""}
{"phrase":"La noche llega","isGood":true,"reason":""}
{"phrase":"El río fluye","isGood":true,"reason":""}
{"phrase":"La montaña alta","isGood":true,"reason":""}
{"phrase":"El cielo claro","isGood":true,"reason":""}
{"phrase":"La comida huele","isGood":true,"reason":""}
{"phrase":"El bosque oscuro","isGood":true,"reason":""}

Devuelve el resultado estrictamente en formato JSON, sin explicaciones ni texto fuera de la estructura.
Esquema JSON:
{
  "phrase": "string",
  "isGood": "boolean",
  "reason": "string"
}
""".strip()


def chunked(iterable: Iterable[str], size: int) -> Iterable[List[str]]:
    batch: List[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_user_message(batch: List[str]) -> str:
    return (
        f"{PROMPT}\n\n"
        "Обработай следующий список фраз и верни JSON-массив с объектами в той же последовательности.\n"
        "Каждый объект должен соответствовать схеме и включать исходную строку.\n\n"
        "Входные фразы:\n"
        f"{json.dumps(batch, ensure_ascii=False)}"
    )


def _strip_code_fences(content: str) -> str:
    fenced = re.match(r"^```(?:json)?\s*(.*)```\s*$", content, flags=re.DOTALL)
    return fenced.group(1).strip() if fenced else content


def _parse_json_stream(content: str):
    """Extract JSON objects even if they are not part of a valid array.

    Some LLMs return a series of objects without commas or wrap the payload in
    prose. This helper scans the string for any decodable JSON object or array
    and returns them in the order encountered.
    """

    decoder = json.JSONDecoder()
    idx = 0
    results = []

    while idx < len(content):
        while idx < len(content) and content[idx].isspace():
            idx += 1
        if idx >= len(content):
            break

        try:
            obj, end = decoder.raw_decode(content, idx)
        except json.JSONDecodeError:
            idx += 1
            continue

        if isinstance(obj, (dict, list)):
            results.append(obj)
        idx = end

    return results


def extract_json(content: str):
    content = _strip_code_fences(content.strip())

    for candidate in (content,):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    start_candidates = [pos for pos in (content.find("["), content.find("{")) if pos != -1]
    end_candidates = [content.rfind("]"), content.rfind("}")]
    start = min(start_candidates) if start_candidates else -1
    end = max(end_candidates) if end_candidates else -1
    if start != -1 and end != -1 and end > start:
        snippet = content[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

    streamed = _parse_json_stream(content)
    if streamed:
        return streamed if len(streamed) > 1 else streamed[0]

    raise ValueError("Cannot parse JSON from LLM response")


def normalize_items(raw_items, batch: List[str]):
    if isinstance(raw_items, dict):
        raw_items = [raw_items]
    if not isinstance(raw_items, list):
        raise ValueError("LLM response is not a JSON array")

    normalized = []
    for idx, orig in enumerate(batch):
        item = raw_items[idx] if idx < len(raw_items) else {}
        orig_text = item.get("orig_text") or item.get("orig") or orig
        clean_text = (item.get("clean_text") or item.get("clean") or "").strip()
        try:
            score_val = float(item.get("score", 0))
        except Exception:
            score_val = 0.0
        score_val = max(0.0, min(1.0, score_val))
        if math.isnan(score_val):
            score_val = 0.0
        normalized.append(
            {
                "orig_text": orig_text,
                "clean_text": clean_text,
                "score": score_val,
            }
        )
    return normalized


def call_llm(batch: List[str]):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": build_user_message(batch)}],
        "temperature": 0.0,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            parsed = extract_json(content)
            return normalize_items(parsed, batch)
        except Exception as exc:  # noqa: PERF203
            if attempt >= MAX_RETRIES:
                raise
            sleep_for = 2 * attempt
            print(f"[warn] retry {attempt}/{MAX_RETRIES} after error: {exc}")
            time.sleep(sleep_for)


def load_batches(path: Path) -> List[List[str]]:
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return list(chunked(lines, BATCH_SIZE))


def process_batches(batches: List[List[str]]):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    buffer = {}
    next_to_write = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(call_llm, batch): idx for idx, batch in enumerate(batches)}

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result_items = future.result()
            except Exception as exc:
                print(f"[error] batch {idx} failed: {exc}")
                result_items = [
                    {"orig_text": orig, "clean_text": "", "score": 0.0}
                    for orig in batches[idx]
                ]

            buffer[idx] = result_items

            while next_to_write in buffer:
                for item in buffer.pop(next_to_write):
                    out.write(json.dumps(item, ensure_ascii=False) + "\n")
                next_to_write += 1



def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    batches = load_batches(INPUT_PATH)
    print(f"Loaded {len(batches)} batches of size up to {BATCH_SIZE} from {INPUT_PATH}")
    process_batches(batches)
    print(f"Results written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
