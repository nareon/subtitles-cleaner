#!/usr/bin/env python
import json, re, os, time, requests
from pathlib import Path
from tqdm import tqdm

# ==========================
#   ПУТИ
# ==========================

# вход — результат префильтра (на sda2)
PHRASE_INDEX = Path(
    "/media/ol/SSD2T_Photo/hablai/corpus/jsonl/phrase_index_prefiltered.jsonl"
)

# итоговый индекс (на sda2)
OUTPUT_INDEX = Path(
    "/media/ol/SSD2T_Photo/hablai/corpus/jsonl/phrase_index_llm_filtered.jsonl"
)

# чекпоинт — номер строки во входном PHRASE_INDEX
CHECKPOINT = Path("/media/ol/SSD2T_Photo/hablai/llm_filter_checkpoint.json")

# ==========================
#   ПАРАМЕТРЫ
# ==========================

BATCH_SIZE = 16  # сколько фраз отправляем в LLM за один запрос

# Настройки LLM (OpenAI-совместимый API)
BASE_URL = "http://localhost:8000/v1"
API_KEY = "dummy-key"
MODEL = "gpt-oss-120b"

SESSION = requests.Session()

# ==========================
#   ЧЕКПОИНТ
# ==========================


def load_checkpoint() -> int:
    """
    Вернуть номер последней обработанной строки входного jsonl.
    -1, если чекпоинта нет или он битый.
    """
    if CHECKPOINT.exists():
        try:
            data = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
            return int(data.get("last_line", -1))
        except Exception:
            return -1
    return -1


def save_checkpoint(last_line: int) -> None:
    """
    Атомарно сохранить номер последней полностью обработанной строки.
    """
    tmp = CHECKPOINT.with_suffix(".tmp")
    tmp.write_text(json.dumps({"last_line": last_line}), encoding="utf-8")
    tmp.replace(CHECKPOINT)


# ==========================
#   LLM
# ==========================

SYSTEM_PROMPT = """
Ты — лингвист-носитель испанского языка и методист по преподаванию испанского как иностранного.
Твоя задача — отбирать короткие испанские фразы, подходящие для карточек изучения языка.

КРИТЕРИИ "ХОРОШАЯ ФРАЗА" (keep = true):
- от 2 до 5 слов
- фраза выглядит завершённой или почти завершённой репликой
- полезна в широком круге ситуаций, а не в одном конкретном фильме/истории
- нормальный современный испанский (допустим лёгкий сленг)

КРИТЕРИИ "ПЛОХАЯ ФРАЗА" (keep = false):
- обрывок середины предложения без контекста
- содержит только имена собственные, даты, локации, заголовки
- технический шум субтитров (например, "FRANCIA MARZO DE 1918", "(risas)", "APLAUSOS")
- содержит в основном URL, цифры, коды, форматирование
- не на испанском
- очень узкая/странная фраза, которая почти не пригодится в изучении языка

Формат ответа: JSON-массив объектов:
[
  {"id": 0, "keep": true, "reason": "короткая полезная реплика"},
  {"id": 1, "keep": false, "reason": "шум субтитров"}
]
Не добавляй никакого текста вне JSON.
""".strip()


# ========= безопасный JSON-парсер =========
def safe_parse_json(s: str):
    s = s.strip()

    # отрезать всё до первой '[' (если модель что-то дописала до массива)
    i = s.find("[")
    if i > 0:
        s = s[i:]

    # попытка 1 — как есть
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # попытка 2: обрезать после последней '}'
    last_brace = s.rfind("}")
    if last_brace != -1:
        s2 = s[: last_brace + 1]
        # если массив не закрыт — добавим ']'
        if not s2.strip().endswith("]"):
            s2 += "]"
        try:
            return json.loads(s2)
        except json.JSONDecodeError:
            pass

    # попытка 3: собрать все полные объекты вида {...}
    objs = re.findall(r"\{[^{}]*\}", s, flags=re.DOTALL)
    if objs:
        candidate = "[" + ",".join(objs) + "]"
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError("JSON parse error from LLM, first 500 chars:\n" + s[:500])


def call_llm(batch):
    """
    batch: список словарей {"id": int, "text": str}
    возвращает dict[id] -> {"keep": bool, "reason": str}

    В случае JSON-ошибки:
        - пытается восстановить частично
        - если не получилось → делает индивидуальные LLM-запросы для каждой фразы
    """
    user_payload = {"phrases": batch}

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "temperature": 0.0,
    }

    # --- основная попытка запроса ---
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

    # --- пробуем распарсить батч целиком ---
    try:
        arr = safe_parse_json(raw_content)
    except Exception as e:
        print("[JSON ERROR] batch-level error:", e)
        print("[JSON ERROR] falling back to per-item evaluation…")

        # === Индивидуальная обработка каждого элемента ===
        results = {}
        for obj in batch:
            single_payload = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps({"phrases": [obj]}, ensure_ascii=False),
                    },
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
                        "reason": item.get("reason", ""),
                    }
                    ok = True
                    break
                except Exception:
                    time.sleep(1)

            if not ok:
                results[obj["id"]] = {"keep": False, "reason": "json_error_individual"}

        return results

    # --- JSON нормальный ---
    result = {}
    for item in arr:
        idx = int(item["id"])
        result[idx] = {
            "keep": bool(item.get("keep", False)),
            "reason": item.get("reason", ""),
        }
    return result


# ==========================
#   MAIN
# ==========================


def main():
    OUTPUT_INDEX.parent.mkdir(parents=True, exist_ok=True)

    resume_from = load_checkpoint()
    print("Resume from source line:", resume_from)

    mode = "a" if resume_from >= 0 and OUTPUT_INDEX.exists() else "w"

    with PHRASE_INDEX.open("r", encoding="utf-8") as inp, \
            OUTPUT_INDEX.open(mode, encoding="utf-8") as out:

        batch_records = []   # список (line_no, rec)
        batch_for_llm = []   # список {"id": local_id, "text": phrase}
        last_line_no = resume_from

        for line_no, line in enumerate(tqdm(inp, desc="scanning phrase_index")):
            # пропускаем уже обработанные строки
            if line_no <= resume_from:
                continue

            rec = json.loads(line)
            phrase = rec["phrase"]

            local_id = len(batch_for_llm)
            batch_records.append((line_no, rec))
            batch_for_llm.append({"id": local_id, "text": phrase})
            last_line_no = line_no

            if len(batch_for_llm) >= BATCH_SIZE:
                try:
                    decisions = call_llm(batch_for_llm)
                except Exception as e:
                    print("[JSON ERROR] batch failed, skipping batch:", str(e)[:200])
                    decisions = {
                        obj["id"]: {"keep": False, "reason": "json_error"}
                        for obj in batch_for_llm
                    }

                # редкий мониторинг качества
                if (line_no // BATCH_SIZE) % 1 == 0:
                    kept = sum(1 for d in decisions.values() if d["keep"])
                    print(
                        f"[info] batch #{line_no // BATCH_SIZE}: "
                        f"kept {kept} of {len(batch_for_llm)}"
                    )

                for idx_in_batch, (ln, r) in enumerate(batch_records):
                    dec = decisions.get(idx_in_batch)
                    if dec and dec["keep"]:
                        r["llm_keep"] = True
                        r["llm_reason"] = dec["reason"]
                        out.write(json.dumps(r, ensure_ascii=False) + "\n")
                
                # гарантируем запись на диск
                out.flush()
                os.fsync(out.fileno())

                if last_line_no >= 0:
                    save_checkpoint(last_line_no)

                batch_records.clear()
                batch_for_llm.clear()

        # хвостовый батч
        if batch_for_llm:
            try:
                decisions = call_llm(batch_for_llm)
            except Exception as e:
                print("[JSON ERROR] tail batch failed, skipping:", str(e)[:200])
                decisions = {
                    obj["id"]: {"keep": False, "reason": "json_error"}
                    for obj in batch_for_llm
                }

            for idx_in_batch, (ln, r) in enumerate(batch_records):
                dec = decisions.get(idx_in_batch)
                if dec and dec["keep"]:
                    r["llm_keep"] = True
                    r["llm_reason"] = dec["reason"]
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")

            if last_line_no >= 0:
                save_checkpoint(last_line_no)

    print("Filtered index written to:", OUTPUT_INDEX.resolve())


if __name__ == "__main__":
    main()
