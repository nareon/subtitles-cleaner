#!/usr/bin/env python
"""
preclean_split.py

Мягкая очистка сырого файла субтитров и разбор на файлы по числу слов:
1, 2, 3, 4, 5, 6+.

Что делает:
- читает входной файл построчно (любой размер, потоково);
- убирает управляющие символы и простые артефакты разметки;
- схлопывает последовательности пробелов в один;
- сохраняет пунктуацию и регистр (это важно для LLM);
- считает количество "слов" как последовательностей букв (латиница + испанские акценты);
- в зависимости от количества слов пишет строку в один из 6 выходных файлов.

Пустые строки и строки без буквенных слов не записываются.
"""

import re
from pathlib import Path
from tqdm import tqdm

# ==========================
#   НАСТРОЙКИ ПУТЕЙ
# ==========================

# входной файл с сырыми субтитрами
SOURCE_PATH = Path("corpus/es.sample.txt")

# каталог для выходных файлов
OUTPUT_DIR = Path("corpus/split")

# имена выходных файлов
OUT_1 = OUTPUT_DIR / "es.1w.soft.txt"
OUT_2 = OUTPUT_DIR / "es.2w.soft.txt"
OUT_3 = OUTPUT_DIR / "es.3w.soft.txt"
OUT_4 = OUTPUT_DIR / "es.4w.soft.txt"
OUT_5 = OUTPUT_DIR / "es.5w.soft.txt"
OUT_6P = OUTPUT_DIR / "es.6plusw.soft.txt"

# ==========================
#   ОЧИСТКА И ТОКЕНИЗАЦИЯ
# ==========================

# управляющие/невидимые символы, которые часто встречаются в субтитрах
CONTROL_CHARS_RE = re.compile(
    r"[\u0000-\u001F\u007F\u200B\u200C\u200D\u200E\u200F\u202A-\u202E]"
)

# очень грубый шаблон HTML-тегов
HTML_TAG_RE = re.compile(r"<[^>]+>")

# таймкоды вида 00:00:01,000 --> 00:00:04,000
TIMECODE_RE = re.compile(
    r"\d{1,2}:\d{2}:\d{2}[,\.]\d{2,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,\.]\d{2,3}"
)

# URL
URL_RE = re.compile(r"https?://\S+|www\.\S+")

# слова: группы латинских букв + испанские символы (ñ, ç, á, é, ...)
WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñÇç]+")


def soft_clean_line(line: str) -> str:
    """
    Мягкая очистка строки:
    - убрать управляющие и невидимые символы;
    - убрать html-теги, таймкоды, URL;
    - схлопнуть пробелы;
    - обрезать по краям.

    Пунктуация и регистр сохраняются.
    """
    # убираем перевод строки
    line = line.rstrip("\n\r")

    if not line:
        return ""

    # управляющие / невидимые символы
    line = CONTROL_CHARS_RE.sub("", line)

    # html-теги
    line = HTML_TAG_RE.sub(" ", line)

    # таймкоды и URL
    line = TIMECODE_RE.sub(" ", line)
    line = URL_RE.sub(" ", line)

    # табы и другие whitespace → пробел
    line = re.sub(r"\s+", " ", line)

    line = line.strip()

    return line


def count_words(line: str) -> int:
    """
    Посчитать количество слов как количество буквенных последовательностей.
    Числа, чистые символы и т.п. не считаются словами.
    """
    if not line:
        return 0
    words = WORD_RE.findall(line)
    return len(words)


# ==========================
#   MAIN
# ==========================

def main():
    if not SOURCE_PATH.exists():
        raise SystemExit(f"Source file not found: {SOURCE_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    kept_lines = 0

    with SOURCE_PATH.open("r", encoding="utf-8", errors="ignore") as src, \
            OUT_1.open("w", encoding="utf-8") as f1, \
            OUT_2.open("w", encoding="utf-8") as f2, \
            OUT_3.open("w", encoding="utf-8") as f3, \
            OUT_4.open("w", encoding="utf-8") as f4, \
            OUT_5.open("w", encoding="utf-8") as f5, \
            OUT_6P.open("w", encoding="utf-8") as f6p:

        files = {1: f1, 2: f2, 3: f3, 4: f4, 5: f5, "6+": f6p}

        for line in tqdm(src, desc="preclean & split"):
            total_lines += 1

            cleaned = soft_clean_line(line)
            if not cleaned:
                continue

            n_words = count_words(cleaned)
            if n_words == 0:
                continue

            kept_lines += 1

            if n_words == 1:
                target = files[1]
            elif n_words == 2:
                target = files[2]
            elif n_words == 3:
                target = files[3]
            elif n_words == 4:
                target = files[4]
            elif n_words == 5:
                target = files[5]
            else:
                target = files["6+"]

            target.write(cleaned + "\n")

    print(f"Total lines read:  {total_lines}")
    print(f"Lines with text:   {kept_lines}")
    print(f"Output files:")
    print(f"  1 word : {OUT_1}")
    print(f"  2 words: {OUT_2}")
    print(f"  3 words: {OUT_3}")
    print(f"  4 words: {OUT_4}")
    print(f"  5 words: {OUT_5}")
    print(f"  6+     : {OUT_6P}")


if __name__ == "__main__":
    main()
