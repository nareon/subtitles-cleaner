from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    corpus_dir = repo_root / "corpus"
    source_path = corpus_dir / "es.txt"
    output_path = corpus_dir / "es.sample.txt"

    if not source_path.is_file():
        raise FileNotFoundError(
            f"Input file not found: {source_path}. "
            "Place the Spanish subtitles corpus at this path before running the script."
        )

    corpus_dir.mkdir(parents=True, exist_ok=True)

    sampled = 0
    with source_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for index, line in enumerate(src, start=1):
            if index % 100 == 0:
                dst.write(line)
                sampled += 1

    print(f"Saved {sampled} lines to {output_path}")


if __name__ == "__main__":
    main()
