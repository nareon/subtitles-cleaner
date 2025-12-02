# Spanish subtitles pre-filter

This repository provides a lightweight preprocessing step for a Spanish
subtitles corpus. The goal is to keep sentences that are useful as learning
materials and remove obvious noise before a later LLM-powered cleanup.

## Usage

1. Place the raw sample at `corpus/es.example.txt` (ignored by Git). A small
   placeholder file can be tracked via `corpus/.gitkeep`.
2. Run the filter:

   ```bash
   python filter_es_corpus.py --input corpus/es.example.txt --output corpus/es.filtered.txt
   ```

3. Inspect `corpus/es.filtered.txt` before passing it to downstream tools.

### CLI options

- `--min-length` / `--max-length`: Control the acceptable line length range.
- `--input`: Path to the raw corpus file.
- `--output`: Where to write the filtered lines (directory is created
  automatically).

### Filtering rules

The pre-filter removes lines that match any of the following:

- Too short/long strings after trimming whitespace.
- No Spanish letters present.
- Timestamps (`-->` or `00:00:00` patterns) and numeric time markers.
- URLs, email addresses, or common subtitle credit phrases.
- Markup such as HTML tags or bracketed stage directions.
- Lines dominated by digits or containing many unusual symbols.
- Duplicate lines.

A summary with kept/dropped counts and reasons is printed after execution.

## Development

Run the built-in unit tests with:

```bash
python -m unittest discover tests
```
