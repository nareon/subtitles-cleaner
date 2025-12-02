import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from filter_es_corpus import LineFilter  # noqa: E402


class LineFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.filter = LineFilter(min_length=10, max_length=80)

    def test_accepts_regular_sentence(self) -> None:
        keep, reason = self.filter.should_keep("Hola, ¿cómo estás hoy?")
        self.assertTrue(keep)
        self.assertIsNone(reason)

    def test_rejects_short_line(self) -> None:
        keep, reason = self.filter.should_keep("Hola")
        self.assertFalse(keep)
        self.assertEqual(reason, "too_short")

    def test_rejects_timestamp_line(self) -> None:
        keep, reason = self.filter.should_keep("00:01:23,000 --> 00:01:25,000")
        self.assertFalse(keep)
        self.assertEqual(reason, "timestamp")

    def test_rejects_url_metadata(self) -> None:
        keep, reason = self.filter.should_keep("Subtítulos por www.example.com")
        self.assertFalse(keep)
        self.assertEqual(reason, "metadata")

    def test_rejects_markup(self) -> None:
        keep, reason = self.filter.should_keep("[Música suave]")
        self.assertFalse(keep)
        self.assertEqual(reason, "markup")

    def test_rejects_digit_heavy_line(self) -> None:
        keep, reason = self.filter.should_keep("1234567890 2024")
        self.assertFalse(keep)
        self.assertEqual(reason, "no_spanish_letters")

    def test_strips_dialogue_indicator(self) -> None:
        lines = ["- Hola, ¿qué tal?", "- Hola, ¿qué tal?"]
        kept, dropped = self.filter.filter_lines(lines)
        self.assertEqual(kept, ["Hola, qué tal"])
        self.assertEqual(dropped["duplicate"], 1)

    def test_strips_dash_like_prefixes_without_spaces(self) -> None:
        normalized = self.filter.normalize("—-–Hola, ¿qué tal?")
        self.assertEqual(normalized, "Hola, qué tal")

    def test_considers_dash_like_prefixes_duplicates(self) -> None:
        lines = ["-Hola amigo", "– Hola amigo"]
        kept, dropped = self.filter.filter_lines(lines)
        self.assertEqual(kept, ["Hola amigo"])
        self.assertEqual(dropped["duplicate"], 1)

    def test_replaces_ellipsis_with_comma(self) -> None:
        kept, _ = self.filter.filter_lines(["Bueno... vale entonces"])
        self.assertEqual(kept, ["Bueno, vale entonces"])

    def test_strips_leading_comma_and_space(self) -> None:
        normalized = self.filter.normalize(", después de todo")
        self.assertEqual(normalized, "después de todo")

    def test_considers_comma_stripped_variant_duplicate(self) -> None:
        lines = [", después de todo", "después de todo"]
        kept, dropped = self.filter.filter_lines(lines)
        self.assertEqual(kept, ["después de todo"])
        self.assertEqual(dropped["duplicate"], 1)

    def test_duplicate_detection(self) -> None:
        lines = ["Buenos días", "Buenos días", "Hasta luego"]
        kept, dropped = self.filter.filter_lines(lines)
        self.assertEqual(kept, ["Buenos días", "Hasta luego"])
        self.assertEqual(dropped["duplicate"], 1)

    def test_trims_trailing_spaces_and_dots(self) -> None:
        keep, reason = self.filter.should_keep("  ..Hola amigo...   ")
        self.assertTrue(keep)
        self.assertIsNone(reason)
        self.assertEqual(self.filter.normalize("  ..Hola amigo...   "), "Hola amigo")

    def test_strips_trailing_special_characters_but_keeps_excited_questions(self) -> None:
        normalized = self.filter.normalize("¿Listo?!...   ")
        self.assertEqual(normalized, "Listo")

    def test_removes_double_quotes_everywhere(self) -> None:
        normalized = self.filter.normalize('"Ella dijo \"hola\" hoy"')
        self.assertEqual(normalized, "Ella dijo hola hoy")

    def test_removes_parentheses_but_keeps_content(self) -> None:
        normalized = self.filter.normalize("(Hola) amigo (mío)")
        self.assertEqual(normalized, "Hola amigo mío")

    def test_removes_trailing_questions_exclamations_and_minuses(self) -> None:
        normalized = self.filter.normalize("¿Vienes?!??---   ")
        self.assertEqual(normalized, "Vienes")

    def test_removes_inverted_punctuation(self) -> None:
        normalized = self.filter.normalize("¡Hola! ¿Cómo estás?")
        self.assertEqual(normalized, "Hola! Cómo estás")


if __name__ == "__main__":
    unittest.main()
