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

    def test_duplicate_detection(self) -> None:
        lines = ["Buenos días", "Buenos días", "Hasta luego"]
        kept, dropped = self.filter.filter_lines(lines)
        self.assertEqual(kept, ["Buenos días", "Hasta luego"])
        self.assertEqual(dropped["duplicate"], 1)


if __name__ == "__main__":
    unittest.main()
