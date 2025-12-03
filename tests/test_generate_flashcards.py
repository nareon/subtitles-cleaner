import sys
import types
from pathlib import Path
import unittest

if "requests" not in sys.modules:
    # Provide a lightweight stand-in so the module can be imported without the
    # external dependency present during tests.
    sys.modules["requests"] = types.SimpleNamespace()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from flashcards_es.generate_flashcards import extract_json  # noqa: E402


class ExtractJsonTests(unittest.TestCase):
    def test_accepts_code_fence_array(self) -> None:
        content = """```json
        [
          {"orig_text": "uno dos tres", "clean_text": "uno dos tres", "score": 0.8}
        ]
        ```"""

        parsed = extract_json(content)

        self.assertIsInstance(parsed, list)
        self.assertEqual(parsed[0]["orig_text"], "uno dos tres")
        self.assertAlmostEqual(parsed[0]["score"], 0.8)

    def test_parses_objects_without_commas(self) -> None:
        content = """
        Ответ:
        {"orig_text": "hola amigo", "clean_text": "hola amigo", "score": 0.9}
        {"orig_text": "buenos dias", "clean_text": "buenos días", "score": 1.0}
        """

        parsed = extract_json(content)

        self.assertIsInstance(parsed, list)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["clean_text"], "hola amigo")
        self.assertEqual(parsed[1]["clean_text"], "buenos días")


if __name__ == "__main__":
    unittest.main()
