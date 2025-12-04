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
          {"phrase": "uno dos tres", "isGood": true, "reason": ""}
        ]
        ```"""

        parsed = extract_json(content)

        self.assertIsInstance(parsed, list)
        self.assertEqual(parsed[0]["phrase"], "uno dos tres")
        self.assertTrue(parsed[0]["isGood"])

    def test_parses_objects_without_commas(self) -> None:
        content = """
        Ответ:
        {"phrase": "hola amigo", "isGood": true, "reason": ""}
        {"phrase": "buenos dias", "isGood": false, "reason": "contiene tilde"}
        """

        parsed = extract_json(content)

        self.assertIsInstance(parsed, list)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["phrase"], "hola amigo")
        self.assertTrue(parsed[0]["isGood"])
        self.assertEqual(parsed[1]["phrase"], "buenos dias")
        self.assertFalse(parsed[1]["isGood"])
        self.assertEqual(parsed[1]["reason"], "contiene tilde")


if __name__ == "__main__":
    unittest.main()
