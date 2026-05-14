from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_salience.text import normalize_text, token_frequencies


class TextTests(unittest.TestCase):
    def test_normalize_text_basic(self) -> None:
        self.assertEqual(
            normalize_text("Hello, HELLO world_123 -- Test!"),
            ["hello", "hello", "world", "123", "test"],
        )

    def test_normalize_text_empty(self) -> None:
        self.assertEqual(normalize_text(""), [])

    def test_normalize_text_unicode_words(self) -> None:
        self.assertEqual(
            normalize_text("Šešlak čuva žutu Đurđevdan Überstraße groß"),
            ["šešlak", "čuva", "žutu", "đurđevdan", "überstraße", "groß"],
        )

    def test_normalize_text_underscores_split(self) -> None:
        self.assertEqual(normalize_text("hello_world"), ["hello", "world"])

    def test_token_frequencies_basic(self) -> None:
        self.assertEqual(token_frequencies("a a b"), {"a": 2.0, "b": 1.0})

    def test_token_frequencies_empty(self) -> None:
        self.assertEqual(token_frequencies(""), {})


if __name__ == "__main__":
    unittest.main()


class SignatureAndFuzzyTextTests(unittest.TestCase):
    def test_stable_hash_hex_is_deterministic(self) -> None:
        from agent_salience.text import stable_hash_hex

        self.assertEqual(stable_hash_hex("abc"), stable_hash_hex("abc"))
        self.assertNotEqual(stable_hash_hex("abc"), stable_hash_hex("abcd"))

    def test_token_shingles_empty_and_tiny(self) -> None:
        from agent_salience.text import token_shingles

        self.assertEqual(token_shingles([], size=3), [])
        self.assertEqual(token_shingles(["a", "b"], size=3), [])
        self.assertEqual(token_shingles(["a", "b", "c"], size=3), [("a", "b", "c")])

    def test_shingle_hashes_are_bounded_and_sorted(self) -> None:
        from agent_salience.text import shingle_hashes

        hashes = shingle_hashes(["a", "b", "c", "d", "e"], size=2, max_hashes=2)
        self.assertEqual(len(hashes), 2)
        self.assertEqual(hashes, sorted(hashes))

    def test_text_signature_roundtrip(self) -> None:
        from agent_salience.text import TextSignature, build_text_signature

        sig = build_text_signature("Run validation before release handoff.")
        restored = TextSignature.from_dict(sig.to_dict())
        self.assertEqual(restored, sig)
        self.assertGreater(sig.token_count, 0)
        self.assertTrue(sig.content_hash)
        self.assertTrue(sig.normalized_hash)

    def test_content_hash_uses_full_raw_text_but_signature_caps_tokens(self) -> None:
        from agent_salience.text import build_text_signature

        a = "a " * 100 + "suffix-one"
        b = "a " * 100 + "suffix-two"
        sig_a = build_text_signature(a, max_chars=20)
        sig_b = build_text_signature(b, max_chars=20)
        self.assertNotEqual(sig_a.content_hash, sig_b.content_hash)
        self.assertEqual(sig_a.normalized_hash, sig_b.normalized_hash)

    def test_char_ngram_similarity_handles_morphology(self) -> None:
        from agent_salience.text import char_ngram_similarity

        related = char_ngram_similarity("validation", "validated")
        unrelated = char_ngram_similarity("validation", "apartment")
        self.assertGreater(related, unrelated)
        self.assertGreater(related, 0.25)

    def test_token_prefix_overlap_is_not_alias_expansion(self) -> None:
        from agent_salience.text import normalize_text, token_prefix_overlap

        related = token_prefix_overlap(normalize_text("configuration"), normalize_text("config"), min_prefix=5)
        semantic = token_prefix_overlap(normalize_text("test failure"), normalize_text("broken validation"), min_prefix=5)
        self.assertGreater(related, semantic)
        self.assertEqual(semantic, 0.0)

    def test_expand_tokens_with_aliases(self) -> None:
        from agent_salience.text import expand_tokens_with_aliases, normalize_text

        aliases = {"test_failure": ["test failure", "broken validation", "debug failing test"]}
        source = expand_tokens_with_aliases(normalize_text("debug broken validation run"), aliases)
        target = expand_tokens_with_aliases(normalize_text("test failure triage"), aliases)
        self.assertIn("test_failure", source)
        self.assertIn("test_failure", target)
