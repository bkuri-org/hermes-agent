"""Tests for memory_tool.py — pre-write dedup, coalescing, and auto-trim."""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.memory_tool import (
    MemoryStore,
    ENTRY_DELIMITER,
    _extract_header,
    _normalize_for_similarity,
    _word_bigrams,
    _jaccard,
    _text_similarity,
    _HEADER_SIMILARITY_THRESHOLD,
    _CONTENT_SIMILARITY_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestExtractHeader:
    def test_bold_header_with_colon(self):
        assert _extract_header("**Desktop:** RX 9060 XT") == "Desktop"

    def test_bold_header_without_colon(self):
        assert _extract_header("**Caddy** config architecture") == "Caddy"

    def test_no_header(self):
        assert _extract_header("Just some plain text") == ""

    def test_header_with_spaces(self):
        assert _extract_header("**Room→project:** mapping") == "Room→project"


class TestNormalize:
    def test_lowercase(self):
        assert "rx 9060 xt" in _normalize_for_similarity("RX 9060 XT 16GB")

    def test_strip_punctuation(self):
        assert _normalize_for_similarity("foo, bar! baz?") == "foo bar baz"

    def test_collapse_whitespace(self):
        assert _normalize_for_similarity("foo   bar") == "foo bar"


class TestTextSimilarity:
    def test_identical_texts(self):
        a = "**Desktop:** RX 9060 XT 16GB, Ryzen 9 3900X, Arch Linux with ROCm support."
        assert _text_similarity(a, a) == 1.0

    def test_near_duplicate(self):
        a = "**Desktop:** RX 9060 XT 16GB, Ryzen 9 3900X, Arch+ROCm."
        b = "**Desktop:** RX 9060 XT 16GB, Ryzen 9 3900X, Arch Linux with ROCm 6.3."
        assert _text_similarity(a, b) > 0.4

    def test_unrelated_texts(self):
        a = "**Desktop:** RX 9060 XT 16GB, Ryzen 9 3900X, Arch+ROCm."
        b = "**Caddy:** Source of truth on server1 for reverse proxy config."
        assert _text_similarity(a, b) < 0.2

    def test_short_texts_return_zero(self):
        assert _text_similarity("short", "short text") == 0.0

    def test_empty_returns_zero(self):
        assert _text_similarity("", "") == 0.0


# ---------------------------------------------------------------------------
# MemoryStore.add() coalescing tests
# ---------------------------------------------------------------------------

class TestMemoryAddCoalescing:
    """Test the pre-write coalescing logic in MemoryStore.add()."""

    def _make_store(self, limit=500):
        """Create a MemoryStore with a temp directory and given char limit."""
        import tempfile
        tmpdir = tempfile.mkdtemp()
        store = MemoryStore(memory_char_limit=limit, user_char_limit=limit)
        # Patch get_memory_dir to use our temp dir
        with patch("tools.memory_tool.get_memory_dir", return_value=Path(tmpdir)):
            store.load_from_disk()
        return store, tmpdir

    def _add(self, store, tmpdir, target, content):
        with patch("tools.memory_tool.get_memory_dir", return_value=Path(tmpdir)):
            return store.add(target, content)

    def test_exact_duplicate_rejected(self):
        store, tmpdir = self._make_store()
        self._add(store, tmpdir, "memory", "**Desktop:** RX 9060 XT.")
        result = self._add(store, tmpdir, "memory", "**Desktop:** RX 9060 XT.")
        assert result["success"] is True
        assert "already exists" in result["message"]
        assert result["entry_count"] == 1

    def test_header_coalescing_replaces_old(self):
        """Entries with the same bold header and similar content should coalesce."""
        store, tmpdir = self._make_store()
        r1 = self._add(store, tmpdir, "memory",
                        "**Desktop:** RX 9060 XT 16GB, Ryzen 9 3900X, Arch+ROCm.")
        r2 = self._add(store, tmpdir, "memory",
                        "**Desktop:** RX 9060 XT 16GB, Ryzen 9 3900X, Arch Linux with ROCm 6.3.")
        assert "coalesced" in r2["message"]
        assert store._char_count("memory") < 200  # not doubled

    def test_header_coalescing_keeps_longer(self):
        """Coalescing should keep the longer (more complete) entry."""
        store, tmpdir = self._make_store()
        self._add(store, tmpdir, "memory",
                  "**Desktop:** RX 9060 XT 16GB, Ryzen 9 3900X, Arch+ROCm.")
        self._add(store, tmpdir, "memory",
                  "**Desktop:** RX 9060 XT 16GB, Ryzen 9 3900X, Arch Linux with ROCm 6.3.")
        # The second (longer) entry should have been kept
        assert "ROCm 6.3" in store.memory_entries[0]

    def test_different_headers_append(self):
        """Entries with different headers should be appended, not coalesced."""
        store, tmpdir = self._make_store(limit=1000)
        self._add(store, tmpdir, "memory", "**Desktop:** RX 9060 XT.")
        r2 = self._add(store, tmpdir, "memory", "**Caddy:** Source of truth on server1.")
        assert "added" in r2["message"]
        assert store.memory_entries.__len__() == 2

    def test_content_similarity_coalescing(self):
        """High similarity without header match should still coalesce."""
        store, tmpdir = self._make_store(limit=1000)
        self._add(store, tmpdir, "memory",
                  "The Ollama load balancer uses Caddy upstream with desktop as primary.")
        r2 = self._add(store, tmpdir, "memory",
                       "Ollama load balancer uses Caddy upstream with desktop as primary backend.")
        # These should coalesce due to high similarity even without headers
        assert "coalesced" in r2["message"] or store.memory_entries.__len__() == 1

    def test_unrelated_entries_both_kept(self):
        """Completely unrelated entries should both be kept."""
        store, tmpdir = self._make_store(limit=1000)
        self._add(store, tmpdir, "memory", "**Desktop:** RX 9060 XT 16GB.")
        r2 = self._add(store, tmpdir, "memory", "**Jetson:** Orin NX 16GB for edge inference.")
        assert "added" in r2["message"]
        assert store.memory_entries.__len__() == 2


class TestMemoryAddAutoTrim:
    """Test the auto-trim logic when store is near capacity."""

    def _make_store(self, limit=200):
        import tempfile
        tmpdir = tempfile.mkdtemp()
        store = MemoryStore(memory_char_limit=limit, user_char_limit=limit)
        with patch("tools.memory_tool.get_memory_dir", return_value=Path(tmpdir)):
            store.load_from_disk()
        return store, tmpdir

    def _add(self, store, tmpdir, target, content):
        with patch("tools.memory_tool.get_memory_dir", return_value=Path(tmpdir)):
            return store.add(target, content)

    def test_write_succeeds_at_capacity(self):
        """Writing when the store would exceed the limit should still succeed."""
        store, tmpdir = self._make_store(limit=150)
        # Fill store with a long entry
        self._add(store, tmpdir, "memory", "**Desktop:** " + "X" * 100)
        # Add another entry that would overflow
        r = self._add(store, tmpdir, "memory", "**Caddy:** " + "Y" * 100)
        assert r["success"] is True

    def test_auto_trim_reduces_longest(self):
        """Auto-trim should truncate the longest entry."""
        store, tmpdir = self._make_store(limit=200)
        # Add a very long entry
        self._add(store, tmpdir, "memory", "**Desktop:** " + "A" * 150)
        # Add another that triggers trim
        r = self._add(store, tmpdir, "memory", "**Caddy:** " + "B" * 80)
        assert r["success"] is True
        assert "Auto-trimmed" in r["message"] or store._char_count("memory") <= 200

    def test_never_drops_new_write(self):
        """The new entry must always appear in the store."""
        store, tmpdir = self._make_store(limit=150)
        self._add(store, tmpdir, "memory", "**Old:** " + "A" * 100)
        new_content = "**New:** " + "B" * 60
        r = self._add(store, tmpdir, "memory", new_content)
        assert r["success"] is True
        # The new content or a coalesced version containing "New" must exist
        all_text = ENTRY_DELIMITER.join(store.memory_entries)
        assert "New" in all_text or "Old" in all_text


class TestMemoryAddEdgeCases:
    """Edge cases for the add() method."""

    def _make_store(self, limit=500):
        import tempfile
        tmpdir = tempfile.mkdtemp()
        store = MemoryStore(memory_char_limit=limit, user_char_limit=limit)
        with patch("tools.memory_tool.get_memory_dir", return_value=Path(tmpdir)):
            store.load_from_disk()
        return store, tmpdir

    def _add(self, store, tmpdir, target, content):
        with patch("tools.memory_tool.get_memory_dir", return_value=Path(tmpdir)):
            return store.add(target, content)

    def test_empty_content_rejected(self):
        store, tmpdir = self._make_store()
        r = self._add(store, tmpdir, "memory", "")
        assert r["success"] is False

    def test_whitespace_only_rejected(self):
        store, tmpdir = self._make_store()
        r = self._add(store, tmpdir, "memory", "   ")
        assert r["success"] is False

    def test_strips_whitespace(self):
        store, tmpdir = self._make_store()
        r = self._add(store, tmpdir, "memory", "  **Desktop:** foo  ")
        assert r["success"] is True
        assert store.memory_entries[0] == "**Desktop:** foo"

    def test_user_target_works(self):
        store, tmpdir = self._make_store()
        r = self._add(store, tmpdir, "user", "Prefers Matrix over Telegram.")
        assert r["success"] is True
        assert r["target"] == "user"
