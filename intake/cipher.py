"""
intake/cipher.py
Generic sequence encoder / decoder.

Works on ANY kind of string or categorical data:
  - character-level text  (names, prose, code)
  - categorical sequences (weather patterns, log events, labels)
  - numeric sequences     (quantised into buckets)

Internal names:
  registry   — ordered list of unique symbols (the vocabulary)
  seal       — integer id of the special start-of-sequence token
  glyphs     — the encoded integer sequence
  encipher()  — string → list[int]
  decipher()  — list[int] → string
"""

from __future__ import annotations
import math as _math
from typing import Iterable


# ── Sentinel label for the boundary token ────────────────────────────────────
_BOUNDARY_LABEL = "<|seq|>"


class Cipher:
    """
    Build a vocabulary from a list of documents, then encode / decode.

    Parameters
    ----------
    documents   : list of raw strings (each is one sample)
    mode        : "char" | "word" | "token"
                  char  — split each doc into characters
                  word  — split on whitespace
                  token — treat each doc as a single pre-tokenised label
    n_buckets   : if > 0, numeric sequences are quantised into this many bins
    """

    def __init__(
        self,
        documents:  list[str],
        mode:       str = "char",
        n_buckets:  int = 0,
    ):
        self.mode      = mode
        self.n_buckets = n_buckets
        self._bnd_lbl  = _BOUNDARY_LABEL

        # Build flat list of all symbols seen in the corpus
        raw_symbols: list[str] = []
        for doc in documents:
            raw_symbols.extend(self._split(doc))

        # Build a sorted, deduplicated vocabulary
        unique = sorted(set(raw_symbols))
        unique.append(self._bnd_lbl)          # boundary token always last

        self.registry: list[str] = unique
        self._enc: dict[str, int] = {s: i for i, s in enumerate(unique)}
        self._dec: dict[int, str] = {i: s for i, s in enumerate(unique)}

        self.seal: int      = self._enc[self._bnd_lbl]
        self.vocab_sz: int  = len(unique)

    # ── Encoding / decoding ───────────────────────────────────────────────────

    def encipher(self, text: str) -> list[int]:
        """Convert a raw string to a list of integer token IDs."""
        return [self._enc[s] for s in self._split(text) if s in self._enc]

    def decipher(self, ids: Iterable[int]) -> str:
        """Convert integer token IDs back to the original symbol sequence."""
        parts = [self._dec.get(i, "?") for i in ids if i != self.seal]
        if self.mode == "word":
            return " ".join(parts)
        if self.mode == "token":
            return " → ".join(parts)
        return "".join(parts)

    def make_sequences(
        self, documents: list[str], horizon: int
    ) -> list[list[int]]:
        """
        Build sliding-window training sequences from a list of documents.

        Each sequence starts with the seal (BOS) token.
        Returns a flat list of fixed-length integer sequences.
        """
        seqs: list[list[int]] = []
        for doc in documents:
            glyphs = [self.seal] + self.encipher(doc) + [self.seal]
            n      = min(horizon, len(glyphs) - 1)
            seqs.append(glyphs[:n + 1])
        return seqs

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _split(self, text: str) -> list[str]:
        if self.mode == "word":
            return text.strip().split()
        if self.mode == "token":
            return [text.strip()]
        return list(text)   # char-level (default)

    def summary(self) -> dict:
        return {
            "vocab_size":  self.vocab_sz,
            "mode":        self.mode,
            "seal_id":     self.seal,
            "sample_vocab": self.registry[:10],
        }
