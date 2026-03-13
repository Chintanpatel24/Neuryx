"""
intake/portal.py
Multi-format data ingestion portal.

Supported input formats:
  1  → Plain text  (.txt)           one document per line
  2  → CSV         (.csv)           specify a text column
  3  → Excel       (.xlsx / .xls)   specify a text column
  4  → JSON        (.json)          array of strings or objects
  5  → TSV         (.tsv)           tab-separated, specify a column
  6  → Auto-detect                  guesses from file extension

Every loader returns list[str] — a flat list of document strings.
"""

from __future__ import annotations
import os
import csv
import json
from typing import Any


# ── Format constants (mirrored in the UI menu) ─────────────────────────────

FMT_TXT  = 1
FMT_CSV  = 2
FMT_XLSX = 3
FMT_JSON = 4
FMT_TSV  = 5
FMT_AUTO = 6

FORMAT_LABELS = {
    FMT_TXT:  "Plain Text  (.txt)   — one document per line",
    FMT_CSV:  "CSV         (.csv)   — choose a column",
    FMT_XLSX: "Excel       (.xlsx)  — choose a column",
    FMT_JSON: "JSON        (.json)  — array of strings or records",
    FMT_TSV:  "TSV         (.tsv)   — tab-separated, choose a column",
    FMT_AUTO: "Auto-detect          — guess from file extension",
}


class Portal:
    """
    Unified entry point for loading documents from various file formats.

    Usage
    -----
    portal = Portal()
    docs   = portal.ingest(path, fmt=FMT_CSV, column="text")
    """

    # ── Main entry point ──────────────────────────────────────────────────────

    def ingest(
        self,
        path:      str,
        fmt:       int   = FMT_AUTO,
        column:    str | None = None,   # CSV / XLSX / TSV column name
        sheet:     int   = 0,           # XLSX sheet index
        encoding:  str   = "utf-8-sig",
    ) -> list[str]:
        """
        Load a file and return a flat list of document strings.

        Raises ValueError with a human-friendly message on failure.
        """
        if not os.path.exists(path):
            raise ValueError(f"File not found: '{path}'")

        if fmt == FMT_AUTO:
            fmt = self._detect_fmt(path)

        dispatch = {
            FMT_TXT:  self._read_txt,
            FMT_CSV:  self._read_csv,
            FMT_XLSX: self._read_xlsx,
            FMT_JSON: self._read_json,
            FMT_TSV:  self._read_tsv,
        }
        if fmt not in dispatch:
            raise ValueError(f"Unknown format id: {fmt}")

        loader = dispatch[fmt]
        kwargs: dict[str, Any] = {"path": path, "encoding": encoding}
        if fmt in (FMT_CSV, FMT_XLSX, FMT_TSV):
            kwargs["column"] = column
        if fmt == FMT_XLSX:
            kwargs["sheet"] = sheet

        docs = loader(**kwargs)
        docs = [d.strip() for d in docs if d and d.strip()]

        if not docs:
            raise ValueError(
                "No documents loaded — file appears empty or the column is missing."
            )
        return docs

    # ── Per-format readers ────────────────────────────────────────────────────

    @staticmethod
    def _read_txt(path: str, encoding: str = "utf-8-sig", **_) -> list[str]:
        with open(path, encoding=encoding, errors="replace") as fh:
            return [line.rstrip("\n") for line in fh]

    @staticmethod
    def _read_csv(
        path:     str,
        column:   str | None = None,
        encoding: str = "utf-8-sig",
        **_,
    ) -> list[str]:
        with open(path, newline="", encoding=encoding, errors="replace") as fh:
            rdr  = csv.DictReader(fh)
            hdrs = [h.strip() for h in (rdr.fieldnames or [])]

            # Pick column
            col = Portal._pick_column(column, hdrs)

            rows: list[str] = []
            for row in rdr:
                val = row.get(col, "")
                if val:
                    rows.append(str(val).strip())
            return rows

    @staticmethod
    def _read_xlsx(
        path:     str,
        column:   str | None = None,
        sheet:    int  = 0,
        encoding: str  = "utf-8-sig",   # unused but kept for uniform signature
        **_,
    ) -> list[str]:
        try:
            import openpyxl  # type: ignore
        except ImportError:
            raise ImportError(
                "openpyxl is required for Excel support.\n"
                "Install with:  pip install openpyxl"
            )
        wb   = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws   = wb.worksheets[sheet]
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            return []

        # First row = headers
        headers = [str(c).strip() if c is not None else "" for c in rows[0]]
        col     = Portal._pick_column(column, headers)
        col_idx = headers.index(col)

        docs: list[str] = []
        for row in rows[1:]:
            val = row[col_idx] if col_idx < len(row) else None
            if val is not None:
                docs.append(str(val).strip())
        wb.close()
        return docs

    @staticmethod
    def _read_json(path: str, encoding: str = "utf-8-sig", **_) -> list[str]:
        with open(path, encoding=encoding, errors="replace") as fh:
            payload = json.load(fh)

        if isinstance(payload, list):
            docs: list[str] = []
            for item in payload:
                if isinstance(item, str):
                    docs.append(item)
                elif isinstance(item, dict):
                    # Grab first string value found
                    for v in item.values():
                        if isinstance(v, str):
                            docs.append(v)
                            break
            return docs

        if isinstance(payload, dict):
            # Assume {"documents": [...]} or {"data": [...]}
            for key in ("documents", "data", "texts", "records", "items"):
                if key in payload and isinstance(payload[key], list):
                    return Portal._read_json.__func__(   # type: ignore[attr-defined]
                        None, path=path
                    )
            return [str(v) for v in payload.values() if isinstance(v, str)]

        raise ValueError("JSON file must contain an array or an object with a list field.")

    @staticmethod
    def _read_tsv(
        path:     str,
        column:   str | None = None,
        encoding: str = "utf-8-sig",
        **_,
    ) -> list[str]:
        with open(path, newline="", encoding=encoding, errors="replace") as fh:
            rdr  = csv.DictReader(fh, delimiter="\t")
            hdrs = [h.strip() for h in (rdr.fieldnames or [])]
            col  = Portal._pick_column(column, hdrs)
            return [str(row.get(col, "")).strip() for row in rdr]

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _pick_column(requested: str | None, available: list[str]) -> str:
        """
        Return the best matching column name.
        If `requested` is None, auto-pick the first text-ish column.
        """
        if requested:
            if requested in available:
                return requested
            # Case-insensitive fallback
            lower_map = {h.lower(): h for h in available}
            if requested.lower() in lower_map:
                return lower_map[requested.lower()]
            raise ValueError(
                f"Column '{requested}' not found.\n"
                f"Available columns: {available}"
            )

        # Auto-pick heuristic: prefer columns named text/content/value/name/label
        preferred = ["text", "content", "value", "name", "label",
                     "document", "sentence", "description", "event", "category"]
        for p in preferred:
            for h in available:
                if p in h.lower():
                    return h

        # Last resort: first column
        if available:
            return available[0]
        raise ValueError("No columns found in file.")

    @staticmethod
    def _detect_fmt(path: str) -> int:
        ext = os.path.splitext(path)[1].lower()
        mapping = {
            ".txt":  FMT_TXT,
            ".csv":  FMT_CSV,
            ".xlsx": FMT_XLSX,
            ".xls":  FMT_XLSX,
            ".json": FMT_JSON,
            ".tsv":  FMT_TSV,
        }
        if ext not in mapping:
            raise ValueError(
                f"Cannot auto-detect format for extension '{ext}'.\n"
                "Supported: .txt .csv .xlsx .xls .json .tsv"
            )
        return mapping[ext]

    @staticmethod
    def list_formats() -> dict[int, str]:
        return dict(FORMAT_LABELS)
