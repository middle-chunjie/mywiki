#!/usr/bin/env python3
"""For each colliding citation_key across multiple paper.bib files, rewrite the
key in-place to disambiguate: append the last 4 digits of the arxiv_id if the
bib carries one, otherwise a short hash of the slug.

Runs scripts/lint.py check 10 logic inline to detect current collisions.
"""
from __future__ import annotations
import hashlib, re, sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "raw" / "papers"

KEY_RE = re.compile(r"(@\w+\s*\{\s*)([^,\s]+)(\s*,)", re.MULTILINE)
ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})")


def extract_arxiv(bib_text: str) -> str | None:
    m = ARXIV_RE.search(bib_text)
    return m.group(1) if m else None


def short_hash(slug: str) -> str:
    return hashlib.md5(slug.encode()).hexdigest()[:4]


def main() -> int:
    # 1. Collect (key, bib_path) pairs
    by_key: dict[str, list[Path]] = defaultdict(list)
    for bib in RAW.rglob("paper.bib"):
        text = bib.read_text(encoding="utf-8", errors="ignore")
        m = KEY_RE.search(text)
        if m:
            by_key[m.group(2).strip()].append(bib)

    # 2. For each colliding group, assign new unique keys
    changes = []
    for key, paths in by_key.items():
        if len(paths) < 2:
            continue
        for p in paths:
            text = p.read_text(encoding="utf-8", errors="ignore")
            arxiv = extract_arxiv(text)
            if arxiv:
                tail = re.sub(r"[^0-9]", "", arxiv)[-4:]
                new_key = f"{key}{tail}" if tail else None
            else:
                new_key = f"{key}{short_hash(p.parent.name)}"
            if new_key is None:
                print(f"[skip] {p}: no arxiv_id and hash degenerate", file=sys.stderr)
                continue
            if new_key == key:
                continue
            new_text = KEY_RE.sub(
                lambda m, nk=new_key: m.group(1) + nk + m.group(3), text, count=1
            )
            if new_text != text:
                p.write_text(new_text, encoding="utf-8")
                changes.append((str(p.relative_to(REPO)), key, new_key))

    print(f"Rewrote {len(changes)} paper.bib citation_key(s):")
    for path, old, new in changes:
        print(f"  {path}  {old} → {new}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
