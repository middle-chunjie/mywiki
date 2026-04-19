#!/usr/bin/env python3
r"""Build <project>/paper/refs.bib from \cite{} keys in main.tex and raw/papers/*/paper.bib.

Usage:
    python scripts/build_bib.py projects/<project-slug>/

Reads every \cite{...}, \citet{...}, \citep{...} occurrence in <project>/paper/main.tex,
finds matching @article/@inproceedings entries in raw/papers/*/paper.bib, and writes
the concatenated subset to <project>/paper/refs.bib.

Warns about citation keys that are not found in the wiki's paper library.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

WIKI_ROOT = Path(__file__).resolve().parent.parent
RAW_PAPERS = WIKI_ROOT / "raw" / "papers"
CITE_RE = re.compile(r"\\cite[tpsA-Za-z]*\*?(?:\[[^\]]*\])?(?:\[[^\]]*\])?\{([^}]+)\}")
BIB_ENTRY_RE = re.compile(r"@\w+\s*\{\s*([^,]+)\s*,", re.MULTILINE)


def extract_citation_keys(tex_path: Path) -> set[str]:
    keys: set[str] = set()
    text = tex_path.read_text(encoding="utf-8")
    for m in CITE_RE.finditer(text):
        for key in m.group(1).split(","):
            key = key.strip()
            if key:
                keys.add(key)
    return keys


def split_bib_entries(bib_path: Path) -> dict[str, str]:
    """Return mapping {citation_key: full_entry_text}."""
    text = bib_path.read_text(encoding="utf-8")
    entries: dict[str, str] = {}
    matches = list(BIB_ENTRY_RE.finditer(text))
    for i, m in enumerate(matches):
        key = m.group(1).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        entries[key] = text[start:end].strip()
    return entries


def collect_all_bib_entries() -> dict[str, tuple[str, Path]]:
    """Map every citation_key found in raw/papers/*/paper.bib to (entry_text, source_file)."""
    all_entries: dict[str, tuple[str, Path]] = {}
    dupes: dict[str, list[Path]] = defaultdict(list)
    for bib_file in RAW_PAPERS.rglob("paper.bib"):
        for key, entry in split_bib_entries(bib_file).items():
            if key in all_entries:
                dupes[key].append(bib_file)
            else:
                all_entries[key] = (entry, bib_file)
    for key, files in dupes.items():
        print(
            f"[warn] duplicate citation_key '{key}' in {files} "
            f"(already registered from {all_entries[key][1]})",
            file=sys.stderr,
        )
    return all_entries


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Build project refs.bib from wiki citation keys")
    parser.add_argument("project_dir", help="projects/<slug>/ directory")
    args = parser.parse_args(argv[1:])

    project_dir = Path(args.project_dir).expanduser().resolve()
    paper_dir = project_dir / "paper"
    tex_path = paper_dir / "main.tex"
    if not tex_path.is_file():
        print(f"Missing {tex_path}", file=sys.stderr)
        return 1

    wanted = extract_citation_keys(tex_path)
    print(f"[build_bib] {len(wanted)} unique citation keys found in {tex_path}")
    all_entries = collect_all_bib_entries()

    matched: list[str] = []
    missing: list[str] = []
    for key in sorted(wanted):
        if key in all_entries:
            matched.append(all_entries[key][0])
        else:
            missing.append(key)

    refs_path = paper_dir / "refs.bib"
    if matched:
        refs_path.write_text("\n\n".join(matched) + "\n", encoding="utf-8")
        print(f"[build_bib] wrote {len(matched)} entries to {refs_path}")
    else:
        refs_path.write_text("", encoding="utf-8")
        print(f"[build_bib] no matches; wrote empty {refs_path}")

    if missing:
        print(f"[warn] {len(missing)} keys have no matching entry in raw/papers/*/paper.bib:")
        for key in missing:
            print(f"  - {key}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
