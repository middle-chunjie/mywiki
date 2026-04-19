#!/usr/bin/env python3
"""Lint the MyWiki knowledge base.

Runs 10 checks and writes a Markdown report to wiki/outputs/lint-YYYY-MM-DD.md.

Usage:
    python scripts/lint.py

Exit code: 0 if no findings, 1 if any findings, 2 if a check itself failed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # type: ignore

WIKI_ROOT = Path(__file__).resolve().parent.parent
WIKI_DIR = WIKI_ROOT / "wiki"
RAW_DIR = WIKI_ROOT / "raw"

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
CITE_KEY_RE = re.compile(r"@\w+\s*\{\s*([^,]+)\s*,", re.MULTILINE)

VOLATILITY_DAYS = {"high": 90, "medium": 180, "low": 365}
SYSTEM_STEMS = {"index", "log", "overview", "QUESTIONS"}

# ----------------------------- helpers -----------------------------


def parse_frontmatter(text: str) -> tuple[dict, str]:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    fm_text = m.group(1)
    body = text[m.end():]
    if yaml is not None:
        try:
            data = yaml.safe_load(fm_text) or {}
            if not isinstance(data, dict):
                data = {}
            return data, body
        except yaml.YAMLError:
            return {}, body
    # Minimal fallback parser (flat keys only; good enough for validation).
    data: dict[str, Any] = {}
    for line in fm_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in line:
            continue
        k, _, v = line.partition(":")
        v = v.strip()
        if v.startswith("[") and v.endswith("]"):
            v_list = [
                item.strip().strip('"').strip("'")
                for item in v[1:-1].split(",")
                if item.strip()
            ]
            data[k.strip()] = v_list
        else:
            data[k.strip()] = v.strip().strip('"').strip("'")
    return data, body


def jaccard_bigrams(a: str, b: str) -> float:
    def bigrams(s: str) -> set[str]:
        return {s[i:i + 2] for i in range(len(s) - 1)}

    A, B = bigrams(a), bigrams(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def gather_wiki_files(include_system: bool = False) -> list[Path]:
    files = [
        p for p in WIKI_DIR.rglob("*.md")
        if "templates" not in p.parts and "outputs" not in p.parts
    ]
    if not include_system:
        files = [p for p in files if not (p.parent == WIKI_DIR and p.stem in SYSTEM_STEMS)]
    return files


# ----------------------------- checks -----------------------------


def check_1_frontmatter(files: list[Path]) -> list[str]:
    findings: list[str] = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        fm, _ = parse_frontmatter(text)
        if not fm:
            findings.append(f"{f.relative_to(WIKI_ROOT)}: missing or invalid YAML frontmatter")
            continue
        if "type" not in fm:
            findings.append(f"{f.relative_to(WIKI_ROOT)}: missing 'type' in frontmatter")
        if "date" not in fm:
            findings.append(f"{f.relative_to(WIKI_ROOT)}: missing 'date' in frontmatter")
    return findings


def check_2_broken_wikilinks(files: list[Path]) -> list[str]:
    findings: list[str] = []
    stem_index: set[str] = set()
    for p in WIKI_DIR.rglob("*.md"):
        stem_index.add(p.stem)
    for f in files:
        text = f.read_text(encoding="utf-8")
        for m in WIKILINK_RE.finditer(text):
            target = m.group(1).strip()
            stem = Path(target).stem
            if stem not in stem_index:
                findings.append(f"{f.relative_to(WIKI_ROOT)}: broken wikilink [[{target}]]")
    return findings


def check_3_index_consistency() -> list[str]:
    findings: list[str] = []
    index_path = WIKI_DIR / "index.md"
    if not index_path.is_file():
        return ["wiki/index.md is missing"]
    text = index_path.read_text(encoding="utf-8")
    for m in re.finditer(r"(wiki/[\w/\-.]+\.md)", text):
        rel = m.group(1)
        if not (WIKI_ROOT / rel).is_file():
            findings.append(f"wiki/index.md references missing file: {rel}")
    return findings


def check_4_stubs(files: list[Path]) -> list[str]:
    findings: list[str] = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        _, body = parse_frontmatter(text)
        if "redirect:" in body[:200]:
            continue
        stripped = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)
        stripped = re.sub(r"```.*?```", "", stripped, flags=re.DOTALL)
        words = len(stripped.split())
        if words < 100:
            findings.append(f"{f.relative_to(WIKI_ROOT)}: stub page ({words} words)")
    return findings


def check_5_near_duplicates(files: list[Path]) -> list[str]:
    findings: list[str] = []
    concept_slugs = [f.stem for f in files if "concepts" in f.parts]
    for i, a in enumerate(concept_slugs):
        for b in concept_slugs[i + 1:]:
            score = jaccard_bigrams(a, b)
            if score > 0.7:
                findings.append(f"near-duplicate concept slugs (Jaccard={score:.2f}): {a} ~ {b}")
    return findings


def check_6_stale(files: list[Path]) -> list[str]:
    findings: list[str] = []
    today = dt.date.today()
    for f in files:
        if "concepts" not in f.parts:
            continue
        fm, _ = parse_frontmatter(f.read_text(encoding="utf-8"))
        volatility = str(fm.get("domain_volatility", "medium"))
        last_reviewed = fm.get("last_reviewed")
        if not last_reviewed:
            continue
        try:
            then = dt.date.fromisoformat(str(last_reviewed))
        except ValueError:
            continue
        threshold = VOLATILITY_DAYS.get(volatility, 180)
        days = (today - then).days
        if days > threshold:
            findings.append(
                f"{f.relative_to(WIKI_ROOT)}: stale ({days}d > {threshold}d threshold for {volatility})"
            )
    return findings


def check_7_cross_language_duplication(files: list[Path]) -> list[str]:
    findings: list[str] = []
    url_to_files = defaultdict(list)
    for f in files:
        if "sources" not in f.parts:
            continue
        fm, _ = parse_frontmatter(f.read_text(encoding="utf-8"))
        url = fm.get("canonical_source") or fm.get("source_url") or fm.get("url")
        if url:
            url_to_files[str(url)].append(f.relative_to(WIKI_ROOT))
    for url, paths in url_to_files.items():
        if len(paths) > 1:
            findings.append(
                f"duplicate source URL {url} across: {', '.join(str(p) for p in paths)}"
            )

    alias_to_concepts = defaultdict(set)
    for f in files:
        if "concepts" not in f.parts:
            continue
        fm, _ = parse_frontmatter(f.read_text(encoding="utf-8"))
        aliases = fm.get("aliases") or []
        if isinstance(aliases, list):
            for a in aliases:
                alias_to_concepts[str(a).strip().lower()].add(f.stem)
    for alias, concepts in alias_to_concepts.items():
        if len(concepts) > 1:
            findings.append(
                f"alias '{alias}' appears in multiple concepts: {', '.join(sorted(concepts))}"
            )
    return findings


BAD_WIKILINK_RE = re.compile(r"[\u4e00-\u9fff]|[A-Z]|_")


def check_8_wikilink_format(files: list[Path]) -> list[str]:
    findings: list[str] = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        for m in WIKILINK_RE.finditer(text):
            target = m.group(1).strip()
            stem = Path(target).stem
            if BAD_WIKILINK_RE.search(stem):
                findings.append(
                    f"{f.relative_to(WIKI_ROOT)}: non-kebab-case wikilink [[{target}]]"
                )
    return findings


def check_9_paper_folder_integrity(files: list[Path]) -> list[str]:
    findings: list[str] = []
    for f in files:
        if "sources" not in f.parts:
            continue
        fm, _ = parse_frontmatter(f.read_text(encoding="utf-8"))
        if fm.get("subtype") != "paper":
            continue
        slug = f.stem
        folder = RAW_DIR / "papers" / slug
        for needed in ("paper.pdf", "paper.md", "paper.bib"):
            if not (folder / needed).is_file():
                findings.append(
                    f"{f.relative_to(WIKI_ROOT)}: missing raw/papers/{slug}/{needed}"
                )
    return findings


def check_10_bibkey_uniqueness() -> list[str]:
    findings: list[str] = []
    key_to_files = defaultdict(list)
    papers_dir = RAW_DIR / "papers"
    if not papers_dir.is_dir():
        return findings
    for bib in papers_dir.rglob("paper.bib"):
        text = bib.read_text(encoding="utf-8")
        for m in CITE_KEY_RE.finditer(text):
            key_to_files[m.group(1).strip()].append(bib.relative_to(WIKI_ROOT))
    for key, paths in key_to_files.items():
        if len(paths) > 1:
            findings.append(
                f"duplicate citation_key '{key}' in: {', '.join(str(p) for p in paths)}"
            )
    return findings


CHECK_LABELS = {
    1: "YAML frontmatter validity",
    2: "Broken wikilinks",
    3: "Index consistency",
    4: "Stub pages",
    5: "Near-duplicate concept slugs",
    6: "Stale pages by domain_volatility",
    7: "Cross-language duplication",
    8: "Wikilink format",
    9: "Paper folder integrity",
    10: "Citation-key uniqueness across paper.bib files",
}

N_CHECKS = len(CHECK_LABELS)


def render_report(results: dict[int, list[str]]) -> str:
    today = dt.date.today().isoformat()
    total = sum(len(v) for v in results.values())
    out: list[str] = []
    out.append("---")
    out.append("type: lint-report")
    out.append("graph-excluded: true")
    out.append(f"date: {today}")
    out.append(f"findings: {total}")
    out.append("---")
    out.append("")
    out.append(f"# Lint Report — {today}")
    out.append("")
    out.append(f"Total findings: **{total}** across {N_CHECKS} checks.")
    out.append("")
    for i in sorted(results):
        items = results[i]
        out.append(f"## Check {i} — {CHECK_LABELS[i]} ({len(items)})")
        out.append("")
        if not items:
            out.append("_No findings._")
        else:
            for item in items:
                out.append(f"- {item}")
        out.append("")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=f"Lint MyWiki ({N_CHECKS} checks)")
    parser.add_argument("--no-report", action="store_true", help="print to stdout instead of writing a file")
    args = parser.parse_args(argv[1:])

    if yaml is None:
        print(
            "[lint] warning: PyYAML not installed; falling back to minimal parser. "
            "Install with: pip install pyyaml",
            file=sys.stderr,
        )

    files = gather_wiki_files(include_system=False)
    results: dict[int, list[str]] = {
        1: check_1_frontmatter(files),
        2: check_2_broken_wikilinks(files),
        3: check_3_index_consistency(),
        4: check_4_stubs(files),
        5: check_5_near_duplicates(files),
        6: check_6_stale(files),
        7: check_7_cross_language_duplication(files),
        8: check_8_wikilink_format(files),
        9: check_9_paper_folder_integrity(files),
        10: check_10_bibkey_uniqueness(),
    }

    report = render_report(results)
    if args.no_report:
        print(report)
    else:
        outputs_dir = WIKI_DIR / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        report_path = outputs_dir / f"lint-{dt.date.today().isoformat()}.md"
        report_path.write_text(report, encoding="utf-8")
        print(f"[lint] report written to {report_path.relative_to(WIKI_ROOT)}")

    total = sum(len(v) for v in results.values())
    print(f"[lint] {total} findings across {N_CHECKS} checks")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
