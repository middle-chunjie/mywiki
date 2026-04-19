#!/usr/bin/env python3
"""Scaffold raw/papers/<slug>/ from a decisions JSONL.

Consumed by the /batch-import skill after a Sonnet subagent has classified
each paper. This script does **only** the mechanical parts: folder creation,
bibtex writing, arxiv PDF download, DeepXiv markdown fetch, migration report.

Decisions JSONL (one object per line):
  {"action": "include", "arxiv_id": "1706.03762", "csl_item": {...}, "reason": "..."}
  {"action": "include", "arxiv_id": null,         "csl_item": {...}, "reason": "venue paper, no arxiv — manual PDF drop needed"}
  {"action": "uncertain", "csl_item": {...}, "reason": "low citations, not in CCF"}
  {"action": "excluded",  "csl_item": {...}, "reason": "education topic (cognitive diagnosis)"}

For each "include":
  - Generate slug (if not in input)
  - mkdir raw/papers/<slug>/
  - Write paper.bib from csl_item
  - If arxiv_id: download arxiv PDF, fetch paper.md via DeepXiv SDK
  - Else: folder + bib only (user drops paper.pdf manually; /batch-ingest picks it up)

Idempotent — existing files are not overwritten.

Usage:
    python scripts/batch_scaffold.py decisions.jsonl
    python scripts/batch_scaffold.py decisions.jsonl --report-path wiki/outputs/batch-import-2026-04-17.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import sys
import urllib.request
from collections import Counter
from pathlib import Path

try:
    from deepxiv_sdk import Reader
except ImportError:
    Reader = None  # type: ignore

WIKI_ROOT = Path(__file__).resolve().parent.parent
RAW_PAPERS = WIKI_ROOT / "raw" / "papers"
OUTPUTS_DIR = WIKI_ROOT / "wiki" / "outputs"

STOPWORDS = {
    "a", "an", "the", "of", "for", "in", "on", "and", "or", "to", "with",
    "is", "are", "be", "by", "from", "as", "that", "this", "at", "via",
    "towards", "toward", "using", "learning",
}


# ----------------------------- slug / bib helpers -----------------------------


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "").lower())
    return re.sub(r"-+", "-", text).strip("-")


def first_content_word(title: str) -> str:
    for word in (title or "").split():
        w = re.sub(r"[^a-zA-Z]", "", word).lower()
        if w and w not in STOPWORDS:
            return w
    return "paper"


def make_slug(csl_item: dict, arxiv_id: str | None) -> str:
    authors = csl_item.get("author") or []
    first_last = (authors[0].get("family") if authors else "unknown") or "unknown"
    ym = ((csl_item.get("issued") or {}).get("date-parts") or [[None]])[0]
    year = str(ym[0]) if ym and ym[0] else "nd"
    short = first_content_word(csl_item.get("title") or "paper")
    if arxiv_id:
        return slugify(f"{first_last}-{year}-{short}-{arxiv_id.replace('/', '-')}")
    return slugify(f"{first_last}-{year}-{short}")


def make_citation_key(csl_item: dict) -> str:
    authors = csl_item.get("author") or []
    first_last = ((authors[0].get("family") if authors else "unknown") or "unknown").lower()
    first_last = re.sub(r"[^a-z]", "", first_last) or "unknown"
    ym = ((csl_item.get("issued") or {}).get("date-parts") or [[None]])[0]
    year = str(ym[0]) if ym and ym[0] else "nd"
    short = first_content_word(csl_item.get("title") or "paper")
    return f"{first_last}{year}{short}"


def csl_to_bibtex(csl_item: dict, citation_key: str, arxiv_id: str | None) -> str:
    ent_type = "inproceedings" if csl_item.get("container-title") else "article"
    title = (csl_item.get("title") or "").replace("{", "\\{").replace("}", "\\}")
    pairs = []
    for a in csl_item.get("author") or []:
        family = a.get("family", "") or ""
        given = a.get("given", "") or ""
        if family and given:
            pairs.append(f"{family}, {given}")
        elif family:
            pairs.append(family)
        elif given:
            pairs.append(given)
    authors = " and ".join(pairs) or "Unknown"
    ym = ((csl_item.get("issued") or {}).get("date-parts") or [[None]])[0]
    year = str(ym[0]) if ym and ym[0] else ""
    venue = csl_item.get("container-title") or ""
    doi = csl_item.get("DOI") or ""
    url = csl_item.get("URL") or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "")
    lines = [
        f"@{ent_type}{{{citation_key},",
        f"  title = {{{title}}},",
        f"  author = {{{authors}}},",
    ]
    if year:
        lines.append(f"  year = {{{year}}},")
    if venue:
        key = "booktitle" if ent_type == "inproceedings" else "journal"
        lines.append(f"  {key} = {{{venue}}},")
    if arxiv_id:
        lines.append(f"  note = {{arXiv:{arxiv_id}}},")
    if doi:
        lines.append(f"  doi = {{{doi}}},")
    if url:
        lines.append(f"  url = {{{url}}},")
    lines.append("}")
    return "\n".join(lines) + "\n"


# ----------------------------- fetch helpers -----------------------------


def download_arxiv_pdf(arxiv_id: str, dest: Path) -> bool:
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MyWiki/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)
        return True
    except Exception as e:
        print(f"  [err] arxiv PDF download failed ({arxiv_id}): {e}", file=sys.stderr)
        return False


def fetch_deepxiv_md(reader, arxiv_id: str, dest: Path) -> bool:
    try:
        md = reader.raw(arxiv_id)
        if not isinstance(md, str) or len(md) < 500:
            print(f"  [err] DeepXiv content for {arxiv_id} unexpectedly short", file=sys.stderr)
            return False
        dest.write_text(md, encoding="utf-8")
        return True
    except Exception as e:
        print(f"  [err] DeepXiv fetch failed ({arxiv_id}): {e}", file=sys.stderr)
        return False


def load_deepxiv_token() -> str | None:
    """Load DEEPXIV_TOKEN: env → MyWiki/.env → Claude Code MCP config (last resort)."""
    import os, json
    token = os.environ.get("DEEPXIV_TOKEN")
    if token:
        return token
    env_file = WIKI_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("DEEPXIV_TOKEN="):
                val = line.split("=", 1)[1].strip()
                if val:
                    return val
    try:
        claude_config = Path.home() / ".claude.json"
        if claude_config.is_file():
            cfg = json.loads(claude_config.read_text(encoding="utf-8"))
            val = (cfg.get("mcpServers", {}).get("deepxiv", {}).get("env", {}) or {}).get("DEEPXIV_TOKEN")
            if val:
                return val
    except Exception:
        pass
    return None


def make_reader():
    if Reader is None:
        return None
    token = load_deepxiv_token()
    if not token:
        return None
    return Reader(token=token)


# ----------------------------- scaffolding -----------------------------


def scaffold(decision: dict, reader) -> tuple[str, str, str]:
    """Scaffold one paper. Returns (slug, status, note).
    status ∈ {ok, partial-no-md, failed, skipped}."""
    csl_item = decision.get("csl_item") or {}
    arxiv_id = decision.get("arxiv_id")
    slug = decision.get("slug") or make_slug(csl_item, arxiv_id)

    target_dir = RAW_PAPERS / slug
    target_dir.mkdir(parents=True, exist_ok=True)

    citation_key = make_citation_key(csl_item)
    bib_path = target_dir / "paper.bib"
    if not bib_path.exists():
        bib_path.write_text(csl_to_bibtex(csl_item, citation_key, arxiv_id), encoding="utf-8")

    if not arxiv_id:
        return (slug, "partial-no-md", "no arxiv_id — drop paper.pdf manually, then run mineru_ingest.py")

    pdf_path = target_dir / "paper.pdf"
    if not pdf_path.exists():
        if not download_arxiv_pdf(arxiv_id, pdf_path):
            return (slug, "failed", "arxiv PDF download failed")

    md_path = target_dir / "paper.md"
    if md_path.exists():
        return (slug, "ok", "already had paper.md")
    if reader is None:
        return (slug, "partial-no-md", "DeepXiv SDK/token unavailable — paper.md not fetched")
    if fetch_deepxiv_md(reader, arxiv_id, md_path):
        return (slug, "ok", "arxiv PDF + DeepXiv paper.md")
    return (slug, "partial-no-md", "DeepXiv fetch failed — run mineru_ingest.py as fallback")


# ----------------------------- report -----------------------------


def render_table(lines: list[str], title: str, header: list[str], rows: list[list[str]]) -> None:
    if not rows:
        return
    lines.append(f"## {title}")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in row) + " |")
    lines.append("")


def write_report(
    decisions: list[dict],
    scaffolded: list[tuple[str, str, str, dict]],
    report_path: Path,
) -> None:
    today = dt.date.today().isoformat()
    lines: list[str] = []
    lines.append("---")
    lines.append("type: batch-import-report")
    lines.append("graph-excluded: true")
    lines.append(f"date: {today}")
    lines.append(f"total_items: {len(decisions)}")
    lines.append("---")
    lines.append("")
    lines.append(f"# Batch-import report — {today}")
    lines.append("")

    action_counts = Counter(d.get("action") for d in decisions)
    status_counts = Counter(s for _slug, s, _note, _d in scaffolded)

    lines.append(f"- **Total decisions**: {len(decisions)}")
    for action in ("include", "uncertain", "excluded", "skip"):
        if action_counts.get(action):
            lines.append(f"  - `{action}`: {action_counts[action]}")
    if scaffolded:
        lines.append(f"- **Scaffold results**:")
        for status, n in status_counts.most_common():
            lines.append(f"  - `{status}`: {n}")
    lines.append("")

    # Included table
    inc_rows = []
    for slug, status, note, d in scaffolded:
        title = (d.get("csl_item") or {}).get("title", "")[:60]
        inc_rows.append([slug, title, status, note, d.get("reason", "")[:80]])
    render_table(
        lines, "Included — scaffolded",
        ["slug", "title", "status", "note", "reason"], inc_rows,
    )

    # Uncertain
    unc_rows = []
    for d in decisions:
        if d.get("action") != "uncertain":
            continue
        title = (d.get("csl_item") or {}).get("title", "")[:60]
        unc_rows.append([title, d.get("arxiv_id") or "—", d.get("reason", "")[:100]])
    render_table(
        lines, "Uncertain — manual review",
        ["title", "arxiv_id", "reason"], unc_rows,
    )

    # Excluded
    exc_rows = []
    for d in decisions:
        if d.get("action") != "excluded":
            continue
        title = (d.get("csl_item") or {}).get("title", "")[:60]
        exc_rows.append([title, d.get("reason", "")[:100]])
    render_table(
        lines, "Excluded",
        ["title", "reason"], exc_rows,
    )

    lines.append("## Next step")
    lines.append("")
    lines.append("Run `/batch-ingest --auto` in Claude Code to INGEST the scaffolded papers into the wiki.")
    lines.append("")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------- main -----------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Scaffold raw/papers/<slug>/ from a decisions JSONL")
    parser.add_argument("decisions_path", help="Path to decisions JSONL file")
    parser.add_argument("--report-path", help="Path to write the migration report (default: wiki/outputs/batch-import-YYYY-MM-DD.md)")
    args = parser.parse_args(argv[1:])

    decisions_path = Path(args.decisions_path).expanduser()
    if not decisions_path.is_file():
        print(f"not a file: {decisions_path}", file=sys.stderr)
        return 1

    decisions: list[dict] = []
    for ln_no, line in enumerate(decisions_path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            decisions.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"[err] line {ln_no} invalid JSON: {e}", file=sys.stderr)
            return 1

    reader = make_reader()
    if reader is None:
        print("[warn] DeepXiv reader unavailable — arxiv papers will be scaffolded without paper.md", file=sys.stderr)

    scaffolded: list[tuple[str, str, str, dict]] = []
    for d in decisions:
        if d.get("action") != "include":
            continue
        slug, status, note = scaffold(d, reader)
        title = (d.get("csl_item") or {}).get("title", "<no title>")[:60]
        print(f"  [{status:14}] {slug}: {title}")
        scaffolded.append((slug, status, note, d))

    if args.report_path:
        report_path = Path(args.report_path).expanduser()
    else:
        today = dt.date.today().isoformat()
        report_path = OUTPUTS_DIR / f"batch-import-{today}.md"
    write_report(decisions, scaffolded, report_path)
    print(f"\n[done] report written to {report_path.relative_to(WIKI_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
