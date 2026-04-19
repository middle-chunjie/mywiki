#!/usr/bin/env python3
"""Scaffold raw/papers/<slug>/ from an arXiv ID, arXiv URL, or local PDF path.

Usage:
    python scripts/new_paper.py 1706.03762
    python scripts/new_paper.py https://arxiv.org/abs/1706.03762
    python scripts/new_paper.py ~/Downloads/some-paper.pdf

Outputs:
    raw/papers/<slug>/paper.pdf
    raw/papers/<slug>/paper.bib

The slug is <first-author-lastname>-<year>-<first-content-word>-<arxiv_id>
for arXiv papers, or <first-author-lastname>-<year>-<first-content-word>
for local PDFs (interactive prompt for title / author / year).

BibTeX citation_key pattern: <firstauthor><year><firstcontentword>,
e.g. vaswani2017attention.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

WIKI_ROOT = Path(__file__).resolve().parent.parent
RAW_PAPERS = WIKI_ROOT / "raw" / "papers"
ARXIV_API = "http://export.arxiv.org/api/query?id_list={id}"
ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

STOPWORDS = {
    "a", "an", "the", "of", "for", "in", "on", "and", "or", "to", "with",
    "is", "are", "be", "by", "from", "as", "that", "this", "at", "via",
    "towards", "toward", "using", "learning",
}


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower())
    return re.sub(r"-+", "-", text).strip("-")


def first_content_word(title: str) -> str:
    for word in title.split():
        w = re.sub(r"[^a-zA-Z]", "", word).lower()
        if w and w not in STOPWORDS:
            return w
    return "paper"


def normalize_arxiv_id(raw: str) -> str | None:
    # Match modern IDs like 1706.03762 or 2312.00752v2
    m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", raw)
    if m:
        return m.group(1)  # strip version
    # Old-style like cs/0603127
    m = re.search(r"([a-z\-]+/\d{7})", raw)
    if m:
        return m.group(1)
    return None


def fetch_arxiv_metadata(arxiv_id: str) -> dict:
    url = ARXIV_API.format(id=arxiv_id)
    req = urllib.request.Request(url, headers={"User-Agent": "MyWiki/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        xml_text = resp.read().decode("utf-8")
    root = ET.fromstring(xml_text)
    entry = root.find("atom:entry", ATOM_NS)
    if entry is None:
        raise SystemExit(f"arXiv returned no entry for {arxiv_id}")
    title = re.sub(r"\s+", " ", entry.findtext("atom:title", default="", namespaces=ATOM_NS)).strip()
    published = entry.findtext("atom:published", default="", namespaces=ATOM_NS)
    year = published[:4] if published else ""
    summary = re.sub(r"\s+", " ", entry.findtext("atom:summary", default="", namespaces=ATOM_NS)).strip()
    authors = [
        (a.findtext("atom:name", namespaces=ATOM_NS) or "").strip()
        for a in entry.findall("atom:author", ATOM_NS)
    ]
    doi = entry.findtext("arxiv:doi", default="", namespaces=ATOM_NS)
    pdf_url = None
    for link in entry.findall("atom:link", ATOM_NS):
        if link.get("title") == "pdf":
            pdf_url = link.get("href")
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "year": year,
        "authors": authors,
        "summary": summary,
        "doi": doi,
        "pdf_url": pdf_url or f"https://arxiv.org/pdf/{arxiv_id}.pdf",
    }


def download_pdf(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "MyWiki/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def bibtex_escape(s: str) -> str:
    return s.replace("{", r"\{").replace("}", r"\}")


def build_bibtex(meta: dict, citation_key: str) -> str:
    authors = " and ".join(meta["authors"]) or "Unknown"
    title = bibtex_escape(meta["title"])
    year = meta["year"] or ""
    arxiv_id = meta["arxiv_id"]
    doi = meta.get("doi") or ""
    url = f"https://arxiv.org/abs/{arxiv_id}"
    lines = [
        f"@article{{{citation_key},",
        f"  title = {{{title}}},",
        f"  author = {{{authors}}},",
        f"  year = {{{year}}},",
        f"  journal = {{arXiv preprint arXiv:{arxiv_id}}},",
        f"  url = {{{url}}},",
    ]
    if doi:
        lines.append(f"  doi = {{{doi}}},")
    lines.append("}")
    return "\n".join(lines) + "\n"


def make_slug(meta: dict) -> str:
    first_author = meta["authors"][0] if meta["authors"] else "unknown"
    first_author_last = first_author.split()[-1]
    year = meta["year"]
    short_title = first_content_word(meta["title"])
    arxiv_id = meta["arxiv_id"]
    return slugify(f"{first_author_last}-{year}-{short_title}-{arxiv_id}")


def make_citation_key(meta: dict) -> str:
    first_author_last = (
        meta["authors"][0].split()[-1] if meta["authors"] else "unknown"
    ).lower()
    first_author_last = re.sub(r"[^a-z]", "", first_author_last)
    year = meta["year"] or "nd"
    short = first_content_word(meta["title"])
    return f"{first_author_last}{year}{short}"


def ingest_arxiv(arxiv_id: str) -> Path:
    print(f"[arxiv] fetching metadata for {arxiv_id}")
    meta = fetch_arxiv_metadata(arxiv_id)
    slug = make_slug(meta)
    citation_key = make_citation_key(meta)
    target_dir = RAW_PAPERS / slug
    target_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = target_dir / "paper.pdf"
    bib_path = target_dir / "paper.bib"
    if pdf_path.exists():
        print(f"[skip] {pdf_path} already exists")
    else:
        print(f"[download] {meta['pdf_url']}")
        download_pdf(meta["pdf_url"], pdf_path)
    if bib_path.exists():
        print(f"[skip] {bib_path} already exists")
    else:
        bib_path.write_text(build_bibtex(meta, citation_key), encoding="utf-8")
        print(f"[write] {bib_path}")
    print(f"[done] slug = {slug}")
    print(f"[done] citation_key = {citation_key}")
    print(f"[next] fetch paper.md via DeepXiv MCP (mcp__deepxiv__get_full_paper) — or MinerU: python scripts/mineru_ingest.py {slug}")
    return target_dir


def ingest_local(
    pdf_path: Path,
    title: str | None = None,
    first_author_last: str | None = None,
    year: str | None = None,
) -> Path:
    print(f"[local] using {pdf_path}")
    if title is None:
        title = input("Paper title: ").strip() or pdf_path.stem
    if first_author_last is None:
        first_author_last = input("First author's last name: ").strip() or "unknown"
    if year is None:
        year = input("Year: ").strip() or "nd"
    short_title = first_content_word(title)
    slug = slugify(f"{first_author_last}-{year}-{short_title}")
    citation_key = re.sub(r"[^a-z]", "", first_author_last.lower()) + year + short_title
    target_dir = RAW_PAPERS / slug
    target_dir.mkdir(parents=True, exist_ok=True)
    target_pdf = target_dir / "paper.pdf"
    if target_pdf.exists():
        print(f"[skip] {target_pdf} already exists")
    else:
        shutil.copy2(pdf_path, target_pdf)
        print(f"[copy] {pdf_path} -> {target_pdf}")
    bib_path = target_dir / "paper.bib"
    if not bib_path.exists():
        bib_path.write_text(
            f"@article{{{citation_key},\n"
            f"  title = {{{title}}},\n"
            f"  author = {{{first_author_last}}},\n"
            f"  year = {{{year}}},\n"
            f"}}\n",
            encoding="utf-8",
        )
    print(f"[done] slug = {slug}")
    print(f"[done] citation_key = {citation_key}")
    print(f"[next] python scripts/mineru_ingest.py {slug}   # local PDF: MinerU only (DeepXiv needs arxiv_id)")
    return target_dir


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Scaffold raw/papers/<slug>/ from arXiv or local PDF")
    parser.add_argument("source", help="arXiv ID (e.g. 1706.03762), arXiv URL, or local PDF path")
    parser.add_argument("--title", help="paper title (local PDF mode; skips interactive prompt)")
    parser.add_argument("--author", help="first author's last name (local PDF mode; skips prompt)")
    parser.add_argument("--year", help="publication year (local PDF mode; skips prompt)")
    args = parser.parse_args(argv[1:])
    arxiv_id = normalize_arxiv_id(args.source)
    src_path = Path(args.source).expanduser()
    if src_path.is_file() and src_path.suffix.lower() == ".pdf":
        ingest_local(
            src_path.resolve(),
            title=args.title,
            first_author_last=args.author,
            year=args.year,
        )
    elif arxiv_id:
        ingest_arxiv(arxiv_id)
    else:
        print(f"Could not interpret '{args.source}' as arXiv ID or PDF path", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
