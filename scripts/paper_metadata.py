#!/usr/bin/env python3
"""Fresh paper metadata lookup: DBLP + DeepXiv (citations, tldr, keywords) + arxiv + CCF rank.

This is a helper consumed by the /batch-import skill. The Sonnet subagent uses
the enriched JSON to decide whether a paper should be included in the wiki.

Sources per paper:
  - arxiv Atom API   — title, comment (often "Accepted at X"), primary category, abstract
  - DeepXiv brief()  — citation count, tldr, keywords, publish_at (replaces direct S2)
  - DeepXiv social_impact() — tweets/likes/views (optional, via --include-social)
  - DBLP             — title search with first-author disambiguation; venue
  - CCF lookup       — rank for resolved venue (scripts/data/ccf_venues.json)

DeepXiv is accessed through the SDK and requires DEEPXIV_TOKEN (env or .env).
No Semantic Scholar API key needed — DeepXiv fronts S2 data for us.

Usage:
    python scripts/paper_metadata.py --arxiv 1706.03762 [--pretty]
    python scripts/paper_metadata.py --title "Attention Is All You Need"
    python scripts/paper_metadata.py --csl path/to/zotero.json [--sleep 1.0] [--include-social]

In --csl batch mode, one JSON object per line is emitted to stdout (JSONL).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

try:
    from deepxiv_sdk import Reader
except ImportError:
    Reader = None  # type: ignore

WIKI_ROOT = Path(__file__).resolve().parent.parent
CCF_LIST_PATH = Path(__file__).resolve().parent / "data" / "ccf_venues.json"

USER_AGENT = "MyWiki/1.0 (research knowledge base)"
ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?|([a-z\-]+/\d{7})")
GITHUB_RE = re.compile(r"github\.com/[\w\-\.]+/[\w\-\.]+", re.IGNORECASE)
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


# ----------------------------- HTTP helpers -----------------------------


def _http_get_json(url: str, headers: Optional[dict] = None, timeout: int = 20, max_retries: int = 3) -> Optional[dict]:
    all_headers = {"User-Agent": USER_AGENT, "Accept": "application/json", **(headers or {})}
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=all_headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                wait = 2 ** (attempt + 1)
                print(f"  [rate-limit] {e.code} on {url[:60]}...; sleep {wait}s (attempt {attempt+1}/{max_retries})", file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"  [warn] GET {url[:60]}... failed: HTTP {e.code}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"  [warn] GET {url[:60]}... failed: {e}", file=sys.stderr)
            return None
    return None


def _http_get_text(url: str, timeout: int = 20) -> Optional[str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except Exception as e:
        print(f"  [warn] GET {url[:80]}... failed: {e}", file=sys.stderr)
        return None


# ----------------------------- source fetchers -----------------------------


def fetch_arxiv(arxiv_id: str) -> dict:
    """Fetch from arxiv Atom API. Returns title, comment, primary_category, abstract."""
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    text = _http_get_text(url)
    if not text:
        return {"found": False}
    try:
        root = ET.fromstring(text)
        entry = root.find("atom:entry", ATOM_NS)
        if entry is None:
            return {"found": False}
        title = re.sub(r"\s+", " ", entry.findtext("atom:title", default="", namespaces=ATOM_NS)).strip()
        summary = re.sub(r"\s+", " ", entry.findtext("atom:summary", default="", namespaces=ATOM_NS)).strip()
        published = entry.findtext("atom:published", default="", namespaces=ATOM_NS)
        comment = (entry.findtext("arxiv:comment", default="", namespaces=ATOM_NS) or "").strip()
        cat_el = entry.find("arxiv:primary_category", ATOM_NS)
        primary_cat = cat_el.get("term") if cat_el is not None else ""
        return {
            "found": True,
            "title": title,
            "published": published,
            "comment": comment,
            "primary_category": primary_cat,
            "abstract": summary,
        }
    except ET.ParseError as e:
        return {"found": False, "error": f"xml parse: {e}"}


def _load_deepxiv_token() -> Optional[str]:
    """Load DEEPXIV_TOKEN, checking env → MyWiki/.env → Claude Code MCP config."""
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
    # Last resort: read from Claude Code MCP server config
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


_reader_cache: Optional["Reader"] = None
_reader_attempted: bool = False


def _get_reader() -> Optional["Reader"]:
    global _reader_cache, _reader_attempted
    if _reader_attempted:
        return _reader_cache
    _reader_attempted = True
    if Reader is None:
        print("[warn] deepxiv_sdk not installed; DeepXiv enrichment disabled", file=sys.stderr)
        return None
    token = _load_deepxiv_token()
    if not token:
        print("[warn] DEEPXIV_TOKEN not set (env or .env); DeepXiv enrichment disabled", file=sys.stderr)
        return None
    _reader_cache = Reader(token=token)
    return _reader_cache


def fetch_deepxiv_brief(arxiv_id: str) -> dict:
    """Fetch DeepXiv brief: citation count, tldr, keywords, publish_at.

    Replaces direct Semantic Scholar calls — no S2 API key needed.
    """
    reader = _get_reader()
    if reader is None:
        return {"found": False, "error": "no reader"}
    try:
        brief = reader.brief(arxiv_id)
    except Exception as e:
        return {"found": False, "error": f"{type(e).__name__}: {e}"}
    if not isinstance(brief, dict):
        return {"found": False, "error": f"unexpected response type {type(brief).__name__}"}
    return {
        "found": True,
        "title": brief.get("title") or "",
        "tldr": brief.get("tldr") or "",
        "keywords": brief.get("keywords") or [],
        "citation_count": brief.get("citations", 0),
        "publish_at": brief.get("publish_at") or "",
        "src_url": brief.get("src_url") or "",
    }


def fetch_deepxiv_social(arxiv_id: str) -> dict:
    """Fetch DeepXiv social-media impact metrics (trending signal)."""
    reader = _get_reader()
    if reader is None:
        return {"found": False, "error": "no reader"}
    try:
        s = reader.social_impact(arxiv_id)
    except Exception as e:
        return {"found": False, "error": f"{type(e).__name__}: {e}"}
    if not s:
        return {"found": False}
    return {
        "found": True,
        "total_tweets": s.get("total_tweets", 0),
        "total_likes": s.get("total_likes", 0),
        "total_views": s.get("total_views", 0),
        "total_replies": s.get("total_replies", 0),
        "first_seen_date": s.get("first_seen_date"),
        "last_seen_date": s.get("last_seen_date"),
    }


def _extract_hit_authors(info: dict) -> list[str]:
    a = (info.get("authors") or {}).get("author") or []
    if isinstance(a, dict):
        a = [a]
    out = []
    for item in a:
        if isinstance(item, dict):
            out.append(item.get("text", "") or "")
        else:
            out.append(str(item))
    return [re.sub(r"\s+\d+$", "", s).strip() for s in out if s]


def _dblp_hit_to_dict(info: dict) -> dict:
    return {
        "title": (info.get("title") or "").strip(". "),
        "venue": info.get("venue", "") or "",
        "year": info.get("year", "") or "",
        "type": info.get("type", "") or "",  # "Conference and Workshop Papers" | "Journal Articles" | "Informal and Other Publications"
        "key": info.get("key", "") or "",
        "authors": _extract_hit_authors(info),
    }


def fetch_dblp(title: str, first_author_surname: Optional[str] = None, top_k: int = 5) -> dict:
    """Search DBLP by title (optionally scoped by first author's surname).

    Returns top-k hits and a chosen 'best' hit that excludes obvious false matches
    (wrong author, or 'Informal and Other Publications' = arxiv mirror).
    """
    query = title
    if first_author_surname:
        query = f"{title} {first_author_surname}"
    q = urllib.parse.quote_plus(query)
    url = f"https://dblp.org/search/publ/api?q={q}&format=json&h={top_k}"
    data = _http_get_json(url)
    if not data:
        return {"found": False}
    try:
        hits = data.get("result", {}).get("hits", {}).get("hit") or []
    except (KeyError, AttributeError):
        hits = []
    if not hits:
        return {"found": False}
    hit_dicts = [_dblp_hit_to_dict((h.get("info") or {})) for h in hits]

    best = None
    author_lc = (first_author_surname or "").lower()
    title_keywords = {w for w in _normalize(title).split() if len(w) > 3} if title else set()

    def title_similar(candidate: str) -> bool:
        if not title:
            return True
        a, b = _normalize(title), _normalize(candidate)
        if not a or not b:
            return False
        # substring or high Jaccard on word bigrams
        return a in b or b in a or (
            len(set(a.split()) & set(b.split())) >= max(3, len(a.split()) // 2)
        )

    # Prefer non-arxiv (peer-reviewed) hits matching author + title
    for h in hit_dicts:
        if "Informal" in h["type"]:
            continue
        if author_lc:
            if not any(author_lc in a.lower() for a in h["authors"]):
                continue
        if not title_similar(h["title"]):
            continue
        best = h
        break

    # Relax: drop author constraint
    if best is None:
        for h in hit_dicts:
            if "Informal" in h["type"]:
                continue
            if not title_similar(h["title"]):
                continue
            best = h
            break

    return {
        "found": True,
        "hits": hit_dicts[:top_k],
        "best": best,  # may be None if no peer-reviewed hit with matching title
        "best_is_published": bool(best and "Informal" not in best["type"]),
    }


# ----------------------------- CCF lookup -----------------------------


_ccf_cache: Optional[list[dict]] = None


def _load_ccf() -> list[dict]:
    global _ccf_cache
    if _ccf_cache is not None:
        return _ccf_cache
    if not CCF_LIST_PATH.is_file():
        _ccf_cache = []
        return _ccf_cache
    _ccf_cache = json.loads(CCF_LIST_PATH.read_text(encoding="utf-8"))
    return _ccf_cache


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def ccf_lookup(venue_name: str, venue_type_hint: Optional[str] = None) -> dict:
    """Match a venue string against the CCF list. Returns a dict with rank/field/matched_abbr or {"rank": None}."""
    if not venue_name:
        return {"rank": None}
    ccf = _load_ccf()
    target = _normalize(venue_name)
    # Exact abbr or alias match
    for entry in ccf:
        candidates = [entry.get("abbr", "")] + list(entry.get("aliases") or [])
        if any(_normalize(c) == target for c in candidates if c):
            return _ccf_result(entry)
    # Exact name match
    for entry in ccf:
        if _normalize(entry.get("name", "")) == target:
            return _ccf_result(entry)
    # Substring match — abbr/alias appears inside the venue string
    best = None
    for entry in ccf:
        candidates = [entry.get("abbr", "")] + list(entry.get("aliases") or [])
        for c in candidates:
            if c and len(c) >= 3 and _normalize(c) in target:
                # Prefer longer matches
                if best is None or len(c) > len(best[1]):
                    best = (entry, c)
    if best:
        return _ccf_result(best[0])
    return {"rank": None}


def _ccf_result(entry: dict) -> dict:
    return {
        "rank": entry.get("rank"),
        "matched_abbr": entry.get("abbr"),
        "matched_name": entry.get("name"),
        "matched_type": entry.get("type"),
        "field": entry.get("field"),
        "note": entry.get("note"),
    }


def detect_github(text: str) -> Optional[str]:
    m = GITHUB_RE.search(text or "")
    return m.group(0) if m else None


# ----------------------------- enrichment -----------------------------


def enrich(
    arxiv_id: Optional[str],
    title: Optional[str],
    abstract_hint: Optional[str] = None,
    first_author_surname: Optional[str] = None,
    csl_idx: Optional[int] = None,
    include_social: bool = False,
) -> dict:
    out: dict[str, Any] = {
        "input": {"arxiv_id": arxiv_id, "title": title, "csl_idx": csl_idx}
    }

    arxiv = fetch_arxiv(arxiv_id) if arxiv_id else {"found": False}
    out["arxiv"] = arxiv

    resolved_title = (arxiv.get("title") if arxiv.get("found") else None) or title or ""

    deepxiv_brief = fetch_deepxiv_brief(arxiv_id) if arxiv_id else {"found": False}
    out["deepxiv_brief"] = deepxiv_brief

    if include_social and arxiv_id:
        out["deepxiv_social"] = fetch_deepxiv_social(arxiv_id)

    dblp = fetch_dblp(resolved_title, first_author_surname=first_author_surname) if resolved_title else {"found": False}
    out["dblp"] = dblp

    # Pick canonical venue: DBLP is the only source that gives a clean venue abbr.
    venue = ""
    venue_source = ""
    if dblp.get("found") and dblp.get("best") and dblp["best"].get("venue"):
        venue = dblp["best"]["venue"]
        venue_source = "dblp"

    out["resolved_venue"] = venue
    out["venue_source"] = venue_source
    out["is_published"] = bool(venue)
    out["ccf"] = ccf_lookup(venue) if venue else {"rank": None}

    # GitHub detection across arxiv abstract, CSL abstract hint, DeepXiv tldr
    blob = " ".join(
        [
            arxiv.get("abstract") or "",
            abstract_hint or "",
            deepxiv_brief.get("tldr") or "",
        ]
    )
    out["github_link"] = detect_github(blob)

    # Terse summary string
    parts: list[str] = []
    if venue:
        rank = out["ccf"].get("rank")
        parts.append(f"venue={venue}" + (f" (CCF-{rank})" if rank else " (not in CCF list)"))
    if deepxiv_brief.get("found"):
        cites = deepxiv_brief.get("citation_count")
        if cites is not None:
            parts.append(f"cites={cites}")
    if arxiv.get("comment"):
        parts.append(f"arxiv_comment={arxiv['comment'][:80]}")
    if out["github_link"]:
        parts.append(f"github={out['github_link']}")
    if include_social and out.get("deepxiv_social", {}).get("found"):
        s = out["deepxiv_social"]
        parts.append(f"tweets={s.get('total_tweets')}/views={s.get('total_views')}")
    out["summary"] = " | ".join(parts) or "no public metadata found"

    return out


# ----------------------------- CSL parsing -----------------------------


def _arxiv_id_from_csl(item: dict) -> Optional[str]:
    for field in ("URL", "note", "extra", "container-title"):
        value = item.get(field, "") or ""
        if isinstance(value, list):
            value = " ".join(str(x) for x in value)
        m = ARXIV_ID_RE.search(str(value))
        if m:
            return m.group(1) or m.group(3)
    return None


def _csl_context(item: dict) -> dict:
    """Compact CSL-derived context to feed Sonnet."""
    ym = ((item.get("issued") or {}).get("date-parts") or [[None]])[0]
    year = ym[0] if ym else None
    return {
        "title_csl": item.get("title") or "",
        "venue_csl": item.get("container-title") or "",
        "year_csl": year,
        "authors_csl": [
            {"given": a.get("given"), "family": a.get("family")}
            for a in (item.get("author") or [])[:5]
        ],
        "doi_csl": item.get("DOI") or "",
        "keyword_csl": item.get("keyword") or "",
        "abstract_csl": (item.get("abstract") or "")[:800],
    }


# ----------------------------- main -----------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Fresh paper metadata lookup (DBLP + Semantic Scholar + arxiv + CCF)")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--arxiv", help="single arxiv ID")
    src.add_argument("--title", help="single title (skips arxiv)")
    src.add_argument("--csl", help="CSL JSON path; emits JSONL to stdout (one enriched record per line)")
    parser.add_argument("--sleep", type=float, default=1.0, help="seconds between papers in batch mode (default 1.0)")
    parser.add_argument("--pretty", action="store_true", help="pretty-print single-item output")
    parser.add_argument("--limit", type=int, default=0, help="cap items processed in --csl mode (0 = all)")
    parser.add_argument("--author", help="first author's surname hint (helps DBLP disambiguation)")
    parser.add_argument("--include-social", action="store_true", help="also fetch DeepXiv social_impact per paper (tweets/likes/views)")
    args = parser.parse_args(argv[1:])

    if args.arxiv or args.title:
        result = enrich(args.arxiv, args.title, first_author_surname=args.author, include_social=args.include_social)
        print(json.dumps(result, indent=2 if args.pretty else None, ensure_ascii=False))
        return 0

    csl_path = Path(args.csl).expanduser()
    if not csl_path.is_file():
        print(f"not a file: {csl_path}", file=sys.stderr)
        return 1
    try:
        items = json.loads(csl_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"CSL JSON parse error: {e}", file=sys.stderr)
        return 1
    if not isinstance(items, list):
        print("CSL JSON must be a top-level array", file=sys.stderr)
        return 1

    if args.limit:
        items = items[: args.limit]
    total = len(items)
    for idx, item in enumerate(items):
        arxiv_id = _arxiv_id_from_csl(item)
        title = item.get("title") or ""
        abstract = (item.get("abstract") or "")[:2000]
        authors = item.get("author") or []
        surname = (authors[0].get("family") if authors else None) or None
        print(f"[{idx+1}/{total}] {title[:60]} (arxiv={arxiv_id})", file=sys.stderr)
        result = enrich(arxiv_id, title, abstract_hint=abstract, first_author_surname=surname, csl_idx=idx, include_social=args.include_social)
        result["csl"] = _csl_context(item)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if idx < total - 1 and args.sleep > 0:
            time.sleep(args.sleep)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
