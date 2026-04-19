#!/usr/bin/env python3
"""Resolve titles -> arxiv_ids via arxiv Search API with fuzzy match.

Input JSONL: {"key": str, "title": str, "doi": str|null, "url": str|null}
Output JSONL: {"key", "input_title", "cleaned", "arxiv_id"|null, "matched_title", "score", "error"?}
"""
from __future__ import annotations
import argparse, json, re, socket, sys, time, urllib.parse, urllib.request, urllib.error
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher

ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom"}

# Build a no-proxy opener so macOS system proxy (Clash etc) does not stall requests.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def clean_title(t: str) -> str:
    if not t:
        return ""
    t = t.strip()
    t = re.sub(r"\.pdf\s*$", "", t, flags=re.I)
    t = t.replace("：", ":").replace("，", ",").replace("（", "(").replace("）", ")")
    t = t.replace("_", " ")
    for _ in range(3):
        t = re.sub(r"^\[[^\]]{1,10}\]\s*", "", t)
    t = re.sub(r"^\d+_", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _phrase(title: str) -> str:
    s = re.sub(r"[^A-Za-z0-9\s\-]", " ", title)
    s = re.sub(r"\s+", " ", s).strip()
    words = s.split()
    return " ".join(words[:16])


def query_arxiv(title: str, max_results: int = 5, retries: int = 2):
    phrase = _phrase(title)
    if not phrase:
        return [], "empty query"
    # Single token → unquoted ti: search; multi-word → quoted phrase.
    if len(phrase.split()) == 1:
        q = f"ti:{phrase}"
    else:
        q = f'ti:"{phrase}"'
    url = f"{ARXIV_API}?search_query={urllib.parse.quote(q)}&max_results={max_results}"
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MyWiki-resolver/1.0"})
            with _OPENER.open(req, timeout=10) as resp:
                data = resp.read()
            root = ET.fromstring(data)
            out = []
            for e in root.findall("a:entry", ATOM):
                aid_url = (e.find("a:id", ATOM).text or "").strip()
                m = re.search(r"abs/(.+?)(v\d+)?$", aid_url)
                aid = m.group(1) if m else None
                etitle = re.sub(r"\s+", " ", (e.find("a:title", ATOM).text or "").strip())
                out.append({"arxiv_id": aid, "title": etitle})
            return out, None
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}"
            # Arxiv 429 rate-limit: back off aggressively.
            if e.code == 429:
                backoff = 60 * (attempt + 1)
                print(f"  [rate-limit 429, sleeping {backoff}s]", file=sys.stderr, flush=True)
                time.sleep(backoff)
            else:
                time.sleep(5 + attempt * 5)
        except (urllib.error.URLError, ET.ParseError,
                TimeoutError, socket.timeout, ConnectionError, OSError) as e:
            last_err = str(e)
            time.sleep(3 + attempt * 3)
    return [], last_err


_STOP = {"a","an","the","of","for","on","in","to","and","with","via","from","by","at","is","are","be","our","we","this"}


def _jaccard(a: str, b: str) -> float:
    qw = {w for w in a.split() if len(w) > 2 and w not in _STOP}
    tw = {w for w in b.split() if len(w) > 2 and w not in _STOP}
    if not qw or not tw:
        return 0.0
    return len(qw & tw) / len(qw | tw)


def _is_acronym_like(raw_first: str) -> bool:
    # All-caps (≥4) or mixedcase with digit (e.g. "GPT4", "B-Coder") or long unusual token.
    if not raw_first or len(raw_first) < 4:
        return False
    stripped = re.sub(r"[^A-Za-z0-9]", "", raw_first)
    if not stripped:
        return False
    if stripped.isupper() and len(stripped) >= 4:
        return True
    if re.search(r"[A-Z]", stripped) and re.search(r"\d", stripped):
        return True
    # CamelCase with 2+ caps: CodeChain, MathCoder
    caps = sum(1 for c in stripped if c.isupper())
    if caps >= 2 and stripped[0].isupper():
        return True
    return False


def best_match(q: str, entries, q_raw_first: str = ""):
    nq = norm(q)
    qw = nq.split()
    q_first = qw[0] if qw else ""
    acronym_query = _is_acronym_like(q_raw_first)
    best, best_score = None, 0.0
    for e in entries:
        nt = norm(e["title"])
        if not nt:
            continue
        tw = nt.split()
        t_first = tw[0] if tw else ""
        # Acronym-lead match: both titles start with the same distinctive acronym token.
        acronym_lead = (
            acronym_query and q_first and t_first and q_first == t_first
            and len(q_first) >= 4 and q_first not in _STOP
        )
        if nq and (nq in nt or nt in nq):
            score = max(0.9, SequenceMatcher(None, nq, nt).ratio())
        else:
            sm = SequenceMatcher(None, nq, nt).ratio()
            jc = _jaccard(nq, nt)
            if acronym_lead:
                score = max(sm, 0.85 + 0.1 * jc)
            else:
                distinctive = q_first and q_first not in _STOP and q_first in tw
                score = max(sm, jc * (0.95 if distinctive else 0.8))
        if score > best_score:
            best, best_score = e, score
    return best, best_score


def resolve_one(item: dict, threshold: float, sleep: float = 1.2):
    title = item.get("title") or ""
    cleaned = clean_title(title)
    if not cleaned:
        return {"arxiv_id": None, "cleaned": cleaned, "score": 0, "matched_title": None, "error": "empty title"}
    # Skip obviously non-paper entries (notes, filenames, emoji-heavy).
    ascii_words = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", cleaned)
    if len(ascii_words) < 3:
        return {"arxiv_id": None, "cleaned": cleaned, "score": 0, "matched_title": None, "error": "too few ascii words"}
    non_ascii_ratio = sum(1 for c in cleaned if ord(c) > 127) / max(1, len(cleaned))
    if non_ascii_ratio > 0.3:
        return {"arxiv_id": None, "cleaned": cleaned, "score": 0, "matched_title": None, "error": "mostly non-ascii"}
    attempts = []
    # Use punctuation-stripped words for truncation so `Foo:Bar` → `Foo Bar`.
    words = _phrase(cleaned).split()
    # Primary: full phrase. Fallback: single distinctive leading token (if acronym-like).
    if words:
        attempts.append(" ".join(words[:16]))
        w0 = re.sub(r"[^A-Za-z0-9]", "", words[0])
        if _is_acronym_like(words[0]) and len(w0) >= 4 and w0.lower() not in _STOP:
            attempts.append(w0)
    # Dedupe order-preserving
    seen = set()
    uniq = [a for a in attempts if not (a in seen or seen.add(a))]

    best, best_score, last_err = None, 0.0, None
    raw_first = words[0] if words else ""
    for q in uniq:
        entries, err = query_arxiv(q)
        if err and not entries:
            last_err = err
            time.sleep(sleep)
            continue
        m, s = best_match(cleaned, entries, q_raw_first=raw_first)
        if m and s > best_score:
            best, best_score = m, s
        if best_score >= 0.92:
            break
        time.sleep(sleep)

    if best and best_score >= threshold:
        return {"arxiv_id": best["arxiv_id"], "cleaned": cleaned, "score": round(best_score, 3), "matched_title": best["title"]}
    return {"arxiv_id": None, "cleaned": cleaned, "score": round(best_score, 3), "matched_title": (best or {}).get("title"), "error": last_err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sleep", type=float, default=1.5)
    ap.add_argument("--threshold", type=float, default=0.72)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    mode = "a" if args.start > 0 else "w"
    seen = 0
    with open(args.input) as f, open(args.output, mode) as out:
        for i, line in enumerate(f):
            if i < args.start:
                continue
            if args.limit and seen >= args.limit:
                break
            item = json.loads(line)
            r = resolve_one(item, args.threshold, sleep=args.sleep)
            r["key"] = item.get("key")
            r["input_title"] = item.get("title")
            r["kind"] = item.get("kind")
            r["csl_idx"] = item.get("csl_idx")
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            out.flush()
            aid = r.get("arxiv_id") or "MISS"
            print(f"[{i+1}] {aid:>14}  score={r.get('score'):.2f}  {item.get('title','')[:70]}", file=sys.stderr, flush=True)
            seen += 1
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
