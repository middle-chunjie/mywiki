#!/usr/bin/env python3
"""Fetch PDF for a paper that lacks an arxiv mirror.

Tries, in order:
  1. CSL/bib URL if it ends in .pdf or is aclanthology.org (append .pdf if missing)
  2. Unpaywall (DOI → OA PDF URL; best-quality signal)
  3. OpenAlex (DOI → `open_access.oa_url`; falls through to `primary_location.pdf_url`)
  4. Semantic Scholar (DOI → `openAccessPdf`)

Modes:
  Single: python fetch_nonarxiv_pdf.py --doi 10.18653/v1/2022.acl-long.353 --out /tmp/p.pdf
  Batch:  python fetch_nonarxiv_pdf.py --decisions <jsonl> --out-dir raw/papers/

Each decisions entry: {"csl_idx", "arxiv_id", "csl_item", "slug"?}. Looks at CSL for DOI / URL / container / title.
"""
from __future__ import annotations
import argparse, json, re, sys, time, socket, urllib.parse, urllib.request, urllib.error
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMAIL = "misterarrow6c@gmail.com"
UA = "MyWiki-fetcher/1.0"
TIMEOUT = 15

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http_get(url: str, accept: str = "*/*") -> tuple[Optional[bytes], Optional[str]]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": accept})
    try:
        with _OPENER.open(req, timeout=TIMEOUT) as resp:
            return resp.read(), resp.headers.get("Content-Type", "")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, socket.timeout, OSError) as e:
        return None, f"ERROR: {e}"


def download_pdf(url: str, dest: Path) -> tuple[bool, str]:
    if not url:
        return False, "empty url"
    data, ctype = http_get(url, accept="application/pdf")
    if data is None:
        return False, ctype or "no data"
    # Detect PDF
    if not data.startswith(b"%PDF"):
        # Sometimes CDN redirects HTML; try parsing for pdf link later; bail here.
        return False, f"not a pdf (ctype={ctype}, first bytes={data[:8]!r})"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return True, "ok"


def clean_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    doi = doi.strip()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    doi = doi.strip("/")
    return doi or None


def try_direct_url(csl: dict) -> Optional[str]:
    url = (csl.get("URL") or "").strip()
    if not url:
        return None
    if url.endswith(".pdf"):
        return url
    # aclanthology.org/2022.acl-long.353 → append .pdf
    if "aclanthology.org" in url:
        return url.rstrip("/") + ".pdf"
    # openreview.net/forum?id=XYZ → pdf?id=XYZ
    if "openreview.net/forum" in url:
        return url.replace("openreview.net/forum", "openreview.net/pdf")
    return None


def try_unpaywall(doi: str, email: str) -> Optional[str]:
    api = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={urllib.parse.quote(email)}"
    data, _ = http_get(api, accept="application/json")
    if not data:
        return None
    try:
        j = json.loads(data)
    except Exception:
        return None
    best = j.get("best_oa_location") or {}
    pdf = best.get("url_for_pdf") or best.get("url")
    if pdf:
        return pdf
    # Check other locations
    for loc in (j.get("oa_locations") or []):
        if loc.get("url_for_pdf"):
            return loc["url_for_pdf"]
    return None


def try_openalex(doi: str) -> Optional[str]:
    api = f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi)}"
    data, _ = http_get(api, accept="application/json")
    if not data:
        return None
    try:
        j = json.loads(data)
    except Exception:
        return None
    oa = (j.get("open_access") or {}).get("oa_url")
    if oa and oa.endswith(".pdf"):
        return oa
    primary = j.get("primary_location") or {}
    if primary.get("pdf_url"):
        return primary["pdf_url"]
    for loc in (j.get("locations") or []):
        if loc.get("pdf_url"):
            return loc["pdf_url"]
    return oa


def try_semantic_scholar(doi: str) -> Optional[str]:
    api = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{urllib.parse.quote(doi)}?fields=openAccessPdf"
    data, _ = http_get(api, accept="application/json")
    if not data:
        return None
    try:
        j = json.loads(data)
    except Exception:
        return None
    oa = (j.get("openAccessPdf") or {}).get("url")
    return oa or None


def try_openalex_by_title(title: str) -> Optional[str]:
    if not title or len(title) < 10:
        return None
    q = urllib.parse.quote(title[:200])
    api = f"https://api.openalex.org/works?search={q}&per-page=3"
    data, _ = http_get(api, accept="application/json")
    if not data:
        return None
    try:
        j = json.loads(data)
    except Exception:
        return None
    for w in (j.get("results") or [])[:3]:
        # Match title loosely
        wt = (w.get("title") or "").lower()
        tt = title.lower()
        if len(set(wt.split()) & set(tt.split())) < max(2, len(tt.split()) // 3):
            continue
        oa = (w.get("open_access") or {}).get("oa_url")
        if oa:
            return oa
        primary = w.get("primary_location") or {}
        if primary.get("pdf_url"):
            return primary["pdf_url"]
        for loc in (w.get("locations") or []):
            if loc.get("pdf_url"):
                return loc["pdf_url"]
    return None


def try_s2_by_title(title: str) -> Optional[str]:
    if not title or len(title) < 10:
        return None
    q = urllib.parse.quote(title[:200])
    api = f"https://api.semanticscholar.org/graph/v1/paper/search?query={q}&limit=3&fields=title,openAccessPdf"
    data, _ = http_get(api, accept="application/json")
    if not data:
        return None
    try:
        j = json.loads(data)
    except Exception:
        return None
    for p in (j.get("data") or [])[:3]:
        pt = (p.get("title") or "").lower()
        tt = title.lower()
        if len(set(pt.split()) & set(tt.split())) < max(2, len(tt.split()) // 3):
            continue
        oa = (p.get("openAccessPdf") or {}).get("url")
        if oa:
            return oa
    return None


def fetch_one(csl: dict, dest: Path, email: str, verbose: bool = False) -> tuple[bool, str, str]:
    """Returns (ok, source, detail)."""
    # 1. Direct URL (PDF / ACL / OpenReview)
    direct = try_direct_url(csl)
    if direct:
        ok, msg = download_pdf(direct, dest)
        if ok:
            return True, "direct", direct
        if verbose:
            print(f"    direct({direct}) → {msg}", file=sys.stderr)

    doi = clean_doi(csl.get("DOI"))
    title = (csl.get("title") or "").strip()

    if not doi:
        # No DOI → try title search via OpenAlex then S2
        if title:
            url = try_openalex_by_title(title)
            if url:
                ok, msg = download_pdf(url, dest)
                if ok:
                    return True, "openalex-title", url
                if verbose:
                    print(f"    openalex-title({url}) → {msg}", file=sys.stderr)
            url = try_s2_by_title(title)
            if url:
                ok, msg = download_pdf(url, dest)
                if ok:
                    return True, "s2-title", url
                if verbose:
                    print(f"    s2-title({url}) → {msg}", file=sys.stderr)
        return False, "no-doi", "no DOI and no usable direct URL"

    # 2. Unpaywall
    url = try_unpaywall(doi, email)
    if url:
        ok, msg = download_pdf(url, dest)
        if ok:
            return True, "unpaywall", url
        if verbose:
            print(f"    unpaywall({url}) → {msg}", file=sys.stderr)

    # 3. OpenAlex
    url = try_openalex(doi)
    if url:
        ok, msg = download_pdf(url, dest)
        if ok:
            return True, "openalex", url
        if verbose:
            print(f"    openalex({url}) → {msg}", file=sys.stderr)

    # 4. Semantic Scholar
    url = try_semantic_scholar(doi)
    if url:
        ok, msg = download_pdf(url, dest)
        if ok:
            return True, "s2", url
        if verbose:
            print(f"    s2({url}) → {msg}", file=sys.stderr)

    return False, "paywalled", f"DOI {doi} has no OA copy reachable via Unpaywall/OpenAlex/S2"


def slug_from_decision(d: dict) -> str:
    if d.get("slug"):
        return d["slug"]
    # fallback to batch_scaffold.make_slug
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from batch_scaffold import make_slug
    return make_slug(d.get("csl_item") or {}, d.get("arxiv_id"))


def batch_main(args):
    email = args.email
    out_dir = Path(args.out_dir).resolve()
    report = []
    decisions = [json.loads(l) for l in open(args.decisions) if l.strip()]
    n_ok = n_fail = n_skip = 0
    for i, d in enumerate(decisions, 1):
        csl = d.get("csl_item") or {}
        slug = slug_from_decision(d)
        dest_dir = out_dir / slug
        pdf = dest_dir / "paper.pdf"
        title = (csl.get("title") or "?")[:70]
        if pdf.exists() and pdf.stat().st_size > 1000:
            print(f"[{i}/{len(decisions)}] SKIP (pdf exists) {slug}: {title}", file=sys.stderr)
            n_skip += 1
            report.append({"slug": slug, "title": csl.get("title"), "status": "skip-exists"})
            continue
        ok, src, detail = fetch_one(csl, pdf, email, verbose=args.verbose)
        tag = "OK" if ok else "FAIL"
        print(f"[{i}/{len(decisions)}] {tag:4} [{src}] {slug}: {title}", file=sys.stderr)
        report.append({"slug": slug, "title": csl.get("title"), "status": "ok" if ok else "fail", "source": src, "detail": detail})
        if ok:
            n_ok += 1
        else:
            n_fail += 1
        time.sleep(args.sleep)

    # Write report
    if args.report:
        with open(args.report, "w") as f:
            json.dump({"ok": n_ok, "fail": n_fail, "skip": n_skip, "items": report}, f, indent=2, ensure_ascii=False)
    print(f"\nsummary: ok={n_ok} fail={n_fail} skip={n_skip}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doi")
    ap.add_argument("--csl-json", help="path to single-item CSL JSON")
    ap.add_argument("--out")
    ap.add_argument("--decisions", help="JSONL of batch_scaffold-style decisions")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "raw" / "papers"))
    ap.add_argument("--email", default=DEFAULT_EMAIL)
    ap.add_argument("--report", help="path to write JSON summary (batch mode)")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.decisions:
        batch_main(args)
        return

    # Single-item mode
    csl = {}
    if args.csl_json:
        csl = json.load(open(args.csl_json))
    if args.doi:
        csl["DOI"] = args.doi
    if not args.out:
        ap.error("--out required in single-item mode")
    ok, src, detail = fetch_one(csl, Path(args.out), args.email, verbose=args.verbose)
    print(json.dumps({"ok": ok, "source": src, "detail": detail}, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
