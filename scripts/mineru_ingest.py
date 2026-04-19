#!/usr/bin/env python3
"""Parse raw/papers/<slug>/paper.pdf via the MinerU cloud API.

Usage:
    python scripts/mineru_ingest.py <slug> [--model vlm|pipeline]

Reads the PDF at raw/papers/<slug>/paper.pdf, uploads it to MinerU, polls until
parsing completes, downloads the result ZIP, and extracts the contents in-place
so that raw/papers/<slug>/ ends up with paper.md, images/, content_list.json,
and any other artefacts produced by MinerU.

Requires:
    MINERU_API_TOKEN set in the environment, or present in MyWiki/.env.
    Python packages: requests.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import requests

WIKI_ROOT = Path(__file__).resolve().parent.parent
RAW_PAPERS = WIKI_ROOT / "raw" / "papers"
API_BASE = "https://mineru.net/api/v4"

TERMINAL_STATES = {"done", "failed"}


def load_token() -> str:
    token = os.environ.get("MINERU_API_TOKEN")
    if token:
        return token
    env_file = WIKI_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("MINERU_API_TOKEN="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("MINERU_API_TOKEN not found. Set it in your environment or in MyWiki/.env.")


def request_upload_slot(session: requests.Session, slug: str, model: str) -> tuple[str, str]:
    url = f"{API_BASE}/file-urls/batch"
    payload = {
        "files": [{"name": "paper.pdf", "data_id": slug}],
        "model_version": model,
        "enable_formula": True,
        "enable_table": True,
    }
    r = session.post(url, json=payload, timeout=30)
    r.raise_for_status()
    body = r.json()
    if body.get("code") != 0:
        raise SystemExit(f"MinerU API error at upload-slot request: {body.get('msg')}")
    data = body["data"]
    file_urls = data.get("file_urls") or []
    if not file_urls:
        raise SystemExit(f"MinerU returned no file_urls: {json.dumps(body)}")
    return data["batch_id"], file_urls[0]


def upload_pdf(upload_url: str, pdf_path: Path) -> None:
    # Note: MinerU docs explicitly say "do not set Content-Type" on the PUT.
    with open(pdf_path, "rb") as f:
        r = requests.put(upload_url, data=f, timeout=300)
    r.raise_for_status()


def poll_batch(
    session: requests.Session,
    batch_id: str,
    interval: float = 5.0,
    timeout: float = 1800.0,
) -> dict[str, Any]:
    url = f"{API_BASE}/extract-results/batch/{batch_id}"
    deadline = time.time() + timeout
    last_state: str | None = None
    while time.time() < deadline:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        body = r.json()
        if body.get("code") != 0:
            raise SystemExit(f"MinerU poll error: {body.get('msg')}")
        results = body.get("data", {}).get("extract_result") or []
        if not results:
            time.sleep(interval)
            continue
        result = results[0]
        state = result.get("state", "unknown")
        if state != last_state:
            progress = result.get("extract_progress") or {}
            extra = ""
            if progress:
                extra = f" ({progress.get('extracted_pages')}/{progress.get('total_pages')} pages)"
            print(f"[mineru] state: {state}{extra}")
            last_state = state
        if state in TERMINAL_STATES:
            if state == "failed":
                raise SystemExit(f"MinerU parsing failed: {result.get('err_msg', '<no err_msg>')}")
            return result
        time.sleep(interval)
    raise SystemExit(f"MinerU parsing timed out after {timeout:.0f}s")


def download_and_extract(zip_url: str, target_dir: Path) -> list[Path]:
    print(f"[mineru] downloading result ZIP from {zip_url}")
    r = requests.get(zip_url, timeout=600)
    r.raise_for_status()
    written: list[Path] = []
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            # Strip any top-level folder prefix so files land directly in target_dir.
            parts = Path(name).parts
            rel = Path(*parts[1:]) if len(parts) > 1 else Path(parts[0])
            dest = target_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(dest, "wb") as out:
                out.write(src.read())
            written.append(dest)
    return written


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"}


def normalize_filenames(target_dir: Path) -> None:
    # Rename full.md -> paper.md.
    full_md = target_dir / "full.md"
    paper_md = target_dir / "paper.md"
    if full_md.exists() and not paper_md.exists():
        full_md.rename(paper_md)
        print("[rename] full.md -> paper.md")

    # MinerU's ZIP scatters extracted figures flat at the root while paper.md
    # references them as `images/<hash>.jpg`. Move them into images/ so links resolve.
    images_dir = target_dir / "images"
    images_dir.mkdir(exist_ok=True)
    moved = 0
    for child in list(target_dir.iterdir()):
        if child.is_file() and child.suffix.lower() in IMAGE_EXTS:
            if child.name == "paper.pdf":
                continue  # defensive — .pdf isn't in IMAGE_EXTS anyway
            dest = images_dir / child.name
            if dest.exists():
                continue
            child.rename(dest)
            moved += 1
    if moved:
        print(f"[move] {moved} image(s) into images/")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Parse raw/papers/<slug>/paper.pdf via MinerU")
    parser.add_argument("slug", help="paper slug (raw/papers/<slug>/paper.pdf must exist)")
    parser.add_argument(
        "--model",
        choices=["vlm", "pipeline", "MinerU-HTML"],
        default="vlm",
        help="parsing backend (default: vlm)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="seconds between polls (default: 5.0)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="overall timeout in seconds (default: 1800)",
    )
    args = parser.parse_args(argv[1:])

    target_dir = RAW_PAPERS / args.slug
    pdf_path = target_dir / "paper.pdf"
    if not pdf_path.is_file():
        print(f"Missing {pdf_path}", file=sys.stderr)
        return 1

    if (target_dir / "paper.md").is_file():
        print(f"[skip] {target_dir / 'paper.md'} already exists (re-run would overwrite).")
        print("Delete paper.md to force re-parsing.")
        return 0

    token = load_token()
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    session.headers["Content-Type"] = "application/json"

    print(f"[mineru] requesting upload slot (slug={args.slug}, model={args.model})")
    batch_id, upload_url = request_upload_slot(session, args.slug, args.model)
    print(f"[mineru] batch_id = {batch_id}")

    print(f"[mineru] uploading {pdf_path} ({pdf_path.stat().st_size/1024/1024:.2f} MB)")
    upload_pdf(upload_url, pdf_path)

    print("[mineru] waiting for parsing to complete")
    result = poll_batch(
        session,
        batch_id,
        interval=args.poll_interval,
        timeout=args.timeout,
    )

    zip_url = result.get("full_zip_url")
    if not zip_url:
        print(json.dumps(result, indent=2), file=sys.stderr)
        raise SystemExit("No full_zip_url in successful result")

    written = download_and_extract(zip_url, target_dir)
    normalize_filenames(target_dir)
    print(f"[done] wrote {len(written)} files into {target_dir}")
    md_path = target_dir / "paper.md"
    if md_path.is_file():
        print(f"[done] paper.md ready at {md_path}")
    else:
        print(f"[warn] expected {md_path} but no file named paper.md was produced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
