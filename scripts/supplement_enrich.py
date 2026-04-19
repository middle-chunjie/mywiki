#!/usr/bin/env python3
"""Enrich a list of arxiv_ids by calling paper_metadata.enrich()."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_metadata import enrich  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="path to newline-separated arxiv_ids")
    ap.add_argument("--out", required=True, help="output JSONL")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--map", required=True, help="JSON map arxiv_id -> csl_idx to preserve original csl_idx")
    args = ap.parse_args()

    ids = [l.strip() for l in open(args.ids) if l.strip()]
    arxiv_to_idx = json.load(open(args.map))

    with open(args.out, "w") as out:
        for i, aid in enumerate(ids, 1):
            try:
                rec = enrich(arxiv_id=aid, title=None, csl_idx=arxiv_to_idx.get(aid), include_social=False)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                summ = rec.get("summary", "")[:80]
                print(f"[{i}/{len(ids)}] {aid}  {summ}", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[{i}/{len(ids)}] {aid}  ERROR: {e}", file=sys.stderr, flush=True)
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
