#!/usr/bin/env python3
"""Merge resolved arxiv_ids back into decisions, emit supplementary jsonl.

Inputs:
  --resolved /tmp/mywiki-batch/resolved.jsonl  (from resolve_titles.py)
  --decisions /tmp/mywiki-batch/decisions.jsonl  (original)

Outputs:
  --partial-out  /tmp/mywiki-batch/supplement_partial_decisions.jsonl
     (only partial-no-md items with a newly resolved arxiv_id; action=include, arxiv_id set)
  --uncertain-ids  /tmp/mywiki-batch/supplement_uncertain_arxiv_ids.txt
     (newline-separated arxiv_ids for items to re-enrich + re-classify)
  --uncertain-map  /tmp/mywiki-batch/supplement_uncertain_map.json
     (arxiv_id -> csl_idx so we can re-join after classification)
  --stats-out  /tmp/mywiki-batch/supplement_stats.json

Also prints stats to stdout.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch_scaffold import make_slug  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolved", required=True)
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--partial-out", required=True)
    ap.add_argument("--uncertain-ids", required=True)
    ap.add_argument("--uncertain-map", required=True)
    ap.add_argument("--stats-out", required=True)
    args = ap.parse_args()

    # Key resolved by (kind, csl_idx) → arxiv_id.
    resolved: dict[tuple[str, int], dict] = {}
    with open(args.resolved) as f:
        for line in f:
            r = json.loads(line)
            if r.get("arxiv_id"):
                resolved[(r["kind"], r["csl_idx"])] = r

    decisions = [json.loads(l) for l in open(args.decisions)]

    partial_patched = []
    uncertain_patched = []
    unresolved_partial = []
    unresolved_uncertain = []

    for d in decisions:
        idx = d.get("csl_idx")
        action = d.get("action")
        if action == "include" and not d.get("arxiv_id"):
            key = ("partial", idx)
            if key in resolved:
                r = resolved[key]
                new = dict(d)
                new["arxiv_id"] = r["arxiv_id"]
                # Preserve original slug (computed without arxiv_id) so we update the existing folder.
                new["slug"] = make_slug(new.get("csl_item") or {}, None)
                new["_resolved_from_title"] = True
                new["_matched_title"] = r.get("matched_title")
                new["_resolve_score"] = r.get("score")
                partial_patched.append(new)
            else:
                unresolved_partial.append(d)
        elif action == "uncertain":
            key = ("uncertain", idx)
            if key in resolved:
                r = resolved[key]
                new = dict(d)
                new["arxiv_id"] = r["arxiv_id"]
                new["_resolved_from_title"] = True
                new["_matched_title"] = r.get("matched_title")
                new["_resolve_score"] = r.get("score")
                uncertain_patched.append(new)
            else:
                unresolved_uncertain.append(d)

    # Write partial-out
    with open(args.partial_out, "w") as f:
        for d in partial_patched:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Write uncertain ids and map
    with open(args.uncertain_ids, "w") as f:
        for d in uncertain_patched:
            f.write(d["arxiv_id"] + "\n")
    umap = {d["arxiv_id"]: d["csl_idx"] for d in uncertain_patched}
    with open(args.uncertain_map, "w") as f:
        json.dump(umap, f, indent=2)

    stats = {
        "partial_total": sum(1 for d in decisions if d.get("action") == "include" and not d.get("arxiv_id")),
        "partial_resolved": len(partial_patched),
        "partial_still_unresolved": len(unresolved_partial),
        "uncertain_total": sum(1 for d in decisions if d.get("action") == "uncertain"),
        "uncertain_resolved": len(uncertain_patched),
        "uncertain_still_unresolved": len(unresolved_uncertain),
    }
    with open(args.stats_out, "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
