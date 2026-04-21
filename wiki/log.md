---
type: system-log
graph-excluded: true
date: 2026-04-16
---

# Wiki Operation Log

Append-only chronological record. Format: `YYYY-MM-DD HH:MM | <op> | <target>`.

Operations: `ingest`, `query`, `lint`, `reflect`, `merge`, `add-question`, `resolve-question`, `add-note`, `config`.

Use plain paths (e.g., `wiki/sources/foo.md`), never wikilinks.

---

2026-04-16 22:25 | config | wiki initialized per plan peppy-swimming-pony
2026-04-17 00:00 | ingest | Attention Is All You Need
2026-04-21 21:34 | batch-ingest | 550 sources, 2141 concepts, 4278 entities promoted from staging to wiki/
2026-04-21 22:08 | lint-fix | Check1:0 Check2:0 Check8:0 Check11:0; merged 22 duplicate concepts; created 35 stubs; fixed lint.py Path.stem bug for versioned slugs
2026-04-22 02:43 | lint-llm | Phase2: L6 27 merged + 47 rejected; L3 no issues; L5 110 hubs flagged; L1 3 concepts sampled (no contradictions); Check 5 167→92, Check 7 90→78
2026-04-22 02:59 | lint | Check 12 added (empty/missing required sections); 1 finding auto-fixed (mistral-7b-instruct-v0 empty ## Sources)
