---
name: digest-chat
description: Digest the current Claude Code conversation into wiki-bound actions. Use when the user says "digest", "digest this chat", "消化对话", "把聊的内容入库", or "/digest-chat". Scans the session transcript since the last digest, extracts candidate papers/URLs/concepts/positions/questions, and routes each approved item to the appropriate existing skill (/save-paper, concept creation, /promote-notes, ADD-QUESTION).
---

# DIGEST-CHAT

Turn a free-form conversation (reading a paper, discussing a blog, debating a technique) into structured wiki contributions. Cross-cutting rules live in `CLAUDE.md`.

This skill is a **dispatcher**, not a new ingest pipeline — every concrete write hits an existing operation (`/save-paper`, concept creation, `/promote-notes`, ADD-QUESTION). Its value is scoping + classification + relationship capture.

## Scope

**Scan range**: messages in the **current session** (identified via `.claude/sessions/session-<pid>.json` registry + process-tree walk) with `timestamp > last_digest_ts` (per-session cursor in `.claude/digest-state.json`).

Concurrent-session correctness: the SessionStart `session-start-register.sh` hook writes a per-Claude-PID registry entry; `digest_scan.py` walks up from its own PID to find which Claude Code instance spawned it, then uses that session's transcript. Running `/digest-chat` in window A never looks at window B's messages.

If no state exists for this session, treat as "since session start" and warn the user before proceeding.

## Workflow

### 1. Scan the conversation

```
python scripts/digest_scan.py scan
```

The scanner auto-resolves the current session's transcript. If you want to sanity-check which transcript it would pick, run `python scripts/digest_scan.py resolve` first — it prints the resolved path and a note (e.g., `matched ancestor pid=12345` vs `FALLBACK (latest-mtime, unsafe …)`). If you see the FALLBACK note, stop and investigate — the SessionStart register hook did not run.

This returns JSON:
- `arxiv` — arxiv IDs with turn snippet
- `dois` — DOIs with snippet
- `urls` — non-arxiv URLs with snippet
- `transcript` — filtered body of user + assistant text messages (tool calls + system reminders stripped) since the cutoff

If `message_count == 0`, tell the user nothing to digest, exit.

### 2. Classify candidates (four buckets)

Read the transcript body and produce a **proposal table** with four categories. For each item, cite the turn(s) where it came up so the user can verify the extraction is faithful.

**(a) Sources** — papers / blogs / articles the user showed interest in saving:
- All `arxiv` / `dois` hits → default include
- URLs whose domains look like papers/blogs (arxiv.org, openreview.net, acl anthology, neurips.cc, *.github.io, personal sites, Substack, Medium, lilianweng, karpathy.github.io, etc.) → include
- URLs that are clearly tooling / docs (github repo READMEs for dependencies, Stack Overflow, MDN) → exclude unless user explicitly said "save this"
- For each: propose slug, infer type (paper/blog), note one-line context ("came up when we discussed X")

**(b) Concepts** — technical terms that got substantive discussion:
- Named ≥ 2 times AND were the subject (not just a passing mention)
- Propose: create new `wiki/concepts/<slug>.md` OR append to existing concept's `## Key Points` / `## Contradictions`
- Glob existing `wiki/concepts/*.md` before proposing "new" — check aliases too

**(c) Positions** — first-person judgments the **user** expressed:
- "I think X is better because...", "我认为...", "I prefer A over B", "the cleaner approach is..."
- Only user turns, not assistant-generated. Assistant recommendations are not positions until user endorses.
- Route to a concept page's `## My Position` via `/promote-notes` flow (interactive confirm)

**(d) Open questions** — genuine unresolved questions:
- User asked something substantive that did NOT get a clean answer in later turns
- Exclude rhetorical questions and questions already answered in-chat
- Route to ADD-QUESTION (append to `wiki/QUESTIONS.md`)

### 3. Capture relationships (if any)

While classifying, check for **explicit relationship statements** the user made between two sources (e.g., "paper A is an improved version of B", "A and B are alternatives under different constraints"). If found, propose adding to the source page's `## Related Work` section with typed wikilink (`extends :: [[b-slug]]`, `alternative-to :: [[b-slug]]`, `supersedes :: [[b-slug]]`) — per the agreed Related Work convention.

If the `## Related Work` section does not yet exist in `wiki/templates/source-template.md`, note this and route to a separate follow-up rather than silently inventing the section.

### 4. Present for approval

Render as a single compact table. Example:

```
Proposed digest actions:

┌─── Sources (3) ───────────────────────────────────────────────┐
│ [1] arxiv:2401.12345  — "Paper X"        action: /save-paper │
│ [2] arxiv:2309.67890  — "Paper Y"        action: /save-paper │
│     └─ related to [1]: alternative-to (user said so @ 10:30) │
│ [3] lilianweng.github.io/posts/...       action: /save-paper │
├─── Concepts (2) ──────────────────────────────────────────────┤
│ [4] sparse-attention     action: new concept page            │
│ [5] kv-cache-compression action: append to existing concept  │
├─── Positions (1) ─────────────────────────────────────────────┤
│ [6] "X > Y for long-context"  → concept:sparse-attention     │
├─── Questions (1) ─────────────────────────────────────────────┤
│ [7] "何时 sparse attention 会退化回 full?"                    │
└───────────────────────────────────────────────────────────────┘

Approve: all / none / [1,2,4,6] / edit [n] / skip [n]
```

Accept natural-language responses — default to conservative interpretation (ask back if ambiguous).

### 5. Fan-out to existing skills

For each approved item, route to the correct handler (sequentially, to avoid concurrent wiki writes):

| Category | Handler |
|---|---|
| Source (arxiv/DOI/URL) | Invoke `/save-paper <id-or-url>` — chains into `/ingest` |
| Concept (new) | Create `wiki/concepts/<slug>.md` from `wiki/templates/concept-template.md` with a minimal stub + seed content from the conversation |
| Concept (append) | Edit existing concept's `## Key Points` / `## Contradictions` with new bullets |
| Position | Invoke `/promote-notes <concept-slug>` flow, seeding candidate claims from the conversation |
| Question | Append to `wiki/QUESTIONS.md` `## Open Questions` (inline ADD-QUESTION per CLAUDE.md) |
| Source relationship | After both sources are saved, edit both source pages' `## Related Work` |

Print a one-line progress indicator per item.

### 6. Update state + log

```
python scripts/digest_scan.py mark-done
python scripts/wiki_ops.py log-append digest-chat "<N> sources, <N> concepts, <N> positions, <N> questions"
```

If the user rejected everything, still call `mark-done` (they already reviewed — no point re-scanning the same range next time). If the user cancelled mid-flow, do NOT mark-done — let the next digest see the same candidates again.

## Guardrails

- **Interactive by default**. Never auto-ingest without explicit approval, even for high-confidence arxiv IDs.
- **Faithful extraction**. Every proposed item must cite the conversation turn(s) it came from. If you cannot point to a specific turn, do not propose it.
- **No fabrication**. If the user mentioned a paper by name without giving arxiv/URL/DOI, list it under "needs identifier" and ask — do not guess the ID.
- **One digest per invocation**. Don't loop. If the user wants another pass on the same range, they can say so explicitly.
