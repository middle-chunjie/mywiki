---
name: create-project
description: Create a new research project with wiki background compilation. Use when the user says "create project", "新建项目", or "/create-project <slug> <direction>". Scaffolds projects/<slug>/, compiles wiki background into lit-review/BACKGROUND.md, and updates hot.md.
---

# CREATE-PROJECT

Scaffold a new research project and compile wiki background. Cross-cutting rules live in `CLAUDE.md`.

## Steps

1. **Parse input**: Extract slug (kebab-case) and direction (1-2 sentence research goal) from user input. Optionally extract `--domains` (comma-separated wiki domain tags to scope background reading).

2. **Scaffold**: Run:
   ```
   python scripts/project_ops.py create --slug <slug> --direction "<direction>" [--domains "<d1>,<d2>"]
   ```
   Verify the JSON output shows `"status": "created"`.

3. **Wiki read protocol** — compile `projects/<slug>/lit-review/BACKGROUND.md`:

   **3a. Match concepts**: Glob `wiki/concepts/*.md`. For each, read frontmatter (`tags`, `domain`, `aliases`) and match against `wiki_read_domains` and direction keywords. For matched concepts: extract `## Key Points`, `## Sources`, `## Contradictions`, `## My Position`.

   **3b. Match sources**: Glob `wiki/sources/*.md`. For each, read frontmatter (`title`, `tags`, `domain`) and match against direction keywords. Collect matched source slugs, titles, and `## Summary` first lines. This is the user's existing paper library — the primary literature base.

   **3c. Read synthesis + questions**: Read matching `wiki/synthesis/*.md` pages. Check `wiki/QUESTIONS.md` for relevant open questions.

   **3d. Write** `projects/<slug>/lit-review/BACKGROUND.md` with sections:
     - `## Related Papers in Wiki` — matched sources with title + one-line summary (this is the largest and most useful section)
     - `## Established Knowledge` — key points from matched concepts (confidence ≥ medium first)
     - `## Open Debates` — contradictions across matched concepts
     - `## Knowledge Gaps` — relevant open questions from QUESTIONS.md
     - `## User Positions` — content from `## My Position` sections

4. **Update hot.md**: Edit `wiki/hot.md` `## Current Focus` to: `Active project: <slug> (<direction>)`

5. **Log**: `python scripts/wiki_ops.py log-append create-project "<slug>"`

6. **Advance to lit-review**: `python scripts/project_ops.py advance-stage --slug <slug> --to lit-review`

7. **Report**: Show the user the project path, compiled background summary, and next steps.
