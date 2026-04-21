---
name: project
description: "Unified project management: status, switch, and resume. Use when the user says \"project status\", \"项目状态\", \"resume\", \"继续\", \"switch to <slug>\", \"切换项目\", or \"/project [slug]\". Without arguments shows all projects; with a slug shows that project's dashboard and offers to resume."
---

# PROJECT — Status / Switch / Resume

Unified entry point for project management. Cross-cutting rules live in `CLAUDE.md`.

## Routing

- **No arguments** (or `project status`, `项目状态`): go to **Status** flow.
- **With slug** (or `resume`, `继续`, `switch to <slug>`): go to **Resume** flow for that project.

---

## Status Flow

1. Run `python scripts/project_ops.py status` → get JSON for all projects.

2. For each project, count non-empty entries in `projects/<slug>/wiki-contributions/pending-*.md`.

3. Render table:

   | Project | Stage | Updated | Stale? | Pending Writebacks |
   |---|---|---|---|---|
   | `<slug>` | `<stage>` | `<date>` | Yes/No (>7 days) | N sources, N concepts, N synthesis |

4. Flag stale projects (>7 days) — suggest resume or archive.

---

## Resume Flow

1. **Resolve slug**: If no slug, read `wiki/hot.md` `## Current Focus`. If no active project, run `python scripts/project_ops.py list` and ask the user.

2. **Switch context**: Update `wiki/hot.md` `## Current Focus` to `Active project: <slug> (<direction>)`.

3. **Read project state** from `projects/<slug>/PROJECT.md`:
   - Frontmatter → stage, direction, domains, venue
   - `## Current Blockers` → what's blocking
   - `## Stage History` → past transitions

4. **Check stage artifacts**:
   - `lit-review`: `lit-review/BACKGROUND.md`
   - `ideation`: `idea-stage/` outputs
   - `refinement`: `refine-logs/` outputs
   - `planning`: `refine-logs/EXPERIMENT_PLAN.md`
   - `implementation`: `experiments/` code
   - `experimentation`: `experiments/results/`
   - `review`: `review-stage/` artifacts
   - `writing`: `paper/` drafts

5. **Present status**: Concise summary — stage, what exists, what's missing, blockers.

6. **Invoke global skill** for current stage:

   | Stage | Skill |
   |---|---|
   | lit-review | `/research-lit` (then compile BACKGROUND.md) |
   | ideation | `/idea-discovery "<direction>"` |
   | refinement | `/research-refine` |
   | planning | `/experiment-plan` |
   | implementation | `/experiment-bridge` |
   | experimentation | `/run-experiment` |
   | review | `/auto-review-loop` |
   | writing | `/paper-writing` |
   | submitted | `/rebuttal` (if reviews received) |

7. **On stage completion**:
   - `python scripts/project_ops.py advance-stage --slug <slug> --to <next>` (auto-updates frontmatter `stage` + appends to `## Stage History`)
   - Append discoveries to `## Findings` in `projects/<slug>/PROJECT.md`
