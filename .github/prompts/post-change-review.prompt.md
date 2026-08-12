---
mode: agent
description: "Multi-role review after a single major change to refresh agent instructions, knowledge base, and domain trackers. Narrower than session-wrapup 鈥?no hygiene/archiving, no human wait between phases unless a blocker surfaces."
---
# Post-Change Documentation Review Workflow

**Trigger:** User requests a "post-change review" or a major feature / architecture change has just landed.

**Scope vs. session-wrapup.** This workflow is scoped to *documentation refresh after one change*. It does **not** cover hygiene, archiving, or sign-off gates 鈥?those belong to `session-wrapup.md`. Do not invoke both back-to-back for the same change.

Role prefix (`@`, `#`, `/`) is a project choice per `base/instructions/base.md` 搂5; this file uses bare names.

## Execution Protocol

- `admin` orchestrates and produces the final changelog (governance authority across workflows).
- Each phase below MUST be run as a **spawned subagent** per `base/instructions/base.md` 搂6 鈥?do not adopt personas in-place. Rationale: keeps review context isolated from orchestrator context, matches the post-implementation review pattern in `design-flow.md`.
- Execute phases sequentially in a single continuous chain. **Do not wait for user input between phases** unless a role surfaces a blocker 鈥?the workflow is designed for fast doc refresh, not iterative approval.

### Autonomy & Restructuring

Each subagent has full autonomy to reorganize, consolidate, or split documentation files within its domain. Priority: structural efficiency and clarity for downstream agent consumption.

---

### Phase 1 鈥?`admin` (Agent & Meta-Documentation)

- **Spawn `admin`** with a brief naming the change under review.
- **Responsibility:** Review and update the project's top-level instructions (`CLAUDE.md` or equivalent), `.claude/commands/` (or the project's slash-command surface), `.agents/skills/`, `.agents/workflows/`, and any shared base assets inherited from upstream.
- **Action:** Check whether the recent code changes invalidate existing rules, require new constraints, or warrant a new skill. Update the files accordingly.
- **Output:** Return list of files updated and rationale. Orchestrator verifies and proceeds to Phase 2.

### Phase 2 鈥?`tl` (Architecture & Knowledge Base)

- **Spawn `tl`** with the same change context plus Phase 1 findings.
- **Responsibility:** Review and update the project's architecture documentation (typically `docs/knowledge_base/`, design diagrams, architecture markdown).
- **Action:** Ensure new design patterns, schema changes, or pipeline shifts are documented.
- **Output:** Return list of files updated.

### Phase 3 鈥?`researcher` / Domain Expert (Experiments & Domain Trackers)

- **Spawn `researcher`** (or the project's equivalent domain-expert role) with the same context.
- **Responsibility:** Review and update the project's domain trackers 鈥?experiment logs, research trackers, hybrid ledgers, or whichever artifacts the project uses to record domain-specific state.
- **Action:** Document how the recent changes affect ongoing experiments, metrics, or domain-specific configurations.
- **Output:** Return list of files updated.

*Projects without a dedicated domain-expert role may skip Phase 3 鈥?note the skip in the changelog.*

### Phase 4 鈥?`admin` Final Changelog

After the prior phases, the orchestrator (`admin`) aggregates each returned summary into a single changelog for the user:

```
## Post-Change Review 鈥?YYYY-MM-DD
**Change under review:** <short description>

### Admin (meta/skills/workflows)
- <files + rationale>

### TL (architecture/knowledge base)
- <files + rationale>

### Researcher / Domain expert (experiments/trackers)
- <files + rationale, or "skipped 鈥?no domain-expert role">

### Blockers raised
- <item or "none">
```

Present the changelog and stop. The user is free to request further edits; this workflow does not gate a merge.
