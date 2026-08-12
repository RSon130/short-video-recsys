---
mode: agent
description: "End-of-session or end-of-project wrap-up protocol. Triggers all roles to review changes, close out experiments or milestones, update documentation, enforce hygiene, and archive. Use when the user says "wrap up", "look back", "close out", "end of sprint", or "archive the session". Must use if `#admin` and wrapping up."
---
# Session / Project Wrap-Up Protocol

The wrap-up is a multi-role review gate that runs at the end of a session or milestone. It is the authoritative checkpoint before a branch is merged, a project is shelved, or the next cycle begins.

**Trigger conditions** (any one suffices):
- User says "wrap up", "look back", "close out", "end of sprint", or "archive".
- A long-running job or experiment completes and no new phase has been declared.
- A PR is about to be merged into the default branch.

---

## Phase 1 鈥?Change Review (`#admin` orchestrates)

`#admin` opens the review gate. For each role, check whether they were *actively involved* in this session. If a role was absent or their domain was touched without their involvement, they MUST review now before Phase 2 begins.

### 1a. `#tl` / Lead Architecture Review
- To enumerate all changes this session, use **both** of the following 鈥?do not rely on `git diff` alone, as work may span multiple commits and there may also be uncommitted changes:
  - `git log <default-branch>..HEAD --oneline` 鈥?lists every commit pushed since the branch diverged from the default branch. (Substitute `<default-branch>` with the project's actual branch name, e.g. `main` or `master`. The angle-bracket form is a placeholder, not literal syntax.)
  - `git status --short` + `git diff HEAD` 鈥?captures any uncommitted edits or staged changes on top of those commits.
- Verify every code change was covered by a design artifact in `.agents/plans/`. If an implementation exists without a corresponding `_design.md`, flag it as a **design debt item** and create a post-hoc note.
- Check that module-level docstrings and architecture invariants are still accurate after the changes.
- Confirm no unintentional scope creep (features added without a requirement).
- Sign-off line: `TL OK 鈥?<date> 鈥?<one-line summary>`.

### 1b. Domain Expert / Researcher Review
- Confirm every experiment, analysis, or domain-specific task has a closed entry in the relevant documentation with an empirical or analytical conclusion.
- Verify that any changes to data pipelines, input contracts, domain logic, or output interpretation were properly reviewed during development. If not, audit now.
- Flag any open comparisons or open-ended analyses that were never concluded, and record them as blockers in the relevant doc.
- Confirm the knowledge base reflects any new domain quirks, edge cases, or conventions discovered this session.
- Sign-off line: `Researcher OK 鈥?<date> 鈥?<one-line summary>`.

### 1c. `#developer` Implementation Review
- Run the linter / static checks across all modified files and resolve any warnings.
- Confirm all `tmp/` scripts written during the session are either deleted (if throwaway) or documented and moved to a reusable location (e.g., `.agents/scripts/`).
- Verify no secrets, credentials, or tokens were committed (`git log -p | grep -i "password\|secret\|token\|api_key"`).
- Confirm dependency manifests (`requirements.txt`, `package.json`, etc.) are in sync with any new imports or packages introduced.
- Sign-off line: `Developer OK 鈥?<date> 鈥?<one-line summary>`.

---

## Phase 2 鈥?Documentation Updates (`#researcher` / domain expert leads, `#admin` reviews)

Complete these in order. Do not mark a section done until the file is saved and verified on disk.

| # | Target | What to update |
|---|--------|----------------|
| 1 | Experiment / project docs | Write or finalize the empirical or analytical conclusion for every task or experiment touched this session. Include the key outcome metric and whether the effort is closed or continuing. |
| 2 | Knowledge base (`docs/knowledge_base/` or equivalent) | Add entries for: new data or domain quirks, confirmed edge cases, architectural patterns that worked or failed, infrastructure discoveries. |
| 3 | Active plan 鈫?Context & Handoff section | Check off completed tasks. Record current artifact locations (URLs or relative paths 鈥?no local absolute paths). State the next session starting point. List unresolved blockers explicitly. |
| 4 | Backlog / `todo.md` | Close completed items. Promote any newly discovered issues. Reprioritize the remaining list for the next session. |
| 5 | Changelog / release notes | If a meaningful milestone was reached (e.g., first passing benchmark, new stable pipeline), record a short human-readable entry in the relevant doc. |

---

## Phase 3 鈥?Hygiene & Archiving (`#admin`)

### 3a. Repository Hygiene
- `tmp/` audit: delete one-off debug scripts; move reusable helpers to `.agents/scripts/` with a docstring.
- Confirm no unexpectedly large binary or generated files were added 鈥?use `git log <default-branch>..HEAD --stat` (committed) and `git status --short` / `git diff HEAD --stat` (uncommitted) to surface files above your project's threshold to the user.
- Confirm branch naming still reflects the work. Notify the user if it has drifted.
- Surface any stale partial runs, intermediate artifacts, or output dumps to the user for deletion or archival.

### 3b. Prompt & Skill Hygiene (if new patterns or violations emerged)
- If a recurring agent mistake was observed this session, identify the root cause and propose a surgical update to the relevant instruction or skill file.
- If a new domain or workflow pattern was discovered (e.g., a new framework edge case, a new infrastructure quirk), add it to the relevant skill.
- Present all proposed instruction changes to the user before applying.

### 3c. Archiving (end-of-project only)
Run only when an entire project or feature branch is being fully closed out 鈥?not for routine end-of-session wraps:
- Move the active plan from `.agents/plans/` to `.agents/plans/archive/YYYY-MM-DD-<slug>/`.
- Confirm the branch is merged or has a clear PR open.
- Write a one-paragraph project retrospective at the bottom of the primary documentation covering: what was learned, what would be done differently, and the recommended starting point for future resumption.

---

## Phase 4 鈥?Status Report (`#admin`)

After all phases complete, produce a structured status report for the user:

```
## Wrap-Up Report 鈥?YYYY-MM-DD

### Changes Reviewed
- TL / Lead:    [OK / FLAG] 鈥?<summary>
- Researcher:   [OK / FLAG] 鈥?<summary>
- Developer:    [OK / FLAG] 鈥?<summary>

### Documentation Updates
- [ ] Experiment / milestone docs closed
- [ ] Knowledge base updated
- [ ] Handoff / plan updated
- [ ] Backlog pruned

### Hygiene
- [ ] tmp/ cleaned
- [ ] No secrets in git
- [ ] Prompts / skills updated (if needed)

### Blockers / Open Items for Next Session
1. <item>
2. <item>

### Recommended Next Action
<one-sentence direction>
```

Present this report and wait for user confirmation before closing the session. If flags were raised in Phase 1, do not mark the wrap-up complete until the human acknowledges each flag.
