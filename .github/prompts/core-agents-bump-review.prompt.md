---
mode: agent
description: "Semantic review workflow triggered on a core-agents submodule bump. Classifies each upstream change as already-reflected / needs-inlining / conflict / not-applicable, producing a concrete action list for the maintainer."
---
# core-agents Bump Review

**Trigger:** a bump of the `core-agents` submodule pointer, or a dotfiles-mode pull that updates `~/.claude/base/` and `~/.agents/`.

**Purpose.** Keep the project's inline-duplicated base content semantically aligned with `core-agents` without requiring byte-level mirrors. Byte-diff tooling can't distinguish healthy paraphrase from real drift; this workflow replaces it with a single per-bump semantic pass.

**Policy context.** `core-agents/base/instructions/base.md` 搂0 establishes the drift-over-duplication policy: projects may keep promoted rules inline (self-contained at the decision point) and audit periodically instead of maintaining live pointers. This workflow *is* the periodic audit.

Role prefix (`@`, `#`, `/`) is a project choice per `base/instructions/base.md` 搂5; this file uses bare names.

---

## Step 1 鈥?Build the upstream changelog

From the project root (submodule mode):

```bash
OLD=<previous submodule SHA>
NEW=<new submodule SHA>
git -C core-agents log "$OLD..$NEW" -- base/ .agents/templates/ .agents/workflows/ prompts/
git -C core-agents diff "$OLD..$NEW" -- base/instructions/base.md | head -400
```

Dotfiles mode: substitute `~/.claude/base/...` and `~/.agents/...` paths and use `git log` within the dotfiles clone.

Save the raw changelog to `tmp/core-agents-bump-<OLD>-<NEW>.md` as the input artifact for the review. Do not edit it 鈥?it is the source of truth for what changed upstream.

---

## Step 2 鈥?Spawn `admin` for classification

**Spawn `admin` as a subagent** per `base/instructions/base.md` 搂6. In-place review bleeds bump context into the main session.

**Standard brief:**

```
Role:      Read `.agents/roles/admin.md` for your persona.
Task:      Semantic review of the core-agents bump <OLD> 鈫?<NEW>. For each
           changed base rule, section, template, or workflow listed in the
           changelog artifact, classify as one of:
             (a) already reflected inline in the project's instructions / role
                 files 鈥?no action.
             (b) new or tightened invariant that needs to be inlined 鈥?specify
                 target file(s) and the exact section to edit.
             (c) conflicts with a project-specific delta 鈥?flag for human
                 decision. Do NOT resolve autonomously.
             (d) not applicable to this project.
           Scope of files to review: the project's top-level instructions file
           (`CLAUDE.md` / `copilot-instructions.md` / equivalent), every file
           in `.agents/roles/`, every file in `.agents/skills/`, every file
           in `.agents/workflows/` that mirrors an upstream workflow, and
           `.agents/templates/_template_*.md`.

           For `.agents/skills/`, `.agents/workflows/`, and
           `.agents/templates/`: project forks of upstream assets are the
           norm, not the exception. When bucket (b) applies (upstream
           improvement worth adopting), the edit MUST be SURGICAL 鈥?preserve
           the project's existing scaffolding intact, then merge the upstream
           improvement INTO the existing fork. Never replace a forked file
           wholesale; that strips deliberate project deltas. If the upstream
           change cannot be cleanly merged into the fork, escalate as bucket
           (c) for human resolution.
Artifacts: Changelog input: tmp/core-agents-bump-<OLD>-<NEW>.md
           Write the classification table to the same file under a new
           "## Classification" section. For (b) entries, include the concrete
           edit (file path + before/after snippet). For (c) entries, describe
           the conflict and the two options for resolution.
Output:    Return: count per bucket (a/b/c/d), any blockers surfaced during
           review, and the classification artifact path.
```

---

## Step 3 鈥?Apply non-conflicting edits

The orchestrator reads the classification artifact:

- **Bucket (a):** no action. Note in the commit message.
- **Bucket (b):** apply each inline edit. These should be mechanical 鈥?the classification artifact already specifies the edit. Run the project's linters after.
- **Bucket (c):** HARD STOP. Present each conflict to the human with the two resolution options. Do NOT apply autonomously. Human either picks a side, writes a project-specific delta to override, or defers the bump.
- **Bucket (d):** no action. Note in the commit message that these upstream changes were intentionally not adopted.

---

## Step 4 鈥?Commit the bump + inline updates

Once buckets (a)鈥?d) are dispositioned:

1. Stage the submodule pointer bump alongside the bucket-(b) inline edits (one coherent commit per logical scope 鈥?prefer small commits over a monolithic one).
2. Commit message names the upstream SHA range and the non-trivial inlined updates. Bucket-(c) deferrals are recorded with a `FOLLOW-UP:` note so they aren't forgotten.
3. If any bucket-(c) items were deferred, open a project-specific issue or plan entry capturing the conflict.

---

## What this workflow does NOT do

- **No byte-level drift checking against inline base content.** Byte-diff is not the right tool 鈥?project paraphrase is healthy and legitimate. Use this semantic review instead.
- **No auto-resolution of bucket-(c) conflicts.** Those are policy decisions, not mechanical updates.
- **No mirror of `base/` into `.agents/base/`.** That pattern was retired; projects inline directly into their top-level instructions and role files.

## When byte-diff (`audit-drift.sh`) DOES apply

`audit-drift.sh` is the correct tool ONLY when a project explicitly opts into manifest-sync via `.agents/sync-manifest` 鈥?i.e. the project chose to import a specific upstream skill / workflow / template as a byte-identical mirror, with no project-specific scaffolding on top. This is appropriate for:

- Brand-new projects without forked deltas yet.
- Cases where the project genuinely treats an upstream asset as a generic drop-in (rare in practice).

For projects that **fork** upstream assets to add project-specific scaffolding (the dominant pattern for mature projects), byte-diff produces noise: divergence is intentional, not drift. Running `audit-drift.sh` against a forked asset would mis-flag deliberate scaffolding as something to "fix" via re-sync 鈥?silently stripping the project deltas the workflows depend on.

**Quick decision rule per project:**
- Project has `.agents/sync-manifest` listing the asset 鈫?run `audit-drift.sh` on bumps for that asset; semantic review covers everything else.
- Project does NOT manifest-sync the asset 鈫?semantic review (this workflow) is the only audit. `audit-drift.sh` is wrong for that asset.
- A project may mix the two strategies across asset classes (e.g., manifest-sync templates as generic; fork skills with project deltas).

Templates were historically lumped with manifest-synced assets in this workflow; they are not special 鈥?the same per-asset opt-in rule applies. Templates that the project forks (e.g. extending the design template with a project-specific Acceptance Contract section) are forked content, audited semantically; templates the project treats as generic are manifest-synced.
