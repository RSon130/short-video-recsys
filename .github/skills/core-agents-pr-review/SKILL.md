---
name: core-agents-pr-review
description: >-
  Reviewer contract for isolated-context PR review — input scope, forbidden reads, scope restrictions, and verdict YAML schema. Shared across all reviewer roles spawned by `.agents/workflows/pr-review-cycle.md`.
---

# Skill: PR Review (isolated-context reviewer mode)

Read this skill when spawned as a PR reviewer via `.agents/workflows/pr-review-cycle.md`. It specifies the mechanical contract for PR review — input scope, forbidden reads, scope restrictions, and verdict schema — shared across reviewer invocations regardless of role. The persona file you read before this one (`.agents/roles/<role>.md`) supplies the review **lens**; this skill supplies the **contract**.

Role prefix (`@`, `#`, `/`) is a project choice per `base/instructions/base.md` §5; this file uses bare names.

---

## 1. Reviewer-mode scope restriction (applies to ALL roles invoked under this skill)

When invoked under this skill, your scope is the review ONLY. Regardless of what your persona file says about your normal responsibilities:

- **DO NOT** route work to other roles.
- **DO NOT** orchestrate subsequent workflow phases.
- **DO NOT** edit source files, configs, docs, or the PR itself (no `gh pr review`, no commits, no `git push`).
- **DO NOT** spawn further subagents.
- **DO** read the isolated input set (section 2) and return the verdict YAML (section 5). Then stop.

The main session is the orchestrator; the reviewer subagent is a pure function from `(inputs) → verdict`.

---

## 2. Input scope — what you MAY read

You MAY read ONLY these files, all under `tmp/pr-review-<pr-number>/`:

| File | Source | Purpose |
|---|---|---|
| `req.md` | Copy of the PR's `_req.md` | The user-confirmed intent the diff must satisfy. |
| `contract.md` | Extracted `## Acceptance Contract` section from the PR's `_design.md` | The measurable acceptance bar, API signatures, test plan summary. |
| `diff.patch` | `gh pr diff <pr>` output (or equivalent) | The complete proposed change. |
| `ci.txt` | `gh pr checks <pr>` output plus any failed-job logs and the PR body header | CI / test signal, plus the project's domain-integrity declaration. |

The `admin` reviewer (operational-rule reviewer) MAY additionally read the project's top-level instruction file (`CLAUDE.md` or equivalent) for operational-rule cross-checks, and the PR body text embedded in `ci.txt` for any project-defined declaration verification (e.g. domain-integrity declaration).

No other role has additional read permissions.

---

## 3. Forbidden reads

You MUST NOT read:

- The PR's `_design.md` (rationale / alternatives / dialog log). The acceptance contract is already extracted for you; the rest is reviewer-withheld by design.
- Any other file under `tmp/` outside `tmp/pr-review-<pr-number>/` (e.g. an active RCA artifact, scratch notes, prior-session state).
- The implementer's chat transcript or any session-local scratch.
- Source files in the working tree other than via `diff.patch` context lines. If the diff's context is insufficient to judge a change, that is itself a review finding (→ `request-changes` with a comment requesting more context in the PR description, not a license to read the working tree).

**Rationale:** fresh-context firewall. Reading withheld artifacts re-imports the implementer's framing and defeats the independence the workflow exists to produce.

---

## 4. Review checklist (adapt per role)

Every reviewer MUST evaluate, in order:

1. **Contract satisfaction.** Does the diff meet every bullet in `contract.md`'s `### Success criteria`? For each, cite the diff line that satisfies it or flag as unmet.
2. **Requirement coverage.** Does the diff address every `R<N>` in `req.md`? Missing requirements → `request-changes`.
3. **Lens-specific critique** (this is where your persona matters):
   - **`tl`** (or architecture lens): API contract stability, module boundaries, regression risk, test coverage adequacy, code structure, design coherence.
   - **`researcher`** (or domain-expert lens, when the project has one and the PR's `Domain integrity gate` is `YES`): domain-integrity per the concerns enumerated in `req.md` and `contract.md`. The exact integrity surface is project-policy — the reviewer reads the project's `_req.md` / `_design.md` declarations to know what to inspect, then verifies the diff respects them.
   - **`admin`** (or operational-rule lens): workflow-rule compliance per the project's top-level instructions (scoped commits, working-tree hygiene, configuration protection, no inline-code-in-shell, etc.); declaration verification (e.g. cross-check the PR body's `Domain integrity gate: YES|NO` against semantic signals in the diff — a `NO` declaration with diff signals that suggest YES is a misclassification → `request-changes` with a note).
4. **CI signal.** Does `ci.txt` show any failure, warning, or skipped required check? Any → `request-changes`.

You MAY find issues outside your lens — surface them in the comments, but your verdict weight is primarily on your lens.

---

## 5. Verdict schema (verbatim YAML block as the final content of your reply)

End your response with exactly one fenced YAML block. The orchestrator parses this; no prose after the block.

````yaml
verdict: approve            # or: request-changes
role: <role>                # bare role name — e.g. tl, researcher, admin
summary: <=120 chars        # one-line bottom line
comments:                   # empty list if verdict=approve with no nits
  - file: path/to/file.py
    line: 42                # integer; use 0 for file-level comments
    severity: blocker       # or: nit | question
    body: <short text, one sentence preferred>
````

Rules:
- `verdict: approve` requires zero `severity: blocker` comments. Nits and questions are permitted under `approve`.
- Any `severity: blocker` → verdict MUST be `request-changes`.
- If you cannot produce a verdict (missing inputs, malformed diff, etc.), return `verdict: request-changes` with a single file-level comment explaining the gap. **Fail-closed** — never return `approve` on uncertainty.
- Keep `comments` under 20 entries. Prioritize blockers; consolidate nits.

---

## 6. Output style

- Before the YAML block, you MAY include up to ~20 lines of prose walking through your reasoning against the checklist. This is for the audit trail only; the orchestrator parses only the YAML.
- Do NOT include emojis, decorative headers, or praise. Terse, specific, actionable.
- Do NOT reveal that you did not read `_design.md`; that is operational detail, not review content.
