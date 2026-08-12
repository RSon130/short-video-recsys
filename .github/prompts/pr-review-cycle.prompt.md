---
mode: agent
description: "Automated implementer 鈫?PR 鈫?multi-role isolated review 鈫?human-gated merge cycle. Invoked after `developer` (or the project's implementer role) completes an implementation task tied to an `_req.md` + `_design.md` pair in `.agents/plans/`."
---
# Workflow: PR Review Cycle

Multi-role isolated PR review with a fresh-context firewall, parallel reviewer spawns, and a human-gated merge step. Pairs with the `.agents/skills/pr-review.md` reviewer contract (input scope, forbidden reads, verdict schema).

Role prefix (`@`, `#`, `/`) is a project choice per `base/instructions/base.md` 搂5; this file uses bare names.

---

## Roles in this workflow

- **Main session (orchestrator):** runs phases A鈥揋 below. Typically `admin`; may be the implementer at handoff.
- **Reviewer subagents:** spawned via the host's subagent / `Agent`-tool mechanism in Phase D. Always include the architectural-review role (`tl`); include the domain-expert role (`researcher`) when the PR's `Domain integrity gate` is `YES`; include the operational-rule lens (`admin`) for workflow-rule cross-check.
- **Human:** final merge gate in Phase F.

---

## Invariants (enforced every run)

1. **No auto-merge.** The workflow terminates at the human merge gate. Nothing in this workflow may invoke `gh pr merge` (or equivalent).
2. **Isolated reviewer context.** Reviewers read ONLY the four files in `tmp/pr-review-<pr>/`. The skill enforces the forbidden-reads list.
3. **Unanimous-approve advancement.** All required reviewers must return `verdict: approve`. Any `request-changes` 鈫?revise loop.
4. **Loop cap = 2 review-triggered rewrites.** The third `request-changes` cycle escalates to human; do not re-spawn reviewers.
5. **CI green before review.** If any required CI check is not passing, reviewers are not spawned.

---

## Phases

### Phase A 鈥?Push & PR open (implementer)

Preconditions (the implementer confirms before opening the PR):
- Branch is rebased on the target branch; clean working tree (`git status --porcelain` empty).
- Local probe / validation per the project's pre-dispatch checklist has passed.
- Commit scope is tight (`git add <files>`, not `-A`).

PR body MUST include:
```
Requirements: .agents/plans/<YYYY-MM-DD-slug>_req.md
Design: .agents/plans/<YYYY-MM-DD-slug>_design.md
Domain integrity gate: YES     # or: NO  (per base 搂5 鈥?YES if the project has a domain-expert role
                               #         and the diff touches that role's integrity surface)
```

Open the PR (the project's host CLI; below uses `gh`):
```bash
gh pr create --title "<concise>" --body-file tmp/pr-body-<slug>.md --base <target-branch>
```

Record the returned PR number as `<PR>`.

### Phase B 鈥?CI green gate (orchestrator)

```bash
gh pr checks <PR> --required --watch
```
- Exit 0 鈫?proceed to Phase C.
- Non-zero 鈫?fetch failing job logs, write to `tmp/pr-review-<PR>/ci-failure.txt`, transition back to the implementer with the log. No reviewer spawn.
- Timeout / indeterminate 鈫?surface to human.

### Phase C 鈥?Assemble reviewer input (orchestrator)

Create `tmp/pr-review-<PR>/` and populate exactly four files:

```bash
mkdir -p tmp/pr-review-<PR>
cp .agents/plans/<slug>_req.md                   tmp/pr-review-<PR>/req.md

# Extract the "## Acceptance Contract" section (case-insensitive) 鈥?stop at next top-level "## " header.
# Case-insensitive match guards against typos like "## Acceptance contract" silently producing empty output.
awk 'BEGIN{IGNORECASE=1} /^## acceptance contract[[:space:]]*$/{f=1; next} /^## /{f=0} f' \
    .agents/plans/<slug>_design.md              > tmp/pr-review-<PR>/contract.md

# Pre-flight check: empty contract = missing/misnamed section. Halt before reviewer spawn.
[ -s tmp/pr-review-<PR>/contract.md ] || \
    { echo "ERROR: Acceptance Contract section missing or empty in <slug>_design.md"; exit 1; }

gh pr diff   <PR>                               > tmp/pr-review-<PR>/diff.patch
{ gh pr view <PR> --json body -q .body; echo; echo "---"; gh pr checks <PR>; } \
                                                > tmp/pr-review-<PR>/ci.txt
```

If `contract.md` is empty 鈫?`_design.md` is missing the Acceptance Contract section. Halt and surface to human; do not spawn reviewers against an empty contract.

### Phase D 鈥?Spawn parallel isolated reviewers (orchestrator)

**Domain-integrity gate:** parse `Domain integrity gate:` from `ci.txt` header.
- `YES` 鈫?spawn architectural reviewer + domain-expert reviewer + operational-rule reviewer (typically `tl` + `researcher` + `admin`).
- `NO` 鈫?spawn architectural reviewer + operational-rule reviewer only (typically `tl` + `admin`). The operational-rule reviewer cross-checks the declaration against semantic signals in the diff.

Send ALL reviewer spawns in a **single message** (parallel 鈥?independent invocations, no shared state). Brief template, per reviewer:

```
You are spawned as <role> performing a PR review.

1. Read .agents/roles/<role>.md (your persona).
2. Read .agents/skills/pr-review.md (the review contract 鈥?input scope, forbidden
   reads, verdict schema, reviewer-mode scope restriction).
3. Read the four files in tmp/pr-review-<PR>/. Do NOT read any other file
   (the operational-rule reviewer: the project's top-level instruction file is
   also permitted for operational-rule cross-check).
4. Apply the review checklist from the skill, through your <role> lens.
5. Return the YAML verdict block as specified in the skill and stop.
```

**Subagent parameters per reviewer** (project-specific 鈥?the host's subagent type, model, any heterogeneity strategy, and the verdict parser used in Phase E all live in the project's pr-review-cycle delta, not in this base workflow). One recommendation worth noting at this level: at least one reviewer should run on a *different* model from the others, to break same-model convergence. If only one model is available, note that fact in the Phase F consolidated review body.

### Phase E 鈥?Parse verdicts (orchestrator)

For each returned subagent message, extract the trailing YAML block via the project's verdict parser.

Missing / malformed YAML 鈫?treat as `request-changes` (fail-closed per skill 搂5).

Aggregate:
- All `approve` 鈫?Phase F.
- Any `request-changes` 鈫?Phase G.

### Phase F 鈥?Consolidated approve & human gate (orchestrator)

Build a single consolidated review body:

```markdown
### PR Review 鈥?all reviewers approved

- **<role-1>** (<model>): <summary>
- **<role-2>** (<model>): <summary>
- **<role-3>** (<model>): <summary>   <!-- omit if domain-expert seat skipped; note reason -->

Iteration: <N>/2.  Nit / question comments (if any) listed below; no blockers.

<optional consolidated nits>
```

Post as an approving review (NOT a merge):
```bash
gh pr review <PR> --approve --body-file tmp/pr-review-<PR>/consolidated.md
```

Surface the PR URL to the human: *"PR #<PR> passed multi-role isolated review. Ready for your merge decision."*

**Workflow terminates here.** Do not run `gh pr merge` (or equivalent) under any circumstance.

### Phase G 鈥?Revise loop (orchestrator 鈫?implementer)

Write aggregate blockers to `tmp/pr-review-<PR>/revise-brief-<N>.md` grouped by file, preserving each blocker's (role, severity, body). Increment iteration counter in `tmp/pr-review-<PR>/iteration.txt`.

Transition to the implementer with the brief. Expected response: a new commit pushed to the same branch, then re-enter Phase B.

Loop-cap check:
- Iteration 1 鈫?allowed.
- Iteration 2 鈫?allowed.
- Iteration 3 (would-be) 鈫?STOP. Surface all accumulated blockers to human; do not re-spawn reviewers.

The iteration counter counts **review-triggered rewrites only**, not CI-failure re-pushes (Phase B returning to the implementer does not increment).

---

## Artifacts layout (`tmp/pr-review-<PR>/`)

```
req.md                    # copy of _req.md
contract.md               # extracted Acceptance Contract
diff.patch                # gh pr diff
ci.txt                    # PR body (with Domain integrity gate decl) + gh pr checks
ci-failure.txt            # only when Phase B fails
<role>-response.txt       # raw reviewer outputs (one file per reviewer)
iteration.txt             # integer, 0 initially
consolidated.md           # Phase F review body
revise-brief-<N>.md       # Phase G handoff artifacts
```

Clean up at session-wrapup when the PR merges or closes.

---

## Related

- Skill: `.agents/skills/pr-review.md` (reviewer contract).
- Design-flow integration: `.agents/workflows/design-flow.md` Step 5 (post-implementation review) delegates here when a project's design flow uses this workflow as its review-cycle implementation.
- Verdict parser: project-specific implementation.
