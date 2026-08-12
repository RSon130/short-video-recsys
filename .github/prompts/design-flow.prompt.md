---
mode: agent
description: "End-to-end design flow for non-trivial tasks: requirements 鈫?co-design 鈫?human review 鈫?implementation 鈫?post-implementation review. Any role may initiate. Subagent spawn is mandatory at Steps 4 and 5."
---
# Design Flow

Use this workflow for any task requiring design decisions before coding begins. **Any role may initiate** (`admin`, `researcher`, `tl`). Read this file before proceeding.

This is the full version of the protocol sketched in `base/instructions/base.md` 搂5. Role prefix (`@`, `#`, `/`) is a project choice.

**Templates:** `.agents/templates/_template_requirements.md`, `.agents/templates/_template_design.md`
**Artifact naming:** `YYYY-MM-DD-<task-slug>_req.md` / `YYYY-MM-DD-<task-slug>_design.md` 鈥?tracked in `.agents/plans/`.

**Subagent spawn contract:** The 4-block brief format (Role / Task / Artifacts / Output), outcome-write contract, and orchestrator result-handling protocol are canonical in `base/instructions/base.md` 搂6 鈥?read before Steps 4 and 5. The briefs shown below are design-flow-specific deltas (named sections, artifact paths), layered on that invariant.

---

## Step 1 鈥?Discovery (In-place, initiating role)

Engage the user until the problem is unambiguous. No artifact yet. Goal: enough clarity to write requirements without open design questions leaking into the requirements document.

---

## Step 2 鈥?Requirements Artifact (In-place, initiating role 鈥?Human gate)

Write `YYYY-MM-DD-<task-slug>_req.md` in `.agents/plans/` using `_template_requirements.md`:

- Fill all sections including **Domain integrity gate** and **Open Questions**.
- Present the artifact to the human for review.
- **Hard stop 鈥?do not proceed until the human confirmation checkbox is marked.**
- Transition to `tl` with an explicit message referencing the artifact path.

---

## Step 3 鈥?Co-Design Dialog (In-place 鈥?automated, no human)

`tl` and the requester run an automated co-design loop. Collaborative and adversarial by design.

**TL per round:** Propose a concrete design section. Surface assumptions that could violate requirements. Flag anything out of scope.

**Requester per round:** Challenge assumptions. Contribute alternatives or tighter contracts. Resolve open questions. Do **not** accept a round unless at least one substantive revision was negotiated.

**Termination condition (ALL must be true):**

1. Every requirement in `_req.md` is addressed or explicitly out-of-scope with justification.
2. Every open question is resolved.
3. Tests table has at least one row per functional requirement (every `R<n>` from `_req.md` appears in at least one row's "Maps to" column 鈥?greppable).
4. Success criteria are measurable and unambiguous.
5. If the **Domain integrity gate** is checked YES in `_req.md`: the domain expert (`researcher` or equivalent) has signed off on Data & Model Contracts (or the project's equivalent contract sections).
6. **Non-blocking concerns cost-checked.** For every concern classified as non-blocking (follow-up, post-merge, deferred), record a one-line *predicted cost of failure* in concrete units the project understands (compute time, data corruption scope, re-dispatch cycles, debugging-hours). If the cost crosses the project's blocking threshold, upgrade to blocking. Rationale: the classification is only as strong as the worst cost it tolerates; see `base/instructions/base.md` 搂4 "Predicted-Cost Estimation for Non-Blocking Concerns."

Record every round in the **Design Dialog Log** section of the design artifact.

---

## Step 4 鈥?Human Review & Handoff (Hard stop, conditional)

Present the completed `<task>_design.md` to the human.

- Human requests revisions 鈫?restart Step 3 from current artifact (do not discard dialog log).
- Human approved 鈫?**spawn `developer` as a subagent** using the standard brief below. **Do NOT run `developer` in-place.** In-place execution bleeds implementation context into the design session and bypasses the artifact-driven handoff contract.

**Hard-stop override:** If `tl` sign-off + domain-expert co-sign (when the domain integrity gate is YES) are both captured in the design artifact, `admin` MAY spawn `developer` without re-asking the human, provided the project's top-level rules permit this level of autonomy. Projects may tighten or relax this override.

**Standard spawn brief for `developer`:**

```
Role:      Read `.agents/roles/developer.md` for your persona.
           Read project-designated domain skills before any domain-specific work.
Task:      Implement all tasks in the Implementation Plan of <design artifact path>.
           Work strictly from the Implementation Plan, Tests, and Success Criteria.
           Do not accept verbal changes 鈥?only the artifact is authoritative.
Artifacts: Design: <design artifact path>
           Fill in the Implementation Status section when complete.
Output:    Return: files changed, test results, any approved deviations from the plan.
```

---

## Step 5 鈥?Post-Implementation Review Loop (Spawned subagents 鈥?no human until Phase D)

### Phase A 鈥?Developer implementation

Handled by the spawned `developer` subagent from Step 4. On completion, the orchestrator reads the returned summary and verifies the **Implementation Status** section of the artifact is filled in before proceeding.

### Phase B 鈥?TL architectural review

**Spawn `tl` as a subagent. Do NOT run in-place.**

```
Role:      Read `.agents/roles/tl.md` for your persona.
Task:      Review the implementation against the design artifact at <path>.
           Check Implementation Plan completion, Tests, Success Criteria, and workspace
           hygiene rules from base/instructions/base.md.
           For every finding you classify as non-blocking, record a predicted cost of
           failure in project-concrete units. If cost crosses the project's blocking
           threshold, upgrade to blocking. Trace into integration points 鈥?do not review
           units in isolation.
           Write findings to the TL Review section of the artifact.
           If issues found: list them clearly so developer can action each one.
           If clean: check the TL sign-off box.
Artifacts: Design: <design artifact path>
Output:    Return: pass/fail, list of issues (if any), sign-off status.
```

- Issues found 鈫?orchestrator re-spawns `developer` (Phase A) with the TL review notes. Repeat until clean.
- Clean 鈫?proceed to Phase C if the domain integrity gate is YES, otherwise Phase D.

### Phase C 鈥?Domain/Researcher review *(Domain integrity gate YES only)*

**Spawn `researcher` (or the project's equivalent domain expert) as a subagent. Do NOT run in-place.**

```
Role:      Read `.agents/roles/researcher.md` for your persona.
Task:      Review the implementation against Data & Model Contracts (or the project's
           equivalent contract section) in <design artifact path>.
           Verify domain-specific invariants: data shapes, dtypes, label conventions,
           loss logic, feature semantics, or whichever contracts the project defines.
           For every finding you classify as non-blocking, record a predicted cost of
           failure. If cost crosses the project's blocking threshold, upgrade to blocking.
           Trace into integration points 鈥?do not review units in isolation.
           Write findings to the Researcher Review section of the artifact.
           If issues found: list them clearly so developer can action each one.
           If clean: check the Researcher sign-off box.
Artifacts: Design: <design artifact path>
Output:    Return: pass/fail, list of issues (if any), sign-off status.
```

- Issues found 鈫?orchestrator re-spawns `developer` (Phase A). Repeat until clean.
- Clean 鈫?Phase D.

### Phase D 鈥?Human final approval (hard stop, conditional)

Present the design artifact with a summary: what was implemented, test results, all sign-offs. Wait for human confirmation. If revisions requested, re-enter at the appropriate phase (code fix 鈫?A; architectural dispute 鈫?B; domain contract violation 鈫?C).

**Hard-stop override:** If all required sign-offs are present in the artifact 鈥?TL review checked, domain-expert review checked (when domain integrity gate = YES), developer Implementation Status complete 鈥?`admin` MAY proceed directly to commit + push + downstream dispatch without re-asking the human, provided the project's top-level rules permit this level of autonomy. Project-specific dispatch gates (branch gates, contract checks, clean-tree requirements) still apply independently.
