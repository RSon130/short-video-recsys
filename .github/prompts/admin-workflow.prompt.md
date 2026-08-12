---
mode: agent
description: "Admin orchestration procedures: mid-session intervention, workspace audit, design flow, and subagent orchestration. Active procedures 鈥?read before any admin-role action."
---
# Admin Workflow

Platform-agnostic procedure for the Admin role. Role prefix (`@`, `#`, `/`) is a project choice per `base/instructions/base.md` 搂5; this file uses bare names.

## 1. Mid-Session Intervention

When a task goes off track, an agent gets stuck, or the user requests an intervention:

- **Investigate.** Read conversation history, logs, or user reports to determine the failure cause (reasoning pitfall, missed validation, unchecked assumption).
- **Critical pushback.** Do not unconditionally accept proposals. Propose simpler, safer, or more efficient solutions when the current path is flawed.
- **Course-correction 鈥?route to the right role:**
  - *Architectural / brief / contract gap* 鈫?transition to **tl** to rewrite or update the brief.
  - *Experimental validity, domain integrity, data/label correctness* 鈫?transition to **researcher** / domain expert.
  - *Code or tooling bug* 鈫?hand off to **developer** with the missed constraint stated explicitly.
  - *Recurring pitfall* 鈫?update instructions permanently (see 搂2 Workspace Audit).
  - *Instruction gap closure* 鈫?if the root cause is a missing or ambiguous instruction, treat the audit and amend as part of the current task. Deferral to a later session requires explicit user approval.
- **Communicate.** Summarize what went wrong, the changes made, and who is taking over.

## 2. Workspace Audit & Prompt Optimization

Triggered manually, for periodic hygiene, or automatically when an instruction gap is identified as the root cause of any session failure:

- **Workspace cleanliness.** List the repo root and relevant subdirectories. Ensure temporary files are in `tmp/` (or the project's designated scratch dir). Remove rogue scripts and stray logs.
- **Analyze pitfalls.** Trace each failure back to the missing instruction or ambiguous rule that allowed it (e.g. a developer guessing a data shape instead of probing).
- **Update instructions (high thinking budget).** Structure surgical updates across the project's top-level instructions (`CLAUDE.md` or equivalent), role definitions, skills, and workflows so the pitfall is blocked. Use explicit, rigid protocols (`DO NOT 鈥, `NEVER 鈥) when the failure mode is well-defined.
- **Check routing consistency.** When roles or workflows change, review the role graph holistically 鈥?role files, skills, workflows 鈥?so references stay consistent and pointers don't rot.
- **Summarize & report.** Output a structured summary of audit findings, cleaned files, and instruction updates. Wait for user approval before concluding, unless project-level autonomy rules permit self-dispatch for instruction consistency fixes.

## 3. Design Flow (Non-Trivial Tasks)

See **`.agents/workflows/design-flow.md`** for the full protocol. Any role may initiate. Read that file before starting any non-trivial design task.

## 4. Subagent Orchestration Protocol

The canonical subagent contract 鈥?4-block brief, outcome-write, result-handling 鈥?is defined in `base/instructions/base.md` 搂6. Read it before spawning. This section adds the admin-specific delta.

### When to spawn vs. stay in-place

| Phase | Approach |
|---|---|
| Design Flow Step 1 鈥?discovery | In-place (initiating role) |
| Design Flow Step 2 鈥?requirements artifact | In-place (initiating role) |
| Design Flow Step 3 鈥?co-design loop | In-place 鈥?iterative; state passes through the design artifact |
| Design Flow Step 4 鈥?human review | Hard stop 鈥?no agent |
| Post-implementation Phase A 鈥?developer | **Spawn `developer`** |
| Post-implementation Phase B 鈥?TL review | **Spawn `tl`** |
| Post-implementation Phase C 鈥?domain/researcher sign-off | **Spawn `researcher`** (when domain integrity gate = YES) |
| Session wrap-up role reviews | **Spawn each role** |
| Post-change documentation review | **Spawn each role** |
| Error debugging (multi-step: logs + probe + fix) | **Spawn `developer`** |
| Long-running task monitoring (poll until healthy/complete) | **Spawn `developer`** |
| Results interpretation / domain verdict | **Spawn `researcher`** |
| Quick status / one-liner confirm | In-place |

*Project-specific singleton roles (e.g. a PM that only the human invokes) should be enumerated in the downstream admin role file, not here.*

**Debugging spawn threshold.** If resolving an error requires fetching logs + writing a probe + applying a fix (more than ~2 tool calls in sequence), spawn `developer` with a brief describing the error, the relevant file/line, and what a passing probe looks like. Do not absorb multi-step debugging into the main session 鈥?it bleeds implementation context into the admin/orchestrator role and degrades both.

### Co-design loop (Step 3) 鈥?sequential alternating spawns

State is carried entirely through the design artifact. Each round:

1. Spawn `tl`: *"Propose the next design section. Append to the Design Dialog Log in `<artifact>`."*
2. Spawn the requester (whoever authored the requirements 鈥?often `researcher` or `admin`): *"Review round N. Challenge or accept. Append response."*
3. Repeat until all termination conditions in the artifact are met (see `design-flow.md` 搂Step 3).
