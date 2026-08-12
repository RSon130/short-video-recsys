---
mode: agent
description: "Rules and procedures for reading, updating, and structuring project plans, templates, and agent handoffs. Use this whenever asked to create a plan or manage handoffs."
---
# Project Planning & Handoffs

## Reading Plans

For multi-session projects, actively search for and read the corresponding project plan in `.agents/plans/` at the start of a session or when asked to review project state. Do not infer project state from memory 鈥?the plan file is the source of truth.

## Writing Plans

When asked to start a new project plan, use a structured markdown format. Focus on:

- High-level objectives.
- Step-by-step task breakdown with checkboxes (checked / unchecked).
- Constraints and requirements.
- A **Context & Handoff** section kept current across sessions (see below).

*Note on domain-specific plans.* If the project has a dedicated location for domain artifacts (e.g. experiment logs, research trackers), place domain-specific plans there rather than in `.agents/plans/`. Follow the project's ledger or tracker convention when one exists. Avoid copying volatile domain parameters (hyperparameters, run IDs, connection strings) into long-lived markdown 鈥?reference them by pointer instead.

## Design Flow Artifacts

For tasks that require design before coding, two artifact types are tracked in `.agents/plans/` (templates live in `.agents/templates/`):

**Naming convention:** `YYYY-MM-DD-<task-slug>_req.md` and `YYYY-MM-DD-<task-slug>_design.md`.

| Artifact | Template | When to create |
|----------|----------|---------------|
| `YYYY-MM-DD-<task-slug>_req.md` | `.agents/templates/_template_requirements.md` | After Step 1 discovery, before TL involvement |
| `YYYY-MM-DD-<task-slug>_design.md` | `.agents/templates/_template_design.md` | During TL / requester co-design loop (Step 3) |

A master project plan that spans multiple design-flow cycles should reference the requirements and design artifacts by path once both are produced. Do not copy their content into the master plan 鈥?link to them.

## Updating Handoffs

At the end of a session, or when asked to *prepare handoff*, update the plan's **Context & Handoff** section:

- Check off completed tasks.
- Summarize the current state of variables, infrastructure, or key parameters at a pointer level (file path + short description, not inline values).
- State what the next session needs to start on.
- Explicitly record any blockers, unresolved issues, or external dependencies that may have changed since the last session.
