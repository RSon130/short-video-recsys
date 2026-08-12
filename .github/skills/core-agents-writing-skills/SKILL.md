---
name: core-agents-writing-skills
description: 'Authoring discipline for new skills in core-agents and downstream projects. Use when proposing a new skill file, refactoring an existing one, or deciding whether new content belongs as a skill, a base-instruction rule, or a workflow. Codifies the placement decision, the file structure, and the empirical-validation gate that separates an exercised rule from a speculative one.'
---

# Skill: Writing Skills

Read this skill before authoring or substantially editing a `skills/<name>/SKILL.md` file (upstream `core-agents`) or a `.agents/skills/<name>.md` file (downstream). It defines:

1. The placement decision — whether new content belongs as a skill, a base-instruction rule, or a workflow.
2. The mechanical shape of a skill file — frontmatter, sections, naming, style.
3. The empirical-validation gate that separates an exercised rule from a speculative one.

Companion to `CONTRIBUTING.md` §3 (Promotion pathway). That section governs *when* a downstream rule moves upstream; this skill governs *how* the rule is shaped as a skill, regardless of where it lives.

---

## 1. Placement: skill vs base-instruction rule vs workflow

Three kinds of reusable agent assets coexist. Conflating them dilutes each — skills get treated as optional reading, workflows get treated as background context, base rules get bloated with domain content.

| Asset | Purpose | Choose this home when |
|---|---|---|
| **Base-instruction rule** (`base/instructions/base.md`) | Universal, always-loaded behavior. | The rule applies to every agent operation regardless of task — workspace hygiene, hallucination control, validation discipline. Loaded on cold-start. Budget-constrained. |
| **Skill** (`skills/<name>/SKILL.md` upstream, `.agents/skills/<name>.md` downstream) | Domain-specific reference knowledge, read on demand. | The content applies only when the agent enters a specific domain or uses a specific tool/contract. Reading every session would be wasteful. |
| **Workflow** (`.agents/workflows/<name>.md`) | Active multi-step procedure with sequenced role transitions. | The content prescribes an ordered sequence: do A, then B, then hand off to role X. Each step has a definite ordering relative to the others. |

**Decision heuristic (in order):**

1. Does the content prescribe an **ordered sequence with role transitions**? → workflow.
2. Does the content describe a **constraint that should shape behavior whenever a specific domain / tool / contract is in scope**? → skill.
3. Does the content describe a **constraint that always applies, regardless of task**? → base-instruction rule.

When uncertain between skill and base rule, prefer skill — base rules are loaded every session and budget-constrained; skills are loaded on demand and may be more discursive. When uncertain between skill and workflow, prefer skill — workflows imply orchestration responsibility; without role transitions, a skill is the lighter-weight home.

---

## 2. File structure

A skill is a single Markdown file under its own directory upstream (`skills/<name>/SKILL.md`) or a single file downstream (`.agents/skills/<name>.md`). The single-directory-per-skill upstream layout reserves room for supporting files (reference data, sub-procedures, lookup tables) without renaming the skill.

### 2.1 Frontmatter

```yaml
---
name: <skill-id>
description: '<one paragraph>'
---
```

- `name` — globally-unique identifier within its scope. Upstream skills are prefixed `core-agents-` to keep them disjoint from downstream-authored skills.
- `description` — used by the skill discovery mechanism to decide relevance. The description MUST: (1) lead with a one-sentence purpose statement (matching the existing core-agents skill convention — see `core-agents-deploy`, `core-agents-pr-review`); (2) include a triggering condition somewhere in the description, typically phrased `Use when ...`; (3) NOT summarize the skill's procedure step-by-step — that defeats the discovery contract by encouraging substitution of the summary for the actual file.

### 2.2 Body sections

Required:

- **Opening paragraph** — restates the triggering condition and names the skill's contribution in one or two sentences. A reader should know within five seconds whether they are in the right file.
- **Numbered or named main sections** — the substance. Each section should be self-contained enough to quote in a code review or PR comment without surrounding context.

Recommended where applicable:

- **Common mistakes / red flags** — failure modes the skill exists to prevent. Cite a triggering incident if one exists.
- **Verification checklist** — a short list the agent can self-check against before declaring the skill applied.

Optional:

- **Supporting files** — heavy reference material, large code samples, lookup tables. Live alongside `SKILL.md` in the skill's directory. The main file references them; readers fetch only what they need.

### 2.3 Style

- **Domain-neutral, role-neutral, path-neutral, mechanism-level** when authoring upstream. Downstream skill files MAY use project-specific tool / role / path names — that is the point of being downstream.
- **No emojis, decorative dividers, or marketing language.** Skills are read under attention pressure; ornament costs reading time.
- **Pick one voice and stay there.** Existing core-agents skills use second-person imperative (`Read this skill when ...`, `Do not ...`). Do not oscillate between `we`, `you`, and the passive voice within one file.
- **Cite triggering incidents** for non-obvious rules. A rule a future reader cannot trace back to a real failure is at risk of being deleted as speculative during a future bump-review.

---

## 3. The empirical-validation gate

A skill is **not ready to ship** until it has been exercised against the situation it is meant to address. This is the load-bearing step in `CONTRIBUTING.md` §3 (step 3, "Exercise"). Authoring without this gate produces speculative skills that codify the author's mental model of failure rather than failure modes that have actually surfaced.

### 3.1 What "exercise" means by skill type

Different skill types exercise differently. Choose the test that matches the type:

- **Discipline-enforcing skill** (the agent must honor a rule under pressure — e.g. anti-hallucination, RCA-first debugging, scoped staging). Exercise by running a flow where the temptation to violate the rule is real (a fast-feedback retry loop, a "looks done" moment with disk state unverified, a multi-target write with cwd ambiguous). The skill is exercised if the agent honors the rule and the rule prevents the failure that motivated it.

- **Technique skill** (a reusable approach the agent applies — e.g. wire-format probe, isolated-context PR review). Exercise by applying the technique to at least one concrete case and recording the outcome. The skill is exercised if the technique's output is meaningfully different from the pre-skill behavior on the same input.

- **Pattern-recognition skill** (helps the agent classify situations — e.g. classifying a remote failure as infra / code / resource class). Exercise by feeding the skill at least one positive case (the pattern applies) and one near-miss (the pattern almost applies but doesn't). The skill is exercised if both cases are classified correctly.

- **Reference skill** (lookup table, contract schema, fixed protocol). Exercise by retrieving from it during a real flow and confirming the retrieved content matched what the flow needed.

### 3.2 What does NOT count as exercise

- A self-review pass over the skill text the agent just authored. Skills written from intuition without a real failure to react to are easier to mis-shape than skills written in response to a triggering incident.
- A unit test of the skill text itself. Skills are not code; they are read by an agent that will then act. The test is whether the agent's downstream behavior is actually different.
- The author's confident assertion that "this is obviously the right rule." If the rule were obvious, it would not need a skill file — it would already be common practice.

### 3.3 The exercise record

`CONTRIBUTING.md` §3 expects the PR body to cite the triggering incident and the exercise. A complete record names:

- **Trigger** — the failure or recurring pattern that motivated the skill (date, project, brief description).
- **Exercise** — the flow where the candidate skill was applied (date, what the agent did differently, the outcome).
- **Generalization claim** — why the rule applies beyond this one incident (the scope-test self-assessment per `CONTRIBUTING.md` §2).

A skill that lacks any of the three components is not yet ready for upstream promotion. Meta-skills *about the upstream itself* (e.g. `core-agents-deploy`, `core-agents-pr-review`, this skill) are exempt from the downstream-author-first lineage by precedent — there is no downstream "where" for them to live first — but they MUST still satisfy a generalization claim.

---

## 4. Common mistakes

- **Description that summarizes the procedure.** `Use when reviewing a PR. Read req.md, then contract.md, then ...` — this is procedural content in the discovery slot; agents may substitute the summary for the actual file. Keep descriptions to triggering condition + mechanism-level abstract.
- **Skill written from anticipated-failure imagination.** No triggering incident, no exercise record, no generalization argument — three signals that the candidate is speculation. Hold it as a draft until a real flow triggers it.
- **Procedural sequence buried in a skill.** If the content prescribes role transitions or step ordering, it belongs in a workflow. Skills are read as constraints; workflows are read as procedures.
- **Path-specific or tool-specific wording in upstream skills.** Upstream is consumed by every fork; concrete project-specific names break the inheritance contract. Use generic vocabulary (`the project's configured remote queue`, `the agent's native file-editing tools`) and let downstream projects substitute.
- **Skill that has accreted a workflow.** Sections like `First do X, then hand off to role Y` are workflow content masquerading as skill. Extract to `.agents/workflows/` and leave a pointer.

---

## 5. Verification checklist

Before merging a new skill or a substantive edit:

- [ ] The placement decision (skill vs base rule vs workflow) is correct — verified by walking the §1 heuristic out loud.
- [ ] Frontmatter `description` starts with `Use when ...` and does not summarize the procedure.
- [ ] Each section is self-contained enough to quote without surrounding context.
- [ ] Style is domain-neutral / role-neutral / path-neutral / mechanism-level (upstream only).
- [ ] An exercise record exists per §3.3 (trigger + exercise + generalization claim), or the skill is a meta-skill about the upstream itself.
- [ ] If the skill modifies an existing skill, the SemVer classification per `CONTRIBUTING.md` §5 is correct (typically minor for additive sections, major for removed or renamed rules).
