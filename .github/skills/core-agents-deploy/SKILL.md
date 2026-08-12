---
name: core-agents-deploy
description: 'Install and update the core-agents plugin in a target Claude Code environment. Use when first installing core-agents on a new maintainer host, updating an existing install to a new release, or verifying that plugin-mediated reads resolve correctly. Plugin-only deployment — legacy dotfiles and submodule modes are no longer supported.'
argument-hint: 'Target host or environment to install / update core-agents'
---

# Deploy Core Agents (plugin)

Use this skill when you need to install or update `core-agents` on a Claude Code host. Plugin distribution is the single supported deployment channel; legacy dotfiles installation (`install.sh`) and Git-submodule mode have been retired.

## Critical Rules

- **Plugin install is per-Claude-Code-host, not per-project.** A maintainer installs the plugin once on each Claude Code installation that needs the `/core-agents-*` commands and the plugin-mediated reads. Project repositories do not embed the plugin.
- **Filename-prefixing preserves the project-local namespace.** All commands and skills the plugin ships are named `core-agents-<name>` so they invoke as `/core-agents-<name>`. Project-local `.claude/commands/admin.md` (or any other bare-name) keeps resolving to the project-local file. Plugin commands and project commands are deliberately disjoint.
- **No tracked references to plugin install paths.** Project repositories must not gain `~/.claude/plugins/...` paths in tracked content. Downstream projects route reads through their own resolver helper (e.g. quant's `.agents/scripts/resolve-core-agents-root.sh`), never by hard-coding the plugin install path.
- **Content reconciliation on each upstream change is delegated to a project-side bump-review workflow.** This skill installs and updates the plugin; it does not touch downstream forked content. After a `/plugin update`, the downstream maintainer runs the project's bump-review workflow to classify upstream changes semantically and apply surgical edits to forked assets.
- If any deployment activity requires a project-side commit, the commit still requires explicit user approval per the project's standing commit policy.

## Procedure

### 1. First-time install on a maintainer host

```
/plugin marketplace add https://github.com/fa-mc/core-agents.git
/plugin install core-agents@core-agents
```

The marketplace declaration lives at `.claude-plugin/marketplace.json` in the repository root. The plugin source resolves to a release tag (`source.ref`), so `/plugin install` pins to that tag rather than tracking `main`.

### 2. Update to a new release

```
/plugin update core-agents
```

`/plugin update` pulls whichever release tag the marketplace currently advertises. Maintainers cut a new tag on every merge to `main`; consumers `/plugin update` to opt in.

### 3. Verify the install (three positive observations)

After install or update, confirm:

1. `/help` lists `/core-agents-admin (plugin:core-agents)`, `/core-agents-pm (plugin:core-agents)`, `/core-agents-tl (plugin:core-agents)`.
2. Project-local `/admin` (typed in any project that ships its own `.claude/commands/admin.md`) still resolves to the project-local content. Plugin's `/core-agents-admin` is invocable separately and is distinct.
3. The plugin-install directory's `HEAD` SHA matches the release tag the marketplace advertises:

```
git -C ~/.claude/plugins/marketplaces/core-agents/plugins/core-agents rev-parse HEAD
```

"No error appeared" is **not** sufficient verification — all three observations must come back positive.

### 4. Post-update content reconciliation (downstream concern)

The plugin install is the source of truth for upstream content. Downstream projects that maintain inline duplicates of upstream rules (CLAUDE.md, role files, skill files) reconcile via:

1. The project's own bump-review workflow — semantic classification of upstream changes (already-reflected / needs-merging / conflicts-with-fork / not-applicable) and surgical edits to forked assets.
2. The project's session-wrapup drift audit — flags inline-duplicate gaps between the plugin and the project's CLAUDE.md / role files.

Neither of those is in scope for this skill; this skill stops at "plugin installed and verified."

### 5. Maintainer test loop (unreleased changes)

For maintainers iterating on `core-agents` itself: temporarily override `source.ref` in a local fork of `marketplace.json` to point at `main` (or a feature branch), install from that fork, exercise the change, then revert before shipping. Do not ship a marketplace fork.

## Reporting

Report back:

- The install or update action taken (`/plugin marketplace add` / `/plugin install` / `/plugin update`).
- The commands the install surfaced in `/help`.
- The plugin install path and its HEAD SHA.
- Whether all three positive observations from §3 came back green.
- Any downstream-project follow-up still needed (bump-review run, drift audit, inline-duplicate update commit).

## See also

- `CONTRIBUTING.md` — promotion pathway, scope test, SemVer rules.
- `CHANGELOG.md` — public contract for what changed between releases.
- For an example downstream migration runbook: `docs/knowledge_base/core-agents-plugin-migration.md` in the quant adopter project.
