# Project Agent Kit

An opinionated, cross-harness scaffold for Claude Code and Codex.

It provides:

- a read-only SessionStart check for missing or stale `AGENTS.md` / `CLAUDE.md` guidance;
- a deterministic bootstrap generator that keeps detailed context under `docs/agent/`;
- Claude agent roles and Codex agent templates for scout, architect, implementer, and reviewer;
- cost-aware routing and a reviewed memory-promotion policy.

The generator never overwrites existing instructions and does not silently turn native memory into project policy.

## Local test

From this directory:

```powershell
node scripts/bootstrap.mjs --check --cwd "C:\path\to\repo" --json
node scripts/bootstrap.mjs --plan --cwd "C:\path\to\repo"
node scripts/bootstrap.mjs --init --cwd "C:\path\to\repo"
node scripts/install-local.mjs --repo "C:\path\to\repo" --init --copy-agents
```

Run `--init` only after reviewing the proposed files. The plugin is intentionally not installed globally by the build step.

Codex project hooks are loaded from `.codex/hooks.json`. In the CLI/TUI, `/hooks` is for reviewing and trusting hook definitions; it does not manually execute them. Hook context is model-visible, while the local configuration also emits a small SessionStart confirmation so the hook is observable.

For Codex native memory, review `config/codex-config.toml` and merge it into your existing user config. Do not replace an existing config file wholesale.
