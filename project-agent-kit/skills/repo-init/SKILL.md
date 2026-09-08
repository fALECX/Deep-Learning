---
name: repo-init
description: Bootstrap or review project-specific agent instructions when a repository lacks canonical guidance, or when the session-start check reports missing or stale instructions. Use for first contact with an unfamiliar repository.
---

# Repository initialization

Use the plugin script `scripts/bootstrap.mjs --check --json` for a read-only survey. If guidance is missing or stale:

1. Inspect the repository, package metadata, CI, contributor docs, tests, and existing instructions.
2. Separate verified facts from assumptions. Ask the user only about decisions that cannot be inferred and materially affect the rules.
3. Show the proposed instruction/doc files and get approval before writing durable project guidance.
4. Run `scripts/bootstrap.mjs --init` after approval, then manually refine the generated files with verified facts.
5. Keep root `AGENTS.md` and `CLAUDE.md` short. Put detailed maps, commands, quality rules, voice, and plans under `docs/agent/`.

Never overwrite an existing instruction file automatically. Never encode a one-off user preference as a project rule without evidence that it is durable.
