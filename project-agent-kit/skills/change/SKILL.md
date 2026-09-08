---
name: change
description: Execute a code change with deliberate planning, cost-aware role routing, focused tests, and a final diff review. Use for feature work, bug fixes, refactors, and multi-file changes.
---

# Change workflow

Classify the request before acting:

- Narrow and obvious: inspect, edit, run the narrowest relevant check, review the diff.
- Multi-file or ambiguous: plan first; use a strong planner/architect only for the design, then a workhorse implementer.
- Risky, security-sensitive, schema-changing, or externally visible: require a focused reviewer and explicit verification.

Default to one active implementer and at most one reviewer. Do not spawn parallel agents unless their file scopes are independent. Use worktrees for parallel changes or when rollback isolation matters.

For Claude, `opusplan` is the preferred subscription-friendly route: Opus plans and Sonnet executes. For Codex, use the strongest model for ambiguity and the balanced/lightweight model for bounded implementation or repetitive checks, according to the models currently available in the user's account. Model names are configuration, not project rules.

The final response must state what changed, verification run, and remaining uncertainty. Never hide a failing check behind a successful-looking summary.
