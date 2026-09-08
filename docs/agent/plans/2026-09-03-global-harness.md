# Global harness expansion

## Scope

Add the reusable parts of the local Project Agent Kit to the user-level Claude
Code and Codex configuration so they apply across repositories.

## Guardrails

- Keep global instructions short and tool-agnostic.
- Keep repository facts and verification commands in each repository.
- Use one subagent at a time by default for subscription-based usage.
- Use SessionStart only for read-only repository checks.
- Do not change the existing Claude permission mode without explicit approval.
- Do not install universal formatters, test runners, MCP servers, or automatic
  self-learning that could misroute or mutate unrelated repositories.

## Verification

- Validate JSON and skill frontmatter.
- Run the global bootstrap hook from this repository.
- Run Codex doctor and Claude doctor.
- Confirm the global files are discoverable without changing repository files.
