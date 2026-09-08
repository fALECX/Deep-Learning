# Opinionated routing

Use this small graph for ordinary coding requests:

```text
request
  -> scout (only if the repo or target is unfamiliar)
  -> architect (only if ambiguous, broad, or risky)
  -> implementer
  -> reviewer (only if risk or scope justifies it)
  -> focused verification
```

Claude subscription default: start with `opusplan`, which uses Opus for planning and Sonnet for execution. Use a single Sonnet implementer for bounded work; add one Sonnet reviewer only for meaningful risk.

Codex subscription default: use the strongest available model for planning or difficult diagnosis, then the balanced available model for bounded implementation. Use the lightweight model for reconnaissance and repetitive checks. Current model names and availability are account-dependent; keep them in user-level Codex config rather than hard-coding them into projects.

Parallelism is an exception. Use it only for genuinely independent scopes, preferably in worktrees. Subscription use still has a quota/throughput cost even when there is no separate per-token invoice; every additional agent also adds coordination and context overhead.
