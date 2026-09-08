---
name: promote-memory
description: Convert repeated, project-specific lessons from native Claude or Codex memory into a reviewed instruction change without automatic instruction drift. Use after corrections, repeated preferences, or recurring failures.
---

# Memory promotion

Use native harness memory first. At the end of meaningful work, record a candidate with:

- the observed user correction or repeated behavior;
- the repository and files affected;
- evidence from at least two occurrences or one explicit durable rule;
- the proposed rule and the smallest file where it belongs.

Promote only project-specific, durable, non-sensitive rules. Do not promote temporary task details, private data, model preferences that belong in user settings, or guesses. Show the proposed diff and obtain approval before changing `AGENTS.md`, `CLAUDE.md`, or `docs/agent/`.

If no promotion is justified, leave the observation in native memory and do not edit project instructions.
