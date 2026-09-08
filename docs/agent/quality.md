# Quality bar

- Start with a small, explicit plan when work crosses files, modules, or behavior boundaries.
- Add or update a regression test for behavior changes when the project has a test system.
- Run focused checks first, then the repository's broader checks when practical.
- Review the final diff for accidental files, secrets, debug output, and unrelated formatting churn.
- Verify user-visible flows manually when unit tests cannot cover them.
- Report test failures honestly; distinguish pre-existing failures from regressions.
