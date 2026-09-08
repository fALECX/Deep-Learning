#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const failures = [];
const required = [
  ".codex-plugin/plugin.json",
  "hooks/hooks.json",
  "hooks/session-start.ps1",
  "scripts/bootstrap.mjs",
  "scripts/install-local.mjs",
  "skills/repo-init/SKILL.md",
  "skills/change/SKILL.md",
  "skills/promote-memory/SKILL.md",
  "skills/write/SKILL.md",
];
for (const relative of required) {
  if (!fs.existsSync(path.join(root, relative))) failures.push(`missing ${relative}`);
}

for (const relative of [".codex-plugin/plugin.json", "hooks/hooks.json"]) {
  try { JSON.parse(fs.readFileSync(path.join(root, relative), "utf8")); }
  catch { failures.push(`invalid JSON ${relative}`); }
}

for (const role of ["scout", "architect", "implementer", "reviewer"]) {
  for (const relative of [`agents/${role}.md`, `codex/agents/${role}.toml`]) {
    if (!fs.existsSync(path.join(root, relative))) failures.push(`missing ${relative}`);
  }
}

for (const skill of ["repo-init", "change", "promote-memory", "write"]) {
  const content = fs.readFileSync(path.join(root, "skills", skill, "SKILL.md"), "utf8");
  if (!content.startsWith("---\n") || !content.includes("\nname:") || !content.includes("\ndescription:")) {
    failures.push(`invalid skill frontmatter ${skill}`);
  }
}

const check = spawnSync(process.execPath, [path.join(root, "scripts", "bootstrap.mjs"), "--check", "--cwd", root, "--json"], { encoding: "utf8" });
if (check.status !== 0) failures.push("bootstrap check failed");
try {
  const result = JSON.parse(check.stdout);
  if (!result.currentFingerprint) failures.push("bootstrap did not return a fingerprint");
} catch { failures.push("bootstrap check did not return JSON"); }

if (failures.length) {
  console.error(failures.map((failure) => `- ${failure}`).join("\n"));
  process.exit(1);
}
console.log(`Harness validation passed: ${root}`);
