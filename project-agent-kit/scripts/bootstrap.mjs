#!/usr/bin/env node

/**
 * Project Agent Kit: read-only project survey, instruction bootstrap, and
 * SessionStart context. It deliberately never overwrites existing project
 * instructions and never learns directly into them.
 */

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";

const argv = process.argv.slice(2);

function has(flag) {
  return argv.includes(flag);
}

function value(flag, fallback = undefined) {
  const index = argv.indexOf(flag);
  return index >= 0 && argv[index + 1] ? argv[index + 1] : fallback;
}

function exists(filePath) {
  try {
    fs.accessSync(filePath);
    return true;
  } catch {
    return false;
  }
}

function read(filePath) {
  try {
    return fs.readFileSync(filePath, "utf8");
  } catch {
    return "";
  }
}

function json(filePath) {
  try {
    return JSON.parse(read(filePath));
  } catch {
    return null;
  }
}

function resolveRoot(start) {
  const candidate = path.resolve(start || process.cwd());
  try {
    return execFileSync("git", ["-C", candidate, "rev-parse", "--show-toplevel"], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
  } catch {
    return candidate;
  }
}

function safeEntries(root) {
  try {
    return fs.readdirSync(root, { withFileTypes: true });
  } catch {
    return [];
  }
}

function detectFacts(root) {
  const entries = safeEntries(root);
  const files = new Set(entries.filter((entry) => entry.isFile()).map((entry) => entry.name));
  const dirs = entries
    .filter((entry) => entry.isDirectory()
      && ![".git", ".agent-harness", ".codex", ".claude", ".venv", "venv", "node_modules", "project-agent-kit"].includes(entry.name))
    .map((entry) => entry.name)
    .sort();
  const packageJson = json(path.join(root, "package.json"));
  const packageManager = packageJson?.packageManager?.split("@")[0]
    || (files.has("pnpm-lock.yaml") ? "pnpm"
      : files.has("yarn.lock") ? "yarn"
        : files.has("bun.lockb") || files.has("bun.lock") ? "bun"
          : files.has("package-lock.json") ? "npm" : null);
  const scripts = packageJson?.scripts && typeof packageJson.scripts === "object"
    ? Object.keys(packageJson.scripts).sort() : [];
  const pythonFiles = ["pyproject.toml", "requirements.txt", "setup.py", "Pipfile", "environment.yml"]
    .filter((file) => files.has(file));
  const instructionFiles = ["AGENTS.md", "AGENTS.override.md", "CLAUDE.md", "CONTRIBUTING.md", "README.md"]
    .filter((file) => files.has(file));
  const ciFiles = [];
  const github = path.join(root, ".github", "workflows");
  if (exists(github)) ciFiles.push(".github/workflows/");
  for (const file of [".gitlab-ci.yml", "azure-pipelines.yml", "Jenkinsfile", "Dockerfile", "docker-compose.yml"]) {
    if (files.has(file)) ciFiles.push(file);
  }
  const hasVirtualEnv = entries.some((entry) => entry.isDirectory() && [".venv", "venv"].includes(entry.name))
    || files.has(".python-version");
  return {
    name: path.basename(root),
    root,
    packageManager,
    packageScripts: scripts,
    pythonFiles,
    hasVirtualEnv,
    topLevelDirectories: dirs.slice(0, 20),
    instructionFiles,
    ciFiles,
    packageName: packageJson?.name || null,
    packageType: packageJson?.type || null,
  };
}

function fingerprint(facts) {
  const stable = JSON.stringify({
    name: facts.name,
    packageManager: facts.packageManager,
    packageScripts: facts.packageScripts,
    pythonFiles: facts.pythonFiles,
    hasVirtualEnv: facts.hasVirtualEnv,
    topLevelDirectories: facts.topLevelDirectories,
    ciFiles: facts.ciFiles,
  });
  return crypto.createHash("sha256").update(stable).digest("hex").slice(0, 16);
}

function commandEvidence(facts) {
  const lines = [];
  if (facts.packageManager && facts.packageScripts.length) {
    for (const name of facts.packageScripts) lines.push(`- ${facts.packageManager} run ${name} (package.json script)`);
  }
  if (facts.pythonFiles.length) lines.push(`- Python project evidence: ${facts.pythonFiles.join(", ")}`);
  if (facts.hasVirtualEnv) lines.push("- Python environment evidence: .venv/venv or .python-version");
  if (facts.ciFiles.length) lines.push(`- CI/build evidence: ${facts.ciFiles.join(", ")}`);
  return lines.length ? lines.join("\n") : "- No standard build/test command was safely inferred; inspect the repository before running commands.";
}

function proposedFiles(facts, stamp = new Date().toISOString()) {
  const fp = fingerprint(facts);
  const files = new Map();
  files.set("AGENTS.md", `# Repository agent instructions

## Purpose

This is the canonical project guidance for coding agents. Keep it short and update it only when a durable project rule has been verified.

## Repository map

- Project: ${facts.packageName || facts.name}
- Root: repository root
- Top-level areas: ${facts.topLevelDirectories.join(", ") || "none discovered"}
- Existing guidance: ${facts.instructionFiles.join(", ") || "none at bootstrap time"}

## Working rules

- Inspect the relevant code and nearby tests before editing.
- Record the initial git status and preserve unrelated user changes.
- Preserve existing conventions and public behavior unless the request changes them.
- Make the smallest coherent change; do not refactor unrelated code.
- Treat generated files, credentials, and user data as protected.
- Do not claim success without running the narrowest relevant verification.
- For risky or broad work, write a plan in docs/agent/plans/ before implementation.

## Verification

${commandEvidence(facts)}

See docs/agent/ for the maintained project map, command evidence, quality bar, and writing voice.

## Handoff

Every completed change should report: what changed, files touched, verification run, and any remaining uncertainty.
`);
  files.set("CLAUDE.md", `@AGENTS.md

## Claude-specific operating notes

- Use plan mode for multi-file or risky changes, then execute with the least expensive capable model.
- Prefer one focused implementer and one reviewer; do not create a swarm for ordinary work.
- Ask before changing durable project rules, dependencies, schemas, or deployment behavior.
`);
  files.set("docs/agent/index.md", `# Agent context

This directory contains project-specific context that is too detailed for the root instruction files.

- [Commands](commands.md) — observed commands and evidence
- [Architecture](architecture.md) — the current repository map
- [Quality](quality.md) — definition of done and verification policy
- [Voice](voice.md) — writing and communication conventions
- [Plans](plans/) — durable plans for broad changes
`);
  files.set("docs/agent/commands.md", `# Commands and evidence

Generated from a read-only repository survey on ${stamp}. Reconfirm commands before using them when the project changes.

${commandEvidence(facts)}

Agents must prefer commands already documented by the project. If a command is missing, inspect package metadata, CI, Makefiles, task runners, and contributor documentation before inventing one.
`);
  files.set("docs/agent/architecture.md", `# Architecture map

Generated from the repository root on ${stamp}.

- Project: ${facts.packageName || facts.name}
- Top-level directories: ${facts.topLevelDirectories.join(", ") || "none discovered"}
- CI/build files: ${facts.ciFiles.join(", ") || "none discovered"}
- Python project markers: ${facts.pythonFiles.join(", ") || "none discovered"}

Before making a broad change, update this map with verified ownership boundaries and data flow. Do not turn guesses into rules.
`);
  files.set("docs/agent/quality.md", `# Quality bar

- Start with a small, explicit plan when work crosses files, modules, or behavior boundaries.
- Add or update a regression test for behavior changes when the project has a test system.
- Run focused checks first, then the repository's broader checks when practical.
- Review the final diff for accidental files, secrets, debug output, and unrelated formatting churn.
- Verify user-visible flows manually when unit tests cannot cover them.
- Report test failures honestly; distinguish pre-existing failures from regressions.
`);
  files.set("docs/agent/voice.md", `# Writing voice

- Be concise, concrete, and direct.
- Prefer plain language and active voice.
- Preserve the terminology already used by the project.
- For technical changes, explain the outcome before the implementation details.
- For documentation, outline first, draft second, then perform a fresh-reader pass for unsupported claims and ambiguity.
`);
  files.set(".agent-harness/state.json", JSON.stringify({
    schemaVersion: 1,
    source: "project-agent-kit",
    generatedAt: stamp,
    fingerprint: fp,
    root: facts.root,
    instructionFiles: ["AGENTS.md", "CLAUDE.md"],
    policy: "native-memory-first; promote durable lessons only after review",
  }, null, 2) + "\n");
  return files;
}

function reportFor(root) {
  const facts = detectFacts(root);
  const statePath = path.join(root, ".agent-harness", "state.json");
  const state = json(statePath);
  const missing = ["AGENTS.md", "CLAUDE.md", "docs/agent/index.md", ".agent-harness/state.json"]
    .filter((relative) => !exists(path.join(root, relative)));
  const currentFingerprint = fingerprint(facts);
  const stale = Boolean(state?.fingerprint && state.fingerprint !== currentFingerprint);
  return { facts, state, missing, currentFingerprint, stale, statePath };
}

function context(report) {
  const { facts, missing, stale } = report;
  const lines = [
    "Project Agent Kit session check:",
    `- repository: ${facts.root}`,
    `- project: ${facts.packageName || facts.name}`,
  ];
  if (missing.length) lines.push(`- bootstrap needed: ${missing.join(", ")}`);
  else if (stale) lines.push("- project map changed since the last bootstrap; run repo-init review before relying on stale context");
  else lines.push("- project instructions present and fingerprint is current");
  lines.push("- memory policy: use native harness memory for observations; promote only reviewed, repeated, project-specific rules");
  lines.push("- routing policy: planner for ambiguity, implementer for changes, reviewer for focused verification; keep concurrency at one by default");
  return lines.join("\n");
}

function print(valueToPrint) {
  process.stdout.write(`${valueToPrint}\n`);
}

function main() {
  const root = resolveRoot(value("--cwd", process.cwd()));
  const report = reportFor(root);
  if (has("--fingerprint")) return print(report.currentFingerprint);
  if (has("--check")) {
    if (has("--json")) return print(JSON.stringify({
      root,
      missing: report.missing,
      stale: report.stale,
      currentFingerprint: report.currentFingerprint,
      facts: report.facts,
    }, null, 2));
    return print(report.missing.length || report.stale ? context(report) : "Project Agent Kit: instructions are present and current.");
  }
  if (has("--hook")) {
    const additionalContext = context(report);
    if (value("--format", "json") === "json") {
      const output = { hookSpecificOutput: { hookEventName: "SessionStart", additionalContext } };
      if (has("--visible")) {
        output.systemMessage = report.missing.length || report.stale
          ? "Project Agent Kit: review the project bootstrap status in the session context."
          : "Project Agent Kit: SessionStart hook ran; project instructions are current.";
      }
      return print(JSON.stringify(output));
    }
    return print(additionalContext);
  }
  if (has("--plan")) {
    const files = proposedFiles(report.facts);
    const missingOnly = [...files.keys()].filter((relative) => !exists(path.join(root, relative)));
    return print(JSON.stringify({ root, missing: missingOnly, stale: report.stale, fingerprint: report.currentFingerprint }, null, 2));
  }
  if (has("--init")) {
    if (report.missing.length === 0 && !report.stale) return print("Project Agent Kit: nothing to initialize.");
    if (report.missing.length === 0 && report.stale && !has("--refresh-state")) {
      return print("Project Agent Kit: project facts changed; review --plan, then rerun with --refresh-state after approval.");
    }
    const files = proposedFiles(report.facts);
    const written = [];
    for (const [relative, contents] of files) {
      const target = path.join(root, relative);
      if (exists(target)) continue;
      fs.mkdirSync(path.dirname(target), { recursive: true });
      fs.writeFileSync(target, contents, "utf8");
      written.push(relative);
    }
    if (has("--refresh-state") && exists(path.join(root, ".agent-harness", "state.json"))) {
      const stateTarget = path.join(root, ".agent-harness", "state.json");
      fs.writeFileSync(stateTarget, files.get(".agent-harness/state.json"), "utf8");
      if (!written.includes(".agent-harness/state.json")) written.push(".agent-harness/state.json (refreshed)");
    }
    return print(JSON.stringify({ root, written, fingerprint: report.currentFingerprint }, null, 2));
  }
  print("Usage: bootstrap.mjs --check [--json] | --plan | --init | --hook [--format json] | --fingerprint");
}

main();
