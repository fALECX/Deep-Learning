#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const pluginRoot = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const argv = process.argv.slice(2);
const repoFlag = argv.indexOf("--repo");
const repo = path.resolve(repoFlag >= 0 ? argv[repoFlag + 1] : process.cwd());
const shouldInit = argv.includes("--init");
const copyAgents = argv.includes("--copy-agents");

function copyMissing(source, target) {
  if (fs.existsSync(target)) return false;
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.copyFileSync(source, target);
  return true;
}

if (shouldInit) {
  const result = spawnSync(process.execPath, [
    path.join(pluginRoot, "scripts", "bootstrap.mjs"), "--init", "--cwd", repo,
  ], { stdio: "inherit" });
  if (result.status !== 0) process.exit(result.status || 1);
}

const installed = [];
if (copyAgents) {
  for (const role of ["scout", "architect", "implementer", "reviewer"]) {
    if (copyMissing(
      path.join(pluginRoot, "codex", "agents", `${role}.toml`),
      path.join(repo, ".codex", "agents", `${role}.toml`),
    )) installed.push(`.codex/agents/${role}.toml`);
    if (copyMissing(
      path.join(pluginRoot, "agents", `${role}.md`),
      path.join(repo, ".claude", "agents", `${role}.md`),
    )) installed.push(`.claude/agents/${role}.md`);
  }
}

console.log(JSON.stringify({ repo, initialized: shouldInit, copiedAgents: installed }, null, 2));
