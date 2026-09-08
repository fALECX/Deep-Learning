$raw = [Console]::In.ReadToEnd()
$cwd = $null
try {
  if ($raw) {
    $payload = $raw | ConvertFrom-Json
    $cwd = $payload.cwd
  }
} catch {
  $cwd = $null
}

$pluginRoot = $env:CLAUDE_PLUGIN_ROOT
if (-not $pluginRoot) { $pluginRoot = $env:PLUGIN_ROOT }
if (-not $pluginRoot) { $pluginRoot = Split-Path -Parent $PSScriptRoot }

$nodeArgs = @("$pluginRoot\scripts\bootstrap.mjs", "--hook", "--format", "json")
if ($env:CODEX_HOME) { $nodeArgs += "--visible" }
if ($cwd) { $nodeArgs += @("--cwd", $cwd) }
& node @nodeArgs
exit $LASTEXITCODE
