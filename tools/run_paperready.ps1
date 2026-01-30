param(
  [Parameter(Mandatory = $false)]
  [string]$OutRoot = "D:\\Research\\avalanche_runs",

  [Parameter(Mandatory = $false)]
  [int]$MaxPhase = 8,

  [Parameter(Mandatory = $false)]
  [switch]$GpuSmi,

  [Parameter(Mandatory = $false)]
  [switch]$Background
)

$ErrorActionPreference = "Stop"

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

$log = Join-Path $OutRoot ("paperready_" + $ts + ".log")
$argsList = @(
  "tools/ssot_pipeline.py",
  "--out-root", $OutRoot,
  "--max-phase", $MaxPhase
)
if ($GpuSmi) { $argsList += "--gpu-smi" }

if ($Background) {
  Start-Process -FilePath "python" -ArgumentList $argsList -NoNewWindow `
    -RedirectStandardOutput $log -RedirectStandardError $log | Out-Null
  Write-Host ("Started in background. Log: " + $log)
  exit 0
}

Write-Host ("Logging to: " + $log)
& python @argsList 2>&1 | Tee-Object -FilePath $log
