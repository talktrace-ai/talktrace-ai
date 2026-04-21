# TalkTrace AI — install + launch helper for Windows PowerShell.
#
# Usage (from this folder):
#   .\run.ps1              # install (if needed) and start the app
#   .\run.ps1 -Reinstall   # force-recreate the virtual environment
#   .\run.ps1 -NoWindow    # start the app headless (no desktop window)
#
# The script creates a project-local virtual environment in .\.venv,
# installs dependencies from requirements.txt, and launches the Shiny
# app. It is safe to re-run — dependency installation is idempotent.

[CmdletBinding()]
param(
    [switch]$Reinstall,
    [switch]$NoWindow
)

$ErrorActionPreference = 'Stop'

# Always run from the script's own directory so relative paths resolve.
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ProjectRoot

Write-Host "[TalkTrace] Project root: $ProjectRoot" -ForegroundColor Cyan

# --- 1. Locate Python ----------------------------------------------------
$PythonCmd = $null
foreach ($candidate in @('py -3', 'python', 'python3')) {
    $parts = $candidate -split ' '
    $exe = $parts[0]
    $probe = Get-Command $exe -ErrorAction SilentlyContinue
    if ($probe) {
        try {
            & $exe @($parts[1..($parts.Length - 1)]) --version *> $null
            if ($LASTEXITCODE -eq 0) { $PythonCmd = $candidate; break }
        } catch { }
    }
}
if (-not $PythonCmd) {
    Write-Error "Python 3 not found. Install from https://www.python.org/downloads/ and re-run."
    exit 1
}
Write-Host "[TalkTrace] Using Python launcher: $PythonCmd" -ForegroundColor Cyan

# --- 2. Virtual environment ---------------------------------------------
$VenvDir = Join-Path $ProjectRoot '.venv'
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'

if ($Reinstall -and (Test-Path $VenvDir)) {
    Write-Host "[TalkTrace] Removing existing venv (-Reinstall)..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $VenvDir
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "[TalkTrace] Creating virtual environment in .venv ..." -ForegroundColor Cyan
    $parts = $PythonCmd -split ' '
    & $parts[0] @($parts[1..($parts.Length - 1)]) -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { Write-Error "venv creation failed"; exit 1 }
}

# --- 3. Install dependencies --------------------------------------------
$ReqFile = Join-Path $ProjectRoot 'requirements.txt'
if (-not (Test-Path $ReqFile)) {
    Write-Error "requirements.txt not found at $ReqFile"
    exit 1
}

$StampFile = Join-Path $VenvDir '.requirements.sha256'
$CurrentHash = (Get-FileHash $ReqFile -Algorithm SHA256).Hash
$StoredHash = if (Test-Path $StampFile) { Get-Content $StampFile -Raw } else { '' }

if ($StoredHash.Trim() -ne $CurrentHash) {
    Write-Host "[TalkTrace] Installing/upgrading dependencies ..." -ForegroundColor Cyan
    & $VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { Write-Error "pip upgrade failed"; exit 1 }
    & $VenvPython -m pip install -r $ReqFile
    if ($LASTEXITCODE -ne 0) { Write-Error "pip install failed"; exit 1 }
    $CurrentHash | Out-File -FilePath $StampFile -Encoding ascii -NoNewline
} else {
    Write-Host "[TalkTrace] Dependencies already installed (requirements.txt unchanged)." -ForegroundColor Green
}

# --- 4. Launch the app --------------------------------------------------
Write-Host "[TalkTrace] Starting Shiny app ... press Ctrl+C to stop." -ForegroundColor Cyan

# The app uses relative imports (`from .myfuncs import ...`) so it must
# be launched as a package. We call the package's `main()` function,
# which opens a native desktop window (via pywebview) by default.
$PyCode = if ($NoWindow) {
    'from talktrace_ai.app import main; main(open_window=False)'
} else {
    'from talktrace_ai.app import main; main()'
}

& $VenvPython -c $PyCode
exit $LASTEXITCODE
