param(
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$environmentPath = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $environmentPath "Scripts\python.exe"
$condaPython = Join-Path $environmentPath "python.exe"
if (Test-Path $venvPython) {
    $python = $venvPython
} elseif (Test-Path $condaPython) {
    $python = $condaPython
} else {
    $python = $venvPython
}
if (-not (Test-Path $python)) {
    $systemPython = Get-Command python -ErrorAction SilentlyContinue
    $created = $false
    if ($null -ne $systemPython) {
        $versionText = & $systemPython.Source --version 2>&1
        if ($versionText -match "Python 3\.(10|11|12)") {
            & $systemPython.Source -m venv $environmentPath
            $created = $LASTEXITCODE -eq 0
        }
    }
    if (-not $created) {
        $launcher = Get-Command py -ErrorAction SilentlyContinue
        if ($null -ne $launcher) {
            & $launcher.Source -3.12 -m venv $environmentPath
            $created = $LASTEXITCODE -eq 0
        } else {
            $conda = Get-Command conda -ErrorAction SilentlyContinue
            if ($null -eq $conda) {
                throw "Python 3.10-3.12 is required. Install Python from https://www.python.org/downloads/ or Miniconda, then rerun this script."
            }
            & $conda.Source create --yes --prefix $environmentPath python=3.12
            $created = $LASTEXITCODE -eq 0
            $python = $condaPython
        }
    }
}

if (-not (Test-Path $python)) {
    throw "The virtual environment could not be created. Check that Python 3.10 or newer is installed."
}

& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip."
}

if ($Cpu) {
    & $python -m pip install -r requirements.txt
} else {
    # The CUDA wheel index is optional; pip will fall back to the regular index
    # when a CUDA wheel is unavailable on the current platform.
    & $python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
}
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install the project dependencies."
}

& $python -m ipykernel install --user --name eecs498-wi22 --display-name "Python (EECS 498 WI22)"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install the Jupyter kernel."
}
& $python scripts\check_setup.py
if ($LASTEXITCODE -ne 0) {
    throw "Environment validation failed."
}

if (Test-Path $condaPython) {
    Write-Host "Environment ready. Activate it with conda activate .\.venv"
} else {
    Write-Host "Environment ready. Activate it with .\.venv\Scripts\Activate.ps1"
}
Write-Host "Start notebooks with .\.venv\Scripts\jupyter lab"
