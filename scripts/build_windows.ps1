<#
.SYNOPSIS
Builds the Windows standalone executable and installer for Vaani.
#>
$ErrorActionPreference = "Stop"

$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspace = Split-Path -Parent $workspace
$buildDir = Join-Path $workspace "build_env"
$releaseDir = Join-Path $workspace "release\windows"

Write-Host "=== Vaani Windows Build Script ==="

# 1. Check Python
$pythonVer = python --version
if ($pythonVer -notmatch "Python 3\.12") {
    Write-Warning "Requires Python 3.12+. Found: $pythonVer"
}

# 2. Venv
Write-Host "Creating build venv..."
if (Test-Path $buildDir) {
    Remove-Item -Recurse -Force $buildDir
}
python -m venv $buildDir
$pip = Join-Path $buildDir "Scripts\pip.exe"
$pyinstaller = Join-Path $buildDir "Scripts\pyinstaller.exe"

# 3. Install
Write-Host "Installing Vaani and dependencies..."
& $pip install --upgrade pip
& $pip install "$workspace\[stt,translate,voice,cuda]"
& $pip install pyinstaller

# 4. PyInstaller
Write-Host "Running PyInstaller..."
if (!(Test-Path $releaseDir)) {
    New-Item -ItemType Directory -Force -Path $releaseDir | Out-Null
}

& $pyinstaller --noconfirm --onedir --windowed `
    --name vaani `
    --add-data "$workspace\packaging\vaani.svg;packaging" `
    --hidden-import vaani.system.hardware `
    --hidden-import vaani.system.platform `
    --hidden-import vaani.system.profile `
    --hidden-import vaani.system.ollama_manager `
    --hidden-import vaani.ui.windows_console `
    --hidden-import vaani.audio.backend.windows_backend `
    --hidden-import vaani.devices.windows_manager `
    --distpath $releaseDir `
    "$workspace\src\vaani\ui\__main__.py"

# 5. Inno Setup
$iscc = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if (Test-Path $iscc) {
    Write-Host "Building installer with Inno Setup..."
    & $iscc "$workspace\packaging\windows\vaani_setup.iss"
} else {
    Write-Host "Inno Setup not found, skipping installer build."
}

# 6. Checksum
Write-Host "Generating checksums..."
$exePath = Join-Path $releaseDir "vaani\vaani.exe"
if (Test-Path $exePath) {
    $hash = Get-FileHash -Path $exePath -Algorithm SHA256
    $hash.Hash | Out-File (Join-Path $releaseDir "vaani_exe_sha256.txt")
}

Write-Host "Build complete!"
