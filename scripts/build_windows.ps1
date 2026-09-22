<#
.SYNOPSIS
Builds the Windows standalone executable and installer for Vaani.
#>
$ErrorActionPreference = "Stop"

$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspace = Split-Path -Parent $workspace
$buildDir = Join-Path $workspace "build_env"
$releaseDir = Join-Path $workspace "release\windows"

Write-Host "=== Vaani Windows Build Script (v1.0.1) ==="

# 1. Check Python
$pythonVer = python --version
if ($LASTEXITCODE -ne 0) {
    throw "Python was not found in PATH."
}
Write-Host "Detected Python: $pythonVer"

# 2. Venv creation
Write-Host "Creating build virtual environment..."
if (Test-Path $buildDir) {
    Remove-Item -Recurse -Force $buildDir
}
python -m venv $buildDir
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create build virtual environment."
}

$pyExe = Join-Path $buildDir "Scripts\python.exe"
$pyinstaller = Join-Path $buildDir "Scripts\pyinstaller.exe"

# 3. Upgrade pip and install dependencies
Write-Host "Upgrading pip..."
& $pyExe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip in build environment."
}

Write-Host "Installing Vaani and build dependencies..."
& $pyExe -m pip install "$workspace\[stt,translate,voice,cuda,windows,screen]" pyinstaller
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install dependencies."
}

# 4. Run PyInstaller
Write-Host "Running PyInstaller..."
if (!(Test-Path $releaseDir)) {
    New-Item -ItemType Directory -Force -Path $releaseDir | Out-Null
}

$pyinstallerArgs = @(
    "--noconfirm",
    "--onedir",
    "--windowed",
    "--name", "vaani",
    "--icon", "$workspace\packaging\vaani.ico",
    "--add-data", "$workspace\packaging\vaani.svg;packaging",
    "--add-data", "$workspace\packaging\vaani.ico;packaging",
    "--hidden-import", "vaani.system.hardware",
    "--hidden-import", "vaani.system.platform",
    "--hidden-import", "vaani.system.profile",
    "--hidden-import", "vaani.system.ollama_manager",
    "--hidden-import", "vaani.system.logging",
    "--hidden-import", "vaani.ui.windows_console",
    "--hidden-import", "vaani.audio.backend.windows_backend",
    "--hidden-import", "vaani.devices.windows_manager",
    "--hidden-import", "vaani.diagnostics.startup",
    "--hidden-import", "vaani.vision.screen_recording",
    "--distpath", $releaseDir,
    "$workspace\src\vaani\ui\__main__.py"
)

& $pyinstaller $pyinstallerArgs
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller execution failed with exit code $LASTEXITCODE"
}

$exePath = Join-Path $releaseDir "vaani\vaani.exe"
if (!(Test-Path $exePath)) {
    throw "Expected executable not found at: $exePath"
}
Write-Host "PyInstaller executable built successfully: $exePath"

# 5. Inno Setup Compilation
Write-Host "Checking for Inno Setup compiler (ISCC)..."
$isccCandidates = @(
    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
    "C:\Program Files\Inno Setup 6\ISCC.exe"
)

$isccPath = $null
foreach ($candidate in $isccCandidates) {
    if (Test-Path $candidate) {
        $isccPath = $candidate
        break
    }
}
if ($null -eq $isccPath) {
    $cmdIscc = Get-Command iscc.exe -ErrorAction SilentlyContinue
    if ($cmdIscc) {
        $isccPath = $cmdIscc.Source
    }
}

if ($isccPath) {
    Write-Host "Compiling installer with Inno Setup using: $isccPath"
    $issFile = Join-Path $workspace "packaging\windows\vaani_setup.iss"
    & $isccPath $issFile
    if ($LASTEXITCODE -ne 0) {
        throw "Inno Setup compilation failed with exit code $LASTEXITCODE"
    }

    $installerPath = Join-Path $releaseDir "vaani_installer.exe"
    if (!(Test-Path $installerPath)) {
        throw "Expected installer not found at: $installerPath"
    }
    Write-Host "Inno Setup installer created successfully: $installerPath"

    $instHash = Get-FileHash -Path $installerPath -Algorithm SHA256
    $instHash.Hash | Out-File (Join-Path $releaseDir "vaani_installer_sha256.txt")
} else {
    Write-Warning "Inno Setup (ISCC.exe) was not found. Skipping installer creation."
}

# 6. Checksums
Write-Host "Generating executable checksum..."
$exeHash = Get-FileHash -Path $exePath -Algorithm SHA256
$exeHash.Hash | Out-File (Join-Path $releaseDir "vaani_exe_sha256.txt")

Write-Host "=== Windows Build Completed Successfully! ==="
