# build.ps1 — Compilacion completa de la tesis (pdflatex + biber)
# Uso:
#   .\build.ps1          — compilacion completa (limpia + pdflatex x3 + biber)
#   .\build.ps1 -Quick   — compilacion rapida (pdflatex x2, sin biber)
#   .\build.ps1 -Clean   — solo limpiar archivos auxiliares

param(
    [switch]$Quick,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$thesisDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $thesisDir

function Remove-AuxFiles {
    $extensions = @("*.aux", "*.bbl", "*.bcf", "*.blg", "*.log", "*.out",
                    "*.run.xml", "*.toc", "*.lof", "*.lot")
    foreach ($ext in $extensions) {
        Get-ChildItem -Path . -Filter $ext -ErrorAction SilentlyContinue |
            Remove-Item -Force
    }
    Get-ChildItem -Path chapters -Filter "*.aux" -ErrorAction SilentlyContinue |
        Remove-Item -Force
    Write-Host "[OK] Archivos auxiliares eliminados" -ForegroundColor Green
}

function Invoke-PdfLatex {
    param([string]$Pass)
    Write-Host "[...] pdflatex ($Pass)" -ForegroundColor Cyan
    $output = & pdflatex -interaction=nonstopmode main.tex 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] pdflatex fallo en $Pass" -ForegroundColor Red
        $output | Select-String "^!" | ForEach-Object { Write-Host $_.Line -ForegroundColor Red }
        Pop-Location
        exit 1
    }
    Write-Host "[OK] pdflatex ($Pass)" -ForegroundColor Green
}

function Invoke-Biber {
    Write-Host "[...] biber" -ForegroundColor Cyan
    $output = & biber main 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] biber fallo" -ForegroundColor Red
        $output | ForEach-Object { Write-Host $_ -ForegroundColor Red }
        Pop-Location
        exit 1
    }
    Write-Host "[OK] biber" -ForegroundColor Green
}

# --- Ejecucion ---

if ($Clean) {
    Remove-AuxFiles
    Pop-Location
    exit 0
}

if ($Quick) {
    Write-Host "=== Compilacion rapida (sin biber) ===" -ForegroundColor Yellow
    Invoke-PdfLatex "1/2"
    Invoke-PdfLatex "2/2"
} else {
    Write-Host "=== Compilacion completa ===" -ForegroundColor Yellow
    Remove-AuxFiles
    Invoke-PdfLatex "1/3"
    Invoke-Biber
    Invoke-PdfLatex "2/3"
    Invoke-PdfLatex "3/3"
}

$pdfPath = Join-Path $thesisDir "main.pdf"
if (Test-Path $pdfPath) {
    $size = [math]::Round((Get-Item $pdfPath).Length / 1MB, 2)
    Write-Host "=== Compilacion exitosa: main.pdf ($size MB) ===" -ForegroundColor Green
} else {
    Write-Host "=== ADVERTENCIA: main.pdf no encontrado ===" -ForegroundColor Red
}

Pop-Location
