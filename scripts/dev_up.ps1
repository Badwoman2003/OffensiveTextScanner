# Windows / PowerShell version of scripts/dev_up.sh
$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location ..

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "[dev_up] .env created from .env.example"
}

Write-Host "[dev_up] starting redis + backend + workers"
docker compose up -d redis backend worker-cpu flower

Write-Host "[dev_up] tailing backend logs (Ctrl+C to stop)"
docker compose logs -f backend worker-cpu
