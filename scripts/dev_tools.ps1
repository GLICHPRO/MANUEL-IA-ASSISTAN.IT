# GIDEON Dev Automation Script
# Uso: .\dev_tools.ps1 [comando]
# Comandi: start, stop, restart, test, quick, status, clean

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

$ProjectRoot = "C:\OneDrive\OneDrive - Technetpro\Desktop\gideon2.0"
$BackendDir = Join-Path $ProjectRoot "backend"
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Port = 8001
$Host_ = "127.0.0.1"

function Write-Status($msg, $color = "Cyan") {
    Write-Host "[$((Get-Date).ToString('HH:mm:ss'))] " -NoNewline -ForegroundColor DarkGray
    Write-Host $msg -ForegroundColor $color
}

function Test-ServerRunning {
    try {
        $response = Invoke-RestMethod -Uri "http://${Host_}:${Port}/health" -TimeoutSec 3 -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

function Stop-GideonServer {
    Write-Status "Stopping GIDEON server..." "Yellow"
    
    # Kill processes on port
    $connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    foreach ($conn in $connections) {
        Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
    }
    
    # Kill background jobs
    Get-Job | Where-Object { $_.Command -like "*uvicorn*" } | Remove-Job -Force -ErrorAction SilentlyContinue
    
    Start-Sleep -Seconds 1
    Write-Status "Server stopped" "Green"
}

function Start-GideonServer {
    if (Test-ServerRunning) {
        Write-Status "Server already running!" "Green"
        return
    }
    
    Stop-GideonServer
    Start-Sleep -Seconds 1
    
    Write-Status "Starting GIDEON server..." "Cyan"
    
    $job = Start-Job -ScriptBlock {
        param($python, $backend)
        Set-Location $backend
        & $python -m uvicorn main:app --host 127.0.0.1 --port 8001
    } -ArgumentList $VenvPython, $BackendDir
    
    # Wait for startup
    for ($i = 0; $i -lt 15; $i++) {
        Start-Sleep -Seconds 1
        if (Test-ServerRunning) {
            Write-Status "Server started successfully!" "Green"
            Write-Status "URL: http://${Host_}:${Port}" "White"
            return
        }
        Write-Host "." -NoNewline
    }
    
    Write-Status "Server failed to start!" "Red"
    Receive-Job -Job $job
}

function Get-ServerStatus {
    if (Test-ServerRunning) {
        $health = Invoke-RestMethod -Uri "http://${Host_}:${Port}/health" -TimeoutSec 5
        Write-Status "Server: ONLINE" "Green"
        Write-Host "  Status: $($health.status)"
        Write-Host "  Mode: $($health.mode)"
        Write-Host "  Services:"
        $health.services.PSObject.Properties | ForEach-Object {
            $icon = if ($_.Value) { "[OK]" } else { "[X]" }
            $color = if ($_.Value) { "Green" } else { "Red" }
            Write-Host "    $icon $($_.Name)" -ForegroundColor $color
        }
    } else {
        Write-Status "Server: OFFLINE" "Red"
    }
}

function Invoke-QuickTest {
    Write-Status "Running quick component test..." "Cyan"
    
    Set-Location $BackendDir
    & $VenvPython -c @"
import asyncio
from core.mode_manager import ModeManager
from core.emergency import EmergencySystem
from gideon import GideonCore
from jarvis import JarvisCore

tests = [
    ('ModeManager', lambda: ModeManager()),
    ('EmergencySystem', lambda: EmergencySystem()),
    ('GideonCore', lambda: GideonCore()),
    ('JarvisCore', lambda: JarvisCore())
]

print()
for name, factory in tests:
    try:
        obj = factory()
        print(f'  [OK] {name}')
    except Exception as e:
        print(f'  [X] {name}: {e}')
print()
"@
    
    Write-Status "Quick test complete" "Green"
}

function Invoke-FullTest {
    Write-Status "Running pytest suite..." "Cyan"
    Set-Location $ProjectRoot
    & $VenvPython -m pytest -q --tb=short
}

function Clear-Cache {
    Write-Status "Cleaning Python cache..." "Cyan"
    
    Get-ChildItem -Path $ProjectRoot -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path $ProjectRoot -Recurse -Filter "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue
    
    $pytestCache = Join-Path $ProjectRoot ".pytest_cache"
    if (Test-Path $pytestCache) {
        Remove-Item -Path $pytestCache -Recurse -Force -ErrorAction SilentlyContinue
    }
    
    Write-Status "Cache cleaned" "Green"
}

function Show-Help {
    Write-Host ""
    Write-Host "GIDEON Dev Tools" -ForegroundColor Cyan
    Write-Host "================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Yellow
    Write-Host "  start    - Start GIDEON server (background)"
    Write-Host "  stop     - Stop GIDEON server"
    Write-Host "  restart  - Restart server"
    Write-Host "  status   - Show server status"
    Write-Host "  test     - Run full pytest suite"
    Write-Host "  quick    - Quick component test"
    Write-Host "  clean    - Clean Python cache"
    Write-Host "  help     - Show this help"
    Write-Host ""
    Write-Host "Usage: .\dev_tools.ps1 [command]" -ForegroundColor Gray
    Write-Host ""
}

# Main
switch ($Command.ToLower()) {
    "start"   { Start-GideonServer }
    "stop"    { Stop-GideonServer }
    "restart" { Stop-GideonServer; Start-Sleep -Seconds 2; Start-GideonServer }
    "status"  { Get-ServerStatus }
    "test"    { Invoke-FullTest }
    "quick"   { Invoke-QuickTest }
    "clean"   { Clear-Cache }
    "help"    { Show-Help }
    default   { Write-Host "Unknown command: $Command" -ForegroundColor Red; Show-Help }
}
