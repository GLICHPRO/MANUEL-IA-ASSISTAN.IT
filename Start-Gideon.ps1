# ═══════════════════════════════════════════════════════════════════════════════
#  🤖 GIDEON 4.0 - Ultimate Unified AI System
#  Single Startup Point (PowerShell)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  G.I.D.E.O.N. = Generative Intelligence for Dynamic Executive Operations Network
#
#  This script starts:
#  - Backend API (FastAPI) on port 8001
#  - Frontend Server on port 3000
#
#  Usage:
#    .\Start-Gideon.ps1             # Start both servers
#    .\Start-Gideon.ps1 -Backend    # Start only backend
#    .\Start-Gideon.ps1 -Frontend   # Start only frontend
#    .\Start-Gideon.ps1 -NoBrowser  # Start without opening browser
#
# ═══════════════════════════════════════════════════════════════════════════════

param(
    [switch]$Backend,
    [switch]$Frontend,
    [switch]$NoBrowser,
    [switch]$Help
)

$Host.UI.RawUI.WindowTitle = "GIDEON 4.0 - Ultimate Unified AI"

# Colors
function Write-Banner {
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  🤖 GIDEON 4.0 - Ultimate Unified AI System" -ForegroundColor Green
    Write-Host "  G.I.D.E.O.N. = Generative Intelligence for Dynamic Executive Operations Network" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
}

function Write-VersionHistory {
    Write-Host "  ╔═══════════════════════════════════════════════════════════════════════════╗" -ForegroundColor DarkCyan
    Write-Host "  ║                         VERSION HISTORY                                    ║" -ForegroundColor DarkCyan
    Write-Host "  ╠═══════════════════════════════════════════════════════════════════════════╣" -ForegroundColor DarkCyan
    Write-Host "  ║  v1.0 - Basic Assistant       : Simple Q&A, basic commands                ║" -ForegroundColor Gray
    Write-Host "  ║  v2.0 - Brain Integration     : NLP, Memory, Reasoning, AI Providers      ║" -ForegroundColor Gray
    Write-Host "  ║  v3.0 - Cognitive Module      : Predictor, Analyzer, Simulator            ║" -ForegroundColor Gray
    Write-Host "  ║  v4.0 - Ultimate Unified      : All versions + Executive AI + Autonomy    ║" -ForegroundColor Yellow
    Write-Host "  ╚═══════════════════════════════════════════════════════════════════════════╝" -ForegroundColor DarkCyan
    Write-Host ""
}

function Write-Endpoints {
    Write-Host ""
    Write-Host "  📊 GIDEON 4.0 Endpoints:" -ForegroundColor Cyan
    Write-Host "     • Main API:     " -NoNewline; Write-Host "http://localhost:8001/api/gideon4" -ForegroundColor Green
    Write-Host "     • Health:       " -NoNewline; Write-Host "http://localhost:8001/api/gideon4/health" -ForegroundColor Green
    Write-Host "     • Status:       " -NoNewline; Write-Host "http://localhost:8001/api/gideon4/status" -ForegroundColor Green
    Write-Host "     • Process:      " -NoNewline; Write-Host "http://localhost:8001/api/gideon4/process" -ForegroundColor Green
    Write-Host "     • Analyze:      " -NoNewline; Write-Host "http://localhost:8001/api/gideon4/analyze" -ForegroundColor Green
    Write-Host "     • Predict:      " -NoNewline; Write-Host "http://localhost:8001/api/gideon4/predict" -ForegroundColor Green
    Write-Host "     • Execute:      " -NoNewline; Write-Host "http://localhost:8001/api/gideon4/execute" -ForegroundColor Green
    Write-Host "     • WebSocket:    " -NoNewline; Write-Host "ws://localhost:8001/api/gideon4/ws" -ForegroundColor Green
    Write-Host ""
    Write-Host "  🌐 Frontend:       " -NoNewline; Write-Host "http://localhost:3000/gideon_unified.html" -ForegroundColor Yellow
    Write-Host ""
}

if ($Help) {
    Write-Banner
    Write-Host "  Usage:" -ForegroundColor Cyan
    Write-Host "    .\Start-Gideon.ps1             # Start both servers"
    Write-Host "    .\Start-Gideon.ps1 -Backend    # Start only backend"
    Write-Host "    .\Start-Gideon.ps1 -Frontend   # Start only frontend"
    Write-Host "    .\Start-Gideon.ps1 -NoBrowser  # Start without opening browser"
    Write-Host ""
    exit 0
}

Write-Banner
Write-VersionHistory

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $ScriptDir) {
    $ScriptDir = Get-Location
}
Set-Location $ScriptDir

# Check virtual environment
$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "  ❌ Virtual environment not found!" -ForegroundColor Red
    Write-Host "     Please run: python -m venv .venv" -ForegroundColor Yellow
    Write-Host "     And then: pip install -r backend\requirements.txt" -ForegroundColor Yellow
    exit 1
}

# Determine what to start
$StartBackend = -not $Frontend -or $Backend
$StartFrontend = -not $Backend -or $Frontend

Write-Host "  [1/4] 🔧 Activating virtual environment..." -ForegroundColor Gray

# Stop existing Python processes
Write-Host "  [2/4] 🧹 Stopping existing Python processes..." -ForegroundColor Gray
Get-Process -Name python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

if ($StartBackend) {
    Write-Host "  [3/4] 🚀 Starting GIDEON 4.0 Backend (port 8001)..." -ForegroundColor Green
    
    $BackendPath = Join-Path $ScriptDir "backend\main.py"
    Start-Process -FilePath $VenvPython -ArgumentList $BackendPath -WorkingDirectory $ScriptDir -WindowStyle Normal
}

if ($StartFrontend) {
    Write-Host "  [4/4] 🌐 Starting Frontend Server (port 3000)..." -ForegroundColor Green
    Start-Sleep -Seconds 3
    
    $FrontendPath = Join-Path $ScriptDir "frontend\server.py"
    Start-Process -FilePath $VenvPython -ArgumentList $FrontendPath -WorkingDirectory $ScriptDir -WindowStyle Normal
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "  ✅ GIDEON 4.0 is starting!" -ForegroundColor Green

Write-Endpoints

Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Open browser after a delay
if (-not $NoBrowser) {
    Write-Host "  Opening browser in 5 seconds..." -ForegroundColor Gray
    Start-Sleep -Seconds 5
    Start-Process "http://localhost:3000/gideon_unified.html"
}

Write-Host ""
Write-Host "  Press Enter to open browser again, or Ctrl+C to exit." -ForegroundColor DarkGray
Read-Host
Start-Process "http://localhost:3000/gideon_unified.html"
