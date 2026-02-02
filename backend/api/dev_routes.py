"""
🛠️ Development API Routes - GIDEON 3.0

API per automazioni di sviluppo e monitoraggio sistema
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
import asyncio

router = APIRouter(prefix="/dev", tags=["Development"])

# Import DevAutomations
try:
    from jarvis.dev_automations import DevAutomations, SystemLoadMonitor, get_dev_automations
    DEV_AVAILABLE = True
except ImportError:
    DEV_AVAILABLE = False
    DevAutomations = None
    SystemLoadMonitor = None

# Singleton instance
_dev_instance: Optional['DevAutomations'] = None
_load_monitor: Optional['SystemLoadMonitor'] = None

def get_dev() -> 'DevAutomations':
    global _dev_instance
    if not DEV_AVAILABLE:
        raise HTTPException(status_code=503, detail="DevAutomations not available")
    if _dev_instance is None:
        _dev_instance = get_dev_automations()
    return _dev_instance

def get_monitor() -> 'SystemLoadMonitor':
    global _load_monitor
    if not DEV_AVAILABLE:
        raise HTTPException(status_code=503, detail="SystemLoadMonitor not available")
    if _load_monitor is None:
        _load_monitor = SystemLoadMonitor()
    return _load_monitor


# ============ Models ============

class OperationRequest(BaseModel):
    operation: str
    force: bool = False
    options: Dict[str, Any] = {}


class TestRequest(BaseModel):
    pattern: Optional[str] = None
    quick: bool = True
    verbose: bool = False


# ============ System Load Endpoints ============

@router.get("/system/load")
async def get_system_load():
    """Ottiene il carico attuale del sistema"""
    monitor = get_monitor()
    load = monitor.get_system_load()
    return {
        "success": True,
        "load": load,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/system/can-execute/{operation}")
async def can_execute_operation(operation: str):
    """Verifica se un'operazione può essere eseguita dato il carico"""
    monitor = get_monitor()
    result = monitor.can_execute(operation)
    return {
        "operation": operation,
        "allowed": result.get("allowed", False),
        "reason": result.get("reason"),
        "load": result.get("load"),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/system/thresholds")
async def get_thresholds():
    """Ottiene le soglie di carico configurate"""
    monitor = get_monitor()
    return {
        "thresholds": monitor.THRESHOLDS,
        "operation_costs": monitor.OPERATION_COSTS
    }


# ============ Server Management ============

@router.post("/server/start")
async def start_server(port: int = 8001, force: bool = False):
    """Avvia il server backend"""
    dev = get_dev()
    result = await dev.start_server(port=port, force=force)
    return result


@router.post("/server/stop")
async def stop_server():
    """Ferma il server backend"""
    dev = get_dev()
    result = await dev.stop_server()
    return result


@router.post("/server/restart")
async def restart_server(port: int = 8001, force: bool = False):
    """Riavvia il server backend"""
    dev = get_dev()
    result = await dev.restart_server(port=port, force=force)
    return result


@router.get("/server/health")
async def server_health():
    """Controlla lo stato di salute del server"""
    dev = get_dev()
    result = await dev.health_check()
    return result


# ============ Testing ============

@router.post("/test/run")
async def run_tests(request: TestRequest):
    """Esegue i test"""
    dev = get_dev()
    if request.quick:
        result = await dev.run_quick_test(pattern=request.pattern)
    else:
        result = await dev.run_tests(pattern=request.pattern, verbose=request.verbose)
    return result


@router.get("/test/quick")
async def quick_test():
    """Esegue un test rapido del sistema"""
    dev = get_dev()
    result = await dev.run_quick_test()
    return result


# ============ Maintenance ============

@router.post("/clean/cache")
async def clean_cache(force: bool = False):
    """Pulisce la cache Python"""
    dev = get_dev()
    result = await dev.clean_cache(force=force)
    return result


@router.post("/clean/logs")
async def clean_logs(days: int = 7):
    """Pulisce i log vecchi"""
    dev = get_dev()
    result = await dev.clean_logs(days_old=days)
    return result


@router.post("/backup")
async def backup_project(force: bool = False):
    """Crea un backup del progetto"""
    dev = get_dev()
    result = await dev.backup_project(force=force)
    return result


# ============ Git Operations ============

@router.get("/git/status")
async def git_status():
    """Ottiene lo stato git"""
    dev = get_dev()
    result = await dev.git_status()
    return result


@router.post("/git/commit")
async def git_commit(message: str):
    """Crea un commit"""
    dev = get_dev()
    result = await dev.git_commit(message)
    return result


# ============ Development Workflows ============

@router.post("/workflow/startup")
async def dev_startup(force: bool = False):
    """Esegue lo startup completo di sviluppo"""
    dev = get_dev()
    result = await dev.dev_startup(force=force)
    return result


@router.post("/workflow/full-test")
async def full_test_suite(force: bool = False):
    """Esegue la suite completa di test"""
    dev = get_dev()
    result = await dev.full_test_suite(force=force)
    return result


@router.get("/status")
async def dev_status():
    """Stato completo del sistema di sviluppo"""
    dev = get_dev()
    monitor = get_monitor()
    
    load = monitor.get_system_load()
    health = await dev.health_check()
    
    return {
        "success": True,
        "system_load": load,
        "server_health": health,
        "available_operations": list(monitor.OPERATION_COSTS.keys()),
        "timestamp": datetime.now().isoformat()
    }
