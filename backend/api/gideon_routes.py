"""
🚀 GIDEON 3.0 System - API Routes INTEGRATE
Architettura: Jarvis (Supervisor) + Gideon (Cognitive) + Automation (Executive) + Dev Tools
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

# Router principale Gideon (Modulo Cognitivo)
gideon_router = APIRouter(prefix="/gideon", tags=["Gideon (Cognitive Module)"])

# Router Jarvis (Supervisor)
jarvis_router = APIRouter(prefix="/jarvis", tags=["Jarvis (Supervisor)"])

# Router Sistema e Automazione
system_router = APIRouter(prefix="/system", tags=["System & Automation"])

# ==================== DEV TOOLS INTEGRATION ====================
try:
    from jarvis.dev_automations import DevAutomations, SystemLoadMonitor, get_dev_automations
    DEV_AVAILABLE = True
except ImportError:
    DEV_AVAILABLE = False
    DevAutomations = None
    SystemLoadMonitor = None

_dev_instance = None
_load_monitor = None

def _get_dev():
    global _dev_instance
    if not DEV_AVAILABLE:
        return None
    if _dev_instance is None:
        _dev_instance = get_dev_automations()
    return _dev_instance

def _get_monitor():
    global _load_monitor
    if not DEV_AVAILABLE:
        return None
    if _load_monitor is None:
        _load_monitor = SystemLoadMonitor()
    return _load_monitor


# ==================== MODELS ====================

class ProcessRequest(BaseModel):
    """Richiesta di elaborazione"""
    text: str
    context: Optional[Dict[str, Any]] = None
    trace: bool = False  # Include cognitive trace

class ModeRequest(BaseModel):
    """Cambio modalità"""
    mode: str  # passive, copilot, pilot, executive

class LevelRequest(BaseModel):
    """Cambio livello risposta"""
    level: str  # normal, advanced

class ConfirmRequest(BaseModel):
    """Conferma/rifiuta azione pending"""
    decision_id: str
    confirm: bool

class ActionRequest(BaseModel):
    """Esecuzione azione diretta"""
    action: str
    params: Optional[Dict[str, Any]] = None
    bypass_security: bool = False

class AutomationRequest(BaseModel):
    """Creazione automazione"""
    name: str
    trigger_type: str  # schedule, event, condition
    trigger_config: Dict[str, Any]
    actions: List[Dict[str, Any]]

class EmergencyRequest(BaseModel):
    """Azione emergenza"""
    action: str  # kill, safe_mode, lockdown, resume
    reason: Optional[str] = None


# ==================== GLOBAL INSTANCES ====================

_orchestrator = None
_gideon_core = None
_jarvis_core = None
_automation_layer = None
_mode_manager = None
_emergency_system = None


def set_system_instances(orchestrator=None, gideon=None, jarvis=None, 
                         automation=None, mode_manager=None, emergency=None):
    """Imposta le istanze globali dal main.py"""
    global _orchestrator, _gideon_core, _jarvis_core, _automation_layer, _mode_manager, _emergency_system
    _orchestrator = orchestrator
    _gideon_core = gideon
    _jarvis_core = jarvis
    _automation_layer = automation
    _mode_manager = mode_manager
    _emergency_system = emergency


# ==================== ORCHESTRATOR ROUTES ====================

@gideon_router.post("/process")
async def process_input(request: ProcessRequest):
    """
    🧠 Pipeline cognitiva completa
    
    Flusso: Input → Intent → [Gideon Analysis] → Decision → Execute → Response
    """
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator non inizializzato")
    
    try:
        if request.trace:
            result = await _orchestrator.process_with_trace(request.text, request.context)
        else:
            result = await _orchestrator.process(request.text, request.context)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@gideon_router.post("/quick")
async def quick_command(request: ProcessRequest):
    """
    ⚡ Comando rapido (senza analisi approfondita)
    
    Per comandi semplici: time, date, greeting, help
    """
    if not _jarvis_core:
        raise HTTPException(status_code=503, detail="Jarvis non inizializzato")
    
    try:
        result = await _jarvis_core.quick_command(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@gideon_router.get("/status")
async def get_orchestrator_status():
    """📊 Stato dell'orchestratore e pipeline"""
    status = {
        "orchestrator": _orchestrator.get_status() if _orchestrator else None,
        "gideon": _gideon_core is not None,
        "jarvis": _jarvis_core.get_supervisor_status() if _jarvis_core else None,
        "mode": _mode_manager.get_mode_info() if _mode_manager else None,
        "emergency": _emergency_system.get_status() if _emergency_system else None,
        "timestamp": datetime.now().isoformat()
    }
    return status


@gideon_router.get("/history")
async def get_processing_history(limit: int = 20):
    """📜 Storico elaborazioni"""
    if not _orchestrator:
        return {"history": [], "error": "Orchestrator non inizializzato"}
    
    return {"history": _orchestrator.get_history(limit)}


# ==================== JARVIS CORE ROUTES ====================

@jarvis_router.post("/interpret")
async def interpret_intent(request: ProcessRequest):
    """
    🎯 Solo interpretazione intent (senza esecuzione)
    """
    if not _jarvis_core or not hasattr(_jarvis_core, 'interpreter'):
        raise HTTPException(status_code=503, detail="Interpreter non inizializzato")
    
    intent = _jarvis_core.interpreter.interpret(request.text, request.context)
    return intent.to_dict()


@jarvis_router.post("/decide")
async def make_decision(request: ProcessRequest):
    """
    ⚖️ Solo decisione (senza esecuzione)
    """
    if not _jarvis_core:
        raise HTTPException(status_code=503, detail="Jarvis non inizializzato")
    
    try:
        intent = _jarvis_core.interpreter.interpret(request.text, request.context)
        decision = await _jarvis_core.decision_maker.decide(intent.to_dict(), request.context)
        return decision.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@jarvis_router.post("/execute")
async def execute_action(request: ActionRequest):
    """
    ⚡ Esegue azione diretta
    """
    if not _jarvis_core:
        raise HTTPException(status_code=503, detail="Jarvis non inizializzato")
    
    action = {"action": request.action, "params": request.params or {}}
    
    try:
        result = await _jarvis_core.execute_direct(action, request.bypass_security)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@jarvis_router.get("/pending")
async def get_pending_actions():
    """📋 Azioni in attesa di conferma"""
    if not _jarvis_core:
        return {"pending": []}
    
    pending = _jarvis_core.get_pending_actions()
    decisions = _jarvis_core.decision_maker.get_pending_decisions() if hasattr(_jarvis_core, 'decision_maker') else []
    
    return {
        "pending_actions": pending,
        "pending_decisions": [d.to_dict() for d in decisions]
    }


@jarvis_router.post("/confirm")
async def confirm_or_reject(request: ConfirmRequest):
    """
    ✅/❌ Conferma o rifiuta azione pending
    """
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator non inizializzato")
    
    if request.confirm:
        result = await _orchestrator.confirm_pending(request.decision_id)
    else:
        result = await _orchestrator.reject_pending(request.decision_id)
    
    return result


@jarvis_router.get("/decisions/history")
async def get_decision_history(limit: int = 20):
    """📊 Storico decisioni"""
    if not _jarvis_core or not hasattr(_jarvis_core, 'decision_maker'):
        return {"history": [], "stats": {}}
    
    return {
        "history": _jarvis_core.decision_maker.get_history(limit),
        "stats": _jarvis_core.decision_maker.get_stats()
    }


# ==================== MODE MANAGER ROUTES ====================

@system_router.get("/mode")
async def get_current_mode():
    """🎛️ Modalità corrente"""
    if not _mode_manager:
        return {"mode": "unknown", "level": "normal"}
    
    return _mode_manager.get_mode_info()


@system_router.post("/mode")
async def set_mode(request: ModeRequest):
    """
    🔄 Cambia modalità operativa
    
    Modi: passive, copilot, pilot, executive
    """
    if not _mode_manager:
        raise HTTPException(status_code=503, detail="Mode manager non inizializzato")
    
    try:
        from core.mode_manager import OperatingMode
        mode_map = {
            "passive": OperatingMode.PASSIVE,
            "copilot": OperatingMode.COPILOT,
            "pilot": OperatingMode.PILOT,
            "executive": OperatingMode.EXECUTIVE
        }
        
        if request.mode.lower() not in mode_map:
            raise HTTPException(status_code=400, detail=f"Modalità non valida: {request.mode}")
        
        _mode_manager.set_mode(mode_map[request.mode.lower()])
        return _mode_manager.get_mode_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@system_router.post("/level")
async def set_response_level(request: LevelRequest):
    """
    📊 Cambia livello risposta
    
    Livelli: normal, advanced
    """
    if not _mode_manager:
        raise HTTPException(status_code=503, detail="Mode manager non inizializzato")
    
    try:
        from core.mode_manager import ResponseLevel
        level_map = {
            "normal": ResponseLevel.NORMAL,
            "advanced": ResponseLevel.ADVANCED
        }
        
        if request.level.lower() not in level_map:
            raise HTTPException(status_code=400, detail=f"Livello non valido: {request.level}")
        
        _mode_manager.set_response_level(level_map[request.level.lower()])
        return _mode_manager.get_mode_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== EMERGENCY ROUTES ====================

@system_router.post("/emergency")
async def emergency_action(request: EmergencyRequest):
    """
    🚨 Azioni di emergenza
    
    Azioni: kill, safe_mode, lockdown, resume
    """
    if not _emergency_system:
        raise HTTPException(status_code=503, detail="Emergency system non inizializzato")
    
    try:
        action_map = {
            "kill": lambda: _emergency_system.activate_kill_switch(request.reason or "Manual kill"),
            "safe_mode": lambda: _emergency_system.enter_safe_mode(request.reason or "Manual safe mode"),
            "lockdown": lambda: _emergency_system.lockdown(request.reason or "Manual lockdown"),
            "resume": lambda: _emergency_system.resume_normal()
        }
        
        if request.action not in action_map:
            raise HTTPException(status_code=400, detail=f"Azione non valida: {request.action}")
        
        result = await action_map[request.action]()
        return {"success": True, "result": result, "status": _emergency_system.get_status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@system_router.get("/emergency/status")
async def get_emergency_status():
    """🚨 Stato sistema emergenza"""
    if not _emergency_system:
        return {"active": False, "error": "Emergency system non inizializzato"}
    
    return _emergency_system.get_status()


# ==================== AUTOMATIONS ROUTES ====================

@jarvis_router.get("/automations")
async def get_automations():
    """🤖 Lista automazioni"""
    if not _jarvis_core or not hasattr(_jarvis_core, 'automator'):
        return {"automations": []}
    
    return {"automations": _jarvis_core.automator.list_automations()}


@jarvis_router.post("/automations")
async def create_automation(request: AutomationRequest):
    """➕ Crea nuova automazione"""
    if not _jarvis_core or not hasattr(_jarvis_core, 'automator'):
        raise HTTPException(status_code=503, detail="Automator non inizializzato")
    
    try:
        automation_id = await _jarvis_core.automator.create_automation(
            name=request.name,
            trigger_type=request.trigger_type,
            trigger_config=request.trigger_config,
            actions=request.actions
        )
        return {"success": True, "automation_id": automation_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@jarvis_router.delete("/automations/{automation_id}")
async def delete_automation(automation_id: str):
    """🗑️ Elimina automazione"""
    if not _jarvis_core or not hasattr(_jarvis_core, 'automator'):
        raise HTTPException(status_code=503, detail="Automator non inizializzato")
    
    success = await _jarvis_core.automator.delete_automation(automation_id)
    return {"success": success}


# ==================== SECURITY ROUTES ====================

@jarvis_router.get("/security/status")
async def get_security_status():
    """🔒 Stato sicurezza"""
    if not _jarvis_core or not hasattr(_jarvis_core, 'security'):
        return {"error": "Security non inizializzato"}
    
    return _jarvis_core.security.get_security_status()


@jarvis_router.post("/security/verify-pin")
async def verify_pin(pin: str):
    """🔐 Verifica PIN"""
    if not _jarvis_core or not hasattr(_jarvis_core, 'security'):
        raise HTTPException(status_code=503, detail="Security non inizializzato")
    
    valid = _jarvis_core.security.verify_pin(pin)
    return {"valid": valid}


# ==================== HEALTH CHECK ====================

@gideon_router.get("/health")
async def health_check():
    """❤️ Health check completo"""
    # Carico sistema integrato
    monitor = _get_monitor()
    system_load = monitor.get_system_load() if monitor else None
    
    return {
        "status": "healthy",
        "version": "3.0.0",
        "components": {
            "orchestrator": _orchestrator is not None,
            "gideon_core": _gideon_core is not None,
            "jarvis_core": _jarvis_core is not None,
            "mode_manager": _mode_manager is not None,
            "emergency": _emergency_system is not None,
            "dev_tools": DEV_AVAILABLE
        },
        "system_load": system_load,
        "timestamp": datetime.now().isoformat()
    }


# ==================== DEV TOOLS ROUTES (INTEGRATI) ====================

@system_router.get("/load")
async def get_system_load():
    """📊 Carico attuale del sistema (CPU, RAM, Disco)"""
    monitor = _get_monitor()
    if not monitor:
        raise HTTPException(status_code=503, detail="SystemLoadMonitor non disponibile")
    
    load = monitor.get_system_load()
    return {
        "success": True,
        "load": load,
        "timestamp": datetime.now().isoformat()
    }


@system_router.get("/can-execute/{operation}")
async def can_execute_operation(operation: str):
    """✅ Verifica se un'operazione può essere eseguita dato il carico"""
    monitor = _get_monitor()
    if not monitor:
        raise HTTPException(status_code=503, detail="SystemLoadMonitor non disponibile")
    
    result = monitor.can_execute(operation)
    return {
        "operation": operation,
        "allowed": result.get("allowed", False),
        "reason": result.get("reason"),
        "load": result.get("load"),
        "timestamp": datetime.now().isoformat()
    }


@system_router.get("/thresholds")
async def get_thresholds():
    """⚙️ Soglie di carico configurate"""
    monitor = _get_monitor()
    if not monitor:
        raise HTTPException(status_code=503, detail="SystemLoadMonitor non disponibile")
    
    return {
        "thresholds": monitor.THRESHOLDS,
        "operation_costs": monitor.OPERATION_COSTS
    }


@system_router.post("/server/restart")
async def restart_server(port: int = 8001, force: bool = False):
    """🔄 Riavvia il server backend"""
    dev = _get_dev()
    if not dev:
        raise HTTPException(status_code=503, detail="DevAutomations non disponibile")
    
    result = await dev.restart_server(port=port, force=force)
    return result


@system_router.get("/server/health")
async def server_health():
    """💓 Stato di salute del server"""
    dev = _get_dev()
    if not dev:
        raise HTTPException(status_code=503, detail="DevAutomations non disponibile")
    
    result = await dev.health_check()
    return result


@system_router.post("/test/run")
async def run_tests(pattern: str = None, quick: bool = True):
    """🧪 Esegue i test"""
    dev = _get_dev()
    if not dev:
        raise HTTPException(status_code=503, detail="DevAutomations non disponibile")
    
    if quick:
        result = await dev.run_quick_test(pattern=pattern)
    else:
        result = await dev.run_tests(pattern=pattern)
    return result


@system_router.post("/clean/cache")
async def clean_cache(force: bool = False):
    """🧹 Pulisce la cache Python"""
    dev = _get_dev()
    if not dev:
        raise HTTPException(status_code=503, detail="DevAutomations non disponibile")
    
    result = await dev.clean_cache(force=force)
    return result


@system_router.post("/backup")
async def backup_project(force: bool = False):
    """💾 Crea un backup del progetto"""
    dev = _get_dev()
    if not dev:
        raise HTTPException(status_code=503, detail="DevAutomations non disponibile")
    
    result = await dev.backup_project(force=force)
    return result


@system_router.get("/git/status")
async def git_status():
    """📂 Stato Git"""
    dev = _get_dev()
    if not dev:
        raise HTTPException(status_code=503, detail="DevAutomations non disponibile")
    
    result = await dev.git_status()
    return result


@system_router.post("/git/commit")
async def git_commit(message: str):
    """📝 Crea un commit Git"""
    dev = _get_dev()
    if not dev:
        raise HTTPException(status_code=503, detail="DevAutomations non disponibile")
    
    result = await dev.git_commit(message)
    return result


@system_router.get("/dev/status")
async def dev_status():
    """🛠️ Stato completo sistema di sviluppo"""
    dev = _get_dev()
    monitor = _get_monitor()
    
    load = monitor.get_system_load() if monitor else {"error": "monitor unavailable"}
    health = await dev.health_check() if dev else {"error": "dev unavailable"}
    
    return {
        "success": True,
        "dev_available": DEV_AVAILABLE,
        "system_load": load,
        "server_health": health,
        "available_operations": list(monitor.OPERATION_COSTS.keys()) if monitor else [],
        "timestamp": datetime.now().isoformat()
    }


# ==================== GIDEON TOOLS ROUTES ====================

class ToolRequest(BaseModel):
    """Richiesta esecuzione tool"""
    category: str
    action: str
    session_id: Optional[str] = None
    timestamp: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


@gideon_router.post("/tools/execute")
async def execute_tool(request: ToolRequest):
    """
    🛠️ Esecuzione tool GIDEON
    
    Categories: security, health, science, chemistry, archaeology, military, monitor, cyber
    """
    import platform
    import psutil
    import subprocess
    import json
    
    category = request.category
    action = request.action
    
    result = {
        "success": True,
        "category": category,
        "action": action,
        "timestamp": datetime.now().isoformat(),
        "data": {},
        "summary": "",
        "recommendations": []
    }
    
    try:
        # ========== SECURITY ==========
        if category == "security":
            if action == "full_scan":
                # Scansione sicurezza sistema
                processes = []
                for proc in psutil.process_iter(['pid', 'name', 'username', 'status']):
                    try:
                        processes.append(proc.info)
                    except:
                        pass
                
                # Connessioni di rete
                connections = []
                for conn in psutil.net_connections(kind='inet')[:50]:
                    connections.append({
                        "laddr": f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else "N/A",
                        "raddr": f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else "N/A",
                        "status": conn.status
                    })
                
                result["data"] = {
                    "total_processes": len(processes),
                    "active_connections": len(connections),
                    "sample_processes": processes[:20],
                    "sample_connections": connections[:10]
                }
                result["summary"] = f"Scansione completata: {len(processes)} processi, {len(connections)} connessioni attive"
                result["recommendations"] = ["Monitora le connessioni sconosciute", "Verifica processi con privilegi elevati"]
                
            elif action == "vuln_check":
                # Verifica vulnerabilità base
                checks = {
                    "firewall_active": True,  # Placeholder
                    "antivirus_active": True,
                    "updates_pending": False,
                    "open_ports": []
                }
                
                # Check porte aperte
                for conn in psutil.net_connections(kind='inet'):
                    if conn.status == 'LISTEN':
                        checks["open_ports"].append(conn.laddr.port if conn.laddr else 0)
                
                result["data"] = checks
                result["summary"] = f"Trovate {len(checks['open_ports'])} porte in ascolto"
                result["status"] = "warning" if len(checks["open_ports"]) > 10 else "success"
                
            elif action == "firewall":
                # Stato firewall (Windows)
                if platform.system() == "Windows":
                    try:
                        fw_status = subprocess.run(
                            ["netsh", "advfirewall", "show", "allprofiles", "state"],
                            capture_output=True, text=True, timeout=10
                        )
                        result["data"] = {"output": fw_status.stdout}
                        result["summary"] = "Stato firewall Windows recuperato"
                    except:
                        result["data"] = {"error": "Impossibile ottenere stato firewall"}
                else:
                    result["data"] = {"platform": platform.system(), "note": "Firewall check non disponibile"}
                    
            elif action == "audit":
                # Audit sistema
                result["data"] = {
                    "platform": platform.platform(),
                    "architecture": platform.architecture(),
                    "processor": platform.processor(),
                    "python_version": platform.python_version(),
                    "users_logged": len(psutil.users()),
                    "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
                }
                result["summary"] = "Audit di sistema completato"
        
        # ========== MONITOR ==========
        elif category == "monitor":
            if action == "system":
                cpu = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                result["data"] = {
                    "cpu_percent": cpu,
                    "memory_total_gb": round(memory.total / (1024**3), 2),
                    "memory_used_gb": round(memory.used / (1024**3), 2),
                    "memory_percent": memory.percent,
                    "disk_total_gb": round(disk.total / (1024**3), 2),
                    "disk_used_gb": round(disk.used / (1024**3), 2),
                    "disk_percent": disk.percent
                }
                result["summary"] = f"CPU: {cpu}%, RAM: {memory.percent}%, Disco: {disk.percent}%"
                
            elif action == "network":
                net = psutil.net_io_counters()
                result["data"] = {
                    "bytes_sent_mb": round(net.bytes_sent / (1024**2), 2),
                    "bytes_recv_mb": round(net.bytes_recv / (1024**2), 2),
                    "packets_sent": net.packets_sent,
                    "packets_recv": net.packets_recv,
                    "errors_in": net.errin,
                    "errors_out": net.errout
                }
                result["summary"] = f"Traffico: {result['data']['bytes_recv_mb']} MB ricevuti, {result['data']['bytes_sent_mb']} MB inviati"
                
            elif action == "performance":
                cpu_times = psutil.cpu_times_percent()
                result["data"] = {
                    "cpu_user": cpu_times.user,
                    "cpu_system": cpu_times.system,
                    "cpu_idle": cpu_times.idle,
                    "load_avg": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0],
                    "process_count": len(psutil.pids())
                }
                result["summary"] = f"{result['data']['process_count']} processi attivi"
                
            elif action == "logs":
                result["data"] = {
                    "log_sources": ["Windows Event Log", "Application Logs", "Security Logs"],
                    "note": "Per analisi dettagliata dei log, specifica la sorgente"
                }
                result["summary"] = "Fonti log disponibili identificate"
        
        # ========== CYBER ==========
        elif category == "cyber":
            if action == "threat_scan":
                # Scansione minacce base
                suspicious = []
                for proc in psutil.process_iter(['pid', 'name', 'exe']):
                    try:
                        info = proc.info
                        # Check processi senza path (potenzialmente sospetti)
                        if info['exe'] is None and info['name'] not in ['System', 'Registry', 'Idle']:
                            suspicious.append(info)
                    except:
                        pass
                
                result["data"] = {
                    "suspicious_processes": suspicious[:10],
                    "total_suspicious": len(suspicious)
                }
                result["summary"] = f"Trovati {len(suspicious)} processi potenzialmente sospetti"
                result["status"] = "warning" if len(suspicious) > 5 else "success"
                
            elif action == "intrusion":
                # Check connessioni insolite
                unusual = []
                for conn in psutil.net_connections(kind='inet'):
                    if conn.status == 'ESTABLISHED' and conn.raddr:
                        # Check porte alte non standard
                        if conn.raddr.port > 50000 or conn.laddr.port > 50000:
                            unusual.append({
                                "local": f"{conn.laddr.ip}:{conn.laddr.port}",
                                "remote": f"{conn.raddr.ip}:{conn.raddr.port}",
                                "pid": conn.pid
                            })
                
                result["data"] = {
                    "unusual_connections": unusual[:10],
                    "total_unusual": len(unusual)
                }
                result["summary"] = f"{len(unusual)} connessioni su porte non standard rilevate"
                
            elif action == "encryption":
                result["data"] = {
                    "available_algorithms": ["AES-256", "RSA-4096", "ChaCha20", "SHA-512"],
                    "ssl_version": "TLS 1.3",
                    "recommendation": "Usa AES-256 per dati sensibili"
                }
                result["summary"] = "Strumenti crittografia disponibili"
                
            elif action == "pentest":
                result["data"] = {
                    "scan_types": ["Port Scan", "Service Enumeration", "Vulnerability Assessment"],
                    "note": "Penetration test richiede autorizzazione esplicita",
                    "tools_available": ["nmap", "nikto", "sqlmap"]
                }
                result["summary"] = "Configurazione pentest pronta"
                result["recommendations"] = ["Esegui solo su sistemi autorizzati", "Documenta tutti i test"]
        
        # ========== HEALTH/SCIENCE/CHEMISTRY/ARCHAEOLOGY/MILITARY ==========
        # Questi richiedono elaborazione AI, restituiamo info base
        elif category in ["health", "science", "chemistry", "archaeology", "military"]:
            result["data"] = {
                "category": category,
                "action": action,
                "requires_ai": True,
                "message": f"Funzionalità {category}/{action} richiede elaborazione cognitiva GIDEON"
            }
            result["summary"] = f"Usa la chat per query dettagliate su {category}"
            # Trigger AI fallback nel frontend
            result["use_ai_fallback"] = True
        
        # ========== ANALYSIS ==========
        elif category == "analysis":
            if action == "text":
                result["data"] = {
                    "features": ["Analisi linguistica", "Estrazione keywords", "Conteggio parole", "Readability score"],
                    "instruction": "Incolla il testo da analizzare nella chat"
                }
                result["summary"] = "Pronto per analisi testuale"
                result["use_ai_fallback"] = True
                
            elif action == "sentiment":
                result["data"] = {
                    "features": ["Sentiment positivo/negativo/neutro", "Emozioni rilevate", "Tono generale"],
                    "instruction": "Incolla il testo da analizzare"
                }
                result["summary"] = "Analisi sentiment pronta"
                result["use_ai_fallback"] = True
                
            elif action == "summary":
                result["data"] = {
                    "features": ["Riassunto automatico", "Punti chiave", "Abstract"],
                    "instruction": "Incolla il testo da riassumere"
                }
                result["summary"] = "Funzione riassunto attiva"
                result["use_ai_fallback"] = True
                
            elif action == "translate":
                result["data"] = {
                    "languages": ["Italiano", "English", "Español", "Français", "Deutsch", "中文", "日本語", "العربية"],
                    "instruction": "Incolla il testo e specifica la lingua target"
                }
                result["summary"] = "Traduttore multilingua pronto"
                result["use_ai_fallback"] = True
                
            elif action == "code":
                result["data"] = {
                    "features": ["Analisi qualità codice", "Bug detection", "Security scan", "Ottimizzazioni"],
                    "languages_supported": ["Python", "JavaScript", "Java", "C++", "Go", "Rust"]
                }
                result["summary"] = "Analizzatore codice pronto"
                result["use_ai_fallback"] = True
        
        # ========== UTILITIES ==========
        elif category == "utilities":
            if action == "calculator":
                result["data"] = {
                    "features": ["Calcoli matematici", "Equazioni", "Conversioni", "Statistiche"],
                    "examples": ["2+2", "sqrt(144)", "sin(45)", "15% di 200"]
                }
                result["summary"] = "Calcolatrice scientifica pronta - scrivi l'operazione in chat"
                result["use_ai_fallback"] = True
                
            elif action == "converter":
                result["data"] = {
                    "categories": {
                        "lunghezza": ["m", "km", "mi", "ft", "in"],
                        "peso": ["kg", "g", "lb", "oz"],
                        "temperatura": ["°C", "°F", "K"],
                        "valuta": ["EUR", "USD", "GBP", "JPY"],
                        "dati": ["KB", "MB", "GB", "TB"]
                    }
                }
                result["summary"] = "Convertitore universale - specifica valore e unità"
                result["use_ai_fallback"] = True
                
            elif action == "timer":
                result["data"] = {
                    "active_timers": [],
                    "features": ["Timer", "Sveglia", "Promemoria", "Countdown"],
                    "instruction": "Di' 'imposta timer 5 minuti' o 'ricordami alle 15:00'"
                }
                result["summary"] = "Sistema timer pronto"
                result["use_ai_fallback"] = True
                
            elif action == "weather":
                result["data"] = {
                    "instruction": "Specifica la città per il meteo",
                    "features": ["Temperatura attuale", "Previsioni 7 giorni", "Umidità", "Vento"]
                }
                result["summary"] = "Servizio meteo - indica la località"
                result["use_ai_fallback"] = True
                
            elif action == "qrcode":
                result["data"] = {
                    "features": ["Genera QR Code", "Leggi QR Code", "Formati: URL, testo, vCard, WiFi"],
                    "instruction": "Specifica il contenuto per il QR Code"
                }
                result["summary"] = "Generatore QR Code pronto"
                result["use_ai_fallback"] = True
        
        # ========== AI TOOLS ==========
        elif category == "ai":
            if action == "image_gen":
                result["data"] = {
                    "models": ["DALL-E 3", "Stable Diffusion", "Midjourney style"],
                    "instruction": "Descrivi l'immagine da generare"
                }
                result["summary"] = "Generatore immagini AI - descrivi cosa vuoi creare"
                result["use_ai_fallback"] = True
                
            elif action == "image_analyze":
                result["data"] = {
                    "features": ["Riconoscimento oggetti", "OCR", "Descrizione scena", "Face detection"],
                    "instruction": "Carica un'immagine per l'analisi"
                }
                result["summary"] = "Analizzatore immagini pronto"
                result["use_ai_fallback"] = True
                
            elif action == "voice":
                result["data"] = {
                    "features": ["Speech-to-Text", "Text-to-Speech", "Comandi vocali"],
                    "status": "Attivo - usa il pulsante microfono"
                }
                result["summary"] = "Assistente vocale attivo"
                
            elif action == "reasoning":
                result["data"] = {
                    "features": ["Chain of Thought", "Multi-step reasoning", "Problem decomposition"],
                    "instruction": "Poni una domanda complessa per il deep reasoning"
                }
                result["summary"] = "Deep reasoning mode - poni la tua domanda"
                result["use_ai_fallback"] = True
                
            elif action == "creative":
                result["data"] = {
                    "modes": ["Poesia", "Racconto", "Script", "Lyrics", "Blog post"],
                    "instruction": "Indica il tipo di contenuto e il tema"
                }
                result["summary"] = "Scrittura creativa - indica genere e argomento"
                result["use_ai_fallback"] = True
        
        # ========== DATA ==========
        elif category == "data":
            if action == "export":
                # Esporta la conversazione corrente
                result["data"] = {
                    "formats": ["JSON", "TXT", "Markdown", "PDF"],
                    "instruction": "La chat verrà esportata nel formato scelto"
                }
                result["summary"] = "Esportazione chat - scegli il formato"
                
            elif action == "import":
                result["data"] = {
                    "supported_formats": ["JSON", "CSV", "TXT"],
                    "instruction": "Carica un file per importare dati"
                }
                result["summary"] = "Import dati pronto"
                
            elif action == "backup":
                # Crea backup
                import os
                backup_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'backups')
                os.makedirs(backup_dir, exist_ok=True)
                
                result["data"] = {
                    "backup_location": backup_dir,
                    "last_backup": "N/A",
                    "auto_backup": True
                }
                result["summary"] = "Sistema backup configurato"
                
            elif action == "history":
                result["data"] = {
                    "features": ["Cronologia conversazioni", "Ricerca messaggi", "Filtri data"],
                    "instruction": "Accedi alla cronologia completa"
                }
                result["summary"] = "Cronologia disponibile"
                
            elif action == "settings":
                result["data"] = {
                    "categories": ["Aspetto", "Notifiche", "Privacy", "AI Preferences", "Shortcuts"],
                    "instruction": "Seleziona la categoria da configurare"
                }
                result["summary"] = "Pannello impostazioni"
        
        else:
            result["success"] = False
            result["summary"] = f"Categoria '{category}' non riconosciuta"
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "category": category,
            "action": action,
            "timestamp": datetime.now().isoformat()
        }


@gideon_router.get("/tools/categories")
async def get_tool_categories():
    """📋 Lista categorie e azioni disponibili"""
    return {
        "categories": {
            "security": {
                "name": "Sicurezza",
                "icon": "🔒",
                "actions": ["full_scan", "vuln_check", "firewall", "audit"],
                "advanced_actions": ["predictive_risk_mapping", "anomaly_narrator", "defensive_scenario_simulator"]
            },
            "health": {
                "name": "Sanità",
                "icon": "💊",
                "actions": ["diagnostics", "medications", "wellness", "emergency"],
                "requires_ai": True
            },
            "science": {
                "name": "Scienza",
                "icon": "🔬",
                "actions": ["research", "physics", "biology", "simulation"],
                "advanced_actions": ["molecular_pattern_validator", "environmental_contamination_scan", "scientific_cross_check"],
                "requires_ai": True,
                "safe_mode": True
            },
            "chemistry": {
                "name": "Chimica",
                "icon": "🧪",
                "actions": ["compounds", "reactions", "periodic", "molecular"],
                "requires_ai": True,
                "safe_mode": True
            },
            "archaeology": {
                "name": "Archeologia",
                "icon": "🏛️",
                "actions": ["artifacts", "dating", "civilizations", "sites"],
                "advanced_actions": ["predictive_reconstruction", "temporal_layer_fusion", "authenticity_risk_assessment"],
                "requires_ai": True
            },
            "military": {
                "name": "Militare",
                "icon": "⚔️",
                "actions": ["strategy", "intel", "logistics", "defense"],
                "requires_ai": True
            },
            "monitor": {
                "name": "Monitoraggio",
                "icon": "📊",
                "actions": ["system", "network", "performance", "logs"]
            },
            "cyber": {
                "name": "Cybersecurity",
                "icon": "🛡️",
                "actions": ["threat_scan", "intrusion", "encryption", "pentest"],
                "advanced_actions": ["behavioral_baseline_builder", "incident_explainability_engine", "supply_chain_trust_scanner"]
            },
            "analysis": {
                "name": "Analisi",
                "icon": "📈",
                "actions": ["text", "sentiment", "summary", "translate", "code"],
                "requires_ai": True
            },
            "utilities": {
                "name": "Utilità",
                "icon": "🛠️",
                "actions": ["calculator", "converter", "timer", "weather", "qrcode"]
            },
            "ai": {
                "name": "AI Tools",
                "icon": "🤖",
                "actions": ["image_gen", "image_analyze", "voice", "reasoning", "creative"],
                "requires_ai": True
            },
            "data": {
                "name": "Dati",
                "icon": "💾",
                "actions": ["export", "import", "backup", "history", "settings"]
            },
            "core": {
                "name": "Gideon Core",
                "icon": "🧠",
                "actions": ["multi_tool_reasoning", "confidence_weighted_output", "human_override_gate"],
                "advanced_actions": ["audit_trail", "bias_and_drift_monitor", "failsafe_trigger"]
            }
        },
        "timestamp": datetime.now().isoformat()
    }


# ==================== GIDEON ADVANCED TOOLS ROUTES ====================

class AdvancedToolRequest(BaseModel):
    """Richiesta esecuzione tool avanzato"""
    tool: str  # security, cyber, science, archaeology, core
    action: str
    params: Optional[Dict[str, Any]] = None


@gideon_router.post("/tools/advanced/execute")
async def execute_advanced_tool(request: AdvancedToolRequest):
    """
    🛠️ Esecuzione tool AVANZATI GIDEON
    
    Tools:
    - security: predictive_risk_mapping, anomaly_narrator, defensive_scenario_simulator
    - cyber: behavioral_baseline_builder, incident_explainability_engine, supply_chain_trust_scanner
    - science: molecular_pattern_validator, environmental_contamination_scan, scientific_cross_check
    - archaeology: predictive_reconstruction, temporal_layer_fusion, authenticity_risk_assessment
    - core: multi_tool_reasoning, confidence_weighted_output, human_override_gate
    """
    
    try:
        # Import tools
        from tools import (
            get_security_tool, get_cyber_tool, get_science_tool,
            get_archaeology_tool, get_gideon_core
        )
        
        params = request.params or {}
        tool_name = request.tool.lower()
        action = request.action
        
        # ========== SECURITY TOOL ==========
        if tool_name == "security":
            security = get_security_tool()
            
            if action == "predictive_risk_mapping":
                result = await security.predictive_risk_mapping(
                    target_system=params.get("target_system", "general"),
                    time_horizon=params.get("time_horizon", "24h")
                )
            elif action == "anomaly_narrator":
                result = await security.anomaly_narrator(
                    anomaly_type=params.get("anomaly_type", "general"),
                    audience=params.get("audience", "technical")
                )
            elif action == "defensive_scenario_simulator":
                result = await security.defensive_scenario_simulator(
                    scenario_type=params.get("scenario_type", "infrastructure_failure"),
                    complexity=params.get("complexity", "medium")
                )
            else:
                return {"success": False, "error": f"Action '{action}' not found in security tool"}
        
        # ========== CYBER TOOL ==========
        elif tool_name == "cyber":
            cyber = get_cyber_tool()
            
            if action == "behavioral_baseline_builder":
                result = await cyber.behavioral_baseline_builder(
                    target_type=params.get("target_type", "network"),
                    observation_period=params.get("observation_period", "24h")
                )
            elif action == "incident_explainability_engine":
                result = await cyber.incident_explainability_engine(
                    incident_type=params.get("incident_type", "unauthorized_access"),
                    severity=params.get("severity", "medium"),
                    audience=params.get("audience", "all")
                )
            elif action == "supply_chain_trust_scanner":
                result = await cyber.supply_chain_trust_scanner(
                    target=params.get("target", "dependencies"),
                    depth=params.get("depth", "shallow")
                )
            else:
                return {"success": False, "error": f"Action '{action}' not found in cyber tool"}
        
        # ========== SCIENCE TOOL ==========
        elif tool_name == "science":
            science = get_science_tool()
            
            if action == "molecular_pattern_validator":
                result = await science.molecular_pattern_validator(
                    molecule_query=params.get("molecule_query", "water"),
                    analysis_type=params.get("analysis_type", "stability")
                )
            elif action == "environmental_contamination_scan":
                result = await science.environmental_contamination_scan(
                    medium=params.get("medium", "air"),
                    sensor_data=params.get("sensor_data"),
                    location=params.get("location", "unknown")
                )
            elif action == "scientific_cross_check":
                result = await science.scientific_cross_check(
                    claim=params.get("claim", ""),
                    field=params.get("field", "general"),
                    strictness=params.get("strictness", "moderate")
                )
            else:
                return {"success": False, "error": f"Action '{action}' not found in science tool"}
        
        # ========== ARCHAEOLOGY TOOL ==========
        elif tool_name == "archaeology":
            archaeology = get_archaeology_tool()
            
            if action == "predictive_reconstruction":
                result = await archaeology.predictive_reconstruction(
                    artifact_description=params.get("artifact_description", "unknown artifact"),
                    known_data=params.get("known_data", {}),
                    reconstruction_type=params.get("reconstruction_type", "visual")
                )
            elif action == "temporal_layer_fusion":
                result = await archaeology.temporal_layer_fusion(
                    subject=params.get("subject", "historical site"),
                    data_layers=params.get("data_layers", {}),
                    time_range=params.get("time_range", "all")
                )
            elif action == "authenticity_risk_assessment":
                result = await archaeology.authenticity_risk_assessment(
                    artifact_id=params.get("artifact_id", "artifact_001"),
                    artifact_data=params.get("artifact_data", {}),
                    assessment_level=params.get("assessment_level", "standard")
                )
            else:
                return {"success": False, "error": f"Action '{action}' not found in archaeology tool"}
        
        # ========== CORE TOOL ==========
        elif tool_name == "core":
            core = get_gideon_core()
            
            if action == "multi_tool_reasoning":
                result = await core.multi_tool_reasoning(
                    query=params.get("query", ""),
                    tools_to_consult=params.get("tools_to_consult", ["security", "analysis"]),
                    comparison_mode=params.get("comparison_mode", "synthesize")
                )
            elif action == "confidence_weighted_output":
                result = await core.confidence_weighted_output(
                    content=params.get("content", ""),
                    analysis_type=params.get("analysis_type", "general"),
                    sources=params.get("sources", [])
                )
            elif action == "human_override_gate":
                result = await core.human_override_gate(
                    action=params.get("gate_action", "unknown"),
                    action_details=params.get("action_details", {}),
                    criticality=params.get("criticality", "auto")
                )
            elif action == "audit_trail":
                result = await core.get_audit_trail(
                    filter_action=params.get("filter_action"),
                    limit=params.get("limit", 100)
                )
            elif action == "bias_and_drift_monitor":
                result = await core.bias_and_drift_monitor(
                    check_type=params.get("check_type", "full")
                )
            elif action == "failsafe_trigger":
                result = await core.failsafe_trigger(
                    reason=params.get("reason", "Manual trigger"),
                    context=params.get("context", {})
                )
            else:
                return {"success": False, "error": f"Action '{action}' not found in core tool"}
        
        else:
            return {"success": False, "error": f"Tool '{tool_name}' not found"}
        
        return result
        
    except ImportError as ie:
        return {
            "success": False,
            "error": f"Tool module not available: {str(ie)}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "tool": request.tool,
            "action": request.action,
            "timestamp": datetime.now().isoformat()
        }


@gideon_router.get("/tools/advanced/capabilities")
async def get_advanced_capabilities():
    """📋 Capacità avanzate dei tool GIDEON"""
    
    try:
        from tools import get_tool_capabilities
        capabilities = get_tool_capabilities()
        return {
            "success": True,
            "capabilities": capabilities,
            "timestamp": datetime.now().isoformat()
        }
    except ImportError:
        # Fallback se il modulo non è disponibile
        return {
            "success": True,
            "capabilities": {
                "security": {
                    "name": "Security Tool",
                    "icon": "🔒",
                    "actions": ["predictive_risk_mapping", "anomaly_narrator", "defensive_scenario_simulator"],
                    "description": "Physical & Infrastructure Defense"
                },
                "cyber": {
                    "name": "Cyber Tool",
                    "icon": "🛡️",
                    "actions": ["behavioral_baseline_builder", "incident_explainability_engine", "supply_chain_trust_scanner"],
                    "description": "Defensive AI-SOC"
                },
                "science": {
                    "name": "Science Tool (SAFE)",
                    "icon": "🧬",
                    "actions": ["molecular_pattern_validator", "environmental_contamination_scan", "scientific_cross_check"],
                    "description": "Health/Chemistry Analysis - NO SYNTHESIS",
                    "safety_mode": True
                },
                "archaeology": {
                    "name": "Archaeology Tool",
                    "icon": "🏛️",
                    "actions": ["predictive_reconstruction", "temporal_layer_fusion", "authenticity_risk_assessment"],
                    "description": "Digital Heritage Analysis"
                },
                "core": {
                    "name": "Gideon Core",
                    "icon": "🧠",
                    "actions": ["multi_tool_reasoning", "confidence_weighted_output", "human_override_gate", "audit_trail", "bias_and_drift_monitor", "failsafe_trigger"],
                    "description": "Central Reasoning & Safety Systems"
                }
            },
            "timestamp": datetime.now().isoformat()
        }
