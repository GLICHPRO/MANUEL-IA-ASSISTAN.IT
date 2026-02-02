"""
🛠️ JARVIS CORE - Development Automations
Automazioni specifiche per lo sviluppo e debug di GIDEON
Con sistema di throttling dinamico basato sul carico
"""

import asyncio
import subprocess
import os
import sys
from datetime import datetime
from typing import Dict, Optional, List
from pathlib import Path

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class SystemLoadMonitor:
    """
    Monitora il carico del sistema e decide se permettere operazioni pesanti
    """
    
    # Soglie di carico (personalizzabili)
    THRESHOLDS = {
        "cpu_critical": 95,      # CPU > 95% = critico
        "cpu_high": 85,          # CPU > 85% = alto
        "cpu_medium": 60,        # CPU > 60% = medio
        "memory_critical": 95,   # RAM > 95% = critico
        "memory_high": 90,       # RAM > 90% = alto
        "memory_medium": 80,     # RAM > 80% = medio
    }
    
    # Costi delle operazioni (1-10)
    OPERATION_COSTS = {
        "start_server": 7,
        "stop_server": 2,
        "restart_server": 8,
        "run_tests": 9,
        "run_quick_test": 4,
        "clean_cache": 3,
        "backup_project": 8,
        "git_status": 1,
        "health_check": 1,
        "dev_startup": 10,
        "full_test_suite": 10,
    }
    
    def __init__(self):
        self.last_check = None
        self.cached_load = None
        self.cache_duration = 2  # secondi
    
    def get_system_load(self) -> Dict:
        """Ottiene il carico attuale del sistema"""
        if not PSUTIL_AVAILABLE:
            return {"cpu": 0, "memory": 0, "available": True, "reason": "psutil non disponibile"}
        
        # Usa cache se recente
        now = datetime.now()
        if self.last_check and self.cached_load:
            elapsed = (now - self.last_check).total_seconds()
            if elapsed < self.cache_duration:
                return self.cached_load
        
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        net_io = psutil.net_io_counters()
        
        # Calcola network speed (approssimato)
        download_kbs = getattr(self, '_last_bytes_recv', 0)
        upload_kbs = getattr(self, '_last_bytes_sent', 0)
        if hasattr(self, '_last_net_check'):
            elapsed_net = (now - self._last_net_check).total_seconds()
            if elapsed_net > 0:
                download_kbs = (net_io.bytes_recv - getattr(self, '_last_bytes_recv', net_io.bytes_recv)) / 1024 / elapsed_net
                upload_kbs = (net_io.bytes_sent - getattr(self, '_last_bytes_sent', net_io.bytes_sent)) / 1024 / elapsed_net
        
        self._last_bytes_recv = net_io.bytes_recv
        self._last_bytes_sent = net_io.bytes_sent
        self._last_net_check = now
        
        self.cached_load = {
            # Formato legacy
            "cpu": cpu_percent,
            "memory": memory.percent,
            "cpu_level": self._get_level(cpu_percent, "cpu"),
            "memory_level": self._get_level(memory.percent, "memory"),
            # Formato esteso per frontend
            "cpu_percent": cpu_percent,
            "cpu_cores": psutil.cpu_count(),
            "cpu_freq": f"{psutil.cpu_freq().current:.0f} MHz" if psutil.cpu_freq() else "N/A",
            "ram_percent": memory.percent,
            "ram_used_gb": round(memory.used / (1024**3), 1),
            "ram_total_gb": round(memory.total / (1024**3), 1),
            "disk_percent": disk.percent,
            "disk_used_gb": round(disk.used / (1024**3), 0),
            "disk_total_gb": round(disk.total / (1024**3), 0),
            "network_download_kbs": round(max(0, download_kbs), 0),
            "network_upload_kbs": round(max(0, upload_kbs), 0),
            "timestamp": now.isoformat()
        }
        self.last_check = now
        
        return self.cached_load
    
    def _get_level(self, value: float, metric: str) -> str:
        """Determina il livello di carico"""
        if value >= self.THRESHOLDS[f"{metric}_critical"]:
            return "critical"
        elif value >= self.THRESHOLDS[f"{metric}_high"]:
            return "high"
        elif value >= self.THRESHOLDS[f"{metric}_medium"]:
            return "medium"
        return "low"
    
    def can_execute(self, operation: str) -> Dict:
        """
        Verifica se un'operazione può essere eseguita dato il carico attuale
        
        Returns:
            Dict con:
            - allowed: bool
            - reason: str (se non permesso)
            - wait_seconds: int (suggerimento attesa)
            - load: Dict (carico attuale)
        """
        load = self.get_system_load()
        cost = self.OPERATION_COSTS.get(operation, 5)
        
        result = {
            "operation": operation,
            "cost": cost,
            "load": load,
        }
        
        # Decisioni basate sul livello di stress
        # CRITICO: blocca tutto tranne operazioni leggere (cost <= 2)
        if load["cpu_level"] == "critical" or load["memory_level"] == "critical":
            if cost <= 2:
                result["allowed"] = True
                result["reason"] = "Operazione leggera permessa anche in stato critico"
            else:
                result["allowed"] = False
                result["reason"] = f"Sistema CRITICO (CPU: {load['cpu']:.0f}%, RAM: {load['memory']:.0f}%)"
                result["wait_seconds"] = 30
                result["suggestion"] = "Attendi che il carico si riduca"
            return result
        
        # ALTO: blocca operazioni molto pesanti (cost >= 9)
        if load["cpu_level"] == "high" or load["memory_level"] == "high":
            if cost >= 9:
                result["allowed"] = False
                result["reason"] = f"Carico ALTO per operazione pesante (costo {cost}/10)"
                result["wait_seconds"] = 15
                result["suggestion"] = "Prova un'operazione più leggera"
            else:
                result["allowed"] = True
                result["reason"] = "OK (carico alto ma operazione gestibile)"
            return result
        
        # MEDIO: permetti tutto tranne i workflow completi (cost = 10)
        if load["cpu_level"] == "medium" or load["memory_level"] == "medium":
            if cost >= 10:
                result["allowed"] = False
                result["reason"] = f"Carico MEDIO - workflow completo non raccomandato"
                result["wait_seconds"] = 10
                result["suggestion"] = "Esegui le operazioni singolarmente"
            else:
                result["allowed"] = True
                result["reason"] = "OK (carico medio)"
            return result
        
        # BASSO: permetti tutto
        result["allowed"] = True
        result["reason"] = "OK (sistema disponibile)"
        result["wait_seconds"] = 0
        
        return result
    
    def get_recommended_delay(self, operation: str) -> float:
        """Calcola un ritardo raccomandato basato sul carico"""
        load = self.get_system_load()
        cost = self.OPERATION_COSTS.get(operation, 5)
        
        # Base delay per operazioni costose
        base_delay = 0
        
        if load["cpu_level"] == "high":
            base_delay += 2
        elif load["cpu_level"] == "medium":
            base_delay += 1
            
        if load["memory_level"] == "high":
            base_delay += 3
        elif load["memory_level"] == "medium":
            base_delay += 1
        
        # Moltiplica per il costo
        return base_delay * (cost / 5)


class DevAutomations:
    """
    Automazioni per sviluppo, testing e manutenzione GIDEON
    Con throttling dinamico basato sul carico del sistema
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.backend_dir = self.project_root / "backend"
        self.venv_python = self.project_root / ".venv" / "Scripts" / "python.exe"
        self.server_process: Optional[subprocess.Popen] = None
        self.server_port = 8001
        self.server_host = "127.0.0.1"
        
        # Sistema di monitoraggio carico
        self.load_monitor = SystemLoadMonitor()
        self.enforce_limits = True  # Abilita/disabilita controlli
        
    def _check_load(self, operation: str) -> Dict:
        """Verifica se l'operazione è permessa dato il carico"""
        if not self.enforce_limits:
            return {"allowed": True, "reason": "Limiti disabilitati"}
        return self.load_monitor.can_execute(operation)
    
    async def _wait_for_capacity(self, operation: str, max_wait: int = 60) -> bool:
        """Attende finché c'è capacità per l'operazione"""
        waited = 0
        while waited < max_wait:
            check = self._check_load(operation)
            if check["allowed"]:
                return True
            
            wait_time = min(check.get("wait_seconds", 5), max_wait - waited)
            await asyncio.sleep(wait_time)
            waited += wait_time
        
        return False
    
    def get_system_status(self) -> Dict:
        """Stato del sistema con metriche di carico"""
        load = self.load_monitor.get_system_load()
        
        return {
            "cpu_percent": load.get("cpu", 0),
            "memory_percent": load.get("memory", 0),
            "cpu_level": load.get("cpu_level", "unknown"),
            "memory_level": load.get("memory_level", "unknown"),
            "limits_enabled": self.enforce_limits,
            "thresholds": self.load_monitor.THRESHOLDS
        }
        
    # ========== SERVER MANAGEMENT ==========
    
    async def start_server(self, background: bool = True, force: bool = False) -> Dict:
        """
        Avvia il server GIDEON in modo affidabile
        
        Args:
            background: Se True, avvia in background
            force: Se True, ignora i limiti di carico
        """
        # Verifica limiti di carico
        if not force:
            check = self._check_load("start_server")
            if not check["allowed"]:
                return {
                    "success": False,
                    "blocked_by_load": True,
                    "reason": check["reason"],
                    "suggestion": check.get("suggestion"),
                    "load": check.get("load"),
                    "wait_seconds": check.get("wait_seconds", 0)
                }
        
        # Prima verifica se già in esecuzione
        if await self.health_check():
            return {
                "success": True,
                "message": "Server già in esecuzione",
                "url": f"http://{self.server_host}:{self.server_port}"
            }
        
        # Termina eventuali processi zombie
        await self.stop_server()
        await asyncio.sleep(1)
        
        try:
            cmd = [
                str(self.venv_python),
                "-m", "uvicorn",
                "main:app",
                "--host", self.server_host,
                "--port", str(self.server_port)
            ]
            
            if background:
                # Avvia in background
                self.server_process = subprocess.Popen(
                    cmd,
                    cwd=str(self.backend_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )
                
                # Attendi avvio
                for _ in range(10):
                    await asyncio.sleep(1)
                    if await self.health_check():
                        return {
                            "success": True,
                            "message": "Server avviato con successo",
                            "pid": self.server_process.pid,
                            "url": f"http://{self.server_host}:{self.server_port}"
                        }
                
                return {
                    "success": False,
                    "error": "Server avviato ma health check fallito"
                }
            else:
                # Avvia in foreground (blocca)
                result = subprocess.run(cmd, cwd=str(self.backend_dir))
                return {"success": result.returncode == 0}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def stop_server(self) -> Dict:
        """Ferma il server GIDEON"""
        try:
            if os.name == 'nt':
                # Windows: trova e termina processo sulla porta
                result = subprocess.run(
                    ["netstat", "-ano"],
                    capture_output=True, text=True
                )
                
                for line in result.stdout.split('\n'):
                    if f":{self.server_port}" in line and "LISTENING" in line:
                        parts = line.split()
                        pid = parts[-1]
                        subprocess.run(["taskkill", "/PID", pid, "/F"], 
                                      capture_output=True)
            
            if self.server_process:
                self.server_process.terminate()
                self.server_process = None
                
            return {"success": True, "message": "Server fermato"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def restart_server(self, force: bool = False) -> Dict:
        """Riavvia il server"""
        if not force:
            check = self._check_load("restart_server")
            if not check["allowed"]:
                return {
                    "success": False,
                    "blocked_by_load": True,
                    "reason": check["reason"],
                    "load": check.get("load")
                }
        
        await self.stop_server()
        await asyncio.sleep(2)
        return await self.start_server(force=True)  # Forza perché già verificato
    
    async def health_check(self) -> bool:
        """Verifica se il server è attivo e funzionante"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{self.server_host}:{self.server_port}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    return resp.status == 200
        except:
            # Fallback con requests sync
            try:
                import requests
                resp = requests.get(
                    f"http://{self.server_host}:{self.server_port}/health",
                    timeout=5
                )
                return resp.status_code == 200
            except:
                return False
    
    async def get_server_status(self) -> Dict:
        """Stato dettagliato del server"""
        try:
            import requests
            resp = requests.get(
                f"http://{self.server_host}:{self.server_port}/health",
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "online": True,
                    "status": data.get("status"),
                    "services": data.get("services", {}),
                    "mode": data.get("mode"),
                    "metrics": data.get("metrics", {})
                }
        except:
            pass
        
        return {"online": False}
    
    # ========== TESTING ==========
    
    async def run_tests(self, 
                        pattern: str = None,
                        verbose: bool = False,
                        stop_on_first_fail: bool = False,
                        force: bool = False) -> Dict:
        """
        Esegue i test pytest
        
        Args:
            pattern: Pattern per filtrare test (es: "test_gideon")
            verbose: Output dettagliato
            stop_on_first_fail: Ferma al primo fallimento
            force: Ignora limiti di carico
        """
        # Verifica limiti di carico (test sono pesanti)
        if not force:
            check = self._check_load("run_tests")
            if not check["allowed"]:
                return {
                    "success": False,
                    "blocked_by_load": True,
                    "reason": check["reason"],
                    "suggestion": check.get("suggestion"),
                    "load": check.get("load"),
                    "wait_seconds": check.get("wait_seconds", 0)
                }
        
        cmd = [str(self.venv_python), "-m", "pytest"]
        
        if not verbose:
            cmd.append("-q")
        else:
            cmd.append("-v")
            
        if stop_on_first_fail:
            cmd.append("-x")
            
        if pattern:
            cmd.extend(["-k", pattern])
        
        cmd.append("--tb=short")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # Parse output
            lines = result.stdout.split('\n')
            summary = ""
            for line in lines:
                if "passed" in line or "failed" in line or "error" in line:
                    summary = line.strip()
                    break
            
            return {
                "success": result.returncode == 0,
                "exit_code": result.returncode,
                "summary": summary,
                "output": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
                "errors": result.stderr[-1000:] if result.stderr else None
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Test timeout (5 min)"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def run_quick_test(self, force: bool = False) -> Dict:
        """Test rapido dei componenti core"""
        # Quick test è più leggero ma verifichiamo comunque
        if not force:
            check = self._check_load("run_quick_test")
            if not check["allowed"]:
                return {
                    "success": False,
                    "blocked_by_load": True,
                    "reason": check["reason"],
                    "load": check.get("load")
                }
        
        test_code = '''
import asyncio
from core.mode_manager import ModeManager
from core.emergency import EmergencySystem
from gideon import GideonCore
from jarvis import JarvisCore

async def quick_test():
    results = []
    
    # ModeManager
    try:
        mm = ModeManager()
        results.append(("ModeManager", True, mm.mode_name))
    except Exception as e:
        results.append(("ModeManager", False, str(e)))
    
    # Emergency
    try:
        em = EmergencySystem()
        results.append(("EmergencySystem", True, f"active={not em.is_killed}"))
    except Exception as e:
        results.append(("EmergencySystem", False, str(e)))
    
    # GideonCore
    try:
        gc = GideonCore()
        results.append(("GideonCore", True, "ready"))
    except Exception as e:
        results.append(("GideonCore", False, str(e)))
    
    # JarvisCore
    try:
        jc = JarvisCore()
        results.append(("JarvisCore", True, "ready"))
    except Exception as e:
        results.append(("JarvisCore", False, str(e)))
    
    return results

results = asyncio.run(quick_test())
for name, ok, msg in results:
    status = "✓" if ok else "✗"
    print(f"{status} {name}: {msg}")
'''
        
        try:
            result = subprocess.run(
                [str(self.venv_python), "-c", test_code],
                cwd=str(self.backend_dir),
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr if result.stderr else None
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ========== CACHE & CLEANUP ==========
    
    async def clean_cache(self) -> Dict:
        """Pulisce cache Python e file temporanei"""
        cleaned = []
        
        # __pycache__ directories
        for pycache in self.project_root.rglob("__pycache__"):
            try:
                import shutil
                shutil.rmtree(pycache)
                cleaned.append(str(pycache))
            except:
                pass
        
        # .pyc files
        for pyc in self.project_root.rglob("*.pyc"):
            try:
                pyc.unlink()
                cleaned.append(str(pyc))
            except:
                pass
        
        # pytest cache
        pytest_cache = self.project_root / ".pytest_cache"
        if pytest_cache.exists():
            try:
                import shutil
                shutil.rmtree(pytest_cache)
                cleaned.append(str(pytest_cache))
            except:
                pass
        
        return {
            "success": True,
            "cleaned_count": len(cleaned),
            "items": cleaned[:20]  # Mostra solo primi 20
        }
    
    # ========== GIT OPERATIONS ==========
    
    async def git_status(self) -> Dict:
        """Stato del repository git"""
        try:
            # Status
            status = subprocess.run(
                ["git", "status", "--short"],
                cwd=str(self.project_root),
                capture_output=True, text=True
            )
            
            # Branch corrente
            branch = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=str(self.project_root),
                capture_output=True, text=True
            )
            
            # Ultimo commit
            last_commit = subprocess.run(
                ["git", "log", "-1", "--oneline"],
                cwd=str(self.project_root),
                capture_output=True, text=True
            )
            
            changes = status.stdout.strip().split('\n') if status.stdout.strip() else []
            
            return {
                "success": True,
                "branch": branch.stdout.strip(),
                "last_commit": last_commit.stdout.strip(),
                "changes_count": len(changes),
                "changes": changes[:10]  # Prime 10 modifiche
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ========== BACKUP ==========
    
    async def backup_project(self, include_venv: bool = False, force: bool = False) -> Dict:
        """
        Crea backup del progetto
        
        Args:
            include_venv: Se includere la cartella .venv
            force: Ignora limiti di carico
        """
        # Backup è operazione pesante (I/O intensivo)
        if not force:
            check = self._check_load("backup_project")
            if not check["allowed"]:
                return {
                    "success": False,
                    "blocked_by_load": True,
                    "reason": check["reason"],
                    "suggestion": "Il backup richiede molte risorse I/O",
                    "load": check.get("load")
                }
        
        import shutil
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"gideon_backup_{timestamp}"
        backup_path = self.project_root.parent / backup_name
        
        try:
            # Escludi cartelle
            exclude = {'.venv', '__pycache__', '.git', 'node_modules', '.pytest_cache'}
            if include_venv:
                exclude.discard('.venv')
            
            def ignore_patterns(dir, files):
                return [f for f in files if f in exclude]
            
            shutil.copytree(
                self.project_root, 
                backup_path,
                ignore=ignore_patterns
            )
            
            # Comprimi
            archive_path = shutil.make_archive(
                str(backup_path),
                'zip',
                backup_path.parent,
                backup_name
            )
            
            # Rimuovi cartella non compressa
            shutil.rmtree(backup_path)
            
            return {
                "success": True,
                "backup_path": archive_path,
                "size_mb": round(os.path.getsize(archive_path) / (1024*1024), 2)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ========== WORKFLOW COMPOSITI ==========
    
    async def dev_startup(self, force: bool = False) -> Dict:
        """
        Workflow completo di avvio sviluppo:
        1. Verifica carico sistema
        2. Pulisce cache
        3. Avvia server
        4. Verifica health
        5. Esegue quick test
        """
        # Questo è il workflow più pesante
        if not force:
            check = self._check_load("dev_startup")
            if not check["allowed"]:
                return {
                    "success": False,
                    "blocked_by_load": True,
                    "reason": check["reason"],
                    "suggestion": "Workflow completo richiede risorse significative",
                    "load": check.get("load"),
                    "alternative": "Prova con singole operazioni: clean, start, quick"
                }
        
        results = {
            "steps": [],
            "success": True,
            "load_at_start": self.load_monitor.get_system_load()
        }
        
        # Step 1: Clean cache
        clean = await self.clean_cache()
        results["steps"].append({
            "name": "clean_cache",
            "success": clean["success"],
            "cleaned": clean.get("cleaned_count", 0)
        })
        
        # Piccola pausa tra operazioni pesanti
        await asyncio.sleep(1)
        
        # Step 2: Start server
        server = await self.start_server(force=True)  # Già verificato carico all'inizio
        results["steps"].append({
            "name": "start_server",
            "success": server["success"],
            "message": server.get("message") or server.get("error")
        })
        
        if not server["success"]:
            results["success"] = False
            return results
        
        # Step 3: Health check
        healthy = await self.health_check()
        results["steps"].append({
            "name": "health_check",
            "success": healthy
        })
        
        # Step 4: Quick test
        test = await self.run_quick_test(force=True)
        results["steps"].append({
            "name": "quick_test",
            "success": test["success"],
            "output": test.get("output", "")[:500]
        })
        
        results["success"] = all(s["success"] for s in results["steps"])
        results["load_at_end"] = self.load_monitor.get_system_load()
        return results
    
    async def full_test_suite(self, force: bool = False) -> Dict:
        """
        Esegue suite completa di test:
        1. Verifica carico sistema
        2. Verifica server attivo
        3. Esegue tutti i test
        4. Genera report
        """
        if not force:
            check = self._check_load("full_test_suite")
            if not check["allowed"]:
                return {
                    "success": False,
                    "blocked_by_load": True,
                    "reason": check["reason"],
                    "load": check.get("load")
                }
        
        results = {"steps": [], "success": True}
        
        # Verifica server
        if not await self.health_check():
            # Avvia server se non attivo
            await self.start_server(force=True)
            await asyncio.sleep(3)
        
        # Esegui test
        test_result = await self.run_tests(verbose=True, force=True)
        results["test_result"] = test_result
        results["success"] = test_result["success"]
        
        return results
    
    # ========== UTILITIES ==========
    
    def set_limits_enabled(self, enabled: bool):
        """Abilita/disabilita i controlli di carico"""
        self.enforce_limits = enabled
        return {"limits_enabled": self.enforce_limits}
    
    def adjust_thresholds(self, **kwargs):
        """
        Modifica le soglie di carico
        
        Esempio: adjust_thresholds(cpu_high=80, memory_critical=90)
        """
        for key, value in kwargs.items():
            if key in self.load_monitor.THRESHOLDS:
                self.load_monitor.THRESHOLDS[key] = value
        
        return {"thresholds": self.load_monitor.THRESHOLDS}
    
    def check_operation_feasibility(self, operation: str) -> Dict:
        """Verifica se un'operazione è fattibile ora"""
        return self._check_load(operation)


# Singleton instance
_dev_automations: Optional[DevAutomations] = None

def get_dev_automations() -> DevAutomations:
    """Ottiene istanza singleton delle automazioni dev"""
    global _dev_automations
    if _dev_automations is None:
        _dev_automations = DevAutomations()
    return _dev_automations
