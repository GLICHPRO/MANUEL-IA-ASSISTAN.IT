"""
🎮 JARVIS CORE - System Controller
Controllo sistema operativo, processi e risorse
"""

import asyncio
import subprocess
import os
import platform
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import win32gui
    import win32con
    import win32process
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False


class SystemController:
    """
    Controller per il sistema operativo
    Gestisce processi, finestre, risorse e comandi di sistema
    """
    
    def __init__(self, executor=None, security_manager=None):
        self.executor = executor
        self.security = security_manager
        self.os_type = platform.system()  # Windows, Linux, Darwin
        self.monitored_processes = {}
        self.window_history = []
        
    # === INFORMAZIONI SISTEMA ===
    
    def get_system_info(self) -> dict:
        """Informazioni complete sul sistema"""
        info = {
            "os": self.os_type,
            "os_version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "hostname": platform.node()
        }
        
        if PSUTIL_AVAILABLE:
            info.update({
                "cpu_count": psutil.cpu_count(),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "disk_total_gb": round(psutil.disk_usage('/').total / (1024**3), 2)
            })
        
        return info
    
    def get_resource_usage(self) -> dict:
        """Utilizzo risorse in tempo reale"""
        if not PSUTIL_AVAILABLE:
            return {"error": "psutil non disponibile"}
        
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu": {
                "percent": cpu_percent,
                "per_core": psutil.cpu_percent(interval=0.1, percpu=True)
            },
            "memory": {
                "percent": memory.percent,
                "used_gb": round(memory.used / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "total_gb": round(memory.total / (1024**3), 2)
            },
            "disk": {
                "percent": disk.percent,
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "total_gb": round(disk.total / (1024**3), 2)
            },
            "network": self._get_network_stats()
        }
    
    def _get_network_stats(self) -> dict:
        """Statistiche rete"""
        if not PSUTIL_AVAILABLE:
            return {}
        
        net = psutil.net_io_counters()
        return {
            "bytes_sent_mb": round(net.bytes_sent / (1024**2), 2),
            "bytes_recv_mb": round(net.bytes_recv / (1024**2), 2),
            "packets_sent": net.packets_sent,
            "packets_recv": net.packets_recv
        }
    
    # === GESTIONE PROCESSI ===
    
    def get_running_processes(self, limit: int = 50) -> List[dict]:
        """Lista processi in esecuzione"""
        if not PSUTIL_AVAILABLE:
            return []
        
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                pinfo = proc.info
                processes.append({
                    "pid": pinfo['pid'],
                    "name": pinfo['name'],
                    "cpu_percent": pinfo['cpu_percent'] or 0,
                    "memory_percent": round(pinfo['memory_percent'] or 0, 2)
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Ordina per CPU usage
        processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
        return processes[:limit]
    
    def get_process_by_name(self, name: str) -> List[dict]:
        """Trova processi per nome"""
        if not PSUTIL_AVAILABLE:
            return []
        
        processes = []
        name_lower = name.lower()
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
            try:
                if name_lower in proc.info['name'].lower():
                    processes.append({
                        "pid": proc.info['pid'],
                        "name": proc.info['name'],
                        "cpu_percent": proc.info['cpu_percent'] or 0,
                        "memory_percent": round(proc.info['memory_percent'] or 0, 2),
                        "status": proc.info['status']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return processes
    
    async def kill_process(self, pid: int = None, name: str = None) -> dict:
        """Termina un processo"""
        if not PSUTIL_AVAILABLE:
            return {"success": False, "error": "psutil non disponibile"}
        
        try:
            if pid:
                proc = psutil.Process(pid)
                proc_name = proc.name()
                proc.terminate()
                return {"success": True, "killed": {"pid": pid, "name": proc_name}}
            
            elif name:
                killed = []
                for proc in psutil.process_iter(['pid', 'name']):
                    if name.lower() in proc.info['name'].lower():
                        proc.terminate()
                        killed.append({"pid": proc.info['pid'], "name": proc.info['name']})
                
                return {"success": True, "killed": killed, "count": len(killed)}
            
            else:
                return {"success": False, "error": "Specificare pid o name"}
                
        except psutil.NoSuchProcess:
            return {"success": False, "error": "Processo non trovato"}
        except psutil.AccessDenied:
            return {"success": False, "error": "Accesso negato"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # === GESTIONE FINESTRE (Windows) ===
    
    def get_open_windows(self) -> List[dict]:
        """Lista finestre aperte"""
        if not WIN32_AVAILABLE:
            return self._get_windows_fallback()
        
        windows = []
        
        def enum_handler(hwnd, results):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title:
                    _, pid = win32process.GetWindowThreadProcessId(hwnd)
                    results.append({
                        "hwnd": hwnd,
                        "title": title,
                        "pid": pid
                    })
        
        win32gui.EnumWindows(enum_handler, windows)
        return windows
    
    def _get_windows_fallback(self) -> List[dict]:
        """Fallback per lista finestre senza win32"""
        if self.os_type == "Windows":
            try:
                result = subprocess.run(
                    ['powershell', '-Command', 
                     'Get-Process | Where-Object {$_.MainWindowTitle} | Select-Object Id, MainWindowTitle | ConvertTo-Json'],
                    capture_output=True, text=True, timeout=5
                )
                import json
                data = json.loads(result.stdout)
                if isinstance(data, dict):
                    data = [data]
                return [{"pid": p["Id"], "title": p["MainWindowTitle"]} for p in data]
            except:
                pass
        return []
    
    async def focus_window(self, title: str = None, hwnd: int = None) -> dict:
        """Porta una finestra in primo piano"""
        if not WIN32_AVAILABLE:
            return {"success": False, "error": "win32gui non disponibile"}
        
        try:
            if hwnd:
                win32gui.SetForegroundWindow(hwnd)
                return {"success": True, "focused": hwnd}
            
            elif title:
                def find_window(hwnd, target):
                    if target.lower() in win32gui.GetWindowText(hwnd).lower():
                        win32gui.SetForegroundWindow(hwnd)
                        return False  # Stop enumeration
                    return True
                
                win32gui.EnumWindows(find_window, title)
                return {"success": True, "focused": title}
            
            else:
                return {"success": False, "error": "Specificare title o hwnd"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def minimize_window(self, title: str = None, hwnd: int = None) -> dict:
        """Minimizza una finestra"""
        if not WIN32_AVAILABLE:
            return {"success": False, "error": "win32gui non disponibile"}
        
        try:
            target_hwnd = hwnd
            if title and not hwnd:
                target_hwnd = win32gui.FindWindow(None, title)
            
            if target_hwnd:
                win32gui.ShowWindow(target_hwnd, win32con.SW_MINIMIZE)
                return {"success": True}
            
            return {"success": False, "error": "Finestra non trovata"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def maximize_window(self, title: str = None, hwnd: int = None) -> dict:
        """Massimizza una finestra"""
        if not WIN32_AVAILABLE:
            return {"success": False, "error": "win32gui non disponibile"}
        
        try:
            target_hwnd = hwnd
            if title and not hwnd:
                target_hwnd = win32gui.FindWindow(None, title)
            
            if target_hwnd:
                win32gui.ShowWindow(target_hwnd, win32con.SW_MAXIMIZE)
                return {"success": True}
            
            return {"success": False, "error": "Finestra non trovata"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # === COMANDI SISTEMA ===
    
    async def run_system_command(self, command: str, timeout: int = 30) -> dict:
        """Esegue un comando di sistema"""
        try:
            if self.os_type == "Windows":
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    shell=True
                )
            else:
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            return {
                "success": process.returncode == 0,
                "returncode": process.returncode,
                "stdout": stdout.decode('utf-8', errors='ignore'),
                "stderr": stderr.decode('utf-8', errors='ignore')
            }
            
        except asyncio.TimeoutError:
            return {"success": False, "error": f"Timeout dopo {timeout} secondi"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # === MONITORAGGIO ===
    
    async def start_monitoring(self, interval: float = 5.0):
        """Avvia monitoraggio continuo delle risorse"""
        while True:
            usage = self.get_resource_usage()
            
            # Controlla soglie critiche
            if usage.get("cpu", {}).get("percent", 0) > 90:
                await self._alert("CPU alta", f"Utilizzo CPU: {usage['cpu']['percent']}%")
            
            if usage.get("memory", {}).get("percent", 0) > 85:
                await self._alert("Memoria alta", f"Utilizzo RAM: {usage['memory']['percent']}%")
            
            if usage.get("disk", {}).get("percent", 0) > 90:
                await self._alert("Disco quasi pieno", f"Utilizzo disco: {usage['disk']['percent']}%")
            
            await asyncio.sleep(interval)
    
    async def _alert(self, title: str, message: str):
        """Invia alert (da collegare al sistema notifiche)"""
        print(f"⚠️ ALERT: {title} - {message}")
        # TODO: Collegare al sistema di notifiche Jarvis
    
    def get_controller_status(self) -> dict:
        """Stato del controller"""
        return {
            "os": self.os_type,
            "psutil_available": PSUTIL_AVAILABLE,
            "win32_available": WIN32_AVAILABLE,
            "monitored_processes": len(self.monitored_processes),
            "features": {
                "process_management": PSUTIL_AVAILABLE,
                "window_management": WIN32_AVAILABLE,
                "resource_monitoring": PSUTIL_AVAILABLE,
                "system_commands": True
            }
        }
