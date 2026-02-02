# /backend/jarvis/execution_controller.py
"""
JARVIS Execution Controller - Controllo Unificato di Esecuzione
Gestisce automazioni, routine e controlli applicazioni in modo centralizzato.
"""

import asyncio
import psutil
import subprocess
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class ExecutionState(Enum):
    """Stato di esecuzione"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING = "waiting"
    BLOCKED = "blocked"


class RoutineType(Enum):
    """Tipi di routine"""
    SCHEDULED = "scheduled"      # Orario specifico
    INTERVAL = "interval"        # Ogni X minuti
    TRIGGERED = "triggered"      # Su evento
    CONDITIONAL = "conditional"  # Su condizione
    SEQUENTIAL = "sequential"    # Sequenza di azioni
    PARALLEL = "parallel"        # Azioni parallele
    WORKFLOW = "workflow"        # Workflow complesso


class AppControlAction(Enum):
    """Azioni di controllo app"""
    LAUNCH = "launch"
    CLOSE = "close"
    FOCUS = "focus"
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    RESTORE = "restore"
    RESTART = "restart"
    MONITOR = "monitor"


@dataclass
class ExecutionTask:
    """Task di esecuzione"""
    id: str
    name: str
    action: dict
    priority: int = 5
    state: ExecutionState = ExecutionState.IDLE
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    timeout: int = 30
    dependencies: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def is_complete(self) -> bool:
        return self.state in [ExecutionState.COMPLETED, ExecutionState.FAILED, ExecutionState.CANCELLED]


@dataclass
class Routine:
    """Definizione di una routine"""
    id: str
    name: str
    description: str
    routine_type: RoutineType
    actions: List[dict]
    trigger: dict  # Configurazione trigger
    enabled: bool = True
    priority: int = 5
    created_at: datetime = field(default_factory=datetime.now)
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    success_count: int = 0
    error_count: int = 0
    avg_duration: float = 0.0
    conditions: List[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    def should_run(self, context: dict) -> bool:
        """Verifica se la routine deve essere eseguita"""
        if not self.enabled:
            return False
        
        for condition in self.conditions:
            if not self._evaluate_condition(condition, context):
                return False
        return True
    
    def _evaluate_condition(self, condition: dict, context: dict) -> bool:
        """Valuta una singola condizione"""
        cond_type = condition.get("type")
        
        if cond_type == "time_range":
            now = datetime.now()
            start = datetime.strptime(condition.get("start", "00:00"), "%H:%M").time()
            end = datetime.strptime(condition.get("end", "23:59"), "%H:%M").time()
            return start <= now.time() <= end
        
        if cond_type == "day_of_week":
            days = condition.get("days", [])
            return datetime.now().weekday() in days
        
        if cond_type == "system_idle":
            threshold = condition.get("idle_minutes", 5)
            return context.get("system_idle_minutes", 0) >= threshold
        
        if cond_type == "resource_threshold":
            resource = condition.get("resource", "cpu")
            threshold = condition.get("threshold", 50)
            operator = condition.get("operator", "lt")
            value = context.get(f"system_{resource}", 0)
            
            if operator == "lt":
                return value < threshold
            elif operator == "gt":
                return value > threshold
            elif operator == "eq":
                return value == threshold
        
        if cond_type == "app_running":
            app_name = condition.get("app")
            return context.get("running_apps", {}).get(app_name, False)
        
        return True


class AppController:
    """Controller per gestione applicazioni"""
    
    def __init__(self):
        self.monitored_apps: Dict[str, dict] = {}
        self.app_history: List[dict] = []
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
    
    async def launch_app(self, app_name: str, args: List[str] = None, 
                         wait: bool = False) -> dict:
        """Avvia un'applicazione"""
        try:
            # Mappatura app comuni
            app_paths = {
                "notepad": "notepad.exe",
                "calc": "calc.exe",
                "explorer": "explorer.exe",
                "chrome": "chrome.exe",
                "firefox": "firefox.exe",
                "code": "code.exe",
                "spotify": "spotify.exe",
                "discord": "discord.exe",
                "slack": "slack.exe",
                "terminal": "wt.exe",
                "cmd": "cmd.exe",
                "powershell": "powershell.exe"
            }
            
            executable = app_paths.get(app_name.lower(), app_name)
            cmd = [executable] + (args or [])
            
            if wait:
                process = subprocess.run(cmd, capture_output=True, text=True)
                success = process.returncode == 0
            else:
                process = subprocess.Popen(cmd)
                success = True
            
            result = {
                "success": success,
                "app": app_name,
                "action": AppControlAction.LAUNCH.value,
                "pid": process.pid if hasattr(process, 'pid') else None,
                "timestamp": datetime.now().isoformat()
            }
            
            self._log_action(result)
            return result
            
        except Exception as e:
            return {
                "success": False,
                "app": app_name,
                "action": AppControlAction.LAUNCH.value,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def close_app(self, app_name: str, force: bool = False) -> dict:
        """Chiude un'applicazione"""
        try:
            flag = "/F" if force else ""
            cmd = ["taskkill", "/IM", f"{app_name}.exe"]
            if force:
                cmd.append("/F")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            success = result.returncode == 0
            
            result_dict = {
                "success": success,
                "app": app_name,
                "action": AppControlAction.CLOSE.value,
                "forced": force,
                "timestamp": datetime.now().isoformat()
            }
            
            self._log_action(result_dict)
            return result_dict
            
        except Exception as e:
            return {
                "success": False,
                "app": app_name,
                "action": AppControlAction.CLOSE.value,
                "error": str(e)
            }
    
    async def focus_app(self, app_name: str) -> dict:
        """Porta un'applicazione in primo piano"""
        try:
            import win32gui
            import win32con
            
            def callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if app_name.lower() in title.lower():
                        windows.append(hwnd)
                return True
            
            windows = []
            win32gui.EnumWindows(callback, windows)
            
            if windows:
                hwnd = windows[0]
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd)
                
                return {
                    "success": True,
                    "app": app_name,
                    "action": AppControlAction.FOCUS.value
                }
            else:
                return {
                    "success": False,
                    "app": app_name,
                    "action": AppControlAction.FOCUS.value,
                    "error": "Finestra non trovata"
                }
                
        except Exception as e:
            return {
                "success": False,
                "app": app_name,
                "action": AppControlAction.FOCUS.value,
                "error": str(e)
            }
    
    async def minimize_app(self, app_name: str) -> dict:
        """Minimizza un'applicazione"""
        try:
            import win32gui
            import win32con
            
            def callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if app_name.lower() in title.lower():
                        windows.append(hwnd)
                return True
            
            windows = []
            win32gui.EnumWindows(callback, windows)
            
            if windows:
                for hwnd in windows:
                    win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
                
                return {
                    "success": True,
                    "app": app_name,
                    "action": AppControlAction.MINIMIZE.value,
                    "windows_affected": len(windows)
                }
            
            return {
                "success": False,
                "app": app_name,
                "error": "Nessuna finestra trovata"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def maximize_app(self, app_name: str) -> dict:
        """Massimizza un'applicazione"""
        try:
            import win32gui
            import win32con
            
            def callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if app_name.lower() in title.lower():
                        windows.append(hwnd)
                return True
            
            windows = []
            win32gui.EnumWindows(callback, windows)
            
            if windows:
                hwnd = windows[0]
                win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
                win32gui.SetForegroundWindow(hwnd)
                
                return {
                    "success": True,
                    "app": app_name,
                    "action": AppControlAction.MAXIMIZE.value
                }
            
            return {"success": False, "error": "Finestra non trovata"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def restart_app(self, app_name: str, args: List[str] = None) -> dict:
        """Riavvia un'applicazione"""
        close_result = await self.close_app(app_name, force=True)
        await asyncio.sleep(1)  # Attendi chiusura
        launch_result = await self.launch_app(app_name, args)
        
        return {
            "success": close_result.get("success", False) and launch_result.get("success", False),
            "app": app_name,
            "action": AppControlAction.RESTART.value,
            "close_result": close_result,
            "launch_result": launch_result
        }
    
    def is_app_running(self, app_name: str) -> bool:
        """Verifica se un'app è in esecuzione"""
        for proc in psutil.process_iter(['name']):
            try:
                if app_name.lower() in proc.info['name'].lower():
                    return True
            except:
                pass
        return False
    
    def get_running_apps(self) -> List[dict]:
        """Ottiene lista app in esecuzione"""
        apps = []
        seen = set()
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                name = proc.info['name']
                if name not in seen:
                    seen.add(name)
                    apps.append({
                        "name": name,
                        "pid": proc.info['pid'],
                        "cpu": proc.info['cpu_percent'],
                        "memory": proc.info['memory_percent']
                    })
            except:
                pass
        
        return sorted(apps, key=lambda x: x['name'])
    
    async def start_monitoring(self, app_name: str, interval: int = 30):
        """Avvia monitoraggio di un'app"""
        if app_name in self._monitoring_tasks:
            return
        
        async def monitor_loop():
            while True:
                is_running = self.is_app_running(app_name)
                self.monitored_apps[app_name] = {
                    "running": is_running,
                    "last_check": datetime.now().isoformat()
                }
                await asyncio.sleep(interval)
        
        self._monitoring_tasks[app_name] = asyncio.create_task(monitor_loop())
    
    async def stop_monitoring(self, app_name: str):
        """Ferma monitoraggio di un'app"""
        if app_name in self._monitoring_tasks:
            self._monitoring_tasks[app_name].cancel()
            del self._monitoring_tasks[app_name]
    
    def _log_action(self, action: dict):
        """Logga un'azione"""
        self.app_history.append(action)
        # Mantieni solo le ultime 500 azioni
        if len(self.app_history) > 500:
            self.app_history = self.app_history[-500:]


class ExecutionController:
    """
    Controller centrale di esecuzione.
    Gestisce automazioni, routine e controllo applicazioni.
    """
    
    def __init__(self, executor=None, automator=None, security=None):
        self.executor = executor
        self.automator = automator
        self.security = security
        
        # App Controller
        self.app_controller = AppController()
        
        # Task Management
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.active_tasks: Dict[str, ExecutionTask] = {}
        self.completed_tasks: List[ExecutionTask] = []
        self.task_counter = 0
        
        # Routine Management
        self.routines: Dict[str, Routine] = {}
        
        # Execution State
        self.is_running = False
        self._worker_task: Optional[asyncio.Task] = None
        self._routine_scheduler: Optional[asyncio.Task] = None
        self.max_concurrent_tasks = 5
        self.semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        
        # Hooks & Callbacks
        self.pre_execution_hooks: List[Callable] = []
        self.post_execution_hooks: List[Callable] = []
        self.error_handlers: List[Callable] = []
        
        # Statistics
        self.stats = {
            "total_executed": 0,
            "successful": 0,
            "failed": 0,
            "cancelled": 0,
            "total_duration": 0.0,
            "avg_duration": 0.0
        }
        
        # Default routines
        self._setup_default_routines()
    
    def _setup_default_routines(self):
        """Configura routine predefinite"""
        # Routine mattutina
        self.create_routine(
            routine_id="morning_routine",
            name="Routine Mattutina",
            description="Apre applicazioni di lavoro al mattino",
            routine_type=RoutineType.SCHEDULED,
            actions=[
                {"type": "launch_app", "app": "chrome"},
                {"type": "launch_app", "app": "code"},
                {"type": "notify", "message": "Buongiorno! Ambiente di lavoro pronto."}
            ],
            trigger={"type": "time", "time": "09:00"},
            enabled=False  # Disabilitato di default
        )
        
        # Routine serale
        self.create_routine(
            routine_id="evening_routine",
            name="Routine Serale",
            description="Chiude applicazioni e prepara per la sera",
            routine_type=RoutineType.SCHEDULED,
            actions=[
                {"type": "close_app", "app": "code"},
                {"type": "notify", "message": "Fine giornata lavorativa."}
            ],
            trigger={"type": "time", "time": "18:00"},
            enabled=False
        )
        
        # Routine di pulizia memoria
        self.create_routine(
            routine_id="memory_cleanup",
            name="Pulizia Memoria",
            description="Pulisce processi che usano troppa memoria",
            routine_type=RoutineType.CONDITIONAL,
            actions=[
                {"type": "notify", "message": "Memoria alta, ottimizzazione in corso..."}
            ],
            trigger={"type": "condition", "check": "memory_usage", "threshold": 90},
            enabled=False
        )
    
    # === Task Management ===
    
    async def submit_task(self, name: str, action: dict, 
                          priority: int = 5, **kwargs) -> ExecutionTask:
        """Sottomette un task per l'esecuzione"""
        self.task_counter += 1
        task_id = f"task_{self.task_counter}_{datetime.now().strftime('%H%M%S')}"
        
        task = ExecutionTask(
            id=task_id,
            name=name,
            action=action,
            priority=priority,
            timeout=kwargs.get("timeout", 30),
            max_retries=kwargs.get("max_retries", 3),
            dependencies=kwargs.get("dependencies", []),
            metadata=kwargs.get("metadata", {})
        )
        
        # Priority queue usa (priority, task_id) per ordinamento
        await self.task_queue.put((priority, task_id, task))
        self.active_tasks[task_id] = task
        
        logger.info(f"Task sottomesso: {task_id} - {name}")
        return task
    
    async def execute_task(self, task: ExecutionTask) -> dict:
        """Esegue un singolo task"""
        async with self.semaphore:
            task.state = ExecutionState.RUNNING
            task.started_at = datetime.now()
            
            try:
                # Pre-execution hooks
                for hook in self.pre_execution_hooks:
                    await hook(task)
                
                # Security check
                if self.security:
                    check = await self.security.validate_action(task.action)
                    if not check.get("allowed", True):
                        raise Exception(f"Azione bloccata: {check.get('reason')}")
                
                # Execute
                if self.executor:
                    result = await asyncio.wait_for(
                        self.executor.execute(task.action),
                        timeout=task.timeout
                    )
                else:
                    result = await self._internal_execute(task.action)
                
                task.result = result
                task.state = ExecutionState.COMPLETED
                task.completed_at = datetime.now()
                
                # Update stats
                self.stats["total_executed"] += 1
                self.stats["successful"] += 1
                if task.duration:
                    self.stats["total_duration"] += task.duration
                    self.stats["avg_duration"] = (
                        self.stats["total_duration"] / self.stats["successful"]
                    )
                
                # Post-execution hooks
                for hook in self.post_execution_hooks:
                    await hook(task, result)
                
                return result
                
            except asyncio.TimeoutError:
                task.state = ExecutionState.FAILED
                task.error = "Timeout"
                task.completed_at = datetime.now()
                return {"success": False, "error": "Timeout"}
                
            except Exception as e:
                task.error = str(e)
                
                # Retry logic
                if task.retries < task.max_retries:
                    task.retries += 1
                    task.state = ExecutionState.WAITING
                    logger.warning(f"Retry {task.retries}/{task.max_retries} per {task.id}")
                    await asyncio.sleep(2 ** task.retries)  # Exponential backoff
                    return await self.execute_task(task)
                
                task.state = ExecutionState.FAILED
                task.completed_at = datetime.now()
                self.stats["total_executed"] += 1
                self.stats["failed"] += 1
                
                # Error handlers
                for handler in self.error_handlers:
                    await handler(task, e)
                
                return {"success": False, "error": str(e)}
            
            finally:
                # Move to completed
                if task.is_complete:
                    self.completed_tasks.append(task)
                    if task.id in self.active_tasks:
                        del self.active_tasks[task.id]
    
    async def _internal_execute(self, action: dict) -> dict:
        """Esecuzione interna se executor non disponibile"""
        action_type = action.get("type", "")
        
        # App control actions
        if action_type == "launch_app":
            return await self.app_controller.launch_app(
                action.get("app"),
                action.get("args", [])
            )
        elif action_type == "close_app":
            return await self.app_controller.close_app(
                action.get("app"),
                action.get("force", False)
            )
        elif action_type == "focus_app":
            return await self.app_controller.focus_app(action.get("app"))
        elif action_type == "minimize_app":
            return await self.app_controller.minimize_app(action.get("app"))
        elif action_type == "maximize_app":
            return await self.app_controller.maximize_app(action.get("app"))
        elif action_type == "restart_app":
            return await self.app_controller.restart_app(action.get("app"))
        elif action_type == "notify":
            # Semplice notifica
            message = action.get("message", "")
            logger.info(f"📢 Notifica: {message}")
            return {"success": True, "message": message}
        
        return {"success": False, "error": f"Azione sconosciuta: {action_type}"}
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancella un task"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.state = ExecutionState.CANCELLED
            task.completed_at = datetime.now()
            self.stats["cancelled"] += 1
            return True
        return False
    
    def get_task_status(self, task_id: str) -> Optional[dict]:
        """Ottiene lo stato di un task"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                "id": task.id,
                "name": task.name,
                "state": task.state.value,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "duration": task.duration,
                "retries": task.retries,
                "error": task.error
            }
        return None
    
    # === Routine Management ===
    
    def create_routine(self, routine_id: str, name: str, description: str,
                       routine_type: RoutineType, actions: List[dict],
                       trigger: dict, enabled: bool = True,
                       priority: int = 5, conditions: List[dict] = None) -> Routine:
        """Crea una nuova routine"""
        routine = Routine(
            id=routine_id,
            name=name,
            description=description,
            routine_type=routine_type,
            actions=actions,
            trigger=trigger,
            enabled=enabled,
            priority=priority,
            conditions=conditions or []
        )
        
        # Calcola prossima esecuzione
        routine.next_run = self._calculate_next_run(routine)
        
        self.routines[routine_id] = routine
        logger.info(f"Routine creata: {routine_id} - {name}")
        return routine
    
    def _calculate_next_run(self, routine: Routine) -> Optional[datetime]:
        """Calcola la prossima esecuzione di una routine"""
        now = datetime.now()
        trigger = routine.trigger
        trigger_type = trigger.get("type")
        
        if trigger_type == "time":
            target_time = datetime.strptime(trigger.get("time", "00:00"), "%H:%M")
            next_run = now.replace(
                hour=target_time.hour,
                minute=target_time.minute,
                second=0,
                microsecond=0
            )
            if next_run <= now:
                next_run += timedelta(days=1)
            return next_run
        
        if trigger_type == "interval":
            minutes = trigger.get("minutes", 60)
            return now + timedelta(minutes=minutes)
        
        return None
    
    async def run_routine(self, routine_id: str, context: dict = None) -> dict:
        """Esegue una routine"""
        if routine_id not in self.routines:
            return {"success": False, "error": "Routine non trovata"}
        
        routine = self.routines[routine_id]
        context = context or {}
        
        # Verifica condizioni
        if not routine.should_run(context):
            return {"success": False, "error": "Condizioni non soddisfatte"}
        
        results = []
        start_time = datetime.now()
        
        try:
            if routine.routine_type == RoutineType.PARALLEL:
                # Esecuzione parallela
                tasks = []
                for i, action in enumerate(routine.actions):
                    task = await self.submit_task(
                        f"{routine.name}_step_{i}",
                        action,
                        priority=routine.priority
                    )
                    tasks.append(self.execute_task(task))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # Esecuzione sequenziale
                for i, action in enumerate(routine.actions):
                    task = await self.submit_task(
                        f"{routine.name}_step_{i}",
                        action,
                        priority=routine.priority
                    )
                    result = await self.execute_task(task)
                    results.append(result)
                    
                    # Stop on failure for sequential
                    if not result.get("success", False):
                        break
            
            # Update routine stats
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            routine.last_run = end_time
            routine.run_count += 1
            routine.next_run = self._calculate_next_run(routine)
            
            success = all(r.get("success", False) for r in results if isinstance(r, dict))
            if success:
                routine.success_count += 1
            else:
                routine.error_count += 1
            
            # Update average duration
            routine.avg_duration = (
                (routine.avg_duration * (routine.run_count - 1) + duration) / routine.run_count
            )
            
            return {
                "success": success,
                "routine_id": routine_id,
                "results": results,
                "duration": duration
            }
            
        except Exception as e:
            routine.error_count += 1
            routine.last_run = datetime.now()
            return {"success": False, "error": str(e)}
    
    def enable_routine(self, routine_id: str) -> bool:
        """Abilita una routine"""
        if routine_id in self.routines:
            self.routines[routine_id].enabled = True
            self.routines[routine_id].next_run = self._calculate_next_run(self.routines[routine_id])
            return True
        return False
    
    def disable_routine(self, routine_id: str) -> bool:
        """Disabilita una routine"""
        if routine_id in self.routines:
            self.routines[routine_id].enabled = False
            return True
        return False
    
    def list_routines(self, enabled_only: bool = False) -> List[dict]:
        """Lista tutte le routine"""
        result = []
        for routine in self.routines.values():
            if enabled_only and not routine.enabled:
                continue
            result.append({
                "id": routine.id,
                "name": routine.name,
                "description": routine.description,
                "type": routine.routine_type.value,
                "enabled": routine.enabled,
                "last_run": routine.last_run.isoformat() if routine.last_run else None,
                "next_run": routine.next_run.isoformat() if routine.next_run else None,
                "run_count": routine.run_count,
                "success_rate": (routine.success_count / routine.run_count * 100) 
                               if routine.run_count > 0 else 0
            })
        return result
    
    # === Worker & Scheduler ===
    
    async def start(self):
        """Avvia il controller"""
        self.is_running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        self._routine_scheduler = asyncio.create_task(self._routine_scheduler_loop())
        logger.info("ExecutionController avviato")
    
    async def stop(self):
        """Ferma il controller"""
        self.is_running = False
        
        if self._worker_task:
            self._worker_task.cancel()
        if self._routine_scheduler:
            self._routine_scheduler.cancel()
        
        logger.info("ExecutionController fermato")
    
    async def _worker_loop(self):
        """Loop worker per esecuzione task"""
        while self.is_running:
            try:
                # Timeout per permettere check periodici
                priority, task_id, task = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )
                
                # Verifica dipendenze
                deps_met = all(
                    dep not in self.active_tasks or 
                    self.active_tasks[dep].is_complete
                    for dep in task.dependencies
                )
                
                if deps_met:
                    asyncio.create_task(self.execute_task(task))
                else:
                    # Re-enqueue with lower priority
                    await self.task_queue.put((priority + 1, task_id, task))
                    
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    async def _routine_scheduler_loop(self):
        """Scheduler per routine"""
        while self.is_running:
            try:
                now = datetime.now()
                
                for routine in self.routines.values():
                    if not routine.enabled:
                        continue
                    
                    if routine.next_run and routine.next_run <= now:
                        # Esegui routine
                        asyncio.create_task(self.run_routine(routine.id))
                
                # Check ogni 30 secondi
                await asyncio.sleep(30)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
    
    # === Hooks ===
    
    def add_pre_execution_hook(self, hook: Callable):
        """Aggiunge hook pre-esecuzione"""
        self.pre_execution_hooks.append(hook)
    
    def add_post_execution_hook(self, hook: Callable):
        """Aggiunge hook post-esecuzione"""
        self.post_execution_hooks.append(hook)
    
    def add_error_handler(self, handler: Callable):
        """Aggiunge error handler"""
        self.error_handlers.append(handler)
    
    # === Status & Stats ===
    
    def get_status(self) -> dict:
        """Stato completo del controller"""
        return {
            "running": self.is_running,
            "active_tasks": len(self.active_tasks),
            "queued_tasks": self.task_queue.qsize(),
            "completed_tasks": len(self.completed_tasks),
            "total_routines": len(self.routines),
            "enabled_routines": sum(1 for r in self.routines.values() if r.enabled),
            "monitored_apps": len(self.app_controller.monitored_apps),
            "stats": self.stats
        }
    
    def get_execution_history(self, limit: int = 50) -> List[dict]:
        """History delle esecuzioni"""
        history = []
        for task in self.completed_tasks[-limit:]:
            history.append({
                "id": task.id,
                "name": task.name,
                "state": task.state.value,
                "duration": task.duration,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "result": task.result,
                "error": task.error
            })
        return history
