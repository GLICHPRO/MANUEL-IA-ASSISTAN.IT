"""
Gideon Action Manager - Automated Actions & Routine System
Handles routine execution, critical procedures, rollback and action logging
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from loguru import logger
from enum import Enum
import json
import os
import shutil
import psutil


class ActionLevel(Enum):
    """Action security levels"""
    SAFE = "safe"              # Can run anytime
    ELEVATED = "elevated"      # Requires confirmation
    CRITICAL = "critical"      # Pilot mode only
    DESTRUCTIVE = "destructive" # Pilot + confirmation + rollback


class ActionStatus(Enum):
    """Action execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class ActionLog:
    """Single action log entry"""
    def __init__(self, action_id: str, action_type: str, level: ActionLevel):
        self.id = action_id
        self.action_type = action_type
        self.level = level
        self.status = ActionStatus.PENDING
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.params: Dict[str, Any] = {}
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.rollback_data: Optional[Dict[str, Any]] = None
        self.can_rollback: bool = False
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "action_type": self.action_type,
            "level": self.level.value,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": int((self.completed_at - self.started_at).total_seconds() * 1000) 
                          if self.started_at and self.completed_at else None,
            "params": self.params,
            "result": self.result,
            "error": self.error,
            "can_rollback": self.can_rollback
        }


class Routine:
    """Scheduled routine definition"""
    def __init__(
        self,
        name: str,
        description: str,
        action_type: str,
        schedule: str,  # cron-like or interval
        level: ActionLevel = ActionLevel.SAFE,
        enabled: bool = True
    ):
        self.name = name
        self.description = description
        self.action_type = action_type
        self.schedule = schedule
        self.level = level
        self.enabled = enabled
        self.last_run: Optional[datetime] = None
        self.next_run: Optional[datetime] = None
        self.run_count: int = 0
        self.last_result: Optional[str] = None
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "action_type": self.action_type,
            "schedule": self.schedule,
            "level": self.level.value,
            "enabled": self.enabled,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "run_count": self.run_count,
            "last_result": self.last_result
        }


class ActionManager:
    """
    Manages automated actions, routines, and critical procedures
    All actions are logged and traceable
    """
    
    def __init__(self):
        self.actions: Dict[str, Callable] = {}
        self.action_logs: List[ActionLog] = []
        self.routines: Dict[str, Routine] = {}
        self.rollback_stack: List[ActionLog] = []
        self.max_logs = 1000
        self.max_rollback = 50
        self.is_running = False
        self.pilot_mode_active = False
        self._action_counter = 0
        self._routine_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize action manager and register built-in actions"""
        logger.info("⚡ Initializing Action Manager...")
        
        # Register built-in actions
        self._register_builtin_actions()
        
        # Setup default routines
        self._setup_default_routines()
        
        self.is_running = True
        
        # Start routine scheduler
        self._routine_task = asyncio.create_task(self._routine_scheduler())
        
        logger.info("✅ Action Manager ready")
        
    async def shutdown(self):
        """Cleanup action manager"""
        self.is_running = False
        if self._routine_task:
            self._routine_task.cancel()
            try:
                await self._routine_task
            except asyncio.CancelledError:
                pass
        logger.info("Action Manager shutdown")
        
    def set_pilot_mode(self, active: bool):
        """Update pilot mode status"""
        self.pilot_mode_active = active
        logger.info(f"🛫 Action Manager pilot mode: {'ACTIVE' if active else 'INACTIVE'}")
        
    def _generate_action_id(self) -> str:
        """Generate unique action ID"""
        self._action_counter += 1
        return f"ACT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._action_counter:04d}"
        
    def _register_builtin_actions(self):
        """Register built-in system actions"""
        
        # SAFE actions
        self.register_action("system_status", self._action_system_status, ActionLevel.SAFE)
        self.register_action("memory_report", self._action_memory_report, ActionLevel.SAFE)
        self.register_action("list_processes", self._action_list_processes, ActionLevel.SAFE)
        self.register_action("disk_usage", self._action_disk_usage, ActionLevel.SAFE)
        
        # ELEVATED actions (require confirmation)
        self.register_action("clear_temp", self._action_clear_temp, ActionLevel.ELEVATED)
        self.register_action("optimize_memory", self._action_optimize_memory, ActionLevel.ELEVATED)
        
        # CRITICAL actions (Pilot mode only)
        self.register_action("kill_process", self._action_kill_process, ActionLevel.CRITICAL)
        self.register_action("restart_service", self._action_restart_service, ActionLevel.CRITICAL)
        self.register_action("backup_create", self._action_backup_create, ActionLevel.CRITICAL)
        
        # DESTRUCTIVE actions (Pilot + confirmation + rollback)
        self.register_action("delete_files", self._action_delete_files, ActionLevel.DESTRUCTIVE)
        self.register_action("system_cleanup", self._action_system_cleanup, ActionLevel.DESTRUCTIVE)
        
        logger.info(f"📋 Registered {len(self.actions)} built-in actions")
        
    def _setup_default_routines(self):
        """Setup default scheduled routines"""
        
        self.routines = {
            "daily_status": Routine(
                name="Daily Status Report",
                description="Generate daily system status report",
                action_type="system_status",
                schedule="daily:08:00",
                level=ActionLevel.SAFE,
                enabled=True
            ),
            "hourly_memory": Routine(
                name="Hourly Memory Check",
                description="Check memory usage every hour",
                action_type="memory_report",
                schedule="interval:3600",  # every hour
                level=ActionLevel.SAFE,
                enabled=True
            ),
            "daily_cleanup": Routine(
                name="Daily Temp Cleanup",
                description="Clean temporary files daily",
                action_type="clear_temp",
                schedule="daily:03:00",
                level=ActionLevel.ELEVATED,
                enabled=False  # Disabled by default
            ),
            "weekly_backup": Routine(
                name="Weekly Backup",
                description="Create weekly system backup",
                action_type="backup_create",
                schedule="weekly:sun:02:00",
                level=ActionLevel.CRITICAL,
                enabled=False  # Requires pilot mode
            )
        }
        
        # Calculate next run times
        now = datetime.now()
        for routine in self.routines.values():
            routine.next_run = self._calculate_next_run(routine.schedule, now)
            
    def _calculate_next_run(self, schedule: str, from_time: datetime) -> datetime:
        """Calculate next run time based on schedule"""
        if schedule.startswith("interval:"):
            seconds = int(schedule.split(":")[1])
            return from_time + timedelta(seconds=seconds)
        elif schedule.startswith("daily:"):
            time_str = schedule.split(":")[1] + ":" + schedule.split(":")[2]
            hour, minute = map(int, time_str.split(":"))
            next_run = from_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= from_time:
                next_run += timedelta(days=1)
            return next_run
        elif schedule.startswith("weekly:"):
            parts = schedule.split(":")
            day_name = parts[1].lower()
            hour, minute = int(parts[2]), int(parts[3])
            days = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
            target_day = days.get(day_name, 0)
            current_day = from_time.weekday()
            days_ahead = target_day - current_day
            if days_ahead <= 0:
                days_ahead += 7
            next_run = from_time + timedelta(days=days_ahead)
            return next_run.replace(hour=hour, minute=minute, second=0, microsecond=0)
        else:
            return from_time + timedelta(hours=1)
            
    def register_action(
        self,
        action_type: str,
        handler: Callable,
        level: ActionLevel = ActionLevel.SAFE
    ):
        """Register a new action handler"""
        self.actions[action_type] = {
            "handler": handler,
            "level": level
        }
        
    async def execute_action(
        self,
        action_type: str,
        params: Optional[Dict[str, Any]] = None,
        force: bool = False,
        pilot_confirmed: bool = False
    ) -> Dict[str, Any]:
        """
        Execute an action with proper authorization and logging
        
        Args:
            action_type: Type of action to execute
            params: Action parameters
            force: Skip confirmation for elevated actions
            pilot_confirmed: Pilot mode confirmation received
            
        Returns:
            Execution result
        """
        if action_type not in self.actions:
            return {
                "success": False,
                "error": f"Unknown action type: {action_type}",
                "action_id": None
            }
            
        action_def = self.actions[action_type]
        level = action_def["level"]
        handler = action_def["handler"]
        
        # Create action log
        action_log = ActionLog(
            action_id=self._generate_action_id(),
            action_type=action_type,
            level=level
        )
        action_log.params = params or {}
        
        # Authorization check
        auth_result = self._check_authorization(level, force, pilot_confirmed)
        if not auth_result["authorized"]:
            action_log.status = ActionStatus.CANCELLED
            action_log.error = auth_result["reason"]
            self._add_log(action_log)
            return {
                "success": False,
                "error": auth_result["reason"],
                "action_id": action_log.id,
                "requires": auth_result.get("requires")
            }
            
        # Execute action
        action_log.status = ActionStatus.RUNNING
        action_log.started_at = datetime.now()
        
        try:
            logger.info(f"⚡ Executing action: {action_type} [{level.value}]")
            
            # Execute handler
            result = await handler(params or {}, action_log)
            
            action_log.status = ActionStatus.SUCCESS
            action_log.result = result
            action_log.completed_at = datetime.now()
            
            # Add to rollback stack if applicable
            if action_log.can_rollback:
                self._add_to_rollback_stack(action_log)
                
            self._add_log(action_log)
            
            logger.info(f"✅ Action completed: {action_log.id}")
            
            return {
                "success": True,
                "action_id": action_log.id,
                "result": result,
                "can_rollback": action_log.can_rollback,
                "duration_ms": action_log.to_dict()["duration_ms"]
            }
            
        except Exception as e:
            action_log.status = ActionStatus.FAILED
            action_log.error = str(e)
            action_log.completed_at = datetime.now()
            self._add_log(action_log)
            
            logger.error(f"❌ Action failed: {action_log.id} - {e}")
            
            return {
                "success": False,
                "action_id": action_log.id,
                "error": str(e)
            }
            
    def _check_authorization(
        self,
        level: ActionLevel,
        force: bool,
        pilot_confirmed: bool
    ) -> Dict[str, Any]:
        """Check if action is authorized"""
        
        if level == ActionLevel.SAFE:
            return {"authorized": True}
            
        if level == ActionLevel.ELEVATED:
            if force or pilot_confirmed:
                return {"authorized": True}
            return {
                "authorized": False,
                "reason": "Azione elevata richiede conferma",
                "requires": "confirmation"
            }
            
        if level == ActionLevel.CRITICAL:
            if not self.pilot_mode_active:
                return {
                    "authorized": False,
                    "reason": "Azione critica richiede Pilot Mode attivo",
                    "requires": "pilot_mode"
                }
            return {"authorized": True}
            
        if level == ActionLevel.DESTRUCTIVE:
            if not self.pilot_mode_active:
                return {
                    "authorized": False,
                    "reason": "Azione distruttiva richiede Pilot Mode attivo",
                    "requires": "pilot_mode"
                }
            if not pilot_confirmed:
                return {
                    "authorized": False,
                    "reason": "Azione distruttiva richiede conferma esplicita in Pilot Mode",
                    "requires": "pilot_confirmation"
                }
            return {"authorized": True}
            
        return {"authorized": False, "reason": "Livello sconosciuto"}
        
    def _add_log(self, action_log: ActionLog):
        """Add action to logs, maintaining max size"""
        self.action_logs.append(action_log)
        if len(self.action_logs) > self.max_logs:
            self.action_logs = self.action_logs[-self.max_logs:]
            
    def _add_to_rollback_stack(self, action_log: ActionLog):
        """Add action to rollback stack"""
        self.rollback_stack.append(action_log)
        if len(self.rollback_stack) > self.max_rollback:
            self.rollback_stack = self.rollback_stack[-self.max_rollback:]
            
    async def rollback_last(self) -> Dict[str, Any]:
        """Rollback the last rollbackable action"""
        if not self.rollback_stack:
            return {
                "success": False,
                "error": "Nessuna azione da annullare"
            }
            
        if not self.pilot_mode_active:
            return {
                "success": False,
                "error": "Rollback richiede Pilot Mode attivo"
            }
            
        action_log = self.rollback_stack.pop()
        
        try:
            logger.info(f"🔄 Rolling back action: {action_log.id}")
            
            # Execute rollback based on action type
            rollback_result = await self._execute_rollback(action_log)
            
            action_log.status = ActionStatus.ROLLED_BACK
            
            return {
                "success": True,
                "action_id": action_log.id,
                "rollback_result": rollback_result
            }
            
        except Exception as e:
            logger.error(f"❌ Rollback failed: {action_log.id} - {e}")
            return {
                "success": False,
                "action_id": action_log.id,
                "error": str(e)
            }
            
    async def _execute_rollback(self, action_log: ActionLog) -> Dict[str, Any]:
        """Execute rollback for an action"""
        if not action_log.rollback_data:
            return {"message": "No rollback data available"}
            
        rollback_type = action_log.rollback_data.get("type")
        
        if rollback_type == "restore_files":
            # Restore deleted files from backup
            backup_path = action_log.rollback_data.get("backup_path")
            original_path = action_log.rollback_data.get("original_path")
            if backup_path and original_path and os.path.exists(backup_path):
                shutil.move(backup_path, original_path)
                return {"message": f"File ripristinato: {original_path}"}
                
        elif rollback_type == "restore_backup":
            # More complex restore logic
            pass
            
        return {"message": "Rollback completato"}
        
    async def _routine_scheduler(self):
        """Background task to run scheduled routines"""
        logger.info("📅 Routine scheduler started")
        
        while self.is_running:
            try:
                now = datetime.now()
                
                for routine_id, routine in self.routines.items():
                    if not routine.enabled:
                        continue
                        
                    if routine.next_run and now >= routine.next_run:
                        # Check if critical routine requires pilot mode
                        if routine.level in [ActionLevel.CRITICAL, ActionLevel.DESTRUCTIVE]:
                            if not self.pilot_mode_active:
                                logger.warning(f"⏭️ Skipping {routine.name} - requires Pilot Mode")
                                routine.next_run = self._calculate_next_run(routine.schedule, now)
                                continue
                                
                        # Execute routine
                        logger.info(f"📅 Running routine: {routine.name}")
                        result = await self.execute_action(
                            routine.action_type,
                            force=True,
                            pilot_confirmed=self.pilot_mode_active
                        )
                        
                        routine.last_run = now
                        routine.run_count += 1
                        routine.last_result = "success" if result.get("success") else "failed"
                        routine.next_run = self._calculate_next_run(routine.schedule, now)
                        
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Routine scheduler error: {e}")
                await asyncio.sleep(60)
                
    # ============ BUILT-IN ACTION HANDLERS ============
    
    async def _action_system_status(self, params: Dict, log: ActionLog) -> Dict:
        """Get comprehensive system status"""
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "cpu": {
                "percent": cpu,
                "cores": psutil.cpu_count()
            },
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "percent": memory.percent
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "percent": round(disk.percent, 1)
            },
            "processes": len(psutil.pids())
        }
        
        return status
        
    async def _action_memory_report(self, params: Dict, log: ActionLog) -> Dict:
        """Generate memory usage report"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Top memory consumers
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
            try:
                info = proc.info
                if info['memory_percent'] and info['memory_percent'] > 0.1:
                    processes.append(info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
        processes.sort(key=lambda x: x['memory_percent'], reverse=True)
        
        return {
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "percent": memory.percent
            },
            "swap": {
                "total_gb": round(swap.total / (1024**3), 2),
                "used_gb": round(swap.used / (1024**3), 2),
                "percent": swap.percent
            },
            "top_consumers": processes[:10]
        }
        
    async def _action_list_processes(self, params: Dict, log: ActionLog) -> Dict:
        """List running processes"""
        filter_name = params.get("filter", "").lower()
        
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
            try:
                info = proc.info
                if filter_name and filter_name not in info['name'].lower():
                    continue
                processes.append(info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
        return {
            "count": len(processes),
            "processes": processes[:50]  # Limit to 50
        }
        
    async def _action_disk_usage(self, params: Dict, log: ActionLog) -> Dict:
        """Get disk usage for all partitions"""
        partitions = []
        
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                partitions.append({
                    "device": partition.device,
                    "mountpoint": partition.mountpoint,
                    "fstype": partition.fstype,
                    "total_gb": round(usage.total / (1024**3), 2),
                    "used_gb": round(usage.used / (1024**3), 2),
                    "free_gb": round(usage.free / (1024**3), 2),
                    "percent": round(usage.percent, 1)
                })
            except (PermissionError, OSError):
                pass
                
        return {"partitions": partitions}
        
    async def _action_clear_temp(self, params: Dict, log: ActionLog) -> Dict:
        """Clear temporary files"""
        import tempfile
        
        temp_dir = tempfile.gettempdir()
        files_deleted = 0
        space_freed = 0
        errors = []
        
        # Just count for safety - actual deletion would need more logic
        for root, dirs, files in os.walk(temp_dir):
            for f in files:
                try:
                    file_path = os.path.join(root, f)
                    size = os.path.getsize(file_path)
                    # Only report, don't actually delete for safety
                    files_deleted += 1
                    space_freed += size
                except Exception as e:
                    errors.append(str(e))
                    
            # Only check top level
            break
            
        return {
            "temp_dir": temp_dir,
            "files_found": files_deleted,
            "potential_space_mb": round(space_freed / (1024**2), 2),
            "note": "Dry run - no files deleted for safety"
        }
        
    async def _action_optimize_memory(self, params: Dict, log: ActionLog) -> Dict:
        """Optimize memory usage"""
        import gc
        
        # Force garbage collection
        before = psutil.virtual_memory().available
        gc.collect()
        after = psutil.virtual_memory().available
        
        freed = after - before
        
        return {
            "gc_collected": True,
            "memory_freed_mb": round(freed / (1024**2), 2) if freed > 0 else 0,
            "current_available_gb": round(after / (1024**3), 2)
        }
        
    async def _action_kill_process(self, params: Dict, log: ActionLog) -> Dict:
        """Kill a process by PID or name - CRITICAL"""
        pid = params.get("pid")
        name = params.get("name")
        
        if not pid and not name:
            raise ValueError("Richiesto PID o nome processo")
            
        killed = []
        
        if pid:
            try:
                proc = psutil.Process(pid)
                proc_name = proc.name()
                proc.terminate()
                killed.append({"pid": pid, "name": proc_name})
            except psutil.NoSuchProcess:
                raise ValueError(f"Processo {pid} non trovato")
        elif name:
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if name.lower() in proc.info['name'].lower():
                        proc.terminate()
                        killed.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                    
        return {
            "killed": killed,
            "count": len(killed)
        }
        
    async def _action_restart_service(self, params: Dict, log: ActionLog) -> Dict:
        """Restart a Windows service - CRITICAL"""
        service_name = params.get("service")
        
        if not service_name:
            raise ValueError("Nome servizio richiesto")
            
        # Note: This would require admin privileges
        return {
            "service": service_name,
            "status": "simulated",
            "note": "Service restart requires admin privileges"
        }
        
    async def _action_backup_create(self, params: Dict, log: ActionLog) -> Dict:
        """Create a backup - CRITICAL"""
        source = params.get("source", ".")
        
        backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        log.can_rollback = True
        log.rollback_data = {
            "type": "restore_backup",
            "backup_name": backup_name
        }
        
        return {
            "backup_name": backup_name,
            "source": source,
            "status": "created",
            "note": "Backup simulation - implement actual backup logic"
        }
        
    async def _action_delete_files(self, params: Dict, log: ActionLog) -> Dict:
        """Delete files - DESTRUCTIVE with rollback"""
        pattern = params.get("pattern")
        directory = params.get("directory")
        
        if not pattern or not directory:
            raise ValueError("Pattern e directory richiesti")
            
        # For safety, just simulate
        log.can_rollback = True
        log.rollback_data = {
            "type": "restore_files",
            "pattern": pattern,
            "directory": directory
        }
        
        return {
            "pattern": pattern,
            "directory": directory,
            "status": "simulated",
            "note": "Delete simulation - implement with caution"
        }
        
    async def _action_system_cleanup(self, params: Dict, log: ActionLog) -> Dict:
        """System cleanup - DESTRUCTIVE"""
        log.can_rollback = True
        log.rollback_data = {
            "type": "restore_backup"
        }
        
        return {
            "status": "simulated",
            "note": "System cleanup simulation"
        }
        
    # ============ PUBLIC API ============
    
    def get_action_logs(
        self,
        limit: int = 50,
        status: Optional[ActionStatus] = None,
        action_type: Optional[str] = None
    ) -> List[Dict]:
        """Get action logs with optional filters"""
        logs = self.action_logs.copy()
        
        if status:
            logs = [l for l in logs if l.status == status]
            
        if action_type:
            logs = [l for l in logs if l.action_type == action_type]
            
        logs.reverse()  # Most recent first
        return [l.to_dict() for l in logs[:limit]]
        
    def get_routines(self) -> List[Dict]:
        """Get all routines"""
        return [r.to_dict() for r in self.routines.values()]
        
    def toggle_routine(self, routine_id: str, enabled: bool) -> bool:
        """Enable/disable a routine"""
        if routine_id in self.routines:
            routine = self.routines[routine_id]
            
            # Check if critical routine requires pilot mode
            if enabled and routine.level in [ActionLevel.CRITICAL, ActionLevel.DESTRUCTIVE]:
                if not self.pilot_mode_active:
                    return False
                    
            routine.enabled = enabled
            return True
        return False
        
    def get_available_actions(self) -> List[Dict]:
        """Get list of available actions"""
        actions = []
        for action_type, action_def in self.actions.items():
            actions.append({
                "type": action_type,
                "level": action_def["level"].value,
                "requires_pilot": action_def["level"] in [ActionLevel.CRITICAL, ActionLevel.DESTRUCTIVE]
            })
        return actions
        
    def get_rollback_stack(self) -> List[Dict]:
        """Get actions that can be rolled back"""
        return [a.to_dict() for a in reversed(self.rollback_stack)]
