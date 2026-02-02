"""
⚙️ JARVIS CORE - Workflow Manager
Gestione workflow complessi, catene di azioni e macro
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid


class WorkflowStatus(Enum):
    """Stati di un workflow"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepType(Enum):
    """Tipi di step nel workflow"""
    ACTION = "action"
    CONDITION = "condition"
    LOOP = "loop"
    PARALLEL = "parallel"
    WAIT = "wait"
    SUB_WORKFLOW = "sub_workflow"


@dataclass
class WorkflowStep:
    """Singolo step di un workflow"""
    id: str
    type: StepType
    config: dict
    next_step: Optional[str] = None
    on_error: Optional[str] = None
    timeout: Optional[int] = None
    retries: int = 0
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "config": self.config,
            "next_step": self.next_step,
            "on_error": self.on_error,
            "timeout": self.timeout,
            "retries": self.retries
        }


@dataclass 
class WorkflowExecution:
    """Esecuzione di un workflow"""
    id: str
    workflow_id: str
    status: WorkflowStatus
    current_step: Optional[str]
    started_at: datetime
    completed_at: Optional[datetime] = None
    step_results: Dict[str, dict] = field(default_factory=dict)
    error: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "status": self.status.value,
            "current_step": self.current_step,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "step_results": self.step_results,
            "error": self.error,
            "progress": len(self.step_results)
        }


@dataclass
class Workflow:
    """Definizione di un workflow"""
    id: str
    name: str
    description: str
    steps: Dict[str, WorkflowStep]
    start_step: str
    created_at: datetime = field(default_factory=datetime.now)
    variables: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": {k: v.to_dict() for k, v in self.steps.items()},
            "start_step": self.start_step,
            "created_at": self.created_at.isoformat(),
            "step_count": len(self.steps)
        }


class WorkflowManager:
    """
    Gestisce workflow complessi e catene di azioni
    
    Features:
    - Workflow multi-step con branching
    - Esecuzione parallela
    - Condizioni e loop
    - Sub-workflow annidati
    - Error handling e retry
    """
    
    def __init__(self, executor=None):
        self.executor = executor
        self.workflows: Dict[str, Workflow] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        # Inizializza workflow predefiniti
        self._init_default_workflows()
    
    def _init_default_workflows(self):
        """Crea workflow predefiniti"""
        # Workflow: Pulizia sistema
        cleanup_workflow = self.create_workflow(
            "system_cleanup",
            "Pulizia Sistema",
            "Workflow per pulire file temporanei e liberare risorse"
        )
        self.add_step(cleanup_workflow.id, WorkflowStep(
            id="notify_start",
            type=StepType.ACTION,
            config={"action": "notify", "message": "Inizio pulizia sistema..."},
            next_step="clear_temp"
        ))
        self.add_step(cleanup_workflow.id, WorkflowStep(
            id="clear_temp",
            type=StepType.ACTION,
            config={"action": "clear_temp_files"},
            next_step="check_memory"
        ))
        self.add_step(cleanup_workflow.id, WorkflowStep(
            id="check_memory",
            type=StepType.CONDITION,
            config={
                "condition": "memory_percent > 80",
                "true_step": "free_memory",
                "false_step": "notify_end"
            }
        ))
        self.add_step(cleanup_workflow.id, WorkflowStep(
            id="free_memory",
            type=StepType.ACTION,
            config={"action": "free_memory"},
            next_step="notify_end"
        ))
        self.add_step(cleanup_workflow.id, WorkflowStep(
            id="notify_end",
            type=StepType.ACTION,
            config={"action": "notify", "message": "Pulizia completata!"}
        ))
        
        # Workflow: Backup rapido
        backup_workflow = self.create_workflow(
            "quick_backup",
            "Backup Rapido",
            "Backup dei file importanti"
        )
        self.add_step(backup_workflow.id, WorkflowStep(
            id="start",
            type=StepType.ACTION,
            config={"action": "notify", "message": "Avvio backup..."},
            next_step="backup_docs"
        ))
        self.add_step(backup_workflow.id, WorkflowStep(
            id="backup_docs",
            type=StepType.ACTION,
            config={"action": "backup", "folder": "Documents"},
            next_step="backup_desktop",
            timeout=300,
            retries=2
        ))
        self.add_step(backup_workflow.id, WorkflowStep(
            id="backup_desktop",
            type=StepType.ACTION,
            config={"action": "backup", "folder": "Desktop"},
            next_step="complete"
        ))
        self.add_step(backup_workflow.id, WorkflowStep(
            id="complete",
            type=StepType.ACTION,
            config={"action": "notify", "message": "Backup completato!"}
        ))
    
    # ============================================
    # WORKFLOW CREATION
    # ============================================
    
    def create_workflow(self, workflow_id: str, name: str, 
                        description: str = "") -> Workflow:
        """Crea un nuovo workflow"""
        workflow = Workflow(
            id=workflow_id,
            name=name,
            description=description,
            steps={},
            start_step=""
        )
        self.workflows[workflow_id] = workflow
        return workflow
    
    def add_step(self, workflow_id: str, step: WorkflowStep, 
                 is_start: bool = False) -> bool:
        """Aggiunge uno step a un workflow"""
        if workflow_id not in self.workflows:
            return False
        
        workflow = self.workflows[workflow_id]
        workflow.steps[step.id] = step
        
        if is_start or not workflow.start_step:
            workflow.start_step = step.id
        
        return True
    
    def remove_step(self, workflow_id: str, step_id: str) -> bool:
        """Rimuove uno step da un workflow"""
        if workflow_id not in self.workflows:
            return False
        
        workflow = self.workflows[workflow_id]
        if step_id in workflow.steps:
            del workflow.steps[step_id]
            return True
        return False
    
    def create_from_actions(self, workflow_id: str, name: str,
                            actions: List[dict]) -> Workflow:
        """Crea un workflow lineare da una lista di azioni"""
        workflow = self.create_workflow(workflow_id, name, 
                                        f"Workflow generato da {len(actions)} azioni")
        
        prev_step_id = None
        for i, action in enumerate(actions):
            step_id = f"step_{i}"
            step = WorkflowStep(
                id=step_id,
                type=StepType.ACTION,
                config=action,
                next_step=f"step_{i+1}" if i < len(actions) - 1 else None
            )
            self.add_step(workflow_id, step, is_start=(i == 0))
            prev_step_id = step_id
        
        return workflow
    
    # ============================================
    # WORKFLOW EXECUTION
    # ============================================
    
    async def execute(self, workflow_id: str, 
                      initial_context: dict = None) -> WorkflowExecution:
        """Esegue un workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow '{workflow_id}' non trovato")
        
        workflow = self.workflows[workflow_id]
        
        execution = WorkflowExecution(
            id=f"exec_{uuid.uuid4().hex[:8]}",
            workflow_id=workflow_id,
            status=WorkflowStatus.RUNNING,
            current_step=workflow.start_step,
            started_at=datetime.now(),
            context=initial_context or {}
        )
        
        self.executions[execution.id] = execution
        
        # Esegui in background
        task = asyncio.create_task(self._run_workflow(execution, workflow))
        self.running_tasks[execution.id] = task
        
        return execution
    
    async def _run_workflow(self, execution: WorkflowExecution, 
                            workflow: Workflow):
        """Loop principale di esecuzione workflow"""
        try:
            current_step_id = workflow.start_step
            
            while current_step_id and execution.status == WorkflowStatus.RUNNING:
                if current_step_id not in workflow.steps:
                    raise ValueError(f"Step '{current_step_id}' non trovato")
                
                step = workflow.steps[current_step_id]
                execution.current_step = current_step_id
                
                # Esegui step con retry
                result = await self._execute_step_with_retry(step, execution)
                execution.step_results[current_step_id] = result
                
                if not result.get("success", False):
                    if step.on_error:
                        current_step_id = step.on_error
                    else:
                        execution.status = WorkflowStatus.FAILED
                        execution.error = result.get("error", "Step failed")
                        break
                else:
                    # Determina prossimo step
                    current_step_id = self._get_next_step(step, result, execution)
            
            if execution.status == WorkflowStatus.RUNNING:
                execution.status = WorkflowStatus.COMPLETED
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
        finally:
            execution.completed_at = datetime.now()
            if execution.id in self.running_tasks:
                del self.running_tasks[execution.id]
    
    async def _execute_step_with_retry(self, step: WorkflowStep,
                                        execution: WorkflowExecution) -> dict:
        """Esegue uno step con retry logic"""
        attempts = 0
        max_attempts = step.retries + 1
        
        while attempts < max_attempts:
            try:
                result = await self._execute_step(step, execution)
                if result.get("success", False):
                    return result
            except Exception as e:
                result = {"success": False, "error": str(e)}
            
            attempts += 1
            if attempts < max_attempts:
                await asyncio.sleep(1)  # Wait before retry
        
        return result
    
    async def _execute_step(self, step: WorkflowStep,
                            execution: WorkflowExecution) -> dict:
        """Esegue un singolo step"""
        
        if step.type == StepType.ACTION:
            return await self._execute_action(step.config, execution.context)
        
        elif step.type == StepType.CONDITION:
            return await self._evaluate_condition(step.config, execution.context)
        
        elif step.type == StepType.WAIT:
            seconds = step.config.get("seconds", 1)
            await asyncio.sleep(seconds)
            return {"success": True, "waited": seconds}
        
        elif step.type == StepType.PARALLEL:
            return await self._execute_parallel(step.config, execution.context)
        
        elif step.type == StepType.LOOP:
            return await self._execute_loop(step, execution)
        
        elif step.type == StepType.SUB_WORKFLOW:
            sub_id = step.config.get("workflow_id")
            sub_exec = await self.execute(sub_id, execution.context)
            # Attendi completamento
            while sub_exec.status == WorkflowStatus.RUNNING:
                await asyncio.sleep(0.1)
            return {
                "success": sub_exec.status == WorkflowStatus.COMPLETED,
                "sub_execution": sub_exec.id
            }
        
        return {"success": False, "error": f"Tipo step non supportato: {step.type}"}
    
    async def _execute_action(self, config: dict, context: dict) -> dict:
        """Esegue un'azione"""
        if self.executor:
            return await self.executor.execute(config)
        
        # Simulazione se nessun executor
        action = config.get("action", "unknown")
        return {
            "success": True,
            "action": action,
            "simulated": True,
            "message": f"Azione '{action}' simulata"
        }
    
    async def _evaluate_condition(self, config: dict, context: dict) -> dict:
        """Valuta una condizione"""
        condition = config.get("condition", "True")
        
        # Sostituisci variabili dal contesto
        for key, value in context.items():
            condition = condition.replace(key, str(value))
        
        try:
            result = eval(condition)  # Attenzione: usa solo con input fidato
            return {
                "success": True,
                "condition_result": bool(result),
                "true_step": config.get("true_step"),
                "false_step": config.get("false_step")
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_parallel(self, config: dict, context: dict) -> dict:
        """Esegue azioni in parallelo"""
        actions = config.get("actions", [])
        
        tasks = [self._execute_action(action, context) for action in actions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_success = all(
            isinstance(r, dict) and r.get("success", False) 
            for r in results
        )
        
        return {
            "success": all_success,
            "results": results
        }
    
    async def _execute_loop(self, step: WorkflowStep, 
                           execution: WorkflowExecution) -> dict:
        """Esegue un loop"""
        iterations = step.config.get("iterations", 1)
        body_step = step.config.get("body_step")
        
        if not body_step:
            return {"success": False, "error": "Loop senza body_step"}
        
        results = []
        for i in range(iterations):
            execution.context["loop_index"] = i
            
            if body_step in self.workflows[execution.workflow_id].steps:
                body = self.workflows[execution.workflow_id].steps[body_step]
                result = await self._execute_step(body, execution)
                results.append(result)
                
                if not result.get("success", False):
                    break
        
        return {
            "success": all(r.get("success", False) for r in results),
            "iterations": len(results),
            "results": results
        }
    
    def _get_next_step(self, step: WorkflowStep, result: dict, 
                       execution: WorkflowExecution) -> Optional[str]:
        """Determina il prossimo step"""
        
        if step.type == StepType.CONDITION:
            if result.get("condition_result"):
                return result.get("true_step")
            else:
                return result.get("false_step")
        
        return step.next_step
    
    # ============================================
    # EXECUTION CONTROL
    # ============================================
    
    async def pause(self, execution_id: str) -> bool:
        """Mette in pausa un'esecuzione"""
        if execution_id in self.executions:
            self.executions[execution_id].status = WorkflowStatus.PAUSED
            return True
        return False
    
    async def resume(self, execution_id: str) -> bool:
        """Riprende un'esecuzione in pausa"""
        if execution_id in self.executions:
            exec = self.executions[execution_id]
            if exec.status == WorkflowStatus.PAUSED:
                exec.status = WorkflowStatus.RUNNING
                # Riavvia il task
                workflow = self.workflows[exec.workflow_id]
                task = asyncio.create_task(self._run_workflow(exec, workflow))
                self.running_tasks[execution_id] = task
                return True
        return False
    
    async def cancel(self, execution_id: str) -> bool:
        """Cancella un'esecuzione"""
        if execution_id in self.executions:
            self.executions[execution_id].status = WorkflowStatus.CANCELLED
            if execution_id in self.running_tasks:
                self.running_tasks[execution_id].cancel()
            return True
        return False
    
    # ============================================
    # QUERY METHODS
    # ============================================
    
    def get_workflow(self, workflow_id: str) -> Optional[dict]:
        """Ottiene un workflow"""
        if workflow_id in self.workflows:
            return self.workflows[workflow_id].to_dict()
        return None
    
    def list_workflows(self) -> List[dict]:
        """Lista tutti i workflow"""
        return [w.to_dict() for w in self.workflows.values()]
    
    def get_execution(self, execution_id: str) -> Optional[dict]:
        """Ottiene stato di un'esecuzione"""
        if execution_id in self.executions:
            return self.executions[execution_id].to_dict()
        return None
    
    def list_executions(self, workflow_id: str = None) -> List[dict]:
        """Lista esecuzioni (opzionalmente filtrate per workflow)"""
        executions = []
        for exec in self.executions.values():
            if workflow_id is None or exec.workflow_id == workflow_id:
                executions.append(exec.to_dict())
        return executions
    
    def get_statistics(self) -> dict:
        """Statistiche sui workflow"""
        total_executions = len(self.executions)
        completed = sum(1 for e in self.executions.values() 
                       if e.status == WorkflowStatus.COMPLETED)
        failed = sum(1 for e in self.executions.values() 
                    if e.status == WorkflowStatus.FAILED)
        
        return {
            "total_workflows": len(self.workflows),
            "total_executions": total_executions,
            "completed": completed,
            "failed": failed,
            "success_rate": completed / total_executions if total_executions > 0 else 0,
            "running": len(self.running_tasks)
        }
