"""
⚙️ AUTOMATION LAYER - Modulo Esecutivo

Questo modulo gestisce:
- Esecuzione azioni
- Workflow complessi
- Registrazione e riproduzione macro
- Automazioni e trigger
- Controllo sistema

Riceve comandi da Jarvis (Supervisor) ed esegue.
"""

from jarvis.executor import Executor
from jarvis.automator import Automator
from jarvis.controller import SystemController
from jarvis.security import SecurityManager
from jarvis.workflow_manager import WorkflowManager, Workflow, WorkflowStatus
from jarvis.macro_recorder import MacroRecorder, Macro, MacroAction

__all__ = [
    'AutomationLayer',
    'Executor', 'Automator', 'SystemController', 'SecurityManager',
    'WorkflowManager', 'Workflow', 'WorkflowStatus',
    'MacroRecorder', 'Macro', 'MacroAction'
]


class AutomationLayer:
    """
    Layer di automazione ed esecuzione
    
    Responsabilità:
    - Eseguire azioni richieste da Jarvis
    - Gestire workflow multi-step
    - Registrare e riprodurre macro
    - Controllare il sistema operativo
    - Gestire automazioni schedulate
    """
    
    def __init__(self):
        # Sicurezza
        self.security = SecurityManager()
        
        # Esecutore principale
        self.executor = Executor(self.security)
        
        # Automazioni
        self.automator = Automator(self.executor)
        
        # Controller sistema
        self.controller = SystemController(self.executor, self.security)
        
        # Workflow manager
        self.workflow_manager = WorkflowManager(self.executor)
        
        # Macro recorder
        self.macro_recorder = MacroRecorder(self.executor)
        
        self.is_active = True
        self._action_history = []
    
    # ========== EXECUTION ==========
    
    async def execute(self, action: dict, bypass_security: bool = False) -> dict:
        """
        Esegue un'azione
        
        Args:
            action: Azione da eseguire
            bypass_security: Salta controlli sicurezza (PERICOLOSO)
        """
        if not bypass_security:
            security_check = await self.security.check_action(action)
            if not security_check["allowed"]:
                return {
                    "success": False,
                    "error": "Azione bloccata dalla sicurezza",
                    "reason": security_check.get("reason")
                }
        
        result = await self.executor.execute(action)
        
        # Log azione
        self._action_history.append({
            "action": action,
            "result": result,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        })
        
        return result
    
    async def execute_batch(self, actions: list) -> list:
        """Esegue una serie di azioni in sequenza"""
        results = []
        for action in actions:
            result = await self.execute(action)
            results.append(result)
            if not result.get("success"):
                break  # Ferma al primo errore
        return results
    
    # ========== WORKFLOWS ==========
    
    async def run_workflow(self, workflow_id: str, context: dict = None) -> dict:
        """Esegue un workflow predefinito"""
        try:
            execution = await self.workflow_manager.execute(workflow_id, context)
            return {
                "success": True,
                "execution_id": execution.id,
                "status": execution.status.value
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def create_workflow(self, name: str, actions: list) -> dict:
        """Crea un workflow da una lista di azioni"""
        workflow_id = f"wf_{name.lower().replace(' ', '_')}"
        workflow = self.workflow_manager.create_from_actions(workflow_id, name, actions)
        return workflow.to_dict()
    
    def list_workflows(self) -> list:
        """Lista tutti i workflow"""
        return self.workflow_manager.list_workflows()
    
    async def pause_workflow(self, execution_id: str) -> bool:
        return await self.workflow_manager.pause(execution_id)
    
    async def resume_workflow(self, execution_id: str) -> bool:
        return await self.workflow_manager.resume(execution_id)
    
    async def cancel_workflow(self, execution_id: str) -> bool:
        return await self.workflow_manager.cancel(execution_id)
    
    # ========== MACROS ==========
    
    def start_recording(self) -> str:
        """Inizia registrazione macro"""
        return self.macro_recorder.start_recording()
    
    def record_action(self, action_type: str, params: dict = None) -> bool:
        """Registra un'azione"""
        return self.macro_recorder.record_action(action_type, params)
    
    def stop_recording(self, name: str, description: str = "") -> dict:
        """Ferma registrazione e salva macro"""
        macro = self.macro_recorder.stop_recording(name, description)
        return macro.to_dict() if macro else {"error": "Nessuna macro"}
    
    def cancel_recording(self):
        """Cancella registrazione"""
        self.macro_recorder.cancel_recording()
    
    async def play_macro(self, macro_id: str, speed: float = 1.0, 
                         loop: bool = False) -> dict:
        """Riproduce una macro"""
        return await self.macro_recorder.play(macro_id, speed, loop)
    
    def stop_playback(self) -> bool:
        """Ferma riproduzione macro"""
        return self.macro_recorder.stop_playback()
    
    def list_macros(self) -> list:
        """Lista tutte le macro"""
        return self.macro_recorder.list_macros()
    
    def delete_macro(self, macro_id: str) -> bool:
        return self.macro_recorder.delete_macro(macro_id)
    
    # ========== AUTOMATIONS ==========
    
    def create_automation(self, automation_id: str, name: str,
                          trigger: dict, actions: list) -> dict:
        """Crea una nuova automazione"""
        automation = self.automator.create_automation(
            automation_id, name, trigger, actions
        )
        return automation.to_dict()
    
    def enable_automation(self, automation_id: str) -> bool:
        return self.automator.enable_automation(automation_id)
    
    def disable_automation(self, automation_id: str) -> bool:
        return self.automator.disable_automation(automation_id)
    
    async def run_automation(self, automation_id: str) -> dict:
        """Esegue manualmente un'automazione"""
        return await self.automator.run_automation(automation_id)
    
    def list_automations(self) -> list:
        return self.automator.list_automations()
    
    async def start_scheduler(self):
        """Avvia scheduler automazioni"""
        await self.automator.start_scheduler()
    
    async def stop_scheduler(self):
        """Ferma scheduler"""
        await self.automator.stop_scheduler()
    
    # ========== SYSTEM CONTROL ==========
    
    async def system_command(self, command: str, params: dict = None) -> dict:
        """Esegue un comando di sistema"""
        return await self.controller.execute_command(command, params or {})
    
    async def get_system_status(self) -> dict:
        """Ottiene stato del sistema"""
        return await self.controller.get_status()
    
    # ========== TRIGGERS ==========
    
    def set_macro_trigger(self, macro_id: str, trigger_type: str, 
                          config: dict) -> bool:
        """Imposta trigger per macro"""
        return self.macro_recorder.set_trigger(macro_id, trigger_type, config)
    
    async def check_triggers(self, event: dict):
        """Controlla se un evento attiva trigger"""
        await self.macro_recorder.check_triggers(event)
        await self.automator.trigger_event(event.get("type"), event)
    
    # ========== STATUS ==========
    
    def get_status(self) -> dict:
        """Stato completo del layer di automazione"""
        return {
            "is_active": self.is_active,
            "executor": {
                "actions_executed": len(self._action_history)
            },
            "workflows": self.workflow_manager.get_statistics(),
            "macros": self.macro_recorder.get_statistics(),
            "automations": self.automator.get_status()
        }
    
    def get_action_history(self, limit: int = 20) -> list:
        """Storico azioni eseguite"""
        return self._action_history[-limit:]
