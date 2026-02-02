"""
⚙️ GIDEON Workflow Engine - Automazioni stile n8n

Sistema di automazione visuale per creare workflow complessi:
- Trigger (eventi, schedule, webhook)
- Actions (AI, API, file, notifiche)
- Conditions (if/else, switch)
- Loops (for each, while)

Esempi workflow:
1. "Quando ricevo email importante → Riassumi con AI → Invia su WhatsApp"
2. "Ogni giorno alle 9:00 → Leggi news → Genera report → Invia email"
3. "Quando dico 'crea video' → Genera script → HeyGen → Pubblica"
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from loguru import logger
import uuid
import schedule
import threading


class NodeType(Enum):
    """Tipi di nodi nel workflow"""
    TRIGGER = "trigger"      # Avvia il workflow
    ACTION = "action"        # Esegue un'azione
    CONDITION = "condition"  # Decisione if/else
    LOOP = "loop"           # Iterazione
    DELAY = "delay"         # Attesa
    MERGE = "merge"         # Unisce branch


class TriggerType(Enum):
    """Tipi di trigger"""
    MANUAL = "manual"           # Avvio manuale
    SCHEDULE = "schedule"       # Cron/intervallo
    WEBHOOK = "webhook"         # HTTP webhook
    VOICE = "voice"             # Comando vocale
    EVENT = "event"             # Evento interno
    FILE = "file"               # File modificato
    EMAIL = "email"             # Email ricevuta


class ActionType(Enum):
    """Tipi di azioni"""
    # AI Actions
    AI_CHAT = "ai_chat"
    AI_IMAGE = "ai_image"
    AI_VIDEO = "ai_video"
    AI_VOICE = "ai_voice"
    AI_MUSIC = "ai_music"
    AI_ANALYZE = "ai_analyze"
    
    # Communication
    SEND_EMAIL = "send_email"
    SEND_WHATSAPP = "send_whatsapp"
    SEND_TELEGRAM = "send_telegram"
    SEND_NOTIFICATION = "send_notification"
    
    # File Operations
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    DOWNLOAD = "download"
    UPLOAD = "upload"
    
    # Web
    HTTP_REQUEST = "http_request"
    SCRAPE_WEB = "scrape_web"
    
    # System
    RUN_COMMAND = "run_command"
    RUN_SCRIPT = "run_script"
    SET_VARIABLE = "set_variable"
    
    # Custom
    CUSTOM = "custom"


@dataclass
class WorkflowNode:
    """Nodo di un workflow"""
    id: str
    type: NodeType
    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    next_nodes: List[str] = field(default_factory=list)
    position: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0})


@dataclass
class Workflow:
    """Definizione di un workflow"""
    id: str
    name: str
    description: str = ""
    nodes: Dict[str, WorkflowNode] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_run: Optional[datetime] = None
    run_count: int = 0


@dataclass
class WorkflowExecution:
    """Esecuzione di un workflow"""
    id: str
    workflow_id: str
    status: str = "running"  # running, completed, failed, cancelled
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    current_node: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict] = field(default_factory=list)
    error: Optional[str] = None


class WorkflowEngine:
    """
    ⚙️ GIDEON Workflow Engine
    
    Crea ed esegue automazioni complesse con:
    - Editor visuale (API per frontend)
    - Trigger multipli
    - Azioni AI integrate
    - Condizioni e loop
    """
    
    def __init__(self):
        self.workflows: Dict[str, Workflow] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.action_handlers: Dict[ActionType, Callable] = {}
        self.scheduler_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Storage
        self.storage_path = Path(__file__).parent.parent / "data" / "workflows"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Registra handlers di default
        self._register_default_handlers()
        
        # Carica workflows salvati
        self._load_workflows()
    
    def _register_default_handlers(self):
        """Registra gli handler di default per le azioni"""
        
        # AI Actions
        self.action_handlers[ActionType.AI_CHAT] = self._handle_ai_chat
        self.action_handlers[ActionType.AI_IMAGE] = self._handle_ai_image
        self.action_handlers[ActionType.AI_VIDEO] = self._handle_ai_video
        self.action_handlers[ActionType.AI_VOICE] = self._handle_ai_voice
        self.action_handlers[ActionType.AI_ANALYZE] = self._handle_ai_analyze
        
        # Communication
        self.action_handlers[ActionType.SEND_NOTIFICATION] = self._handle_notification
        self.action_handlers[ActionType.SEND_EMAIL] = self._handle_email
        
        # File
        self.action_handlers[ActionType.READ_FILE] = self._handle_read_file
        self.action_handlers[ActionType.WRITE_FILE] = self._handle_write_file
        
        # Web
        self.action_handlers[ActionType.HTTP_REQUEST] = self._handle_http_request
        
        # System
        self.action_handlers[ActionType.SET_VARIABLE] = self._handle_set_variable
        
        logger.info(f"📋 Registrati {len(self.action_handlers)} action handlers")
    
    # ============================================================
    #                    WORKFLOW MANAGEMENT
    # ============================================================
    
    def create_workflow(
        self,
        name: str,
        description: str = "",
        nodes: List[Dict] = None
    ) -> Workflow:
        """Crea un nuovo workflow"""
        workflow_id = str(uuid.uuid4())[:8]
        
        workflow = Workflow(
            id=workflow_id,
            name=name,
            description=description
        )
        
        # Aggiungi nodi se forniti
        if nodes:
            for node_data in nodes:
                node = WorkflowNode(
                    id=node_data.get("id", str(uuid.uuid4())[:8]),
                    type=NodeType(node_data["type"]),
                    name=node_data["name"],
                    config=node_data.get("config", {}),
                    next_nodes=node_data.get("next_nodes", []),
                    position=node_data.get("position", {"x": 0, "y": 0})
                )
                workflow.nodes[node.id] = node
        
        self.workflows[workflow_id] = workflow
        self._save_workflow(workflow)
        
        logger.info(f"✅ Workflow creato: {name} ({workflow_id})")
        return workflow
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Ottieni un workflow"""
        return self.workflows.get(workflow_id)
    
    def list_workflows(self) -> List[Dict]:
        """Lista tutti i workflow"""
        return [
            {
                "id": w.id,
                "name": w.name,
                "description": w.description,
                "enabled": w.enabled,
                "nodes_count": len(w.nodes),
                "last_run": w.last_run.isoformat() if w.last_run else None,
                "run_count": w.run_count
            }
            for w in self.workflows.values()
        ]
    
    def delete_workflow(self, workflow_id: str) -> bool:
        """Elimina un workflow"""
        if workflow_id in self.workflows:
            del self.workflows[workflow_id]
            # Rimuovi file
            file_path = self.storage_path / f"{workflow_id}.json"
            if file_path.exists():
                file_path.unlink()
            logger.info(f"🗑️ Workflow eliminato: {workflow_id}")
            return True
        return False
    
    def add_node(
        self,
        workflow_id: str,
        node_type: str,
        name: str,
        config: Dict = None,
        position: Dict = None
    ) -> Optional[WorkflowNode]:
        """Aggiungi un nodo a un workflow"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return None
        
        node = WorkflowNode(
            id=str(uuid.uuid4())[:8],
            type=NodeType(node_type),
            name=name,
            config=config or {},
            position=position or {"x": 0, "y": 0}
        )
        
        workflow.nodes[node.id] = node
        self._save_workflow(workflow)
        
        return node
    
    def connect_nodes(
        self,
        workflow_id: str,
        source_id: str,
        target_id: str
    ) -> bool:
        """Collega due nodi"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return False
        
        if source_id in workflow.nodes and target_id in workflow.nodes:
            if target_id not in workflow.nodes[source_id].next_nodes:
                workflow.nodes[source_id].next_nodes.append(target_id)
                self._save_workflow(workflow)
                return True
        return False
    
    # ============================================================
    #                    WORKFLOW EXECUTION
    # ============================================================
    
    async def execute_workflow(
        self,
        workflow_id: str,
        input_data: Dict = None,
        trigger_type: str = "manual"
    ) -> WorkflowExecution:
        """Esegue un workflow"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} non trovato")
        
        if not workflow.enabled:
            raise ValueError(f"Workflow {workflow_id} è disabilitato")
        
        # Crea esecuzione
        execution = WorkflowExecution(
            id=str(uuid.uuid4())[:8],
            workflow_id=workflow_id,
            context={
                "input": input_data or {},
                "variables": dict(workflow.variables),
                "trigger": trigger_type
            }
        )
        self.executions[execution.id] = execution
        
        logger.info(f"▶️ Esecuzione workflow: {workflow.name} ({execution.id})")
        
        try:
            # Trova nodo trigger/start
            start_node = self._find_start_node(workflow)
            if not start_node:
                raise ValueError("Nessun nodo di partenza trovato")
            
            # Esegui workflow
            await self._execute_node(workflow, execution, start_node.id)
            
            execution.status = "completed"
            execution.completed_at = datetime.now()
            
            # Aggiorna statistiche workflow
            workflow.last_run = datetime.now()
            workflow.run_count += 1
            self._save_workflow(workflow)
            
            logger.info(f"✅ Workflow completato: {workflow.name}")
            
        except Exception as e:
            execution.status = "failed"
            execution.error = str(e)
            execution.completed_at = datetime.now()
            logger.error(f"❌ Workflow fallito: {e}")
        
        return execution
    
    async def _execute_node(
        self,
        workflow: Workflow,
        execution: WorkflowExecution,
        node_id: str
    ):
        """Esegue un singolo nodo"""
        node = workflow.nodes.get(node_id)
        if not node:
            return
        
        execution.current_node = node_id
        self._log_execution(execution, f"Esecuzione nodo: {node.name}")
        
        try:
            if node.type == NodeType.TRIGGER:
                # I trigger non eseguono azioni, passano ai nodi successivi
                pass
            
            elif node.type == NodeType.ACTION:
                # Esegui l'azione
                action_type = ActionType(node.config.get("action_type", "custom"))
                result = await self._execute_action(action_type, node.config, execution.context)
                
                # Salva risultato nel contesto
                execution.context[f"node_{node_id}_result"] = result
                self._log_execution(execution, f"Risultato: {result.get('success', False)}")
            
            elif node.type == NodeType.CONDITION:
                # Valuta condizione
                condition = node.config.get("condition", "true")
                result = self._evaluate_condition(condition, execution.context)
                
                # Scegli branch
                if result:
                    node.next_nodes = [node.config.get("true_branch")]
                else:
                    node.next_nodes = [node.config.get("false_branch")]
            
            elif node.type == NodeType.DELAY:
                # Attendi
                delay_seconds = node.config.get("seconds", 1)
                await asyncio.sleep(delay_seconds)
            
            elif node.type == NodeType.LOOP:
                # Loop su array
                items = self._resolve_variable(node.config.get("items", []), execution.context)
                body_node = node.config.get("body_node")
                
                for i, item in enumerate(items):
                    execution.context["loop_index"] = i
                    execution.context["loop_item"] = item
                    if body_node:
                        await self._execute_node(workflow, execution, body_node)
            
            # Esegui nodi successivi
            for next_id in node.next_nodes:
                if next_id:
                    await self._execute_node(workflow, execution, next_id)
                    
        except Exception as e:
            self._log_execution(execution, f"Errore: {e}", level="error")
            raise
    
    async def _execute_action(
        self,
        action_type: ActionType,
        config: Dict,
        context: Dict
    ) -> Dict:
        """Esegue un'azione specifica"""
        # Risolvi variabili nel config
        resolved_config = self._resolve_config(config, context)
        
        handler = self.action_handlers.get(action_type)
        if handler:
            return await handler(resolved_config, context)
        
        return {"success": False, "error": f"Handler non trovato per {action_type}"}
    
    # ============================================================
    #                    ACTION HANDLERS
    # ============================================================
    
    async def _handle_ai_chat(self, config: Dict, context: Dict) -> Dict:
        """Handler per AI Chat"""
        from integrations.ai_hub import get_ai_hub
        
        hub = get_ai_hub()
        await hub.initialize()
        
        return await hub.chat(
            message=config.get("message", ""),
            provider=config.get("provider", "auto"),
            model=config.get("model"),
            system_prompt=config.get("system_prompt"),
            max_tokens=config.get("max_tokens", 500)
        )
    
    async def _handle_ai_image(self, config: Dict, context: Dict) -> Dict:
        """Handler per AI Image Generation"""
        from integrations.ai_hub import get_ai_hub
        
        hub = get_ai_hub()
        await hub.initialize()
        
        return await hub.generate_image(
            prompt=config.get("prompt", ""),
            provider=config.get("provider", "auto"),
            size=config.get("size", "1024x1024")
        )
    
    async def _handle_ai_video(self, config: Dict, context: Dict) -> Dict:
        """Handler per AI Video Generation"""
        from integrations.ai_hub import get_ai_hub
        
        hub = get_ai_hub()
        await hub.initialize()
        
        return await hub.generate_video(
            prompt=config.get("prompt"),
            script=config.get("script"),
            avatar_id=config.get("avatar_id")
        )
    
    async def _handle_ai_voice(self, config: Dict, context: Dict) -> Dict:
        """Handler per AI Voice/TTS"""
        from integrations.ai_hub import get_ai_hub
        
        hub = get_ai_hub()
        await hub.initialize()
        
        return await hub.text_to_speech(
            text=config.get("text", ""),
            provider=config.get("provider", "auto"),
            voice=config.get("voice")
        )
    
    async def _handle_ai_analyze(self, config: Dict, context: Dict) -> Dict:
        """Handler per AI Analysis"""
        from automation.smart_actions import smart_actions
        
        if config.get("image"):
            return await smart_actions.analyze_image(
                config["image"],
                config.get("question", "Analizza questa immagine")
            )
        return {"success": False, "error": "Nessuna immagine fornita"}
    
    async def _handle_notification(self, config: Dict, context: Dict) -> Dict:
        """Handler per notifiche"""
        message = config.get("message", "Notifica GIDEON")
        title = config.get("title", "GIDEON")
        
        # Per ora log, in futuro push notification
        logger.info(f"📢 NOTIFICA: {title} - {message}")
        
        return {"success": True, "message": message}
    
    async def _handle_email(self, config: Dict, context: Dict) -> Dict:
        """Handler per email"""
        # Integrazione SMTP/SendGrid
        return {"success": False, "error": "Email non configurata"}
    
    async def _handle_read_file(self, config: Dict, context: Dict) -> Dict:
        """Handler per lettura file"""
        file_path = Path(config.get("path", ""))
        
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            return {"success": True, "content": content}
        
        return {"success": False, "error": f"File non trovato: {file_path}"}
    
    async def _handle_write_file(self, config: Dict, context: Dict) -> Dict:
        """Handler per scrittura file"""
        file_path = Path(config.get("path", ""))
        content = config.get("content", "")
        
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return {"success": True, "path": str(file_path)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _handle_http_request(self, config: Dict, context: Dict) -> Dict:
        """Handler per richieste HTTP"""
        import aiohttp
        
        url = config.get("url", "")
        method = config.get("method", "GET").upper()
        headers = config.get("headers", {})
        body = config.get("body")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, headers=headers, json=body) as response:
                    data = await response.text()
                    return {
                        "success": response.status < 400,
                        "status": response.status,
                        "data": data
                    }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _handle_set_variable(self, config: Dict, context: Dict) -> Dict:
        """Handler per impostare variabili"""
        name = config.get("name", "var")
        value = config.get("value")
        
        context["variables"] = context.get("variables", {})
        context["variables"][name] = value
        
        return {"success": True, "variable": name, "value": value}
    
    # ============================================================
    #                    UTILITY METHODS
    # ============================================================
    
    def _find_start_node(self, workflow: Workflow) -> Optional[WorkflowNode]:
        """Trova il nodo di partenza"""
        for node in workflow.nodes.values():
            if node.type == NodeType.TRIGGER:
                return node
        # Se non c'è trigger, prendi il primo nodo
        if workflow.nodes:
            return list(workflow.nodes.values())[0]
        return None
    
    def _resolve_variable(self, value: Any, context: Dict) -> Any:
        """Risolvi variabili nel valore"""
        if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
            var_path = value[2:-2].strip()
            parts = var_path.split(".")
            
            result = context
            for part in parts:
                if isinstance(result, dict):
                    result = result.get(part)
                else:
                    return value
            return result
        return value
    
    def _resolve_config(self, config: Dict, context: Dict) -> Dict:
        """Risolvi tutte le variabili in un config"""
        resolved = {}
        for key, value in config.items():
            if isinstance(value, str):
                resolved[key] = self._resolve_variable(value, context)
            elif isinstance(value, dict):
                resolved[key] = self._resolve_config(value, context)
            else:
                resolved[key] = value
        return resolved
    
    def _evaluate_condition(self, condition: str, context: Dict) -> bool:
        """Valuta una condizione"""
        try:
            # Semplice valutazione (in produzione usare parser sicuro)
            resolved = self._resolve_variable(condition, context)
            if isinstance(resolved, bool):
                return resolved
            return bool(resolved)
        except:
            return False
    
    def _log_execution(self, execution: WorkflowExecution, message: str, level: str = "info"):
        """Aggiunge log all'esecuzione"""
        execution.logs.append({
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "node": execution.current_node
        })
    
    def _save_workflow(self, workflow: Workflow):
        """Salva workflow su disco"""
        file_path = self.storage_path / f"{workflow.id}.json"
        
        data = {
            "id": workflow.id,
            "name": workflow.name,
            "description": workflow.description,
            "enabled": workflow.enabled,
            "variables": workflow.variables,
            "created_at": workflow.created_at.isoformat(),
            "last_run": workflow.last_run.isoformat() if workflow.last_run else None,
            "run_count": workflow.run_count,
            "nodes": {
                node_id: {
                    "id": node.id,
                    "type": node.type.value,
                    "name": node.name,
                    "config": node.config,
                    "next_nodes": node.next_nodes,
                    "position": node.position
                }
                for node_id, node in workflow.nodes.items()
            }
        }
        
        file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    
    def _load_workflows(self):
        """Carica workflows da disco"""
        for file_path in self.storage_path.glob("*.json"):
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
                
                nodes = {}
                for node_id, node_data in data.get("nodes", {}).items():
                    nodes[node_id] = WorkflowNode(
                        id=node_data["id"],
                        type=NodeType(node_data["type"]),
                        name=node_data["name"],
                        config=node_data.get("config", {}),
                        next_nodes=node_data.get("next_nodes", []),
                        position=node_data.get("position", {"x": 0, "y": 0})
                    )
                
                workflow = Workflow(
                    id=data["id"],
                    name=data["name"],
                    description=data.get("description", ""),
                    nodes=nodes,
                    variables=data.get("variables", {}),
                    enabled=data.get("enabled", True),
                    created_at=datetime.fromisoformat(data["created_at"]),
                    last_run=datetime.fromisoformat(data["last_run"]) if data.get("last_run") else None,
                    run_count=data.get("run_count", 0)
                )
                
                self.workflows[workflow.id] = workflow
                
            except Exception as e:
                logger.error(f"Errore caricamento workflow {file_path}: {e}")
        
        logger.info(f"📂 Caricati {len(self.workflows)} workflows")
    
    # ============================================================
    #                    PRESET WORKFLOWS
    # ============================================================
    
    def create_preset_workflow(self, preset_name: str) -> Optional[Workflow]:
        """Crea un workflow da preset"""
        presets = {
            "daily_summary": self._preset_daily_summary,
            "voice_assistant": self._preset_voice_assistant,
            "content_creator": self._preset_content_creator,
            "email_responder": self._preset_email_responder
        }
        
        creator = presets.get(preset_name)
        if creator:
            return creator()
        return None
    
    def _preset_daily_summary(self) -> Workflow:
        """Preset: Riassunto giornaliero"""
        return self.create_workflow(
            name="Daily Summary",
            description="Genera un riassunto giornaliero alle 9:00",
            nodes=[
                {
                    "id": "trigger",
                    "type": "trigger",
                    "name": "Schedule 9:00",
                    "config": {"trigger_type": "schedule", "cron": "0 9 * * *"},
                    "next_nodes": ["gather"]
                },
                {
                    "id": "gather",
                    "type": "action",
                    "name": "Raccogli Informazioni",
                    "config": {"action_type": "ai_chat", "message": "Dammi un riassunto delle news di oggi"},
                    "next_nodes": ["summarize"]
                },
                {
                    "id": "summarize",
                    "type": "action",
                    "name": "Genera TTS",
                    "config": {"action_type": "ai_voice", "text": "{{node_gather_result.response}}"},
                    "next_nodes": ["notify"]
                },
                {
                    "id": "notify",
                    "type": "action",
                    "name": "Notifica",
                    "config": {"action_type": "send_notification", "message": "Riassunto giornaliero pronto!"}
                }
            ]
        )
    
    def _preset_voice_assistant(self) -> Workflow:
        """Preset: Assistente vocale"""
        return self.create_workflow(
            name="Voice Assistant",
            description="Risponde a comandi vocali",
            nodes=[
                {
                    "id": "trigger",
                    "type": "trigger",
                    "name": "Voice Command",
                    "config": {"trigger_type": "voice", "wake_word": "gideon"},
                    "next_nodes": ["process"]
                },
                {
                    "id": "process",
                    "type": "action",
                    "name": "Elabora Comando",
                    "config": {"action_type": "ai_chat", "message": "{{input.command}}"},
                    "next_nodes": ["speak"]
                },
                {
                    "id": "speak",
                    "type": "action",
                    "name": "Rispondi",
                    "config": {"action_type": "ai_voice", "text": "{{node_process_result.response}}"}
                }
            ]
        )
    
    def _preset_content_creator(self) -> Workflow:
        """Preset: Creatore contenuti"""
        return self.create_workflow(
            name="Content Creator",
            description="Crea contenuti multimediali da un topic",
            nodes=[
                {
                    "id": "trigger",
                    "type": "trigger",
                    "name": "Start",
                    "config": {"trigger_type": "manual"},
                    "next_nodes": ["script"]
                },
                {
                    "id": "script",
                    "type": "action",
                    "name": "Genera Script",
                    "config": {"action_type": "ai_chat", "message": "Scrivi uno script per un video su: {{input.topic}}"},
                    "next_nodes": ["image", "voice"]
                },
                {
                    "id": "image",
                    "type": "action",
                    "name": "Genera Immagine",
                    "config": {"action_type": "ai_image", "prompt": "{{input.topic}}, professional, 4k"}
                },
                {
                    "id": "voice",
                    "type": "action",
                    "name": "Genera Voce",
                    "config": {"action_type": "ai_voice", "text": "{{node_script_result.response}}"},
                    "next_nodes": ["complete"]
                },
                {
                    "id": "complete",
                    "type": "action",
                    "name": "Completa",
                    "config": {"action_type": "send_notification", "message": "Contenuto creato!"}
                }
            ]
        )
    
    def _preset_email_responder(self) -> Workflow:
        """Preset: Risponditore email"""
        return self.create_workflow(
            name="Email Auto-Responder",
            description="Risponde automaticamente alle email",
            nodes=[
                {
                    "id": "trigger",
                    "type": "trigger",
                    "name": "New Email",
                    "config": {"trigger_type": "email"},
                    "next_nodes": ["analyze"]
                },
                {
                    "id": "analyze",
                    "type": "action",
                    "name": "Analizza Email",
                    "config": {"action_type": "ai_chat", "message": "Analizza questa email e suggerisci una risposta: {{input.email_body}}"},
                    "next_nodes": ["respond"]
                },
                {
                    "id": "respond",
                    "type": "action",
                    "name": "Invia Risposta",
                    "config": {"action_type": "send_email", "body": "{{node_analyze_result.response}}"}
                }
            ]
        )


# Singleton
_workflow_engine: Optional[WorkflowEngine] = None

def get_workflow_engine() -> WorkflowEngine:
    """Ottieni istanza Workflow Engine"""
    global _workflow_engine
    if _workflow_engine is None:
        _workflow_engine = WorkflowEngine()
    return _workflow_engine
