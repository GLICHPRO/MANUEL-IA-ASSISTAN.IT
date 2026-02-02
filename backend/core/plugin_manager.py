"""
🔌 GIDEON 3.0 - Plugin Manager
Sistema espandibile con moduli dinamici
"""

import os
import sys
import importlib
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Type
from abc import ABC, abstractmethod
import asyncio
import json


class PluginInterface(ABC):
    """
    Interfaccia base per tutti i plugin
    Ogni plugin deve implementare questa interfaccia
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Nome univoco del plugin"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Versione del plugin"""
        pass
    
    @property
    def description(self) -> str:
        """Descrizione del plugin"""
        return ""
    
    @property
    def author(self) -> str:
        """Autore del plugin"""
        return "Unknown"
    
    @property
    def dependencies(self) -> List[str]:
        """Lista dipendenze (altri plugin richiesti)"""
        return []
    
    @abstractmethod
    async def initialize(self, context: dict) -> bool:
        """
        Inizializza il plugin
        
        Args:
            context: Contesto con riferimenti ai componenti core
            
        Returns:
            True se inizializzato con successo
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """Arresta il plugin"""
        pass
    
    async def on_message(self, message: str, context: dict) -> Optional[dict]:
        """
        Gestisce un messaggio (opzionale)
        
        Returns:
            Response dict o None se non gestito
        """
        return None
    
    async def on_action(self, action: str, params: dict) -> Optional[dict]:
        """
        Gestisce un'azione (opzionale)
        
        Returns:
            Result dict o None se non gestito
        """
        return None
    
    def get_commands(self) -> List[dict]:
        """
        Restituisce comandi disponibili
        
        Returns:
            Lista di {name, description, params}
        """
        return []
    
    def get_status(self) -> dict:
        """Stato del plugin"""
        return {"active": True}


class PluginInfo:
    """Informazioni su un plugin caricato"""
    
    def __init__(self, plugin: PluginInterface, module_path: str):
        self.plugin = plugin
        self.module_path = module_path
        self.loaded_at = datetime.now()
        self.is_active = False
        self.error = None
    
    def to_dict(self) -> dict:
        return {
            "name": self.plugin.name,
            "version": self.plugin.version,
            "description": self.plugin.description,
            "author": self.plugin.author,
            "dependencies": self.plugin.dependencies,
            "module_path": self.module_path,
            "loaded_at": self.loaded_at.isoformat(),
            "is_active": self.is_active,
            "error": self.error
        }


class PluginManager:
    """
    Gestisce il caricamento e l'esecuzione dei plugin
    """
    
    def __init__(self, plugins_dir: str = None):
        self.plugins_dir = Path(plugins_dir or "plugins")
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        
        self._plugins: Dict[str, PluginInfo] = {}
        self._load_order: List[str] = []
        self._context: dict = {}
        
        # Hooks per estensione
        self._message_handlers: List[PluginInterface] = []
        self._action_handlers: List[PluginInterface] = []
    
    def set_context(self, context: dict):
        """Imposta il contesto condiviso con i plugin"""
        self._context = context
    
    async def discover_plugins(self) -> List[str]:
        """Scopre plugin disponibili nella directory"""
        discovered = []
        
        for item in self.plugins_dir.iterdir():
            if item.is_dir() and (item / "__init__.py").exists():
                discovered.append(item.name)
            elif item.suffix == ".py" and item.name != "__init__.py":
                discovered.append(item.stem)
        
        return discovered
    
    async def load_plugin(self, plugin_name: str) -> dict:
        """
        Carica un plugin
        
        Args:
            plugin_name: Nome del plugin da caricare
        """
        if plugin_name in self._plugins:
            return {"success": False, "error": "Plugin già caricato"}
        
        try:
            # Trova il modulo
            plugin_path = self.plugins_dir / plugin_name
            
            if plugin_path.is_dir():
                module_path = plugin_path / "__init__.py"
            else:
                module_path = self.plugins_dir / f"{plugin_name}.py"
            
            if not module_path.exists():
                return {"success": False, "error": f"Plugin non trovato: {plugin_name}"}
            
            # Carica il modulo
            spec = importlib.util.spec_from_file_location(
                f"plugins.{plugin_name}", 
                module_path
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"plugins.{plugin_name}"] = module
            spec.loader.exec_module(module)
            
            # Trova la classe plugin
            plugin_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, PluginInterface) and 
                    attr is not PluginInterface):
                    plugin_class = attr
                    break
            
            if not plugin_class:
                return {"success": False, "error": "Nessuna classe PluginInterface trovata"}
            
            # Istanzia il plugin
            plugin = plugin_class()
            
            # Verifica dipendenze
            for dep in plugin.dependencies:
                if dep not in self._plugins:
                    return {
                        "success": False, 
                        "error": f"Dipendenza mancante: {dep}"
                    }
            
            # Registra
            info = PluginInfo(plugin, str(module_path))
            self._plugins[plugin.name] = info
            self._load_order.append(plugin.name)
            
            return {
                "success": True,
                "plugin": info.to_dict(),
                "message": f"Plugin {plugin.name} caricato"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def initialize_plugin(self, plugin_name: str) -> dict:
        """Inizializza un plugin caricato"""
        if plugin_name not in self._plugins:
            return {"success": False, "error": "Plugin non caricato"}
        
        info = self._plugins[plugin_name]
        
        try:
            success = await info.plugin.initialize(self._context)
            
            if success:
                info.is_active = True
                
                # Registra handlers
                if info.plugin.get_commands():
                    self._message_handlers.append(info.plugin)
                
                return {"success": True, "message": f"Plugin {plugin_name} inizializzato"}
            else:
                return {"success": False, "error": "Inizializzazione fallita"}
                
        except Exception as e:
            info.error = str(e)
            return {"success": False, "error": str(e)}
    
    async def unload_plugin(self, plugin_name: str) -> dict:
        """Scarica un plugin"""
        if plugin_name not in self._plugins:
            return {"success": False, "error": "Plugin non trovato"}
        
        info = self._plugins[plugin_name]
        
        # Verifica che nessun altro plugin dipenda da questo
        for name, other in self._plugins.items():
            if plugin_name in other.plugin.dependencies:
                return {
                    "success": False, 
                    "error": f"Plugin {name} dipende da {plugin_name}"
                }
        
        try:
            await info.plugin.shutdown()
            
            # Rimuovi handlers
            if info.plugin in self._message_handlers:
                self._message_handlers.remove(info.plugin)
            if info.plugin in self._action_handlers:
                self._action_handlers.remove(info.plugin)
            
            # Rimuovi
            del self._plugins[plugin_name]
            self._load_order.remove(plugin_name)
            
            return {"success": True, "message": f"Plugin {plugin_name} scaricato"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def reload_plugin(self, plugin_name: str) -> dict:
        """Ricarica un plugin"""
        await self.unload_plugin(plugin_name)
        result = await self.load_plugin(plugin_name)
        if result["success"]:
            await self.initialize_plugin(plugin_name)
        return result
    
    async def load_all(self) -> dict:
        """Carica e inizializza tutti i plugin"""
        discovered = await self.discover_plugins()
        results = {"loaded": [], "failed": []}
        
        for plugin_name in discovered:
            load_result = await self.load_plugin(plugin_name)
            if load_result["success"]:
                init_result = await self.initialize_plugin(plugin_name)
                if init_result["success"]:
                    results["loaded"].append(plugin_name)
                else:
                    results["failed"].append({
                        "name": plugin_name, 
                        "error": init_result["error"]
                    })
            else:
                results["failed"].append({
                    "name": plugin_name, 
                    "error": load_result["error"]
                })
        
        return results
    
    async def shutdown_all(self):
        """Arresta tutti i plugin"""
        for plugin_name in reversed(self._load_order):
            await self.unload_plugin(plugin_name)
    
    async def handle_message(self, message: str, context: dict) -> Optional[dict]:
        """Passa un messaggio ai plugin"""
        for plugin in self._message_handlers:
            try:
                result = await plugin.on_message(message, context)
                if result:
                    return result
            except Exception:
                pass
        return None
    
    async def handle_action(self, action: str, params: dict) -> Optional[dict]:
        """Passa un'azione ai plugin"""
        for plugin in self._action_handlers:
            try:
                result = await plugin.on_action(action, params)
                if result:
                    return result
            except Exception:
                pass
        return None
    
    def get_plugin(self, name: str) -> Optional[PluginInterface]:
        """Ottiene un plugin per nome"""
        if name in self._plugins:
            return self._plugins[name].plugin
        return None
    
    def list_plugins(self) -> List[dict]:
        """Lista tutti i plugin caricati"""
        return [info.to_dict() for info in self._plugins.values()]
    
    def get_all_commands(self) -> List[dict]:
        """Tutti i comandi disponibili da tutti i plugin"""
        commands = []
        for info in self._plugins.values():
            if info.is_active:
                for cmd in info.plugin.get_commands():
                    cmd["plugin"] = info.plugin.name
                    commands.append(cmd)
        return commands
    
    def get_status(self) -> dict:
        """Stato del plugin manager"""
        return {
            "plugins_dir": str(self.plugins_dir),
            "loaded_count": len(self._plugins),
            "active_count": sum(1 for p in self._plugins.values() if p.is_active),
            "plugins": [p.to_dict() for p in self._plugins.values()]
        }
