"""
📋 Example Plugin - Plugin di esempio per Gideon 3.0
Dimostra come creare un plugin personalizzato
"""

import sys
import os

# Aggiungi path per import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.plugin_manager import PluginInterface
except ImportError:
    # Fallback per import standalone
    from typing import List, Optional
    
    class PluginInterface:
        @property
        def name(self) -> str: pass
        @property  
        def version(self) -> str: pass
        @property
        def description(self) -> str: return ""
        @property
        def author(self) -> str: return "Unknown"
        @property
        def dependencies(self) -> List[str]: return []
        async def initialize(self, context: dict) -> bool: pass
        async def shutdown(self) -> bool: pass
        async def on_message(self, message: str, context: dict) -> Optional[dict]: return None
        async def on_action(self, action: str, params: dict) -> Optional[dict]: return None
        def get_commands(self) -> List[dict]: return []
        def get_status(self) -> dict: return {"active": True}


class ExamplePlugin(PluginInterface):
    """
    Plugin di esempio che mostra le funzionalità base
    """
    
    def __init__(self):
        self._is_initialized = False
        self._call_count = 0
    
    @property
    def name(self) -> str:
        return "example_plugin"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Plugin di esempio per dimostrare il sistema di plugin"
    
    @property
    def author(self) -> str:
        return "Gideon Team"
    
    @property
    def dependencies(self) -> list:
        return []  # Nessuna dipendenza
    
    async def initialize(self, context: dict) -> bool:
        """Inizializza il plugin"""
        print(f"🔌 {self.name} v{self.version} inizializzato!")
        self._is_initialized = True
        return True
    
    async def shutdown(self) -> bool:
        """Arresta il plugin"""
        print(f"🔌 {self.name} arrestato!")
        self._is_initialized = False
        return True
    
    async def on_message(self, message: str, context: dict) -> dict:
        """
        Gestisce messaggi - risponde a "esempio" o "test plugin"
        """
        message_lower = message.lower()
        
        if "esempio" in message_lower or "test plugin" in message_lower:
            self._call_count += 1
            return {
                "handled": True,
                "response": f"🔌 Ciao dal plugin di esempio! Chiamata #{self._call_count}",
                "plugin": self.name
            }
        
        return None  # Non gestito
    
    async def on_action(self, action: str, params: dict) -> dict:
        """Gestisce azioni personalizzate"""
        if action == "example_action":
            return {
                "success": True,
                "message": "Azione di esempio eseguita!",
                "params_received": params
            }
        return None
    
    def get_commands(self) -> list:
        """Comandi disponibili dal plugin"""
        return [
            {
                "name": "esempio",
                "description": "Comando di test del plugin",
                "params": []
            },
            {
                "name": "example_action",
                "description": "Azione di esempio",
                "params": ["param1", "param2"]
            }
        ]
    
    def get_status(self) -> dict:
        """Stato del plugin"""
        return {
            "active": self._is_initialized,
            "call_count": self._call_count,
            "version": self.version
        }
