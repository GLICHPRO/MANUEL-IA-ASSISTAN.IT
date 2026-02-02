"""
🔌 GIDEON 3.0 - Plugins Directory
Posiziona qui i tuoi plugin personalizzati

Struttura plugin:
    plugins/
    ├── my_plugin/
    │   ├── __init__.py    # Deve contenere classe che estende PluginInterface
    │   └── ...
    └── simple_plugin.py   # Oppure file singolo

Esempio plugin minimo:

    from backend.core.plugin_manager import PluginInterface
    
    class MyPlugin(PluginInterface):
        @property
        def name(self) -> str:
            return "my_plugin"
        
        @property
        def version(self) -> str:
            return "1.0.0"
        
        async def initialize(self, context: dict) -> bool:
            print("Plugin inizializzato!")
            return True
        
        async def shutdown(self) -> bool:
            print("Plugin arrestato!")
            return True
"""
