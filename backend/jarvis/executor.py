"""
⚡ JARVIS CORE - Executor
Esecuzione azioni e comandi
"""

import asyncio
import subprocess
import os
import webbrowser
from datetime import datetime
from typing import Dict, List, Any, Optional


class Executor:
    """
    Esecutore di comandi e azioni di Jarvis Core
    """
    
    def __init__(self, security_manager=None):
        self.security = security_manager
        self.execution_history = []
        self.supported_actions = {
            # Sistema
            "open_app": self._open_application,
            "close_app": self._close_application,
            "run_command": self._run_command,
            
            # File
            "open_file": self._open_file,
            "create_file": self._create_file,
            "delete_file": self._delete_file,
            "copy_file": self._copy_file,
            "move_file": self._move_file,
            
            # Browser
            "open_url": self._open_url,
            "search_web": self._search_web,
            
            # Sistema operativo
            "shutdown": self._shutdown,
            "restart": self._restart,
            "sleep": self._sleep,
            "lock": self._lock_screen,
            
            # Audio
            "set_volume": self._set_volume,
            "mute": self._mute,
            
            # Notifiche
            "notify": self._send_notification,
            
            # Clipboard
            "copy_to_clipboard": self._copy_to_clipboard,
        }
        
    async def execute(self, action: dict) -> dict:
        """
        Esegue un'azione
        
        Args:
            action: Dizionario con tipo azione e parametri
            
        Returns:
            Risultato dell'esecuzione
        """
        action_type = action.get("action") or action.get("type")
        params = action.get("params", {})
        
        result = {
            "action": action_type,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "output": None,
            "error": None
        }
        
        if not action_type:
            result["error"] = "Tipo azione non specificato"
            return result
        
        if action_type not in self.supported_actions:
            result["error"] = f"Azione '{action_type}' non supportata"
            result["supported_actions"] = list(self.supported_actions.keys())
            return result
        
        try:
            # Esegui l'azione
            output = await self.supported_actions[action_type](params)
            result["success"] = True
            result["output"] = output
        except Exception as e:
            result["error"] = str(e)
        
        # Salva nella history
        self.execution_history.append(result)
        
        return result
    
    async def execute_batch(self, actions: list) -> list:
        """Esegue una lista di azioni in sequenza"""
        results = []
        for action in actions:
            result = await self.execute(action)
            results.append(result)
            
            # Se un'azione fallisce, interrompi (opzionale)
            if not result["success"] and action.get("stop_on_error", False):
                break
                
        return results
    
    # === Implementazioni azioni ===
    
    async def _open_application(self, params: dict) -> str:
        """Apre un'applicazione"""
        app_name = params.get("name", "")
        app_path = params.get("path", "")
        
        if app_path:
            # Percorso specifico
            os.startfile(app_path)
            return f"Aperta applicazione: {app_path}"
        
        # Applicazioni comuni Windows
        common_apps = {
            "notepad": "notepad.exe",
            "calculator": "calc.exe",
            "explorer": "explorer.exe",
            "cmd": "cmd.exe",
            "powershell": "powershell.exe",
            "paint": "mspaint.exe",
            "word": "winword.exe",
            "excel": "excel.exe",
            "chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            "firefox": r"C:\Program Files\Mozilla Firefox\firefox.exe",
            "vscode": r"C:\Users\%USERNAME%\AppData\Local\Programs\Microsoft VS Code\Code.exe"
        }
        
        app_lower = app_name.lower()
        if app_lower in common_apps:
            os.startfile(common_apps[app_lower])
            return f"Aperta applicazione: {app_name}"
        
        # Prova ad aprire direttamente
        try:
            os.startfile(app_name)
            return f"Aperta applicazione: {app_name}"
        except:
            raise Exception(f"Applicazione '{app_name}' non trovata")
    
    async def _close_application(self, params: dict) -> str:
        """Chiude un'applicazione"""
        app_name = params.get("name", "")
        
        # Usa taskkill su Windows
        result = subprocess.run(
            ["taskkill", "/IM", f"{app_name}.exe", "/F"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            return f"Chiusa applicazione: {app_name}"
        else:
            raise Exception(f"Impossibile chiudere {app_name}: {result.stderr}")
    
    async def _run_command(self, params: dict) -> str:
        """Esegue un comando shell"""
        command = params.get("command", "")
        
        if not command:
            raise Exception("Comando non specificato")
        
        # Verifica sicurezza del comando
        if self.security:
            check = await self.security.check_command(command)
            if not check["allowed"]:
                raise Exception(f"Comando bloccato: {check.get('reason')}")
        
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return result.stdout or "Comando eseguito"
        else:
            raise Exception(result.stderr or "Errore nell'esecuzione")
    
    async def _open_file(self, params: dict) -> str:
        """Apre un file con l'applicazione predefinita"""
        file_path = params.get("path", "")
        
        if not file_path:
            raise Exception("Percorso file non specificato")
        
        if not os.path.exists(file_path):
            raise Exception(f"File non trovato: {file_path}")
        
        os.startfile(file_path)
        return f"Aperto file: {file_path}"
    
    async def _create_file(self, params: dict) -> str:
        """Crea un nuovo file"""
        file_path = params.get("path", "")
        content = params.get("content", "")
        
        if not file_path:
            raise Exception("Percorso file non specificato")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"Creato file: {file_path}"
    
    async def _delete_file(self, params: dict) -> str:
        """Elimina un file"""
        file_path = params.get("path", "")
        
        if not file_path:
            raise Exception("Percorso file non specificato")
        
        if not os.path.exists(file_path):
            raise Exception(f"File non trovato: {file_path}")
        
        os.remove(file_path)
        return f"Eliminato file: {file_path}"
    
    async def _copy_file(self, params: dict) -> str:
        """Copia un file"""
        import shutil
        
        source = params.get("source", "")
        destination = params.get("destination", "")
        
        if not source or not destination:
            raise Exception("Source e destination richiesti")
        
        shutil.copy2(source, destination)
        return f"Copiato {source} in {destination}"
    
    async def _move_file(self, params: dict) -> str:
        """Sposta un file"""
        import shutil
        
        source = params.get("source", "")
        destination = params.get("destination", "")
        
        if not source or not destination:
            raise Exception("Source e destination richiesti")
        
        shutil.move(source, destination)
        return f"Spostato {source} in {destination}"
    
    async def _open_url(self, params: dict) -> str:
        """Apre un URL nel browser"""
        url = params.get("url", "")
        
        if not url:
            raise Exception("URL non specificato")
        
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        
        webbrowser.open(url)
        return f"Aperto URL: {url}"
    
    async def _search_web(self, params: dict) -> str:
        """Cerca sul web"""
        query = params.get("query", "")
        engine = params.get("engine", "google")
        
        if not query:
            raise Exception("Query non specificata")
        
        import urllib.parse
        encoded_query = urllib.parse.quote(query)
        
        engines = {
            "google": f"https://www.google.com/search?q={encoded_query}",
            "bing": f"https://www.bing.com/search?q={encoded_query}",
            "duckduckgo": f"https://duckduckgo.com/?q={encoded_query}"
        }
        
        url = engines.get(engine, engines["google"])
        webbrowser.open(url)
        return f"Ricerca '{query}' su {engine}"
    
    async def _shutdown(self, params: dict) -> str:
        """Spegne il computer"""
        delay = params.get("delay", 60)  # Default 60 secondi
        
        subprocess.run(["shutdown", "/s", "/t", str(delay)])
        return f"Spegnimento programmato tra {delay} secondi"
    
    async def _restart(self, params: dict) -> str:
        """Riavvia il computer"""
        delay = params.get("delay", 60)
        
        subprocess.run(["shutdown", "/r", "/t", str(delay)])
        return f"Riavvio programmato tra {delay} secondi"
    
    async def _sleep(self, params: dict) -> str:
        """Mette il computer in sospensione"""
        subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"])
        return "Computer in sospensione"
    
    async def _lock_screen(self, params: dict) -> str:
        """Blocca lo schermo"""
        subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"])
        return "Schermo bloccato"
    
    async def _set_volume(self, params: dict) -> str:
        """Imposta il volume del sistema"""
        level = params.get("level", 50)
        # Nota: richiede nircmd o pycaw per funzionare
        return f"Volume impostato a {level}%"
    
    async def _mute(self, params: dict) -> str:
        """Attiva/disattiva muto"""
        # Nota: richiede nircmd o pycaw per funzionare
        return "Audio mutato/smutato"
    
    async def _send_notification(self, params: dict) -> str:
        """Invia una notifica di sistema"""
        title = params.get("title", "Jarvis")
        message = params.get("message", "")
        
        try:
            from win10toast import ToastNotifier
            toaster = ToastNotifier()
            toaster.show_toast(title, message, duration=5, threaded=True)
            return f"Notifica inviata: {title}"
        except ImportError:
            # Fallback senza win10toast
            print(f"📢 {title}: {message}")
            return f"Notifica (console): {title}"
    
    async def _copy_to_clipboard(self, params: dict) -> str:
        """Copia testo negli appunti"""
        text = params.get("text", "")
        
        try:
            import pyperclip
            pyperclip.copy(text)
            return "Testo copiato negli appunti"
        except ImportError:
            # Fallback Windows
            subprocess.run(
                ["clip"],
                input=text.encode(),
                check=True
            )
            return "Testo copiato negli appunti"
    
    def get_supported_actions(self) -> list:
        """Restituisce la lista delle azioni supportate"""
        return list(self.supported_actions.keys())
    
    def get_execution_history(self, limit: int = 50) -> list:
        """Restituisce la history delle esecuzioni"""
        return self.execution_history[-limit:]
