"""
🤖 GIDEON Smart Actions - Automazioni Creative

Automazioni intelligenti che semplificano il lavoro:
- ⏰ Timer e Sveglie
- 📱 WhatsApp Messages
- 📧 Email
- 📷 Analisi Immagini (Vision AI)
- 🔔 Notifiche Sistema
"""

import asyncio
import base64
import hashlib
import os
import smtplib
import subprocess
import threading
import time
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import webbrowser
import urllib.parse
import tempfile
import httpx

# Carica .env dalla cartella backend
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)

from loguru import logger


# ============ TIMER & ALARM SYSTEM ============

class TimerStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class TimerTask:
    """Timer/Sveglia task"""
    id: str
    name: str
    duration_seconds: int
    created_at: datetime
    end_time: datetime
    status: TimerStatus = TimerStatus.PENDING
    callback: Optional[Callable] = None
    message: str = ""
    repeat: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "duration_seconds": self.duration_seconds,
            "created_at": self.created_at.isoformat(),
            "end_time": self.end_time.isoformat(),
            "remaining_seconds": max(0, (self.end_time - datetime.now()).total_seconds()),
            "status": self.status.value,
            "message": self.message,
            "repeat": self.repeat
        }


class TimerManager:
    """Gestisce timer e sveglie"""
    
    def __init__(self):
        self.timers: Dict[str, TimerTask] = {}
        self.alarms: Dict[str, TimerTask] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._notification_callback: Optional[Callable] = None
    
    def set_notification_callback(self, callback: Callable):
        """Imposta callback per notifiche (es. TTS)"""
        self._notification_callback = callback
    
    def start(self):
        """Avvia il timer manager"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._timer_loop, daemon=True)
        self._thread.start()
        logger.info("⏰ Timer Manager avviato")
    
    def stop(self):
        """Ferma il timer manager"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
    
    def _timer_loop(self):
        """Loop principale per controllare timer"""
        while self._running:
            now = datetime.now()
            
            # Check timers
            for timer_id, timer in list(self.timers.items()):
                if timer.status == TimerStatus.RUNNING and now >= timer.end_time:
                    self._trigger_timer(timer)
            
            # Check alarms
            for alarm_id, alarm in list(self.alarms.items()):
                if alarm.status == TimerStatus.RUNNING and now >= alarm.end_time:
                    self._trigger_timer(alarm)
                    if alarm.repeat:
                        # Riprogramma per domani
                        alarm.end_time = alarm.end_time + timedelta(days=1)
                    else:
                        alarm.status = TimerStatus.COMPLETED
            
            time.sleep(0.5)  # Check ogni 500ms
    
    def _trigger_timer(self, timer: TimerTask):
        """Attiva il timer/sveglia"""
        timer.status = TimerStatus.COMPLETED
        
        message = timer.message or f"Timer '{timer.name}' completato!"
        logger.info(f"⏰ {message}")
        
        # Notifica sistema Windows
        self._show_notification(timer.name, message)
        
        # Callback (es. TTS)
        if self._notification_callback:
            try:
                self._notification_callback(message)
            except Exception as e:
                logger.error(f"Notification callback error: {e}")
        
        # Suono
        self._play_sound()
    
    def _show_notification(self, title: str, message: str):
        """Mostra notifica Windows"""
        try:
            from win10toast import ToastNotifier
            toaster = ToastNotifier()
            toaster.show_toast(
                title,
                message,
                duration=10,
                threaded=True
            )
        except ImportError:
            # Fallback PowerShell
            ps_script = f'''
            [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
            $template = [Windows.UI.Notifications.ToastTemplateType]::ToastText02
            $xml = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent($template)
            $xml.GetElementsByTagName("text")[0].AppendChild($xml.CreateTextNode("{title}")) | Out-Null
            $xml.GetElementsByTagName("text")[1].AppendChild($xml.CreateTextNode("{message}")) | Out-Null
            $notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Gideon")
            $notifier.Show([Windows.UI.Notifications.ToastNotification]::new($xml))
            '''
            try:
                subprocess.run(["powershell", "-Command", ps_script], 
                             capture_output=True, timeout=5)
            except:
                pass
    
    def _play_sound(self):
        """Suona notifica"""
        try:
            import winsound
            winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC)
        except:
            pass
    
    def create_timer(
        self, 
        minutes: float, 
        name: str = "Timer",
        message: str = "",
        callback: Optional[Callable] = None
    ) -> TimerTask:
        """
        Crea un nuovo timer
        
        Args:
            minutes: Durata in minuti
            name: Nome del timer
            message: Messaggio da mostrare
            callback: Funzione da chiamare alla fine
        """
        timer_id = hashlib.md5(f"{name}_{time.time()}".encode()).hexdigest()[:8]
        
        now = datetime.now()
        duration_seconds = int(minutes * 60)
        end_time = now + timedelta(seconds=duration_seconds)
        
        timer = TimerTask(
            id=timer_id,
            name=name,
            duration_seconds=duration_seconds,
            created_at=now,
            end_time=end_time,
            status=TimerStatus.RUNNING,
            callback=callback,
            message=message or f"Timer '{name}' completato!"
        )
        
        self.timers[timer_id] = timer
        logger.info(f"⏰ Timer creato: {name} ({minutes} minuti)")
        
        return timer
    
    def create_alarm(
        self,
        hour: int,
        minute: int = 0,
        name: str = "Sveglia",
        message: str = "",
        repeat: bool = False
    ) -> TimerTask:
        """
        Crea una sveglia
        
        Args:
            hour: Ora (0-23)
            minute: Minuto (0-59)
            name: Nome sveglia
            message: Messaggio
            repeat: Se ripetere ogni giorno
        """
        alarm_id = hashlib.md5(f"{name}_{hour}_{minute}".encode()).hexdigest()[:8]
        
        now = datetime.now()
        alarm_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # Se l'ora è già passata oggi, imposta per domani
        if alarm_time <= now:
            alarm_time += timedelta(days=1)
        
        duration_seconds = int((alarm_time - now).total_seconds())
        
        alarm = TimerTask(
            id=alarm_id,
            name=name,
            duration_seconds=duration_seconds,
            created_at=now,
            end_time=alarm_time,
            status=TimerStatus.RUNNING,
            message=message or f"Sveglia! È ora di {name}",
            repeat=repeat
        )
        
        self.alarms[alarm_id] = alarm
        logger.info(f"⏰ Sveglia creata: {name} alle {hour:02d}:{minute:02d}")
        
        return alarm
    
    def cancel_timer(self, timer_id: str) -> bool:
        """Cancella un timer"""
        if timer_id in self.timers:
            self.timers[timer_id].status = TimerStatus.CANCELLED
            del self.timers[timer_id]
            return True
        if timer_id in self.alarms:
            self.alarms[timer_id].status = TimerStatus.CANCELLED
            del self.alarms[timer_id]
            return True
        return False
    
    def list_timers(self) -> List[Dict]:
        """Lista tutti i timer attivi"""
        active = []
        for timer in self.timers.values():
            if timer.status == TimerStatus.RUNNING:
                active.append(timer.to_dict())
        for alarm in self.alarms.values():
            if alarm.status == TimerStatus.RUNNING:
                active.append(alarm.to_dict())
        return active


# ============ WHATSAPP INTEGRATION ============

class WhatsAppSender:
    """Invia messaggi WhatsApp via WhatsApp Web"""
    
    @staticmethod
    def send_message(phone_number: str, message: str) -> Dict:
        """
        Invia messaggio WhatsApp aprendo WhatsApp Web
        
        Args:
            phone_number: Numero con prefisso internazionale (es. +393331234567)
            message: Testo del messaggio
        """
        # Pulisci numero (rimuovi spazi, trattini)
        phone = phone_number.replace(" ", "").replace("-", "").replace("+", "")
        
        # Codifica messaggio per URL
        encoded_message = urllib.parse.quote(message)
        
        # URL WhatsApp Web
        url = f"https://web.whatsapp.com/send?phone={phone}&text={encoded_message}"
        
        try:
            webbrowser.open(url)
            logger.info(f"📱 WhatsApp Web aperto per {phone_number}")
            
            return {
                "success": True,
                "message": f"WhatsApp Web aperto. Premi INVIO per inviare il messaggio a {phone_number}",
                "phone": phone_number,
                "text": message
            }
        except Exception as e:
            logger.error(f"WhatsApp error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def send_via_api(phone_number: str, message: str, api_url: str = None, api_key: str = None) -> Dict:
        """
        Invia messaggio via API WhatsApp Business (richiede configurazione)
        """
        if not api_url or not api_key:
            return WhatsAppSender.send_message(phone_number, message)
        
        # Implementazione per WhatsApp Business API
        # Richiede account WhatsApp Business verificato
        return {
            "success": False,
            "error": "WhatsApp Business API non configurata. Uso fallback WhatsApp Web."
        }


# ============ EMAIL SENDER ============

@dataclass
class EmailConfig:
    """Configurazione SMTP"""
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    email: str = ""
    password: str = ""  # Per Gmail: App Password


class EmailSender:
    """Invia email via SMTP"""
    
    def __init__(self, config: Optional[EmailConfig] = None):
        self.config = config
    
    def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        html: bool = False,
        attachments: List[str] = None
    ) -> Dict:
        """
        Invia email
        
        Args:
            to: Destinatario
            subject: Oggetto
            body: Corpo del messaggio
            html: Se il corpo è HTML
            attachments: Lista path file da allegare
        """
        if not self.config or not self.config.email:
            return {
                "success": False,
                "error": "Email non configurata. Configura SMTP in backend/.env"
            }
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.email
            msg['To'] = to
            msg['Subject'] = subject
            
            content_type = 'html' if html else 'plain'
            msg.attach(MIMEText(body, content_type))
            
            # Allegati
            if attachments:
                from email.mime.base import MIMEBase
                from email import encoders
                
                for file_path in attachments:
                    path = Path(file_path)
                    if path.exists():
                        with open(path, 'rb') as f:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(f.read())
                            encoders.encode_base64(part)
                            part.add_header(
                                'Content-Disposition',
                                f'attachment; filename= {path.name}'
                            )
                            msg.attach(part)
            
            # Invia
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.email, self.config.password)
                server.send_message(msg)
            
            logger.info(f"📧 Email inviata a {to}")
            return {
                "success": True,
                "message": f"Email inviata con successo a {to}",
                "to": to,
                "subject": subject
            }
            
        except Exception as e:
            logger.error(f"Email error: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# ============ IMAGE ANALYSIS (VISION AI) ============

class ImageAnalyzer:
    """
    Analizza immagini usando AI Vision
    Supporta: OpenRouter (con modelli vision), OpenAI GPT-4 Vision
    """
    
    def __init__(self, openrouter_api_key: str = None, openai_api_key: str = None):
        # ⚠️ Legge direttamente dal file .env per sicurezza
        env_path = Path(__file__).parent.parent / ".env"
        env_vars = {}
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip().strip('"').strip("'")
        
        # Prova le keys passate, altrimenti usa quelle lette dal file
        env_or_key = env_vars.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY", ""))
        env_oa_key = env_vars.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        
        self.openrouter_key = (openrouter_api_key if openrouter_api_key and len(openrouter_api_key) > 10 
                               else (env_or_key if len(env_or_key) > 10 else None))
        self.openai_key = (openai_api_key if openai_api_key and len(openai_api_key) > 10 
                          else (env_oa_key if len(env_oa_key) > 10 else None))
        
        logger.info(f"📷 ImageAnalyzer init - OpenRouter: {'✅ ' + self.openrouter_key[:15] + '...' if self.openrouter_key else '❌ NOT FOUND'}")
        logger.info(f"📷 ImageAnalyzer init - OpenAI: {'✅' if self.openai_key else '❌'}")
        
        # Modelli vision disponibili su OpenRouter (GRATUITI PRIMA)
        self.vision_models = [
            "google/gemini-2.0-flash-exp:free",  # ⚡ Gratuito con vision
            "qwen/qwen2.5-vl-72b-instruct:free", # ⚡ Qwen Vision gratuito
            "google/gemini-flash-1.5",
            "anthropic/claude-3-haiku",
            "openai/gpt-4o-mini"
        ]
    
    async def capture_screenshot(self) -> Optional[str]:
        """
        Cattura screenshot dello schermo
        Ritorna path del file salvato
        """
        try:
            from PIL import ImageGrab
            
            # Cattura schermo
            screenshot = ImageGrab.grab()
            
            # Salva in temp
            temp_path = Path(tempfile.gettempdir()) / f"gideon_screenshot_{int(time.time())}.png"
            screenshot.save(str(temp_path), "PNG")
            
            logger.info(f"📷 Screenshot salvato: {temp_path}")
            return str(temp_path)
            
        except ImportError:
            logger.error("PIL non installato. Installa con: pip install Pillow")
            return None
        except Exception as e:
            logger.error(f"Screenshot error: {e}")
            return None
    
    async def capture_camera(self) -> Optional[str]:
        """
        Cattura foto dalla webcam
        Ritorna path del file salvato
        """
        try:
            import cv2
            
            # Apri webcam
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                logger.error("Impossibile aprire la webcam")
                return None
            
            # Cattura frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                logger.error("Impossibile catturare frame dalla webcam")
                return None
            
            # Salva
            temp_path = Path(tempfile.gettempdir()) / f"gideon_camera_{int(time.time())}.jpg"
            cv2.imwrite(str(temp_path), frame)
            
            logger.info(f"📷 Foto webcam salvata: {temp_path}")
            return str(temp_path)
            
        except ImportError:
            logger.error("OpenCV non installato. Installa con: pip install opencv-python")
            return None
        except Exception as e:
            logger.error(f"Camera error: {e}")
            return None
    
    def _image_to_base64(self, image_path: str) -> Optional[str]:
        """Converti immagine in base64"""
        try:
            path = Path(image_path)
            if not path.exists():
                return None
            
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Base64 conversion error: {e}")
            return None
    
    def _get_mime_type(self, image_path: str) -> str:
        """Determina MIME type da estensione"""
        ext = Path(image_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        return mime_types.get(ext, 'image/jpeg')
    
    async def analyze_image(
        self,
        image_path: str,
        question: str = "Descrivi dettagliatamente questa immagine.",
        detailed: bool = True
    ) -> Dict:
        """
        Analizza un'immagine con AI Vision
        
        Args:
            image_path: Path all'immagine
            question: Domanda sull'immagine
            detailed: Se fare analisi dettagliata
        """
        # Converti in base64
        image_base64 = self._image_to_base64(image_path)
        if not image_base64:
            return {
                "success": False,
                "error": f"Impossibile leggere l'immagine: {image_path}"
            }
        
        mime_type = self._get_mime_type(image_path)
        
        # Prepara prompt
        if detailed:
            system_prompt = """Sei un esperto analista di immagini. Analizza l'immagine in modo:
1. DETTAGLIATO: Descrivi ogni elemento visibile
2. NUMERICO: Se ci sono numeri, calcoli, dati, riportali con precisione
3. CONTESTUALE: Spiega il contesto e il significato
4. TECNICO: Se rilevante, fornisci dettagli tecnici

Rispondi in italiano in modo chiaro e strutturato."""
        else:
            system_prompt = "Descrivi brevemente l'immagine in italiano."
        
        # Prova OpenRouter
        if self.openrouter_key:
            result = await self._analyze_with_openrouter(
                image_base64, mime_type, question, system_prompt
            )
            if result.get("success"):
                return result
        
        # Prova OpenAI
        if self.openai_key:
            result = await self._analyze_with_openai(
                image_base64, mime_type, question, system_prompt
            )
            if result.get("success"):
                return result
        
        return {
            "success": False,
            "error": "Nessun provider Vision AI configurato. Configura OPENROUTER_API_KEY o OPENAI_API_KEY."
        }
    
    async def _analyze_with_openrouter(
        self,
        image_base64: str,
        mime_type: str,
        question: str,
        system_prompt: str
    ) -> Dict:
        """Analisi con OpenRouter"""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                for model in self.vision_models:
                    try:
                        response = await client.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {self.openrouter_key}",
                                "Content-Type": "application/json"
                            },
                            json={
                                "model": model,
                                "messages": [
                                    {"role": "system", "content": system_prompt},
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "text", "text": question},
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:{mime_type};base64,{image_base64}"
                                                }
                                            }
                                        ]
                                    }
                                ],
                                "max_tokens": 2000
                            }
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            analysis = data["choices"][0]["message"]["content"]
                            
                            logger.info(f"📷 Immagine analizzata con {model}")
                            return {
                                "success": True,
                                "analysis": analysis,
                                "model": model,
                                "provider": "openrouter"
                            }
                    except Exception as e:
                        logger.warning(f"Model {model} failed: {e}")
                        continue
            
            return {"success": False, "error": "Tutti i modelli vision hanno fallito"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _analyze_with_openai(
        self,
        image_base64: str,
        mime_type: str,
        question: str,
        system_prompt: str
    ) -> Dict:
        """Analisi con OpenAI GPT-4 Vision"""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.openai_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": question},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{mime_type};base64,{image_base64}"
                                        }
                                    }
                                ]
                            }
                        ],
                        "max_tokens": 2000
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    analysis = data["choices"][0]["message"]["content"]
                    
                    logger.info("📷 Immagine analizzata con OpenAI")
                    return {
                        "success": True,
                        "analysis": analysis,
                        "model": "gpt-4o-mini",
                        "provider": "openai"
                    }
                else:
                    return {"success": False, "error": f"OpenAI error: {response.status_code}"}
                    
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============ SMART ACTIONS MANAGER ============

class SmartActionsManager:
    """
    Manager centrale per tutte le automazioni smart
    """
    
    def __init__(self):
        self.timer_manager = TimerManager()
        self.whatsapp = WhatsAppSender()
        self.email_sender: Optional[EmailSender] = None
        self.image_analyzer: Optional[ImageAnalyzer] = None
        
        self._initialized = False
    
    def initialize(
        self,
        openrouter_key: str = None,
        openai_key: str = None,
        email_config: EmailConfig = None
    ):
        """Inizializza tutti i servizi"""
        # Timer
        self.timer_manager.start()
        
        # Email
        if email_config:
            self.email_sender = EmailSender(email_config)
        
        # Image Analyzer - usa keys passate o carica da env
        # Filtra stringhe vuote
        or_key = openrouter_key if openrouter_key else os.getenv("OPENROUTER_API_KEY")
        oa_key = openai_key if openai_key else os.getenv("OPENAI_API_KEY")
        
        self.image_analyzer = ImageAnalyzer(
            openrouter_api_key=or_key,
            openai_api_key=oa_key
        )
        
        self._initialized = True
        logger.info("🤖 Smart Actions Manager inizializzato")
    
    def shutdown(self):
        """Ferma tutti i servizi"""
        self.timer_manager.stop()
        self._initialized = False
    
    # === Timer Actions ===
    
    def set_timer(self, minutes: float, name: str = "Timer", message: str = "") -> Dict:
        """Imposta un timer"""
        timer = self.timer_manager.create_timer(minutes, name, message)
        return {
            "success": True,
            "message": f"⏰ Timer '{name}' impostato per {minutes} minuti",
            "timer": timer.to_dict()
        }
    
    def set_alarm(self, hour: int, minute: int = 0, name: str = "Sveglia", repeat: bool = False) -> Dict:
        """Imposta una sveglia"""
        alarm = self.timer_manager.create_alarm(hour, minute, name, repeat=repeat)
        return {
            "success": True,
            "message": f"⏰ Sveglia '{name}' impostata per le {hour:02d}:{minute:02d}",
            "alarm": alarm.to_dict()
        }
    
    def list_timers(self) -> Dict:
        """Lista timer attivi"""
        timers = self.timer_manager.list_timers()
        return {
            "success": True,
            "timers": timers,
            "count": len(timers)
        }
    
    def cancel_timer(self, timer_id: str) -> Dict:
        """Cancella un timer"""
        if self.timer_manager.cancel_timer(timer_id):
            return {"success": True, "message": f"Timer {timer_id} cancellato"}
        return {"success": False, "error": "Timer non trovato"}
    
    # === WhatsApp Actions ===
    
    def send_whatsapp(self, phone: str, message: str) -> Dict:
        """Invia messaggio WhatsApp"""
        return self.whatsapp.send_message(phone, message)
    
    # === Email Actions ===
    
    def send_email(self, to: str, subject: str, body: str) -> Dict:
        """Invia email"""
        if not self.email_sender:
            return {"success": False, "error": "Email non configurata"}
        return self.email_sender.send_email(to, subject, body)
    
    # === Image Analysis ===
    
    def _ensure_image_analyzer(self):
        """Crea ImageAnalyzer on-demand se non esiste"""
        if not self.image_analyzer:
            logger.info("📷 Creazione ImageAnalyzer on-demand...")
            self.image_analyzer = ImageAnalyzer()
    
    async def analyze_screenshot(self, question: str = "Cosa vedi in questa immagine?") -> Dict:
        """Cattura e analizza screenshot"""
        self._ensure_image_analyzer()
        
        if not self.image_analyzer.openrouter_key and not self.image_analyzer.openai_key:
            return {"success": False, "error": "Nessun provider Vision AI configurato"}
        
        # Cattura
        image_path = await self.image_analyzer.capture_screenshot()
        if not image_path:
            return {"success": False, "error": "Impossibile catturare screenshot"}
        
        # Analizza
        result = await self.image_analyzer.analyze_image(image_path, question)
        result["image_path"] = image_path
        return result
    
    async def analyze_camera(self, question: str = "Cosa vedi in questa foto?") -> Dict:
        """Cattura foto da webcam e analizza"""
        self._ensure_image_analyzer()
        
        if not self.image_analyzer.openrouter_key and not self.image_analyzer.openai_key:
            return {"success": False, "error": "Nessun provider Vision AI configurato"}
        
        # Cattura
        image_path = await self.image_analyzer.capture_camera()
        if not image_path:
            return {"success": False, "error": "Impossibile catturare dalla webcam"}
        
        # Analizza
        result = await self.image_analyzer.analyze_image(image_path, question)
        result["image_path"] = image_path
        return result
    
    async def analyze_image(self, image_path: str, question: str = "Analizza questa immagine") -> Dict:
        """Analizza un'immagine esistente"""
        self._ensure_image_analyzer()
        
        if not self.image_analyzer.openrouter_key and not self.image_analyzer.openai_key:
            return {"success": False, "error": "Nessun provider Vision AI configurato. Configura OPENROUTER_API_KEY o OPENAI_API_KEY."}
        
        return await self.image_analyzer.analyze_image(image_path, question)


# Istanza globale
smart_actions = SmartActionsManager()
