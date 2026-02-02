"""
🤖 Smart Actions API Routes

Endpoints per:
- ⏰ Timer e Sveglie
- 📱 WhatsApp
- 📧 Email
- 📷 Analisi Immagini
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import base64
import tempfile
from pathlib import Path
from loguru import logger

router = APIRouter(prefix="/api/smart", tags=["Smart Actions"])

# Import smart actions manager
from automation.smart_actions import smart_actions, EmailConfig


# ============ REQUEST MODELS ============

class TimerRequest(BaseModel):
    """Richiesta creazione timer"""
    minutes: float = Field(..., gt=0, description="Durata in minuti")
    name: str = Field(default="Timer", description="Nome del timer")
    message: str = Field(default="", description="Messaggio alla fine")


class AlarmRequest(BaseModel):
    """Richiesta creazione sveglia"""
    hour: int = Field(..., ge=0, le=23, description="Ora (0-23)")
    minute: int = Field(default=0, ge=0, le=59, description="Minuto (0-59)")
    name: str = Field(default="Sveglia", description="Nome sveglia")
    repeat: bool = Field(default=False, description="Ripeti ogni giorno")


class WhatsAppRequest(BaseModel):
    """Richiesta invio WhatsApp"""
    phone: str = Field(..., description="Numero telefono con prefisso (es. +393331234567)")
    message: str = Field(..., description="Messaggio da inviare")


class EmailRequest(BaseModel):
    """Richiesta invio email"""
    to: str = Field(..., description="Email destinatario")
    subject: str = Field(..., description="Oggetto")
    body: str = Field(..., description="Corpo del messaggio")
    html: bool = Field(default=False, description="Se il corpo è HTML")


class ImageAnalysisRequest(BaseModel):
    """Richiesta analisi immagine"""
    question: str = Field(
        default="Descrivi dettagliatamente questa immagine, inclusi tutti i numeri, testi e dati visibili.",
        description="Domanda sull'immagine"
    )
    image_base64: Optional[str] = Field(default=None, description="Immagine in base64")
    image_url: Optional[str] = Field(default=None, description="URL immagine")


# ============ TIMER ENDPOINTS ============

@router.post("/timer/create")
async def create_timer(request: TimerRequest):
    """
    ⏰ Crea un nuovo timer
    
    Esempio: "Metti un timer di 5 minuti per la pasta"
    """
    result = smart_actions.set_timer(
        minutes=request.minutes,
        name=request.name,
        message=request.message
    )
    return result


@router.post("/alarm/create")
async def create_alarm(request: AlarmRequest):
    """
    ⏰ Crea una nuova sveglia
    
    Esempio: "Svegliami alle 7:30"
    """
    result = smart_actions.set_alarm(
        hour=request.hour,
        minute=request.minute,
        name=request.name,
        repeat=request.repeat
    )
    return result


@router.get("/timers/list")
async def list_timers():
    """
    📋 Lista tutti i timer e sveglie attivi
    """
    return smart_actions.list_timers()


@router.delete("/timer/{timer_id}")
async def cancel_timer(timer_id: str):
    """
    ❌ Cancella un timer o sveglia
    """
    return smart_actions.cancel_timer(timer_id)


# ============ WHATSAPP ENDPOINTS ============

@router.post("/whatsapp/send")
async def send_whatsapp(request: WhatsAppRequest):
    """
    📱 Invia messaggio WhatsApp
    
    Apre WhatsApp Web con il messaggio pre-compilato.
    L'utente deve solo premere INVIO per inviare.
    
    Esempio: "Manda un messaggio WhatsApp a +393331234567 dicendo Ciao!"
    """
    return smart_actions.send_whatsapp(request.phone, request.message)


# ============ EMAIL ENDPOINTS ============

@router.post("/email/send")
async def send_email(request: EmailRequest):
    """
    📧 Invia email
    
    Richiede configurazione SMTP in .env:
    - SMTP_EMAIL
    - SMTP_PASSWORD
    - SMTP_SERVER (default: smtp.gmail.com)
    
    Esempio: "Manda una email a mario@example.com con oggetto Riunione"
    """
    return smart_actions.send_email(
        to=request.to,
        subject=request.subject,
        body=request.body
    )


# ============ IMAGE ANALYSIS ENDPOINTS ============

@router.post("/vision/screenshot")
async def analyze_screenshot(request: ImageAnalysisRequest):
    """
    📷 Cattura screenshot e analizza con AI Vision
    
    Cattura lo schermo corrente e lo analizza con AI.
    Ottimo per:
    - Analizzare dati sullo schermo
    - Leggere testi e numeri
    - Descrivere interfacce
    
    Esempio: "Fai uno screenshot e dimmi cosa c'è scritto"
    """
    result = await smart_actions.analyze_screenshot(request.question)
    return result


@router.post("/vision/camera")
async def analyze_camera(request: ImageAnalysisRequest):
    """
    📷 Cattura foto da webcam e analizza con AI Vision
    
    Scatta una foto dalla webcam e la analizza.
    Ottimo per:
    - Analizzare documenti fisici
    - Leggere codici, etichette
    - Descrivere oggetti
    
    Esempio: "Scatta una foto e dimmi cosa vedi"
    """
    result = await smart_actions.analyze_camera(request.question)
    return result


@router.post("/vision/analyze")
async def analyze_image(request: ImageAnalysisRequest):
    """
    📷 Analizza un'immagine con AI Vision
    
    Supporta:
    - image_base64: Immagine codificata in base64
    - image_url: URL di un'immagine online
    
    Risponde a domande sull'immagine, incluse:
    - Descrizioni dettagliate
    - Lettura testi e numeri
    - Calcoli matematici visibili
    - Analisi dati/grafici
    """
    if not request.image_base64 and not request.image_url:
        raise HTTPException(status_code=400, detail="Fornire image_base64 o image_url")
    
    # Se base64, salva temporaneamente
    if request.image_base64:
        try:
            # Determina formato
            if request.image_base64.startswith("data:"):
                # Rimuovi header data URL
                header, data = request.image_base64.split(",", 1)
                ext = ".png" if "png" in header else ".jpg"
            else:
                data = request.image_base64
                ext = ".jpg"
            
            # Decodifica e salva
            image_bytes = base64.b64decode(data)
            temp_path = Path(tempfile.gettempdir()) / f"gideon_upload_{int(datetime.now().timestamp())}{ext}"
            temp_path.write_bytes(image_bytes)
            
            result = await smart_actions.analyze_image(str(temp_path), request.question)
            return result
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Errore decodifica base64: {str(e)}")
    
    # Se URL, scarica
    if request.image_url:
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(request.image_url)
                if response.status_code != 200:
                    raise HTTPException(status_code=400, detail="Impossibile scaricare l'immagine")
                
                # Determina estensione
                content_type = response.headers.get("content-type", "")
                ext = ".jpg"
                if "png" in content_type:
                    ext = ".png"
                elif "gif" in content_type:
                    ext = ".gif"
                
                temp_path = Path(tempfile.gettempdir()) / f"gideon_url_{int(datetime.now().timestamp())}{ext}"
                temp_path.write_bytes(response.content)
                
                result = await smart_actions.analyze_image(str(temp_path), request.question)
                return result
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Errore download immagine: {str(e)}")


@router.post("/vision/upload")
async def analyze_uploaded_image(
    file: UploadFile = File(...),
    question: str = Form(default="Analizza questa immagine in dettaglio, inclusi tutti i numeri e testi visibili.")
):
    """
    📷 Upload e analizza immagine
    
    Carica un file immagine e lo analizza con AI Vision.
    Supporta: JPG, PNG, GIF, WebP
    """
    # Verifica tipo file
    allowed_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Tipo file non supportato: {file.content_type}")
    
    # Salva temporaneamente
    ext = Path(file.filename).suffix or ".jpg"
    temp_path = Path(tempfile.gettempdir()) / f"gideon_upload_{int(datetime.now().timestamp())}{ext}"
    
    content = await file.read()
    temp_path.write_bytes(content)
    
    result = await smart_actions.analyze_image(str(temp_path), question)
    return result


# ============ INFO ENDPOINT ============

@router.get("/info")
async def get_smart_actions_info():
    """
    ℹ️ Informazioni sulle Smart Actions disponibili
    """
    return {
        "name": "Gideon Smart Actions",
        "version": "1.0",
        "actions": {
            "timer": {
                "description": "Crea timer con notifica",
                "examples": [
                    "Metti un timer di 5 minuti",
                    "Timer 30 minuti per la riunione"
                ]
            },
            "alarm": {
                "description": "Crea sveglie",
                "examples": [
                    "Svegliami alle 7:30",
                    "Metti una sveglia alle 14:00 per la call"
                ]
            },
            "whatsapp": {
                "description": "Invia messaggi WhatsApp",
                "examples": [
                    "Manda un WhatsApp a +393331234567",
                    "Scrivi su WhatsApp a Mario: Ci vediamo alle 8"
                ]
            },
            "email": {
                "description": "Invia email",
                "examples": [
                    "Manda una email a mario@example.com",
                    "Scrivi a info@azienda.it con oggetto Preventivo"
                ]
            },
            "vision": {
                "description": "Analizza immagini con AI",
                "examples": [
                    "Fai uno screenshot e analizzalo",
                    "Scatta una foto e dimmi cosa vedi",
                    "Analizza questa immagine e leggi i numeri"
                ]
            }
        }
    }
