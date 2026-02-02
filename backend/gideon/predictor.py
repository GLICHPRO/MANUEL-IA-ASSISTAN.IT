"""
🔮 GIDEON 3.0 - Predictor
Sistema di previsioni basato su pattern e contesto
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json


class Predictor:
    """
    Motore predittivo di Gideon 3.0
    Analizza pattern storici e contesto per generare previsioni
    """
    
    def __init__(self):
        self.patterns = {}
        self.history = []
        self.models = {
            "time_based": self._predict_time_based,
            "pattern_based": self._predict_pattern_based,
            "context_based": self._predict_context_based,
            "behavior_based": self._predict_behavior_based
        }
        
    async def predict(self, context: dict) -> dict:
        """
        Genera previsioni basate sul contesto
        
        Args:
            context: Dizionario con query, dati storici, stato attuale
            
        Returns:
            Dizionario con previsioni e probabilità
        """
        predictions = []
        
        # Esegui tutti i modelli predittivi
        for model_name, model_func in self.models.items():
            try:
                result = await model_func(context)
                if result:
                    predictions.append({
                        "model": model_name,
                        "prediction": result,
                        "timestamp": datetime.now().isoformat()
                    })
            except Exception as e:
                print(f"⚠️ Errore in {model_name}: {e}")
        
        # Combina le previsioni
        combined = self._combine_predictions(predictions)
        
        return {
            "predictions": predictions,
            "combined": combined,
            "confidence": self._calculate_confidence(predictions),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _predict_time_based(self, context: dict) -> Optional[dict]:
        """Previsioni basate su pattern temporali"""
        now = datetime.now()
        hour = now.hour
        day = now.weekday()
        
        predictions = {
            "time_of_day": self._get_time_period(hour),
            "expected_activity": self._get_expected_activity(hour, day),
            "suggested_actions": []
        }
        
        # Pattern mattutini (6-9)
        if 6 <= hour < 9:
            predictions["suggested_actions"] = [
                {"action": "check_calendar", "reason": "Inizio giornata", "priority": 0.9},
                {"action": "weather_brief", "reason": "Pianificazione giornata", "priority": 0.8},
                {"action": "news_summary", "reason": "Aggiornamento notizie", "priority": 0.6}
            ]
        # Pattern lavorativi (9-18)
        elif 9 <= hour < 18:
            predictions["suggested_actions"] = [
                {"action": "productivity_check", "reason": "Orario lavorativo", "priority": 0.7},
                {"action": "task_reminder", "reason": "Gestione attività", "priority": 0.8}
            ]
        # Pattern serali (18-22)
        elif 18 <= hour < 22:
            predictions["suggested_actions"] = [
                {"action": "daily_summary", "reason": "Fine giornata", "priority": 0.7},
                {"action": "tomorrow_prep", "reason": "Preparazione domani", "priority": 0.6}
            ]
        # Pattern notturni (22-6)
        else:
            predictions["suggested_actions"] = [
                {"action": "quiet_mode", "reason": "Orario notturno", "priority": 0.9},
                {"action": "backup_data", "reason": "Manutenzione notturna", "priority": 0.5}
            ]
            
        return predictions
    
    async def _predict_pattern_based(self, context: dict) -> Optional[dict]:
        """Previsioni basate su pattern comportamentali ricorrenti"""
        # Analizza pattern dalla history
        if not self.history:
            return {"patterns_found": 0, "predictions": []}
        
        # Trova pattern ricorrenti
        patterns = self._find_recurring_patterns()
        
        return {
            "patterns_found": len(patterns),
            "patterns": patterns[:5],  # Top 5 pattern
            "next_likely_action": patterns[0] if patterns else None
        }
    
    async def _predict_context_based(self, context: dict) -> Optional[dict]:
        """Previsioni basate sul contesto attuale"""
        query = context.get("query", "")
        analysis = context.get("analysis", {})
        
        predictions = {
            "intent_predictions": [],
            "follow_up_likely": []
        }
        
        # Predici intenti correlati
        if "sistema" in query.lower() or "performance" in query.lower():
            predictions["follow_up_likely"] = [
                {"query": "ottimizza sistema", "probability": 0.7},
                {"query": "mostra dettagli CPU", "probability": 0.5},
                {"query": "libera memoria", "probability": 0.4}
            ]
        elif "file" in query.lower() or "documento" in query.lower():
            predictions["follow_up_likely"] = [
                {"query": "apri file", "probability": 0.6},
                {"query": "cerca file simili", "probability": 0.4},
                {"query": "backup file", "probability": 0.3}
            ]
            
        return predictions
    
    async def _predict_behavior_based(self, context: dict) -> Optional[dict]:
        """Previsioni basate sul comportamento utente"""
        return {
            "user_state": "active",  # active, idle, busy
            "attention_level": 0.8,
            "likely_needs": [
                {"need": "information", "probability": 0.6},
                {"need": "action", "probability": 0.3},
                {"need": "reminder", "probability": 0.1}
            ]
        }
    
    def _combine_predictions(self, predictions: list) -> dict:
        """Combina previsioni da tutti i modelli"""
        if not predictions:
            return {}
        
        combined = {
            "all_suggested_actions": [],
            "primary_prediction": None,
            "models_used": len(predictions)
        }
        
        # Raccogli tutte le azioni suggerite
        for pred in predictions:
            if "prediction" in pred and pred["prediction"]:
                if "suggested_actions" in pred["prediction"]:
                    combined["all_suggested_actions"].extend(
                        pred["prediction"]["suggested_actions"]
                    )
        
        # Ordina per priorità
        combined["all_suggested_actions"].sort(
            key=lambda x: x.get("priority", 0), 
            reverse=True
        )
        
        # Imposta previsione primaria
        if combined["all_suggested_actions"]:
            combined["primary_prediction"] = combined["all_suggested_actions"][0]
            
        return combined
    
    def _calculate_confidence(self, predictions: list) -> float:
        """Calcola confidenza complessiva delle previsioni"""
        if not predictions:
            return 0.0
        
        # Media delle confidenze dei singoli modelli
        return min(len(predictions) * 0.2, 0.9)
    
    def _get_time_period(self, hour: int) -> str:
        """Determina il periodo della giornata"""
        if 5 <= hour < 12:
            return "mattina"
        elif 12 <= hour < 17:
            return "pomeriggio"
        elif 17 <= hour < 21:
            return "sera"
        else:
            return "notte"
    
    def _get_expected_activity(self, hour: int, day: int) -> str:
        """Stima l'attività prevista"""
        is_weekend = day >= 5
        
        if is_weekend:
            return "bassa" if hour < 10 or hour > 22 else "media"
        else:
            if 9 <= hour < 18:
                return "alta"
            elif 6 <= hour < 9 or 18 <= hour < 22:
                return "media"
            else:
                return "bassa"
    
    def _find_recurring_patterns(self) -> list:
        """Trova pattern ricorrenti nella history"""
        # Implementazione base - espandibile
        return []
    
    def add_to_history(self, event: dict):
        """Aggiunge un evento alla history per apprendimento"""
        event["timestamp"] = datetime.now().isoformat()
        self.history.append(event)
        # Mantieni solo ultimi 1000 eventi
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
