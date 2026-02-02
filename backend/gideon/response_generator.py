"""
Gideon Response Generator - Output narrativi intelligenti
Genera risposte descrittive e informative per Jarvis e l'utente.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
import random


class ResponseTone(Enum):
    """Tono della risposta"""
    FORMAL = "formal"           # Formale e professionale
    CONCISE = "concise"         # Breve e diretto
    DETAILED = "detailed"       # Dettagliato e analitico
    ADVISORY = "advisory"       # Consulenziale
    URGENT = "urgent"           # Urgente/allerta


class ResponseType(Enum):
    """Tipo di risposta"""
    SCENARIO_ANALYSIS = "scenario_analysis"
    SIMULATION_RESULT = "simulation_result"
    RANKING = "ranking"
    RISK_ASSESSMENT = "risk_assessment"
    PREDICTION = "prediction"
    RECOMMENDATION = "recommendation"
    STATUS_UPDATE = "status_update"
    WARNING = "warning"
    CONFIRMATION = "confirmation"


@dataclass
class GideonResponse:
    """Risposta formattata di Gideon"""
    response_type: ResponseType
    primary_message: str
    details: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 0.0
    tone: ResponseTone = ResponseTone.FORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    for_jarvis: bool = False  # Se destinato a Jarvis per azione
    
    def to_dict(self) -> dict:
        return {
            "type": self.response_type.value,
            "message": self.primary_message,
            "details": self.details,
            "metrics": self.metrics,
            "recommendations": self.recommendations,
            "confidence": round(self.confidence, 2),
            "tone": self.tone.value,
            "timestamp": self.timestamp.isoformat(),
            "for_jarvis": self.for_jarvis
        }
    
    def __str__(self) -> str:
        return self.primary_message


class GideonResponseGenerator:
    """
    Generatore di risposte narrative per Gideon.
    Trasforma analisi tecniche in output comprensibili.
    """
    
    def __init__(self):
        self.response_history: List[GideonResponse] = []
        self.default_tone = ResponseTone.FORMAL
        
        # Template per risposte
        self._init_templates()
    
    def _init_templates(self):
        """Inizializza template di risposta"""
        
        # Template scenario
        self.scenario_templates = {
            "optimal": [
                "Scenario ottimale identificato con probabilità {prob:.2f}, rischio {risk}.",
                "Ho individuato lo scenario migliore: probabilità di successo {prob:.0%}, livello di rischio {risk}.",
                "Analisi completata. Scenario raccomandato con {prob:.0%} di probabilità e rischio {risk}.",
            ],
            "comparison": [
                "Confronto completato tra {n} scenari. Il migliore presenta probabilità {prob:.0%}.",
                "Analizzati {n} scenari possibili. Quello ottimale ha {prob:.0%} di successo.",
                "Valutazione di {n} alternative completata. Scenario preferito: {prob:.0%} probabilità.",
            ],
            "warning": [
                "Attenzione: scenario con rischio {risk}. Probabilità di successo solo {prob:.0%}.",
                "Segnalo criticità: rischio {risk} rilevato, successo stimato al {prob:.0%}.",
                "Avviso: lo scenario presenta rischio {risk}. Consiglio cautela.",
            ]
        }
        
        # Template simulazione
        self.simulation_templates = {
            "completed": [
                "Simulazione completata. {outcome}",
                "Ho terminato la simulazione. Risultato: {outcome}",
                "Analisi simulativa conclusa. {outcome}",
            ],
            "with_suggestions": [
                "Simulazione completata, suggerisco modifiche al flusso operativo.",
                "Simulazione terminata. Raccomando ottimizzazioni al processo.",
                "Analisi conclusa. Ho identificato miglioramenti possibili.",
            ],
            "iterations": [
                "Eseguite {n} iterazioni. Convergenza raggiunta con confidenza {conf:.0%}.",
                "Simulazione su {n} cicli completata. Risultati stabili al {conf:.0%}.",
                "{n} simulazioni effettuate. Affidabilità risultati: {conf:.0%}.",
            ]
        }
        
        # Template ranking
        self.ranking_templates = {
            "ready": [
                "Classifica dei {n} scenari più sicuri pronta per Jarvis.",
                "Ho preparato il ranking dei {n} scenari ottimali per l'esecuzione.",
                "Classificati {n} scenari per sicurezza. Lista pronta per Jarvis.",
            ],
            "top_pick": [
                "Scenario #{rank} raccomandato: {name} con score {score:.2f}.",
                "Prima scelta: {name} (posizione #{rank}, punteggio {score:.2f}).",
                "Miglior opzione identificata: {name}, score {score:.2f}.",
            ],
            "comparison": [
                "Top {n}: range di probabilità da {min:.0%} a {max:.0%}.",
                "I migliori {n} scenari hanno probabilità tra {min:.0%} e {max:.0%}.",
                "Classifica pronta. I {n} scenari top variano da {min:.0%} a {max:.0%}.",
            ]
        }
        
        # Template rischio
        self.risk_templates = {
            "low": [
                "Rischio valutato come minimo. Procedi con fiducia.",
                "Analisi rischio: livello basso. Operazione sicura.",
                "Rischio trascurabile identificato. Via libera per l'esecuzione.",
            ],
            "medium": [
                "Rischio moderato rilevato. Consiglio monitoraggio attivo.",
                "Livello di rischio medio. Raccomando precauzioni standard.",
                "Attenzione moderata richiesta. Rischio nella norma ma presente.",
            ],
            "high": [
                "Rischio elevato! Raccomando valutazione approfondita prima di procedere.",
                "Attenzione: rischio alto identificato. Suggerisco alternative più sicure.",
                "Livello di rischio critico. Sconsiglio l'esecuzione senza mitigazioni.",
            ],
            "critical": [
                "RISCHIO CRITICO. Blocco operazione consigliato.",
                "ALLERTA: rischio estremo rilevato. Non procedere.",
                "Situazione critica. Raccomando abort dell'operazione.",
            ]
        }
        
        # Template predizione
        self.prediction_templates = {
            "positive": [
                "Previsione positiva: {metric} stimato a {value} con confidenza {conf:.0%}.",
                "Outlook favorevole. {metric} previsto: {value} (confidenza {conf:.0%}).",
                "Stima ottimistica per {metric}: {value}. Affidabilità: {conf:.0%}.",
            ],
            "negative": [
                "Previsione negativa: {metric} potrebbe scendere a {value}.",
                "Outlook sfavorevole per {metric}. Valore atteso: {value}.",
                "Attenzione: {metric} in calo previsto verso {value}.",
            ],
            "neutral": [
                "Previsione stabile per {metric}: {value} atteso.",
                "{metric} previsto invariato a {value}.",
                "Nessuna variazione significativa attesa per {metric}.",
            ]
        }
        
        # Template raccomandazione
        self.recommendation_templates = {
            "action": [
                "Raccomando: {action}. Probabilità successo: {prob:.0%}.",
                "Suggerimento operativo: {action} (successo stimato {prob:.0%}).",
                "Consiglio di procedere con: {action}. Confidenza: {prob:.0%}.",
            ],
            "wait": [
                "Suggerisco di attendere. Condizioni non ottimali.",
                "Raccomando pausa operativa. Momento non favorevole.",
                "Consiglio di posticipare. Timing non ideale.",
            ],
            "alternative": [
                "Propongo alternativa: {alt}. Migliore del {improvement:.0%}.",
                "Suggerisco opzione B: {alt}. Incremento stimato: {improvement:.0%}.",
                "Valuta questa alternativa: {alt} (+{improvement:.0%} rispetto al piano).",
            ]
        }
    
    # === Generatori Principali ===
    
    def scenario_optimal(self, probability: float, risk_level: str,
                         scenario_name: str = None,
                         details: Dict = None) -> GideonResponse:
        """Genera risposta per scenario ottimale identificato"""
        
        risk_map = {
            "minimal": "minimo",
            "low": "basso", 
            "medium": "moderato",
            "high": "elevato",
            "critical": "critico"
        }
        risk_it = risk_map.get(risk_level, risk_level)
        
        template = random.choice(self.scenario_templates["optimal"])
        message = template.format(prob=probability, risk=risk_it)
        
        response = GideonResponse(
            response_type=ResponseType.SCENARIO_ANALYSIS,
            primary_message=message,
            confidence=probability,
            tone=ResponseTone.ADVISORY,
            for_jarvis=True
        )
        
        if scenario_name:
            response.details.append(f"Scenario: {scenario_name}")
        
        if details:
            response.metrics = details
            if "factors" in details:
                response.details.append(f"Fattori chiave: {', '.join(details['factors'][:3])}")
        
        self.response_history.append(response)
        return response
    
    def simulation_completed(self, outcome: str, 
                            iterations: int = None,
                            confidence: float = 0.85,
                            suggestions: List[str] = None) -> GideonResponse:
        """Genera risposta per simulazione completata"""
        
        if suggestions:
            template = random.choice(self.simulation_templates["with_suggestions"])
            message = template
        elif iterations:
            template = random.choice(self.simulation_templates["iterations"])
            message = template.format(n=iterations, conf=confidence)
        else:
            template = random.choice(self.simulation_templates["completed"])
            message = template.format(outcome=outcome)
        
        response = GideonResponse(
            response_type=ResponseType.SIMULATION_RESULT,
            primary_message=message,
            confidence=confidence,
            tone=ResponseTone.DETAILED,
            for_jarvis=True
        )
        
        response.details.append(f"Esito: {outcome}")
        
        if iterations:
            response.metrics["iterations"] = iterations
        
        if suggestions:
            response.recommendations = suggestions
            for sug in suggestions[:2]:
                response.details.append(f"→ {sug}")
        
        self.response_history.append(response)
        return response
    
    def ranking_ready(self, scenarios: List[Dict], 
                     top_n: int = 5,
                     criterion: str = "sicurezza") -> GideonResponse:
        """Genera risposta per classifica scenari pronta"""
        
        n = min(top_n, len(scenarios))
        template = random.choice(self.ranking_templates["ready"])
        message = template.format(n=n)
        
        response = GideonResponse(
            response_type=ResponseType.RANKING,
            primary_message=message,
            confidence=0.9,
            tone=ResponseTone.FORMAL,
            for_jarvis=True
        )
        
        # Aggiungi dettagli classifica
        probs = [s.get("probability", s.get("score", 0.5)) for s in scenarios[:n]]
        if probs:
            response.metrics["top_scenarios"] = n
            response.metrics["probability_range"] = {
                "min": min(probs),
                "max": max(probs)
            }
            
            comparison = random.choice(self.ranking_templates["comparison"])
            response.details.append(
                comparison.format(n=n, min=min(probs), max=max(probs))
            )
        
        # Top 3 nel dettaglio
        for i, scenario in enumerate(scenarios[:3], 1):
            name = scenario.get("name", f"Scenario {i}")
            score = scenario.get("probability", scenario.get("score", 0))
            response.details.append(f"#{i} {name}: {score:.0%}")
        
        self.response_history.append(response)
        return response
    
    def risk_assessment(self, risk_level: str,
                       risk_score: float,
                       factors: List[str] = None,
                       mitigations: List[str] = None) -> GideonResponse:
        """Genera risposta per valutazione rischio"""
        
        # Determina categoria
        if risk_score < 0.2:
            category = "low"
            tone = ResponseTone.FORMAL
        elif risk_score < 0.4:
            category = "medium"
            tone = ResponseTone.ADVISORY
        elif risk_score < 0.7:
            category = "high"
            tone = ResponseTone.URGENT
        else:
            category = "critical"
            tone = ResponseTone.URGENT
        
        template = random.choice(self.risk_templates[category])
        message = template
        
        response = GideonResponse(
            response_type=ResponseType.RISK_ASSESSMENT,
            primary_message=message,
            confidence=1 - risk_score,
            tone=tone,
            for_jarvis=risk_score >= 0.4
        )
        
        response.metrics["risk_score"] = risk_score
        response.metrics["risk_level"] = risk_level
        
        if factors:
            response.details.append(f"Fattori di rischio: {', '.join(factors[:3])}")
        
        if mitigations:
            response.recommendations = mitigations
            response.details.append("Mitigazioni disponibili.")
        
        self.response_history.append(response)
        return response
    
    def prediction(self, metric: str, value: Any,
                  confidence: float,
                  trend: str = "neutral",
                  context: Dict = None) -> GideonResponse:
        """Genera risposta per previsione"""
        
        if trend == "positive" or (isinstance(value, (int, float)) and confidence > 0.7):
            templates = self.prediction_templates["positive"]
        elif trend == "negative":
            templates = self.prediction_templates["negative"]
        else:
            templates = self.prediction_templates["neutral"]
        
        template = random.choice(templates)
        
        # Formatta valore
        if isinstance(value, float) and value < 1:
            value_str = f"{value:.0%}"
        elif isinstance(value, float):
            value_str = f"{value:.2f}"
        else:
            value_str = str(value)
        
        message = template.format(metric=metric, value=value_str, conf=confidence)
        
        response = GideonResponse(
            response_type=ResponseType.PREDICTION,
            primary_message=message,
            confidence=confidence,
            tone=ResponseTone.ADVISORY
        )
        
        response.metrics["metric"] = metric
        response.metrics["predicted_value"] = value
        response.metrics["trend"] = trend
        
        if context:
            response.metrics.update(context)
        
        self.response_history.append(response)
        return response
    
    def recommendation(self, action: str,
                      probability: float,
                      reasoning: str = None,
                      alternatives: List[Dict] = None) -> GideonResponse:
        """Genera risposta per raccomandazione"""
        
        template = random.choice(self.recommendation_templates["action"])
        message = template.format(action=action, prob=probability)
        
        response = GideonResponse(
            response_type=ResponseType.RECOMMENDATION,
            primary_message=message,
            confidence=probability,
            tone=ResponseTone.ADVISORY,
            for_jarvis=True
        )
        
        if reasoning:
            response.details.append(f"Motivazione: {reasoning}")
        
        if alternatives:
            response.recommendations = [
                f"{alt['action']} ({alt.get('probability', 0):.0%})"
                for alt in alternatives[:3]
            ]
        
        self.response_history.append(response)
        return response
    
    def suggest_wait(self, reason: str,
                    estimated_wait: str = None) -> GideonResponse:
        """Genera risposta per suggerimento di attesa"""
        
        template = random.choice(self.recommendation_templates["wait"])
        message = template
        
        response = GideonResponse(
            response_type=ResponseType.RECOMMENDATION,
            primary_message=message,
            confidence=0.7,
            tone=ResponseTone.ADVISORY,
            for_jarvis=True
        )
        
        response.details.append(f"Motivo: {reason}")
        
        if estimated_wait:
            response.details.append(f"Attesa stimata: {estimated_wait}")
        
        self.response_history.append(response)
        return response
    
    def warning(self, message: str,
               severity: str = "medium",
               action_required: bool = False) -> GideonResponse:
        """Genera risposta di warning"""
        
        if severity == "critical":
            prefix = "🚨 CRITICO: "
            tone = ResponseTone.URGENT
        elif severity == "high":
            prefix = "⚠️ ATTENZIONE: "
            tone = ResponseTone.URGENT
        else:
            prefix = "ℹ️ Nota: "
            tone = ResponseTone.ADVISORY
        
        response = GideonResponse(
            response_type=ResponseType.WARNING,
            primary_message=prefix + message,
            confidence=0.9,
            tone=tone,
            for_jarvis=action_required
        )
        
        response.metrics["severity"] = severity
        response.metrics["action_required"] = action_required
        
        self.response_history.append(response)
        return response
    
    def status_update(self, status: str,
                     progress: float = None,
                     eta: str = None) -> GideonResponse:
        """Genera risposta di aggiornamento stato"""
        
        message = status
        
        if progress is not None:
            message += f" ({progress:.0%} completato)"
        
        if eta:
            message += f" - ETA: {eta}"
        
        response = GideonResponse(
            response_type=ResponseType.STATUS_UPDATE,
            primary_message=message,
            confidence=1.0,
            tone=ResponseTone.CONCISE
        )
        
        if progress:
            response.metrics["progress"] = progress
        if eta:
            response.metrics["eta"] = eta
        
        self.response_history.append(response)
        return response
    
    # === Metodi Combinati ===
    
    def full_analysis_report(self, 
                            scenario: Dict,
                            simulation_result: Dict,
                            risk: Dict,
                            recommendations: List[str]) -> GideonResponse:
        """Genera report completo di analisi"""
        
        prob = scenario.get("probability", 0.5)
        risk_level = risk.get("level", "medium")
        
        # Costruisci messaggio principale
        messages = []
        
        # Scenario
        messages.append(
            f"Scenario analizzato con probabilità {prob:.0%}"
        )
        
        # Rischio
        risk_map = {"low": "basso", "medium": "moderato", "high": "elevato", "critical": "critico"}
        messages.append(f"rischio {risk_map.get(risk_level, risk_level)}")
        
        primary = "Analisi completata. " + ", ".join(messages) + "."
        
        response = GideonResponse(
            response_type=ResponseType.SCENARIO_ANALYSIS,
            primary_message=primary,
            confidence=prob,
            tone=ResponseTone.DETAILED,
            for_jarvis=True
        )
        
        # Dettagli simulazione
        if simulation_result:
            sim_outcome = simulation_result.get("outcome", "completata")
            response.details.append(f"Simulazione: {sim_outcome}")
        
        # Metriche
        response.metrics = {
            "probability": prob,
            "risk_level": risk_level,
            "risk_score": risk.get("score", 0.5)
        }
        
        # Raccomandazioni
        response.recommendations = recommendations
        
        self.response_history.append(response)
        return response
    
    def compare_scenarios(self, scenarios: List[Dict]) -> GideonResponse:
        """Genera confronto tra scenari"""
        
        if not scenarios:
            return self.warning("Nessuno scenario da confrontare", "low")
        
        n = len(scenarios)
        best = max(scenarios, key=lambda s: s.get("probability", 0))
        best_prob = best.get("probability", 0.5)
        
        template = random.choice(self.scenario_templates["comparison"])
        message = template.format(n=n, prob=best_prob)
        
        response = GideonResponse(
            response_type=ResponseType.SCENARIO_ANALYSIS,
            primary_message=message,
            confidence=best_prob,
            tone=ResponseTone.DETAILED,
            for_jarvis=True
        )
        
        # Classifica top 3
        sorted_scenarios = sorted(
            scenarios, 
            key=lambda s: s.get("probability", 0), 
            reverse=True
        )
        
        for i, s in enumerate(sorted_scenarios[:3], 1):
            name = s.get("name", f"Scenario {i}")
            prob = s.get("probability", 0)
            response.details.append(f"#{i} {name}: {prob:.0%}")
        
        response.metrics["scenarios_compared"] = n
        response.metrics["best_scenario"] = best.get("name", "N/A")
        response.metrics["best_probability"] = best_prob
        
        self.response_history.append(response)
        return response
    
    # === Utility ===
    
    def get_history(self, limit: int = 10) -> List[GideonResponse]:
        """Ottiene storia risposte recenti"""
        return self.response_history[-limit:]
    
    def clear_history(self):
        """Pulisce storia"""
        self.response_history.clear()
    
    def get_status(self) -> Dict:
        """Stato del generatore"""
        return {
            "responses_generated": len(self.response_history),
            "default_tone": self.default_tone.value,
            "last_response": self.response_history[-1].to_dict() if self.response_history else None
        }
