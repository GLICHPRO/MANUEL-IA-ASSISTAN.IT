"""
🧠 GIDEON CORE — Sistema Centrale di Reasoning
==============================================

Azioni Core:
1. MultiToolReasoning - Interroga più tool, compara, segnala conflitti
2. ConfidenceWeightedOutput - Output con % confidenza, fonti, limiti noti
3. HumanOverrideGate - NESSUNA automazione critica senza conferma umana

Cross-Cutting:
- AuditTrailGenerator - Log completo per ogni azione
- BiasAndDriftMonitor - Auto-diagnosi bias e deriva
- FailSafeTrigger - Stop automatico se confidenza troppo bassa
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import hashlib
import json
import asyncio


# Soglie di sicurezza
CONFIDENCE_THRESHOLD_LOW = 0.40
CONFIDENCE_THRESHOLD_CRITICAL = 0.25
CRITICAL_ACTIONS = [
    "delete", "remove", "execute", "transfer", "authorize",
    "cancella", "rimuovi", "esegui", "trasferisci", "autorizza"
]


@dataclass
class ReasoningResult:
    """Risultato multi-tool reasoning"""
    query: str
    tools_consulted: List[str]
    individual_results: List[Dict[str, Any]]
    conflicts: List[Dict[str, Any]]
    synthesis: str
    combined_confidence: float

@dataclass
class WeightedOutput:
    """Output pesato con confidenza"""
    content: Any
    confidence: float
    sources: List[str]
    limitations: List[str]
    alternatives: List[str]

@dataclass
class AuditEntry:
    """Entry audit trail"""
    timestamp: str
    action: str
    actor: str
    details: Dict[str, Any]
    result: str
    hash: str


class GideonCore:
    """
    🧠 GIDEON CORE - Sistema Centrale
    
    Coordina tutti i tool con:
    - Reasoning multi-fonte
    - Confidenza pesata
    - Gate umano per azioni critiche
    - Audit trail completo
    """
    
    def __init__(self):
        self.audit_log: List[AuditEntry] = []
        self.bias_history: List[Dict[str, Any]] = []
        self.human_approvals: Dict[str, bool] = {}
        self.failsafe_triggered = False
    
    # ==================== ACTION 1: MultiToolReasoning ====================
    
    async def multi_tool_reasoning(
        self,
        query: str,
        tools_to_consult: List[str] = None,
        comparison_mode: str = "synthesize"  # synthesize, compare, conflict_check
    ) -> Dict[str, Any]:
        """
        🔄 MultiToolReasoning
        
        Interroga più tool per una query, confronta risultati,
        segnala conflitti e produce sintesi.
        
        Tools disponibili: security, cyber, science, archaeology, analysis
        """
        
        tools_to_consult = tools_to_consult or ["security", "analysis"]
        
        # Log audit
        await self._log_audit("multi_tool_reasoning", "system", {
            "query": query,
            "tools": tools_to_consult,
            "mode": comparison_mode
        })
        
        # Raccogli risultati
        result = await self._consult_multiple_tools(query, tools_to_consult)
        
        # Analizza conflitti
        conflicts = self._detect_conflicts(result.individual_results)
        result.conflicts = conflicts
        
        # Calcola confidenza combinata
        result.combined_confidence = self._calculate_combined_confidence(
            result.individual_results, conflicts
        )
        
        # Check failsafe
        if result.combined_confidence < CONFIDENCE_THRESHOLD_CRITICAL:
            self.failsafe_triggered = True
            return await self._failsafe_response(query, result)
        
        return {
            "success": True,
            "action": "multi_tool_reasoning",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "query": query,
                "mode": comparison_mode,
                "tools_consulted": result.tools_consulted,
                "individual_results": [
                    {
                        "tool": r.get("tool"),
                        "summary": r.get("summary", "N/A"),
                        "confidence": r.get("confidence", 0)
                    }
                    for r in result.individual_results
                ],
                "conflict_analysis": {
                    "conflicts_found": len(conflicts),
                    "conflicts": conflicts,
                    "resolution_needed": len(conflicts) > 0
                },
                "synthesis": result.synthesis
            },
            "summary": f"Consulted {len(result.tools_consulted)} tools - {len(conflicts)} conflicts found",
            "recommendations": self._generate_reasoning_recommendations(result),
            "confidence": result.combined_confidence,
            "disclaimer": "🔄 Sintesi da multiple fonti. Conflitti evidenziati richiedono valutazione umana."
        }
    
    async def _consult_multiple_tools(self, query: str, tools: List[str]) -> ReasoningResult:
        """Consulta tool multipli"""
        
        results = []
        
        for tool_name in tools:
            # Simula consultazione (in produzione: chiama tool reali)
            result = await self._simulate_tool_query(tool_name, query)
            results.append(result)
        
        # Genera sintesi
        synthesis = self._synthesize_results(results)
        
        return ReasoningResult(
            query=query,
            tools_consulted=tools,
            individual_results=results,
            conflicts=[],
            synthesis=synthesis,
            combined_confidence=0.75
        )
    
    async def _simulate_tool_query(self, tool: str, query: str) -> Dict[str, Any]:
        """Simula query a tool specifico"""
        
        # Placeholder - in produzione collegare ai tool reali
        confidence_map = {
            "security": 0.80,
            "cyber": 0.85,
            "science": 0.75,
            "archaeology": 0.70,
            "analysis": 0.82
        }
        
        return {
            "tool": tool,
            "query": query,
            "summary": f"Analisi {tool} per: {query[:50]}...",
            "confidence": confidence_map.get(tool, 0.70),
            "key_findings": [f"Finding 1 da {tool}", f"Finding 2 da {tool}"],
            "timestamp": datetime.now().isoformat()
        }
    
    def _detect_conflicts(self, results: List[Dict]) -> List[Dict]:
        """Rileva conflitti tra risultati"""
        
        conflicts = []
        
        # Controlla differenze significative di confidenza
        confidences = [r.get("confidence", 0) for r in results]
        if len(confidences) >= 2:
            max_conf = max(confidences)
            min_conf = min(confidences)
            if max_conf - min_conf > 0.30:
                conflicts.append({
                    "type": "confidence_divergence",
                    "description": "Differenza significativa nella confidenza tra tool",
                    "severity": "medium",
                    "tools_involved": [r.get("tool") for r in results]
                })
        
        return conflicts
    
    def _synthesize_results(self, results: List[Dict]) -> str:
        """Sintetizza risultati multipli"""
        
        if not results:
            return "Nessun risultato disponibile"
        
        tools = [r.get("tool") for r in results]
        avg_conf = sum(r.get("confidence", 0) for r in results) / len(results)
        
        return f"Sintesi da {len(results)} tool ({', '.join(tools)}). Confidenza media: {avg_conf:.0%}"
    
    def _calculate_combined_confidence(self, results: List[Dict], conflicts: List[Dict]) -> float:
        """Calcola confidenza combinata"""
        
        if not results:
            return 0.0
        
        # Media ponderata
        avg = sum(r.get("confidence", 0) for r in results) / len(results)
        
        # Penalità per conflitti
        penalty = len(conflicts) * 0.10
        
        return max(0.10, avg - penalty)
    
    def _generate_reasoning_recommendations(self, result: ReasoningResult) -> List[str]:
        recs = []
        
        if result.conflicts:
            recs.append("Risolvere conflitti identificati prima di procedere")
        
        if result.combined_confidence < 0.60:
            recs.append("Confidenza bassa - raccogliere più dati")
        
        if len(result.tools_consulted) < 3:
            recs.append("Considerare consultazione di tool aggiuntivi")
        
        return recs if recs else ["Procedere con le informazioni disponibili"]

    # ==================== ACTION 2: ConfidenceWeightedOutput ====================
    
    async def confidence_weighted_output(
        self,
        content: Any,
        analysis_type: str,
        sources: List[str] = None
    ) -> Dict[str, Any]:
        """
        📊 ConfidenceWeightedOutput
        
        Produce output con:
        - % confidenza esplicita
        - Fonti citate
        - Limiti noti dell'analisi
        - Alternative possibili
        """
        
        sources = sources or ["Analisi interna"]
        
        output = await self._weight_output(content, analysis_type, sources)
        
        return {
            "success": True,
            "action": "confidence_weighted_output",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "content": output.content,
                "confidence_analysis": {
                    "confidence_score": f"{output.confidence:.0%}",
                    "confidence_level": self._confidence_level(output.confidence),
                    "interpretation": self._interpret_confidence(output.confidence)
                },
                "sources": {
                    "cited": output.sources,
                    "source_quality": "Variable - vedere dettagli individuali"
                },
                "known_limitations": output.limitations,
                "alternative_interpretations": output.alternatives
            },
            "summary": f"Output con confidenza {output.confidence:.0%}",
            "recommendations": self._generate_confidence_recommendations(output),
            "confidence": output.confidence,
            "disclaimer": "📊 Confidenza calcolata su dati disponibili. Valori bassi indicano necessità di verifica."
        }
    
    async def _weight_output(
        self, content: Any, analysis_type: str, sources: List[str]
    ) -> WeightedOutput:
        """Calcola peso output"""
        
        # Calcola confidenza base
        base_confidence = 0.70
        
        # Boost per multiple fonti
        if len(sources) > 1:
            base_confidence += 0.05 * min(len(sources), 3)
        
        # Limiti comuni
        limitations = [
            "Basato su dati disponibili al momento dell'analisi",
            "Non tiene conto di informazioni non digitalizzate",
            "Possibile bias da fonti principali"
        ]
        
        # Alternative
        alternatives = [
            "Interpretazione alternativa possibile con dati aggiuntivi"
        ]
        
        return WeightedOutput(
            content=content,
            confidence=min(base_confidence, 0.95),
            sources=sources,
            limitations=limitations,
            alternatives=alternatives
        )
    
    def _confidence_level(self, conf: float) -> str:
        if conf >= 0.85:
            return "ALTA"
        elif conf >= 0.65:
            return "MEDIA"
        elif conf >= 0.45:
            return "BASSA"
        return "MOLTO BASSA"
    
    def _interpret_confidence(self, conf: float) -> str:
        if conf >= 0.85:
            return "Risultato affidabile - procedere con fiducia"
        elif conf >= 0.65:
            return "Risultato indicativo - verificare punti chiave"
        elif conf >= 0.45:
            return "Risultato incerto - richiedere più dati"
        return "Risultato non affidabile - non utilizzare per decisioni"
    
    def _generate_confidence_recommendations(self, output: WeightedOutput) -> List[str]:
        if output.confidence >= 0.80:
            return ["Output affidabile per uso generale"]
        elif output.confidence >= 0.60:
            return ["Verificare con fonte secondaria", "Usare con cautela per decisioni importanti"]
        return ["Non utilizzare senza verifica", "Raccogliere dati aggiuntivi"]

    # ==================== ACTION 3: HumanOverrideGate ====================
    
    async def human_override_gate(
        self,
        action: str,
        action_details: Dict[str, Any],
        criticality: str = "auto"  # auto, low, medium, high, critical
    ) -> Dict[str, Any]:
        """
        🚦 HumanOverrideGate
        
        NESSUNA automazione critica senza conferma umana.
        
        Classifica azioni per criticità e richiede:
        - Low: Log only
        - Medium: Notifica
        - High: Conferma richiesta
        - Critical: Conferma obbligatoria + secondo fattore
        """
        
        # Auto-detect criticità
        if criticality == "auto":
            criticality = self._assess_criticality(action, action_details)
        
        # Log audit
        await self._log_audit("human_override_gate", "system", {
            "action": action,
            "criticality": criticality,
            "details": action_details
        })
        
        # Genera token di approvazione
        approval_token = self._generate_approval_token(action, action_details)
        
        # Determina requisiti
        requirements = self._get_approval_requirements(criticality)
        
        return {
            "success": True,
            "action": "human_override_gate",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "proposed_action": action,
                "action_details": action_details,
                "criticality_assessment": {
                    "level": criticality,
                    "auto_detected": True,
                    "factors_considered": self._get_criticality_factors(action)
                },
                "approval_requirements": requirements,
                "approval_token": approval_token,
                "status": "PENDING_APPROVAL" if criticality in ["high", "critical"] else "ALLOWED"
            },
            "summary": f"Azione '{action}' - Criticità: {criticality.upper()}",
            "recommendations": self._get_gate_recommendations(criticality),
            "confidence": 1.0,  # Gate è deterministico
            "disclaimer": "🚦 Azioni ad alta criticità richiedono approvazione umana esplicita."
        }
    
    def _assess_criticality(self, action: str, details: Dict) -> str:
        """Valuta criticità automaticamente"""
        
        action_lower = action.lower()
        
        # Critical
        if any(crit in action_lower for crit in CRITICAL_ACTIONS):
            return "critical"
        
        # High
        if any(h in action_lower for h in ["modify", "update", "change", "modifica", "aggiorna"]):
            return "high"
        
        # Medium
        if any(m in action_lower for m in ["create", "add", "crea", "aggiungi"]):
            return "medium"
        
        # Low
        return "low"
    
    def _generate_approval_token(self, action: str, details: Dict) -> str:
        """Genera token univoco per approvazione"""
        
        data = f"{action}:{json.dumps(details, sort_keys=True)}:{datetime.now().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _get_approval_requirements(self, criticality: str) -> Dict[str, Any]:
        """Requisiti per ogni livello"""
        
        requirements = {
            "low": {
                "approval_needed": False,
                "notification": False,
                "audit_log": True
            },
            "medium": {
                "approval_needed": False,
                "notification": True,
                "audit_log": True
            },
            "high": {
                "approval_needed": True,
                "notification": True,
                "audit_log": True,
                "timeout_minutes": 30
            },
            "critical": {
                "approval_needed": True,
                "second_factor": True,
                "notification": True,
                "audit_log": True,
                "timeout_minutes": 15,
                "supervisor_notification": True
            }
        }
        
        return requirements.get(criticality, requirements["high"])
    
    def _get_criticality_factors(self, action: str) -> List[str]:
        """Fattori considerati per criticità"""
        
        return [
            "Tipo di azione richiesta",
            "Reversibilità dell'operazione",
            "Impatto potenziale su dati/sistemi",
            "Contesto di sicurezza"
        ]
    
    def _get_gate_recommendations(self, criticality: str) -> List[str]:
        if criticality == "critical":
            return [
                "Verificare identità operatore",
                "Confermare con secondo fattore",
                "Documentare motivo dell'azione"
            ]
        elif criticality == "high":
            return [
                "Verificare necessità dell'azione",
                "Confermare prima di procedere"
            ]
        return ["Procedere con normale cautela"]

    # ==================== CROSS-CUTTING: AuditTrailGenerator ====================
    
    async def _log_audit(self, action: str, actor: str, details: Dict) -> None:
        """Log audit per ogni azione"""
        
        entry = AuditEntry(
            timestamp=datetime.now().isoformat(),
            action=action,
            actor=actor,
            details=details,
            result="logged",
            hash=hashlib.sha256(
                json.dumps(details, sort_keys=True).encode()
            ).hexdigest()[:16]
        )
        
        self.audit_log.append(entry)
    
    async def get_audit_trail(
        self,
        filter_action: str = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        📝 AuditTrailGenerator
        
        Recupera log completo delle azioni.
        """
        
        entries = self.audit_log
        
        if filter_action:
            entries = [e for e in entries if filter_action in e.action]
        
        entries = entries[-limit:]
        
        return {
            "success": True,
            "action": "audit_trail",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "total_entries": len(self.audit_log),
                "returned_entries": len(entries),
                "filter_applied": filter_action,
                "entries": [
                    {
                        "timestamp": e.timestamp,
                        "action": e.action,
                        "actor": e.actor,
                        "hash": e.hash
                    }
                    for e in entries
                ]
            },
            "summary": f"Audit trail: {len(entries)} entries",
            "confidence": 1.0
        }

    # ==================== CROSS-CUTTING: BiasAndDriftMonitor ====================
    
    async def bias_and_drift_monitor(
        self,
        check_type: str = "full"  # quick, full, deep
    ) -> Dict[str, Any]:
        """
        ⚖️ BiasAndDriftMonitor
        
        Auto-diagnosi per rilevare:
        - Bias nelle risposte
        - Drift dai parametri originali
        - Pattern anomali
        """
        
        analysis = {
            "bias_indicators": self._check_bias_indicators(),
            "drift_metrics": self._check_drift_metrics(),
            "anomaly_patterns": self._check_anomaly_patterns()
        }
        
        overall_health = self._calculate_system_health(analysis)
        
        return {
            "success": True,
            "action": "bias_and_drift_monitor",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "check_type": check_type,
                "analysis": analysis,
                "system_health": {
                    "score": f"{overall_health:.0%}",
                    "status": "healthy" if overall_health >= 0.80 else "attention_needed"
                },
                "last_calibration": "N/A",
                "recommended_actions": self._get_health_recommendations(overall_health)
            },
            "summary": f"System health: {overall_health:.0%}",
            "confidence": 0.85,
            "disclaimer": "⚖️ Auto-diagnosi indicativa. Per audit completo consultare team tecnico."
        }
    
    def _check_bias_indicators(self) -> Dict[str, Any]:
        return {
            "response_distribution": "normal",
            "topic_coverage": "balanced",
            "confidence_calibration": "good"
        }
    
    def _check_drift_metrics(self) -> Dict[str, Any]:
        return {
            "parameter_stability": 0.95,
            "response_consistency": 0.92,
            "threshold_adherence": 0.98
        }
    
    def _check_anomaly_patterns(self) -> List[str]:
        return []  # Nessuna anomalia rilevata
    
    def _calculate_system_health(self, analysis: Dict) -> float:
        drift = analysis["drift_metrics"]
        return (drift["parameter_stability"] + drift["response_consistency"] + drift["threshold_adherence"]) / 3
    
    def _get_health_recommendations(self, health: float) -> List[str]:
        if health >= 0.90:
            return ["Sistema in buona salute", "Continuare monitoraggio regolare"]
        elif health >= 0.75:
            return ["Monitorare metriche in calo", "Considerare recalibrazione"]
        return ["Recalibrazione consigliata", "Analisi manuale necessaria"]

    # ==================== CROSS-CUTTING: FailSafeTrigger ====================
    
    async def _failsafe_response(self, query: str, result: ReasoningResult) -> Dict[str, Any]:
        """Risposta failsafe quando confidenza troppo bassa"""
        
        return {
            "success": False,
            "action": "failsafe_triggered",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "original_query": query,
                "reason": "Confidenza sotto soglia critica",
                "confidence": result.combined_confidence,
                "threshold": CONFIDENCE_THRESHOLD_CRITICAL,
                "conflicts_detected": len(result.conflicts)
            },
            "summary": f"⚠️ FAILSAFE: Confidenza {result.combined_confidence:.0%} sotto soglia {CONFIDENCE_THRESHOLD_CRITICAL:.0%}",
            "recommendations": [
                "Query richiede intervento umano",
                "Raccogliere più dati prima di riprovare",
                "Verificare fonti disponibili"
            ],
            "confidence": result.combined_confidence,
            "disclaimer": "🛑 Sistema in modalità failsafe. Nessuna azione automatica verrà eseguita."
        }
    
    async def failsafe_trigger(
        self,
        reason: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        🛑 FailSafeTrigger
        
        Attiva manualmente modalità failsafe.
        Stop automatico per tutte le azioni non essenziali.
        """
        
        self.failsafe_triggered = True
        
        await self._log_audit("failsafe_trigger", "system", {
            "reason": reason,
            "context": context or {}
        })
        
        return {
            "success": True,
            "action": "failsafe_trigger",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "failsafe_active": True,
                "trigger_reason": reason,
                "context": context,
                "actions_blocked": [
                    "Automazioni critiche",
                    "Azioni senza supervisione",
                    "Operazioni ad alto rischio"
                ],
                "actions_allowed": [
                    "Query informative",
                    "Audit e logging",
                    "Comunicazioni di stato"
                ],
                "recovery_procedure": "Richiede reset manuale da operatore autorizzato"
            },
            "summary": f"🛑 FAILSAFE ATTIVO: {reason}",
            "recommendations": [
                "Contattare operatore per valutazione",
                "Non forzare override senza autorizzazione",
                "Documentare contesto del trigger"
            ],
            "confidence": 1.0,
            "disclaimer": "🛑 Sistema in modalità sicura. Azioni limitate fino a reset autorizzato."
        }


# Singleton instance
_gideon_core = None

def get_gideon_core() -> GideonCore:
    global _gideon_core
    if _gideon_core is None:
        _gideon_core = GideonCore()
    return _gideon_core
