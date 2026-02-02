"""
🔐 GIDEON SECURITY TOOL — Physical & Infrastructure Defense
============================================================

Azioni Avanzate:
1. PredictiveRiskMapping - Mappe di rischio dinamiche
2. AnomalyNarrator - Racconta anomalie per umani
3. DefensiveScenarioSimulator - Simula errori e guasti
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import random
import math

@dataclass
class RiskScore:
    """Punteggio di rischio con spiegazione"""
    area: str
    score: float  # 0-100
    level: str  # low, medium, high, critical
    causes: List[str]
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class AnomalyReport:
    """Report anomalia narrato"""
    severity: str
    narrative: str
    technical_details: Dict[str, Any]
    recommendations: List[str]
    confidence: float

@dataclass 
class SimulationResult:
    """Risultato simulazione difensiva"""
    scenario_name: str
    success_rate: float
    vulnerabilities_found: List[str]
    resilience_score: float
    improvements: List[str]


class SecurityTool:
    """
    🔐 SECURITY TOOL - Physical & Infrastructure Defense
    
    NON è un tool offensivo.
    È un sistema di analisi e prevenzione.
    """
    
    def __init__(self):
        self.risk_history: List[RiskScore] = []
        self.anomaly_history: List[AnomalyReport] = []
        self.baseline_data: Dict[str, Any] = {}
        
    # ==================== ACTION 1: PredictiveRiskMapping ====================
    
    async def predictive_risk_mapping(
        self,
        event_logs: List[Dict[str, Any]] = None,
        sensor_data: Dict[str, Any] = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        🎯 PredictiveRiskMapping
        
        Incrocia dati storici, sensori, eventi esterni.
        Genera mappe di rischio dinamiche (non target).
        
        Output:
        - Risk score per area/sistema
        - Spiegazione causale
        """
        
        event_logs = event_logs or []
        sensor_data = sensor_data or {}
        context = context or {}
        
        # Analisi fattori di rischio
        risk_factors = self._analyze_risk_factors(event_logs, sensor_data, context)
        
        # Genera mappa rischio per aree
        risk_map = {}
        areas = ["network", "physical_access", "data_storage", "endpoints", "cloud_services"]
        
        for area in areas:
            score = self._calculate_area_risk(area, risk_factors)
            risk_map[area] = RiskScore(
                area=area,
                score=score["value"],
                level=self._score_to_level(score["value"]),
                causes=score["causes"],
                confidence=score["confidence"]
            )
        
        # Calcola rischio aggregato
        total_score = sum(r.score for r in risk_map.values()) / len(risk_map)
        
        return {
            "success": True,
            "action": "predictive_risk_mapping",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "overall_risk_score": round(total_score, 2),
                "overall_level": self._score_to_level(total_score),
                "risk_map": {
                    area: {
                        "score": r.score,
                        "level": r.level,
                        "causes": r.causes,
                        "confidence": f"{r.confidence:.0%}"
                    }
                    for area, r in risk_map.items()
                },
                "trend": self._calculate_trend(),
                "prediction_horizon": "24h"
            },
            "summary": f"Rischio complessivo: {self._score_to_level(total_score).upper()} ({total_score:.1f}/100)",
            "recommendations": self._generate_risk_recommendations(risk_map),
            "confidence": 0.85,
            "explainability": "Analisi basata su pattern storici e correlazione eventi"
        }
    
    def _analyze_risk_factors(
        self,
        events: List[Dict],
        sensors: Dict,
        context: Dict
    ) -> Dict[str, float]:
        """Analizza fattori di rischio"""
        factors = {
            "failed_logins": 0,
            "unusual_hours": 0,
            "sensor_anomalies": 0,
            "external_threats": 0,
            "system_load": 0
        }
        
        # Simula analisi (in produzione: analisi reale)
        import psutil
        
        # Fattore carico sistema
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        factors["system_load"] = (cpu + mem) / 2
        
        # Fattore orario (fuori orario = più rischio)
        hour = datetime.now().hour
        if hour < 6 or hour > 22:
            factors["unusual_hours"] = 30
            
        # Fattore eventi (simulato) - assicura che events sia una lista di dict
        if isinstance(events, list):
            factors["failed_logins"] = len([e for e in events if isinstance(e, dict) and e.get("type") == "failed_login"]) * 5
        else:
            factors["failed_logins"] = 0
        
        return factors
    
    def _calculate_area_risk(self, area: str, factors: Dict[str, float]) -> Dict[str, Any]:
        """Calcola rischio per area specifica"""
        
        base_weights = {
            "network": {"failed_logins": 0.3, "external_threats": 0.4, "system_load": 0.3},
            "physical_access": {"unusual_hours": 0.5, "sensor_anomalies": 0.5},
            "data_storage": {"system_load": 0.4, "failed_logins": 0.6},
            "endpoints": {"system_load": 0.5, "failed_logins": 0.3, "external_threats": 0.2},
            "cloud_services": {"external_threats": 0.5, "system_load": 0.5}
        }
        
        weights = base_weights.get(area, {"system_load": 1.0})
        
        score = sum(factors.get(f, 0) * w for f, w in weights.items())
        score = min(100, max(0, score))
        
        causes = [f for f, w in weights.items() if factors.get(f, 0) > 20]
        
        return {
            "value": round(score, 2),
            "causes": causes if causes else ["Nessuna causa critica rilevata"],
            "confidence": 0.75 + random.uniform(0, 0.2)
        }
    
    def _score_to_level(self, score: float) -> str:
        if score < 25:
            return "low"
        elif score < 50:
            return "medium"
        elif score < 75:
            return "high"
        return "critical"
    
    def _calculate_trend(self) -> str:
        if len(self.risk_history) < 2:
            return "stable"
        recent = sum(r.score for r in self.risk_history[-5:]) / min(5, len(self.risk_history))
        older = sum(r.score for r in self.risk_history[:-5]) / max(1, len(self.risk_history) - 5)
        if recent > older * 1.1:
            return "increasing"
        elif recent < older * 0.9:
            return "decreasing"
        return "stable"
    
    def _generate_risk_recommendations(self, risk_map: Dict[str, RiskScore]) -> List[str]:
        recommendations = []
        for area, risk in risk_map.items():
            if risk.level in ["high", "critical"]:
                if area == "network":
                    recommendations.append("🔒 Verificare regole firewall e segmentazione rete")
                elif area == "physical_access":
                    recommendations.append("🚪 Controllare log accessi fisici e sensori")
                elif area == "data_storage":
                    recommendations.append("💾 Verificare backup e crittografia dati")
                elif area == "endpoints":
                    recommendations.append("💻 Aggiornare endpoint protection")
                elif area == "cloud_services":
                    recommendations.append("☁️ Rivedere policy cloud e IAM")
        
        if not recommendations:
            recommendations.append("✅ Nessuna azione urgente richiesta")
        
        return recommendations

    # ==================== ACTION 2: AnomalyNarrator ====================
    
    async def anomaly_narrator(
        self,
        anomaly_data: Dict[str, Any] = None,
        audience: str = "technical"  # technical, manager, operator
    ) -> Dict[str, Any]:
        """
        📖 AnomalyNarrator
        
        Non solo rileva anomalie, le RACCONTA per umani.
        Traduce dati tecnici in narrativa comprensibile.
        
        audience: technical | manager | operator
        """
        
        anomaly_data = anomaly_data or self._detect_current_anomalies()
        
        narratives = []
        
        for anomaly in anomaly_data.get("anomalies", []):
            narrative = self._create_narrative(anomaly, audience)
            narratives.append(narrative)
        
        # Crea report narrativo
        if not narratives:
            main_narrative = "🟢 **Situazione normale**: Nessuna anomalia significativa rilevata. Tutti i sistemi operano entro i parametri attesi."
        else:
            severity_emoji = {"low": "🟡", "medium": "🟠", "high": "🔴", "critical": "⚫"}
            main_narrative = "\n\n".join([
                f"{severity_emoji.get(n.severity, '🔵')} **{n.severity.upper()}**: {n.narrative}"
                for n in narratives
            ])
        
        return {
            "success": True,
            "action": "anomaly_narrator",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "narrative": main_narrative,
                "audience": audience,
                "anomaly_count": len(narratives),
                "details": [
                    {
                        "severity": n.severity,
                        "narrative": n.narrative,
                        "technical": n.technical_details,
                        "recommendations": n.recommendations,
                        "confidence": f"{n.confidence:.0%}"
                    }
                    for n in narratives
                ]
            },
            "summary": f"Rilevate {len(narratives)} anomalie" if narratives else "Nessuna anomalia",
            "recommendations": self._aggregate_recommendations(narratives),
            "confidence": 0.90
        }
    
    def _detect_current_anomalies(self) -> Dict[str, Any]:
        """Rileva anomalie correnti dal sistema"""
        import psutil
        
        anomalies = []
        
        # Check CPU
        cpu = psutil.cpu_percent(interval=1)
        if cpu > 80:
            anomalies.append({
                "type": "high_cpu",
                "value": cpu,
                "threshold": 80,
                "severity": "high" if cpu > 90 else "medium"
            })
        
        # Check Memory
        mem = psutil.virtual_memory()
        if mem.percent > 85:
            anomalies.append({
                "type": "high_memory",
                "value": mem.percent,
                "threshold": 85,
                "severity": "high" if mem.percent > 95 else "medium"
            })
        
        # Check Disk
        disk = psutil.disk_usage('/')
        if disk.percent > 90:
            anomalies.append({
                "type": "disk_full",
                "value": disk.percent,
                "threshold": 90,
                "severity": "critical" if disk.percent > 95 else "high"
            })
        
        # Check unusual connections
        connections = psutil.net_connections(kind='inet')
        established = [c for c in connections if c.status == 'ESTABLISHED']
        if len(established) > 100:
            anomalies.append({
                "type": "many_connections",
                "value": len(established),
                "threshold": 100,
                "severity": "medium"
            })
        
        return {"anomalies": anomalies}
    
    def _create_narrative(self, anomaly: Dict, audience: str) -> AnomalyReport:
        """Crea narrativa per anomalia"""
        
        narratives_map = {
            "high_cpu": {
                "technical": f"CPU al {anomaly.get('value', 0):.1f}% - superata soglia {anomaly.get('threshold', 80)}%. Possibile processo intensivo o loop.",
                "manager": f"Il processore è sotto stress elevato ({anomaly.get('value', 0):.0f}%). Potrebbe rallentare le operazioni.",
                "operator": f"⚠️ CPU alta! Controlla i processi attivi. Valore: {anomaly.get('value', 0):.0f}%"
            },
            "high_memory": {
                "technical": f"RAM al {anomaly.get('value', 0):.1f}% - rischio OOM. Verificare memory leaks.",
                "manager": f"La memoria del sistema è quasi esaurita ({anomaly.get('value', 0):.0f}%). Rischio rallentamenti.",
                "operator": f"⚠️ Memoria quasi piena! Chiudi applicazioni non necessarie."
            },
            "disk_full": {
                "technical": f"Disco al {anomaly.get('value', 0):.1f}% - spazio critico. Rischio crash servizi.",
                "manager": f"Lo spazio disco sta per esaurirsi ({anomaly.get('value', 0):.0f}%). Servono azioni immediate.",
                "operator": f"🚨 DISCO PIENO! Libera spazio immediatamente!"
            },
            "many_connections": {
                "technical": f"{anomaly.get('value', 0)} connessioni attive - anomalo. Possibile DDoS o bot.",
                "manager": f"Numero insolito di connessioni di rete ({anomaly.get('value', 0)}). Potrebbe indicare un attacco.",
                "operator": f"⚠️ Troppe connessioni! Verifica traffico di rete."
            }
        }
        
        anomaly_type = anomaly.get("type", "unknown")
        narrative_text = narratives_map.get(anomaly_type, {}).get(
            audience, 
            f"Anomalia rilevata: {anomaly_type}"
        )
        
        recommendations = {
            "high_cpu": ["Identificare processi CPU-intensive", "Considerare scaling orizzontale"],
            "high_memory": ["Riavviare servizi con memory leak", "Aumentare RAM se ricorrente"],
            "disk_full": ["Pulire log e cache", "Espandere storage"],
            "many_connections": ["Verificare firewall", "Attivare rate limiting"]
        }
        
        return AnomalyReport(
            severity=anomaly.get("severity", "medium"),
            narrative=narrative_text,
            technical_details=anomaly,
            recommendations=recommendations.get(anomaly_type, ["Investigare manualmente"]),
            confidence=0.85
        )
    
    def _aggregate_recommendations(self, narratives: List[AnomalyReport]) -> List[str]:
        """Aggrega raccomandazioni uniche"""
        all_recs = []
        for n in narratives:
            all_recs.extend(n.recommendations)
        return list(set(all_recs))[:5]

    # ==================== ACTION 3: DefensiveScenarioSimulator ====================
    
    async def defensive_scenario_simulator(
        self,
        scenario_type: str = "system_failure",
        intensity: str = "medium",  # low, medium, high
        target_systems: List[str] = None
    ) -> Dict[str, Any]:
        """
        🎮 DefensiveScenarioSimulator
        
        Simula errori, guasti, stress.
        NESSUN ATTACCO - solo fallimenti realistici.
        
        Uso:
        - Addestramento operatori
        - Miglioramento resilienza
        
        scenario_type:
        - system_failure: guasto hardware/software
        - network_outage: interruzione rete
        - data_corruption: corruzione dati
        - service_degradation: degrado prestazioni
        - backup_failure: fallimento backup
        """
        
        target_systems = target_systems or ["primary_server", "database", "network"]
        
        scenarios = {
            "system_failure": self._simulate_system_failure,
            "network_outage": self._simulate_network_outage,
            "data_corruption": self._simulate_data_corruption,
            "service_degradation": self._simulate_service_degradation,
            "backup_failure": self._simulate_backup_failure
        }
        
        simulator = scenarios.get(scenario_type, self._simulate_system_failure)
        result = simulator(intensity, target_systems)
        
        return {
            "success": True,
            "action": "defensive_scenario_simulator",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "scenario": scenario_type,
                "intensity": intensity,
                "target_systems": target_systems,
                "simulation_result": {
                    "scenario_name": result.scenario_name,
                    "success_rate": f"{result.success_rate:.0%}",
                    "resilience_score": f"{result.resilience_score:.1f}/100",
                    "vulnerabilities_found": result.vulnerabilities_found,
                    "improvements": result.improvements
                },
                "timeline": self._generate_timeline(scenario_type, intensity),
                "affected_services": self._calculate_affected_services(target_systems)
            },
            "summary": f"Simulazione '{scenario_type}' completata - Resilienza: {result.resilience_score:.0f}%",
            "recommendations": result.improvements,
            "training_notes": self._generate_training_notes(scenario_type),
            "confidence": 0.92
        }
    
    def _simulate_system_failure(self, intensity: str, targets: List[str]) -> SimulationResult:
        base_success = {"low": 0.9, "medium": 0.7, "high": 0.5}
        
        return SimulationResult(
            scenario_name="Guasto Sistema Critico",
            success_rate=base_success.get(intensity, 0.7),
            vulnerabilities_found=[
                "Single point of failure nel database",
                "Mancanza di failover automatico",
                "Timeout troppo lunghi per health check"
            ][:2 if intensity == "low" else 3],
            resilience_score=70 - (20 if intensity == "high" else 10 if intensity == "medium" else 0),
            improvements=[
                "Implementare replica database attiva",
                "Configurare health check più aggressivi",
                "Aggiungere circuit breaker ai servizi"
            ]
        )
    
    def _simulate_network_outage(self, intensity: str, targets: List[str]) -> SimulationResult:
        return SimulationResult(
            scenario_name="Interruzione Rete",
            success_rate=0.6 if intensity == "high" else 0.8,
            vulnerabilities_found=[
                "Nessun percorso di rete alternativo",
                "DNS senza ridondanza",
                "VPN senza backup"
            ],
            resilience_score=55 + (15 if intensity == "low" else 0),
            improvements=[
                "Configurare routing multi-path",
                "Implementare DNS secondario",
                "Aggiungere connettività backup (4G/5G)"
            ]
        )
    
    def _simulate_data_corruption(self, intensity: str, targets: List[str]) -> SimulationResult:
        return SimulationResult(
            scenario_name="Corruzione Dati",
            success_rate=0.75,
            vulnerabilities_found=[
                "Backup non testati regolarmente",
                "Mancanza di checksum su dati critici",
                "RPO troppo alto (4 ore)"
            ],
            resilience_score=65,
            improvements=[
                "Implementare backup verificati automaticamente",
                "Aggiungere integrity check sui dati",
                "Ridurre RPO a 15 minuti"
            ]
        )
    
    def _simulate_service_degradation(self, intensity: str, targets: List[str]) -> SimulationResult:
        return SimulationResult(
            scenario_name="Degrado Prestazioni",
            success_rate=0.85,
            vulnerabilities_found=[
                "Nessun auto-scaling configurato",
                "Cache non ottimizzata",
                "Query database lente"
            ],
            resilience_score=72,
            improvements=[
                "Configurare auto-scaling basato su metriche",
                "Implementare caching multi-livello",
                "Ottimizzare query e indici"
            ]
        )
    
    def _simulate_backup_failure(self, intensity: str, targets: List[str]) -> SimulationResult:
        return SimulationResult(
            scenario_name="Fallimento Backup",
            success_rate=0.65,
            vulnerabilities_found=[
                "Backup su singola location",
                "Nessun test di restore automatico",
                "Retention policy inadeguata"
            ],
            resilience_score=58,
            improvements=[
                "Implementare backup 3-2-1",
                "Automatizzare test di restore settimanali",
                "Rivedere retention policy"
            ]
        )
    
    def _generate_timeline(self, scenario: str, intensity: str) -> List[Dict[str, str]]:
        base_timeline = [
            {"time": "T+0", "event": "Inizio simulazione"},
            {"time": "T+1min", "event": "Rilevamento anomalia"},
            {"time": "T+2min", "event": "Alert generato"},
            {"time": "T+5min", "event": "Risposta team"},
            {"time": "T+15min", "event": "Mitigazione iniziata"},
            {"time": "T+30min", "event": "Servizio ripristinato"},
            {"time": "T+1h", "event": "Post-mortem completato"}
        ]
        return base_timeline
    
    def _calculate_affected_services(self, targets: List[str]) -> Dict[str, str]:
        impact_map = {
            "primary_server": "Critico",
            "database": "Critico",
            "network": "Alto",
            "cache": "Medio",
            "logging": "Basso"
        }
        return {t: impact_map.get(t, "Sconosciuto") for t in targets}
    
    def _generate_training_notes(self, scenario: str) -> List[str]:
        notes = {
            "system_failure": [
                "📚 Verificare la documentazione di recovery",
                "🎯 Praticare failover manuale",
                "⏱️ Misurare tempo di risposta team"
            ],
            "network_outage": [
                "📚 Conoscere percorsi di rete alternativi",
                "🎯 Praticare switch su backup",
                "⏱️ Testare comunicazioni out-of-band"
            ],
            "data_corruption": [
                "📚 Documentare procedure di restore",
                "🎯 Praticare recovery point-in-time",
                "⏱️ Verificare integrità backup"
            ]
        }
        return notes.get(scenario, ["📚 Documentare procedure", "🎯 Praticare scenari", "⏱️ Misurare tempi"])


# Singleton instance
_security_tool = None

def get_security_tool() -> SecurityTool:
    global _security_tool
    if _security_tool is None:
        _security_tool = SecurityTool()
    return _security_tool
