"""
🔮 COUNTERFACTUAL SIMULATOR
============================
Prima di dare una risposta ad alto rischio, GIDEON simula:
"Cosa succederebbe se sbagliassi?"

Se il danno simulato supera una soglia, blocca o avvisa.
Usato per: sicurezza, automazioni, suggerimenti medici
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging
import asyncio
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class RiskCategory(Enum):
    """Categorie di rischio"""
    SAFETY = "safety"              # Sicurezza fisica
    SECURITY = "security"          # Sicurezza informatica
    FINANCIAL = "financial"        # Impatto economico
    LEGAL = "legal"               # Implicazioni legali
    REPUTATIONAL = "reputational"  # Danno reputazionale
    OPERATIONAL = "operational"    # Impatto operativo
    HEALTH = "health"             # Rischi per la salute
    DATA = "data"                 # Perdita/compromissione dati
    ETHICAL = "ethical"           # Implicazioni etiche


class SeverityLevel(Enum):
    """Livelli di severità"""
    NEGLIGIBLE = 1   # Trascurabile
    LOW = 2          # Basso
    MEDIUM = 3       # Medio
    HIGH = 4         # Alto
    CRITICAL = 5     # Critico
    CATASTROPHIC = 6 # Catastrofico


class SimulationAction(Enum):
    """Azioni raccomandate dalla simulazione"""
    PROCEED = "proceed"           # Procedi normalmente
    PROCEED_WITH_WARNING = "proceed_warning"  # Procedi con avviso
    REQUIRE_CONFIRMATION = "require_confirm"  # Richiedi conferma
    BLOCK = "block"               # Blocca l'azione
    ESCALATE = "escalate"         # Escala a supervisore


@dataclass
class SimulatedOutcome:
    """Esito simulato di uno scenario"""
    scenario_id: str
    description: str
    probability: float           # 0-1
    severity: SeverityLevel
    category: RiskCategory
    affected_entities: List[str]
    reversible: bool
    time_to_impact: str          # "immediate", "hours", "days", "weeks"
    mitigation_possible: bool
    mitigation_effort: str       # "trivial", "moderate", "significant", "impossible"
    detailed_impact: Dict[str, Any]


@dataclass
class CounterfactualResult:
    """Risultato completo della simulazione counterfactual"""
    original_action: str
    simulated_outcomes: List[SimulatedOutcome]
    worst_case: SimulatedOutcome
    expected_damage_score: float  # 0-100
    recommended_action: SimulationAction
    risk_summary: Dict[RiskCategory, float]
    warnings: List[str]
    mitigations: List[str]
    simulation_confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


class ScenarioGenerator(ABC):
    """Generatore astratto di scenari"""
    
    @abstractmethod
    async def generate_scenarios(
        self,
        action: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        pass


class DefaultScenarioGenerator(ScenarioGenerator):
    """Generatore di scenari di default"""
    
    # Template scenari per tipo azione
    SCENARIO_TEMPLATES = {
        'delete': [
            {"type": "wrong_target", "desc": "Eliminazione del target sbagliato", "prob": 0.05, "sev": 4},
            {"type": "cascade_delete", "desc": "Eliminazione a cascata non prevista", "prob": 0.1, "sev": 5},
            {"type": "no_backup", "desc": "Nessun backup disponibile per recovery", "prob": 0.15, "sev": 5},
        ],
        'execute': [
            {"type": "wrong_env", "desc": "Esecuzione nell'ambiente sbagliato", "prob": 0.1, "sev": 5},
            {"type": "side_effects", "desc": "Effetti collaterali non previsti", "prob": 0.2, "sev": 3},
            {"type": "resource_exhaust", "desc": "Esaurimento risorse di sistema", "prob": 0.05, "sev": 4},
        ],
        'send': [
            {"type": "wrong_recipient", "desc": "Invio al destinatario sbagliato", "prob": 0.08, "sev": 4},
            {"type": "data_leak", "desc": "Fuga di dati sensibili", "prob": 0.03, "sev": 6},
            {"type": "spam_flag", "desc": "Messaggio marcato come spam", "prob": 0.1, "sev": 2},
        ],
        'modify': [
            {"type": "data_corruption", "desc": "Corruzione dati esistenti", "prob": 0.1, "sev": 4},
            {"type": "break_dependency", "desc": "Rottura dipendenze", "prob": 0.15, "sev": 3},
            {"type": "rollback_impossible", "desc": "Rollback impossibile", "prob": 0.05, "sev": 5},
        ],
        'security': [
            {"type": "false_positive", "desc": "Falso positivo blocca utenti legittimi", "prob": 0.1, "sev": 3},
            {"type": "false_negative", "desc": "Minaccia reale non rilevata", "prob": 0.05, "sev": 6},
            {"type": "over_response", "desc": "Risposta eccessiva causa downtime", "prob": 0.08, "sev": 4},
        ],
        'health': [
            {"type": "misdiagnosis", "desc": "Diagnosi errata", "prob": 0.1, "sev": 6},
            {"type": "wrong_recommendation", "desc": "Raccomandazione inappropriata", "prob": 0.15, "sev": 5},
            {"type": "delay_treatment", "desc": "Ritardo in trattamento necessario", "prob": 0.05, "sev": 6},
        ]
    }
    
    async def generate_scenarios(
        self,
        action: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Genera scenari basati sul tipo di azione"""
        
        action_lower = action.lower()
        scenarios = []
        
        # Trova template rilevanti
        for action_type, templates in self.SCENARIO_TEMPLATES.items():
            if action_type in action_lower or action_type in str(context).lower():
                for t in templates:
                    scenario = t.copy()
                    scenario['source_action'] = action_type
                    scenarios.append(scenario)
        
        # Aggiungi scenario generico
        scenarios.append({
            "type": "unknown_failure",
            "desc": "Fallimento imprevisto",
            "prob": 0.05,
            "sev": 3,
            "source_action": "generic"
        })
        
        return scenarios


class CounterfactualSimulator:
    """
    Simulatore Counterfactual per GIDEON.
    
    Prima di dare una risposta ad alto rischio, simula:
    "Cosa succederebbe se sbagliassi?"
    
    Se il danno simulato supera una soglia, blocca o avvisa.
    """
    
    # Soglie di danno
    DAMAGE_THRESHOLD_WARNING = 30   # Soglia per warning
    DAMAGE_THRESHOLD_CONFIRM = 50   # Soglia per conferma
    DAMAGE_THRESHOLD_BLOCK = 75     # Soglia per blocco
    DAMAGE_THRESHOLD_ESCALATE = 90  # Soglia per escalation
    
    # Pesi per categoria
    CATEGORY_WEIGHTS = {
        RiskCategory.SAFETY: 2.0,
        RiskCategory.HEALTH: 2.0,
        RiskCategory.SECURITY: 1.5,
        RiskCategory.LEGAL: 1.3,
        RiskCategory.DATA: 1.2,
        RiskCategory.FINANCIAL: 1.1,
        RiskCategory.REPUTATIONAL: 1.0,
        RiskCategory.OPERATIONAL: 0.9,
        RiskCategory.ETHICAL: 1.4
    }
    
    def __init__(self, scenario_generator: Optional[ScenarioGenerator] = None):
        self.scenario_generator = scenario_generator or DefaultScenarioGenerator()
        self.simulation_history: List[CounterfactualResult] = []
        self.custom_thresholds: Dict[str, float] = {}
        self.blocked_patterns: List[str] = []
    
    async def simulate(
        self,
        action: str,
        context: Dict[str, Any],
        confidence: float = 0.8
    ) -> CounterfactualResult:
        """
        Esegue simulazione counterfactual completa.
        
        Args:
            action: Azione proposta
            context: Contesto dell'azione
            confidence: Confidenza nell'azione proposta
            
        Returns:
            CounterfactualResult con analisi completa
        """
        
        logger.info(f"🔮 Simulazione counterfactual: {action[:50]}...")
        
        # Genera scenari
        raw_scenarios = await self.scenario_generator.generate_scenarios(action, context)
        
        # Simula ogni scenario
        outcomes = []
        for scenario in raw_scenarios:
            outcome = await self._simulate_outcome(scenario, action, context, confidence)
            outcomes.append(outcome)
        
        # Analizza risultati
        worst_case = max(outcomes, key=lambda o: o.severity.value * o.probability)
        damage_score = self._calculate_damage_score(outcomes)
        risk_summary = self._summarize_risks(outcomes)
        recommended_action = self._determine_action(damage_score, worst_case, confidence)
        
        # Genera warnings e mitigations
        warnings = self._generate_warnings(outcomes, damage_score)
        mitigations = self._generate_mitigations(outcomes)
        
        result = CounterfactualResult(
            original_action=action,
            simulated_outcomes=outcomes,
            worst_case=worst_case,
            expected_damage_score=damage_score,
            recommended_action=recommended_action,
            risk_summary=risk_summary,
            warnings=warnings,
            mitigations=mitigations,
            simulation_confidence=self._calculate_simulation_confidence(outcomes, confidence)
        )
        
        self.simulation_history.append(result)
        
        logger.info(f"🔮 Simulazione completata: score={damage_score:.1f}, action={recommended_action.value}")
        
        return result
    
    async def _simulate_outcome(
        self,
        scenario: Dict[str, Any],
        action: str,
        context: Dict[str, Any],
        confidence: float
    ) -> SimulatedOutcome:
        """Simula un singolo scenario"""
        
        # Determina categoria
        category = self._infer_category(scenario, context)
        
        # Calcola probabilità aggiustata
        base_prob = scenario.get('prob', 0.1)
        # Bassa confidenza aumenta probabilità errore
        adjusted_prob = min(1.0, base_prob * (1 + (1 - confidence)))
        
        # Determina severità
        severity = SeverityLevel(min(6, max(1, scenario.get('sev', 3))))
        
        # Determina reversibilità
        reversible = severity.value < 5 and scenario.get('type') not in ['data_leak', 'cascade_delete']
        
        # Determina tempo impatto
        if severity.value >= 5:
            time_to_impact = "immediate"
        elif severity.value >= 3:
            time_to_impact = "hours"
        else:
            time_to_impact = "days"
        
        # Determina mitigabilità
        mitigation_possible = severity.value < 6
        if severity.value <= 2:
            mitigation_effort = "trivial"
        elif severity.value <= 4:
            mitigation_effort = "moderate"
        elif severity.value <= 5:
            mitigation_effort = "significant"
        else:
            mitigation_effort = "impossible"
        
        return SimulatedOutcome(
            scenario_id=f"{scenario['type']}_{scenario.get('source_action', 'unknown')}",
            description=scenario.get('desc', 'Scenario non definito'),
            probability=adjusted_prob,
            severity=severity,
            category=category,
            affected_entities=self._identify_affected_entities(scenario, context),
            reversible=reversible,
            time_to_impact=time_to_impact,
            mitigation_possible=mitigation_possible,
            mitigation_effort=mitigation_effort,
            detailed_impact=self._calculate_detailed_impact(scenario, context)
        )
    
    def _infer_category(self, scenario: Dict[str, Any], context: Dict[str, Any]) -> RiskCategory:
        """Inferisce la categoria di rischio"""
        
        scenario_type = scenario.get('type', '').lower()
        context_str = str(context).lower()
        
        if any(w in scenario_type for w in ['health', 'medical', 'diagnosis']):
            return RiskCategory.HEALTH
        elif any(w in scenario_type for w in ['security', 'attack', 'breach', 'leak']):
            return RiskCategory.SECURITY
        elif any(w in scenario_type for w in ['safety', 'danger', 'injury']):
            return RiskCategory.SAFETY
        elif any(w in scenario_type for w in ['money', 'cost', 'financial']):
            return RiskCategory.FINANCIAL
        elif any(w in scenario_type for w in ['legal', 'compliance', 'regulation']):
            return RiskCategory.LEGAL
        elif any(w in scenario_type for w in ['data', 'corruption', 'delete']):
            return RiskCategory.DATA
        elif any(w in scenario_type for w in ['ethical', 'moral']):
            return RiskCategory.ETHICAL
        elif any(w in scenario_type for w in ['reputation', 'trust']):
            return RiskCategory.REPUTATIONAL
        else:
            return RiskCategory.OPERATIONAL
    
    def _identify_affected_entities(
        self,
        scenario: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[str]:
        """Identifica entità potenzialmente impattate"""
        
        entities = []
        
        # Check contesto per entità
        if 'users' in context:
            entities.append(f"Users: {len(context['users'])} affected")
        if 'systems' in context:
            entities.extend(context['systems'])
        if 'database' in str(context):
            entities.append("Database")
        if 'api' in str(context).lower():
            entities.append("API Services")
        
        # Default
        if not entities:
            entities = ["Sistema principale", "Utenti correlati"]
        
        return entities[:5]  # Max 5
    
    def _calculate_detailed_impact(
        self,
        scenario: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calcola impatto dettagliato"""
        
        severity = scenario.get('sev', 3)
        
        return {
            'downtime_estimate': f"{severity * 2} ore" if severity > 3 else "Minimo",
            'recovery_time': f"{severity * 4} ore" if severity > 2 else "< 1 ora",
            'cost_estimate': f"€{severity * 1000}" if severity > 3 else "Trascurabile",
            'user_impact': f"{severity * 10}% utenti" if severity > 2 else "Nessuno",
            'data_at_risk': "Sì" if severity > 4 else "No"
        }
    
    def _calculate_damage_score(self, outcomes: List[SimulatedOutcome]) -> float:
        """Calcola score di danno complessivo (0-100)"""
        
        total_score = 0.0
        
        for outcome in outcomes:
            # Score base: probabilità * severità
            base = outcome.probability * outcome.severity.value * 10
            
            # Peso categoria
            weight = self.CATEGORY_WEIGHTS.get(outcome.category, 1.0)
            
            # Bonus irreversibilità
            irreversible_bonus = 1.5 if not outcome.reversible else 1.0
            
            # Score finale per outcome
            score = base * weight * irreversible_bonus
            total_score += score
        
        # Normalizza a 0-100
        return min(100, total_score)
    
    def _summarize_risks(self, outcomes: List[SimulatedOutcome]) -> Dict[RiskCategory, float]:
        """Riassume rischi per categoria"""
        
        summary = {cat: 0.0 for cat in RiskCategory}
        
        for outcome in outcomes:
            score = outcome.probability * outcome.severity.value * 10
            summary[outcome.category] += score
        
        # Normalizza
        max_score = max(summary.values()) if summary.values() else 1
        if max_score > 0:
            summary = {k: v / max_score * 100 for k, v in summary.items()}
        
        return summary
    
    def _determine_action(
        self,
        damage_score: float,
        worst_case: SimulatedOutcome,
        confidence: float
    ) -> SimulationAction:
        """Determina azione raccomandata"""
        
        # Check worst case catastrophico
        if worst_case.severity == SeverityLevel.CATASTROPHIC and worst_case.probability > 0.01:
            return SimulationAction.ESCALATE
        
        # Check soglie
        if damage_score >= self.DAMAGE_THRESHOLD_ESCALATE:
            return SimulationAction.ESCALATE
        elif damage_score >= self.DAMAGE_THRESHOLD_BLOCK:
            return SimulationAction.BLOCK
        elif damage_score >= self.DAMAGE_THRESHOLD_CONFIRM:
            return SimulationAction.REQUIRE_CONFIRMATION
        elif damage_score >= self.DAMAGE_THRESHOLD_WARNING:
            return SimulationAction.PROCEED_WITH_WARNING
        
        # Anche con score basso, se confidenza bassa richiedi conferma
        if confidence < 0.5:
            return SimulationAction.REQUIRE_CONFIRMATION
        
        return SimulationAction.PROCEED
    
    def _generate_warnings(
        self,
        outcomes: List[SimulatedOutcome],
        damage_score: float
    ) -> List[str]:
        """Genera warning pertinenti"""
        
        warnings = []
        
        # Warning generali
        if damage_score > self.DAMAGE_THRESHOLD_WARNING:
            warnings.append(f"⚠️ Score di rischio elevato: {damage_score:.1f}/100")
        
        # Warning per outcome specifici
        for outcome in outcomes:
            if outcome.severity.value >= 5 and outcome.probability > 0.05:
                warnings.append(f"🔴 Rischio {outcome.category.value}: {outcome.description}")
            elif outcome.severity.value >= 4 and outcome.probability > 0.1:
                warnings.append(f"🟠 Attenzione {outcome.category.value}: {outcome.description}")
        
        # Warning reversibilità
        irreversible = [o for o in outcomes if not o.reversible and o.probability > 0.05]
        if irreversible:
            warnings.append("⚠️ Alcune conseguenze potrebbero essere IRREVERSIBILI")
        
        return warnings[:5]  # Max 5
    
    def _generate_mitigations(self, outcomes: List[SimulatedOutcome]) -> List[str]:
        """Genera suggerimenti di mitigazione"""
        
        mitigations = []
        
        # Mitigazioni generali per categoria
        categories_present = set(o.category for o in outcomes if o.severity.value >= 3)
        
        category_mitigations = {
            RiskCategory.DATA: "Verificare esistenza backup prima di procedere",
            RiskCategory.SECURITY: "Validare input e limitare permessi",
            RiskCategory.SAFETY: "Implementare doppia conferma per azioni critiche",
            RiskCategory.HEALTH: "Consultare professionista qualificato",
            RiskCategory.FINANCIAL: "Impostare limiti di transazione",
            RiskCategory.LEGAL: "Verificare compliance normativa",
        }
        
        for cat in categories_present:
            if cat in category_mitigations:
                mitigations.append(category_mitigations[cat])
        
        # Mitigazioni specifiche per outcome gravi
        for outcome in outcomes:
            if outcome.severity.value >= 5 and outcome.mitigation_possible:
                mitigations.append(f"Per '{outcome.scenario_id}': valutare alternative meno rischiose")
        
        # Mitigazione generale
        mitigations.append("Documentare la decisione e le motivazioni")
        mitigations.append("Predisporre piano di rollback")
        
        return list(set(mitigations))[:6]  # Unique, max 6
    
    def _calculate_simulation_confidence(
        self,
        outcomes: List[SimulatedOutcome],
        original_confidence: float
    ) -> float:
        """Calcola confidenza nella simulazione stessa"""
        
        # Base dalla confidenza originale
        confidence = original_confidence * 0.5
        
        # Bonus per numero scenari considerati
        scenario_bonus = min(0.3, len(outcomes) * 0.05)
        confidence += scenario_bonus
        
        # Penalità per scenari ad alta incertezza
        uncertain = sum(1 for o in outcomes if o.severity.value >= 4 and o.probability < 0.05)
        uncertainty_penalty = uncertain * 0.05
        confidence -= uncertainty_penalty
        
        return max(0.1, min(1.0, confidence + 0.2))  # Floor 10%, cap 100%
    
    def format_result(self, result: CounterfactualResult) -> str:
        """Formatta risultato per visualizzazione"""
        
        # Emoji per azione
        action_emoji = {
            SimulationAction.PROCEED: "✅",
            SimulationAction.PROCEED_WITH_WARNING: "⚠️",
            SimulationAction.REQUIRE_CONFIRMATION: "❓",
            SimulationAction.BLOCK: "🛑",
            SimulationAction.ESCALATE: "🚨"
        }
        
        emoji = action_emoji.get(result.recommended_action, "❓")
        
        output = f"""
# 🔮 Simulazione Counterfactual

**Azione analizzata**: {result.original_action[:100]}

---

## {emoji} Raccomandazione: {result.recommended_action.value.upper()}

**Score di rischio**: {result.expected_damage_score:.1f}/100
**Confidenza simulazione**: {result.simulation_confidence:.1%}

---

## ⚠️ Warnings
{chr(10).join(f"- {w}" for w in result.warnings) or "Nessun warning"}

---

## 🔴 Worst Case Scenario
- **Scenario**: {result.worst_case.description}
- **Probabilità**: {result.worst_case.probability:.1%}
- **Severità**: {result.worst_case.severity.name}
- **Reversibile**: {'Sì' if result.worst_case.reversible else 'No'}
- **Tempo impatto**: {result.worst_case.time_to_impact}

---

## 📊 Rischi per Categoria
{chr(10).join(f"- **{cat.value}**: {score:.1f}%" for cat, score in sorted(result.risk_summary.items(), key=lambda x: -x[1]) if score > 5)}

---

## 🛡️ Mitigazioni Suggerite
{chr(10).join(f"{i+1}. {m}" for i, m in enumerate(result.mitigations))}

---

*Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return output
    
    def set_threshold(self, threshold_name: str, value: float):
        """Imposta soglia custom"""
        valid_thresholds = ['warning', 'confirm', 'block', 'escalate']
        if threshold_name in valid_thresholds:
            attr_name = f"DAMAGE_THRESHOLD_{threshold_name.upper()}"
            setattr(self, attr_name, value)
            logger.info(f"🔮 Threshold {threshold_name} impostato a {value}")
    
    def add_blocked_pattern(self, pattern: str):
        """Aggiunge pattern sempre bloccato"""
        self.blocked_patterns.append(pattern)
        logger.info(f"🔮 Pattern bloccato aggiunto: {pattern}")
    
    def check_blocked_patterns(self, action: str) -> bool:
        """Verifica se azione matcha pattern bloccato"""
        action_lower = action.lower()
        return any(p.lower() in action_lower for p in self.blocked_patterns)
