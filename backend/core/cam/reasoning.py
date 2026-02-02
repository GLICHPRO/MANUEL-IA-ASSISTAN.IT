"""
🧠 CAM REASONING LAYER
======================
Massima potenza cognitiva per crisi.

Componenti:
- MultiPathReasoningEngine: Analizza con logica causale, probabilistica, controfattuale, storica
- UncertaintyMapper: Mappa cosa sappiamo e cosa NO
- NoActionIntelligence: "Cosa succede se non facciamo nulla?"

Se i percorsi divergono → avviso forte
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
import logging
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


class ReasoningPath(Enum):
    """Percorsi di ragionamento"""
    CAUSAL = "causal"  # Logica causa-effetto
    PROBABILISTIC = "probabilistic"  # Analisi probabilistica
    COUNTERFACTUAL = "counterfactual"  # "What if" alternativi
    HISTORICAL = "historical"  # Basato su precedenti
    ADVERSARIAL = "adversarial"  # Considera worst case


class CertaintyLevel(Enum):
    """Livelli di certezza"""
    CERTAIN = "certain"  # >90% confidence
    LIKELY = "likely"  # 70-90%
    POSSIBLE = "possible"  # 50-70%
    UNCERTAIN = "uncertain"  # 30-50%
    UNKNOWN = "unknown"  # <30%


@dataclass
class ReasoningResult:
    """Risultato di un percorso di ragionamento"""
    path: ReasoningPath
    conclusion: str
    confidence: float  # 0-1
    supporting_evidence: List[str]
    counter_evidence: List[str]
    assumptions: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PathDivergence:
    """Divergenza tra percorsi"""
    paths_compared: List[ReasoningPath]
    divergence_score: float  # 0-1
    main_disagreements: List[str]
    recommendation: str


@dataclass
class UncertaintyMap:
    """Mappa delle incertezze"""
    known_facts: List[str]
    unknown_factors: List[str]
    assumptions_made: List[str]
    what_would_change_assessment: List[str]
    confidence_breakdown: Dict[str, float]


@dataclass
class NoActionAnalysis:
    """Analisi scenario "non fare nulla" """
    description: str
    likely_outcome: str
    probability: float
    risks: List[str]
    benefits: List[str]
    time_sensitivity: str  # "urgent", "moderate", "low"
    is_viable_option: bool


# ============================================================
# MULTI-PATH REASONING ENGINE
# ============================================================

class MultiPathReasoningEngine:
    """
    Analizza problemi con percorsi multipli:
    - Logica causale
    - Probabilistica
    - Controfattuale
    - Storica
    
    Se i percorsi divergono → avviso forte
    """
    
    # Soglia per divergenza significativa
    DIVERGENCE_THRESHOLD = 0.3
    
    def __init__(self):
        self.results: Dict[str, Dict[ReasoningPath, ReasoningResult]] = {}
        self.divergences: List[PathDivergence] = []
        
        # Path analyzers (pluggable)
        self.analyzers: Dict[ReasoningPath, Callable] = {}
    
    def register_analyzer(
        self,
        path: ReasoningPath,
        analyzer: Callable[[str, Dict[str, Any]], ReasoningResult]
    ):
        """Registra analyzer per percorso"""
        self.analyzers[path] = analyzer
    
    async def analyze(
        self,
        problem_id: str,
        problem: str,
        context: Dict[str, Any],
        paths: Optional[List[ReasoningPath]] = None
    ) -> Dict[ReasoningPath, ReasoningResult]:
        """
        Analizza problema con percorsi multipli.
        
        Returns:
            Risultati per ogni percorso
        """
        
        paths = paths or list(ReasoningPath)
        results = {}
        
        for path in paths:
            if path in self.analyzers:
                result = await self.analyzers[path](problem, context)
            else:
                result = await self._default_analyzer(path, problem, context)
            
            results[path] = result
        
        self.results[problem_id] = results
        
        # Check divergenze
        divergence = self._check_divergence(results)
        if divergence.divergence_score > self.DIVERGENCE_THRESHOLD:
            self.divergences.append(divergence)
            logger.warning(f"🧠 DIVERGENZA RILEVATA: {divergence.divergence_score:.1%}")
        
        return results
    
    async def _default_analyzer(
        self,
        path: ReasoningPath,
        problem: str,
        context: Dict[str, Any]
    ) -> ReasoningResult:
        """Analyzer di default per percorso"""
        
        # Simulazione analisi (in produzione: AI/logic engine)
        
        if path == ReasoningPath.CAUSAL:
            return ReasoningResult(
                path=path,
                conclusion=f"Analisi causale: causa probabile identificata",
                confidence=0.7,
                supporting_evidence=["Correlazione temporale", "Pattern consistente"],
                counter_evidence=["Possibili fattori confondenti"],
                assumptions=["Dati completi", "Nessuna variabile nascosta"]
            )
        
        elif path == ReasoningPath.PROBABILISTIC:
            return ReasoningResult(
                path=path,
                conclusion=f"Analisi probabilistica: outcome più probabile definito",
                confidence=0.65,
                supporting_evidence=["Distribuzione storica", "Trend attuale"],
                counter_evidence=["Varianza alta"],
                assumptions=["Distribuzione stabile", "Sample rappresentativo"]
            )
        
        elif path == ReasoningPath.COUNTERFACTUAL:
            return ReasoningResult(
                path=path,
                conclusion=f"Analisi controfattuale: alternative valutate",
                confidence=0.5,
                supporting_evidence=["Scenario B plausibile"],
                counter_evidence=["Difficile verificare"],
                assumptions=["Ceteris paribus"]
            )
        
        elif path == ReasoningPath.HISTORICAL:
            return ReasoningResult(
                path=path,
                conclusion=f"Analisi storica: precedenti consultati",
                confidence=0.6,
                supporting_evidence=["Casi simili documentati"],
                counter_evidence=["Contesto differente"],
                assumptions=["Storia rilevante", "Pattern ripetibili"]
            )
        
        elif path == ReasoningPath.ADVERSARIAL:
            return ReasoningResult(
                path=path,
                conclusion=f"Analisi adversarial: worst case valutato",
                confidence=0.55,
                supporting_evidence=["Vulnerabilità identificate"],
                counter_evidence=["Scenario improbabile"],
                assumptions=["Attore razionale", "Informazioni complete"]
            )
        
        return ReasoningResult(
            path=path,
            conclusion="Analisi non disponibile",
            confidence=0.0,
            supporting_evidence=[],
            counter_evidence=[],
            assumptions=[]
        )
    
    def _check_divergence(
        self,
        results: Dict[ReasoningPath, ReasoningResult]
    ) -> PathDivergence:
        """Verifica divergenze tra percorsi"""
        
        if len(results) < 2:
            return PathDivergence(
                paths_compared=list(results.keys()),
                divergence_score=0.0,
                main_disagreements=[],
                recommendation="Insufficienti percorsi per confronto"
            )
        
        # Calcola divergenza basata su confidence
        confidences = [r.confidence for r in results.values()]
        
        if len(confidences) >= 2:
            divergence_score = max(confidences) - min(confidences)
        else:
            divergence_score = 0
        
        # Identifica disagreements
        disagreements = []
        paths_list = list(results.keys())
        
        for i in range(len(paths_list)):
            for j in range(i + 1, len(paths_list)):
                p1, p2 = paths_list[i], paths_list[j]
                r1, r2 = results[p1], results[p2]
                
                # Se confidence molto diversa
                if abs(r1.confidence - r2.confidence) > 0.2:
                    disagreements.append(
                        f"{p1.value} ({r1.confidence:.0%}) vs {p2.value} ({r2.confidence:.0%})"
                    )
        
        # Recommendation
        if divergence_score > 0.4:
            rec = "⚠️ FORTE DIVERGENZA - Richiesta analisi umana"
        elif divergence_score > 0.2:
            rec = "⚡ Divergenza moderata - Considerare percorsi multipli"
        else:
            rec = "✅ Percorsi allineati"
        
        return PathDivergence(
            paths_compared=list(results.keys()),
            divergence_score=divergence_score,
            main_disagreements=disagreements,
            recommendation=rec
        )
    
    def get_consensus(
        self,
        problem_id: str
    ) -> Tuple[str, float, List[str]]:
        """
        Ottiene consenso tra percorsi.
        
        Returns:
            (conclusione, confidence, caveats)
        """
        
        results = self.results.get(problem_id, {})
        
        if not results:
            return "Nessuna analisi disponibile", 0.0, []
        
        # Weighted average confidence
        total_conf = sum(r.confidence for r in results.values())
        avg_conf = total_conf / len(results) if results else 0
        
        # Conclusione dal path con più confidence
        best = max(results.values(), key=lambda r: r.confidence)
        
        # Caveats da percorsi con bassa confidence
        caveats = []
        for path, result in results.items():
            if result.confidence < 0.5:
                caveats.append(f"{path.value}: {result.counter_evidence[0] if result.counter_evidence else 'Confidence bassa'}")
        
        return best.conclusion, avg_conf, caveats
    
    def format_analysis(self, problem_id: str) -> str:
        """Formatta analisi per visualizzazione"""
        
        results = self.results.get(problem_id, {})
        
        if not results:
            return "Nessuna analisi disponibile"
        
        # Check divergenza
        divergence = self._check_divergence(results)
        
        output = f"""
# 🧠 Multi-Path Reasoning Analysis

## Divergenza
**Score:** {divergence.divergence_score:.1%}
**Valutazione:** {divergence.recommendation}

## Percorsi Analizzati
"""
        
        for path, result in results.items():
            conf_bar = "█" * int(result.confidence * 10) + "░" * (10 - int(result.confidence * 10))
            output += f"""
### {path.value.title()}
**Confidence:** {conf_bar} {result.confidence:.0%}
**Conclusione:** {result.conclusion}

**Evidenze a supporto:**
{chr(10).join(f'- {e}' for e in result.supporting_evidence) or '- Nessuna'}

**Contro-evidenze:**
{chr(10).join(f'- {e}' for e in result.counter_evidence) or '- Nessuna'}

**Assunzioni:**
{chr(10).join(f'- {a}' for a in result.assumptions) or '- Nessuna'}

---
"""
        
        # Consensus
        conclusion, conf, caveats = self.get_consensus(problem_id)
        output += f"""
## Consenso
**Conclusione:** {conclusion}
**Confidence media:** {conf:.0%}

**Caveats:**
{chr(10).join(f'- {c}' for c in caveats) or '- Nessuno'}
"""
        
        return output


# ============================================================
# UNCERTAINTY MAPPER
# ============================================================

class UncertaintyMapper:
    """
    Per ogni output mostra:
    - Cosa sappiamo
    - Cosa NON sappiamo
    - Cosa cambierebbe la valutazione
    
    "Uncertainty is a Signal"
    """
    
    def __init__(self):
        self.maps: Dict[str, UncertaintyMap] = {}
    
    def create_map(
        self,
        analysis_id: str,
        known_facts: List[str],
        unknown_factors: List[str],
        assumptions: List[str],
        sensitivity_factors: List[str],
        confidence_breakdown: Optional[Dict[str, float]] = None
    ) -> UncertaintyMap:
        """Crea mappa delle incertezze"""
        
        uncertainty_map = UncertaintyMap(
            known_facts=known_facts,
            unknown_factors=unknown_factors,
            assumptions_made=assumptions,
            what_would_change_assessment=sensitivity_factors,
            confidence_breakdown=confidence_breakdown or {}
        )
        
        self.maps[analysis_id] = uncertainty_map
        
        return uncertainty_map
    
    def analyze_confidence(
        self,
        components: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analizza breakdown della confidence"""
        
        if not components:
            return {'overall': 0, 'weakest_link': None, 'strongest': None}
        
        overall = statistics.mean(components.values())
        weakest = min(components.items(), key=lambda x: x[1])
        strongest = max(components.items(), key=lambda x: x[1])
        
        return {
            'overall': overall,
            'weakest_link': {'component': weakest[0], 'confidence': weakest[1]},
            'strongest': {'component': strongest[0], 'confidence': strongest[1]},
            'variance': statistics.variance(components.values()) if len(components) > 1 else 0
        }
    
    def get_certainty_level(self, confidence: float) -> CertaintyLevel:
        """Converte confidence in livello certezza"""
        
        if confidence >= 0.9:
            return CertaintyLevel.CERTAIN
        elif confidence >= 0.7:
            return CertaintyLevel.LIKELY
        elif confidence >= 0.5:
            return CertaintyLevel.POSSIBLE
        elif confidence >= 0.3:
            return CertaintyLevel.UNCERTAIN
        else:
            return CertaintyLevel.UNKNOWN
    
    def format_map(self, analysis_id: str) -> str:
        """Formatta mappa per visualizzazione"""
        
        umap = self.maps.get(analysis_id)
        
        if not umap:
            return "Mappa non disponibile"
        
        # Analizza confidence
        conf_analysis = self.analyze_confidence(umap.confidence_breakdown)
        
        return f"""
# 🗺️ Uncertainty Map

## ✅ Cosa Sappiamo (Known Facts)
{chr(10).join(f'- {f}' for f in umap.known_facts) or '- Nessun fatto certo'}

## ❓ Cosa NON Sappiamo (Unknown Factors)
{chr(10).join(f'- {f}' for f in umap.unknown_factors) or '- Nessuna incertezza identificata'}

## 📊 Assunzioni Fatte
{chr(10).join(f'- {a}' for a in umap.assumptions_made) or '- Nessuna assunzione esplicita'}

## 🔄 Cosa Cambierebbe la Valutazione
{chr(10).join(f'- {s}' for s in umap.what_would_change_assessment) or '- Nessun fattore critico identificato'}

## 📈 Confidence Breakdown
| Componente | Confidence |
|------------|------------|
{chr(10).join(f'| {k} | {v:.0%} |' for k, v in umap.confidence_breakdown.items()) if umap.confidence_breakdown else '| N/A | N/A |'}

**Overall Confidence:** {conf_analysis.get('overall', 0):.0%}
**Weakest Link:** {conf_analysis.get('weakest_link', {}).get('component', 'N/A')} ({conf_analysis.get('weakest_link', {}).get('confidence', 0):.0%})
"""


# ============================================================
# NO-ACTION INTELLIGENCE
# ============================================================

class NoActionIntelligence:
    """
    Sempre presente in ogni analisi:
    "Cosa succede se non facciamo nulla?"
    
    Spesso è l'opzione migliore in crisi.
    """
    
    def __init__(self):
        self.analyses: Dict[str, NoActionAnalysis] = {}
    
    def analyze_no_action(
        self,
        scenario_id: str,
        current_situation: str,
        context: Dict[str, Any],
        time_constraints: Optional[Dict[str, Any]] = None
    ) -> NoActionAnalysis:
        """
        Analizza scenario "non fare nulla".
        
        Considera:
        - Evoluzione naturale
        - Rischi di inazione
        - Benefici di inazione
        - Pressione temporale
        """
        
        # Determina time sensitivity
        if time_constraints:
            deadline = time_constraints.get('deadline')
            if deadline:
                hours_remaining = (deadline - datetime.now()).total_seconds() / 3600
                if hours_remaining < 1:
                    time_sensitivity = "urgent"
                elif hours_remaining < 24:
                    time_sensitivity = "moderate"
                else:
                    time_sensitivity = "low"
            else:
                time_sensitivity = "moderate"
        else:
            time_sensitivity = "low"
        
        # Analisi base (in produzione: AI analysis)
        analysis = NoActionAnalysis(
            description=f"Analisi scenario di non-intervento per: {current_situation[:100]}",
            likely_outcome="Situazione evolve secondo trend corrente",
            probability=0.6,
            risks=[
                "Perdita finestra di opportunità",
                "Escalation se trend negativo",
                "Percezione di inazione"
            ],
            benefits=[
                "Evita errori da azione precipitosa",
                "Permette raccolta più dati",
                "Conserva risorse",
                "Evita effetti collaterali indesiderati"
            ],
            time_sensitivity=time_sensitivity,
            is_viable_option=time_sensitivity != "urgent"
        )
        
        self.analyses[scenario_id] = analysis
        
        return analysis
    
    def compare_with_action(
        self,
        scenario_id: str,
        proposed_action: str,
        action_risks: List[str],
        action_benefits: List[str]
    ) -> str:
        """Confronta no-action con azione proposta"""
        
        no_action = self.analyses.get(scenario_id)
        
        if not no_action:
            return "Analisi no-action non disponibile"
        
        # Conta pro/contro
        no_action_pros = len(no_action.benefits)
        no_action_cons = len(no_action.risks)
        action_pros = len(action_benefits)
        action_cons = len(action_risks)
        
        # Valutazione semplice
        no_action_score = no_action_pros - no_action_cons * 1.5  # Penalizza rischi
        action_score = action_pros - action_cons * 1.5
        
        if no_action.time_sensitivity == "urgent":
            action_score += 2  # Bonus per azione in urgenza
        
        recommendation = ""
        if no_action_score > action_score + 1:
            recommendation = "💡 **Considerare seriamente NON agire** - I benefici dell'attesa superano i rischi"
        elif action_score > no_action_score + 1:
            recommendation = "⚡ **Azione consigliata** - I benefici dell'azione giustificano i rischi"
        else:
            recommendation = "⚖️ **Decisione equilibrata** - Entrambe le opzioni hanno merit"
        
        return f"""
# 🤔 No-Action vs Action Analysis

## Scenario: Non fare nulla
**Outcome probabile:** {no_action.likely_outcome}
**Probabilità:** {no_action.probability:.0%}
**Time sensitivity:** {no_action.time_sensitivity}
**Viable:** {'✅ Sì' if no_action.is_viable_option else '⚠️ No'}

### Rischi di non agire
{chr(10).join(f'- {r}' for r in no_action.risks)}

### Benefici di non agire
{chr(10).join(f'- {b}' for b in no_action.benefits)}

---

## Scenario: {proposed_action}

### Rischi dell'azione
{chr(10).join(f'- {r}' for r in action_risks)}

### Benefici dell'azione
{chr(10).join(f'- {b}' for b in action_benefits)}

---

## Raccomandazione
{recommendation}

**Score No-Action:** {no_action_score:.1f}
**Score Azione:** {action_score:.1f}
"""
    
    def format_analysis(self, scenario_id: str) -> str:
        """Formatta analisi per visualizzazione"""
        
        analysis = self.analyses.get(scenario_id)
        
        if not analysis:
            return "Analisi non disponibile"
        
        time_emoji = {
            "urgent": "🔴",
            "moderate": "🟡",
            "low": "🟢"
        }
        
        return f"""
# 🤔 No-Action Intelligence

## Descrizione
{analysis.description}

## Outcome Probabile
{analysis.likely_outcome}
**Probabilità:** {analysis.probability:.0%}

## Time Sensitivity
{time_emoji.get(analysis.time_sensitivity, '⚪')} **{analysis.time_sensitivity.upper()}**

## Opzione Viabile?
{'✅ Sì - Non agire è un\'opzione valida' if analysis.is_viable_option else '⚠️ No - Azione probabilmente necessaria'}

## Rischi di Non Agire
{chr(10).join(f'- ⚠️ {r}' for r in analysis.risks)}

## Benefici di Non Agire
{chr(10).join(f'- ✅ {b}' for b in analysis.benefits)}

---

> 💡 *"Sometimes the best action is no action at all"*
"""


# ============================================================
# UNIFIED REASONING LAYER
# ============================================================

class ReasoningLayer:
    """
    Reasoning Layer unificato per CAM.
    
    Integra:
    - MultiPathReasoningEngine
    - UncertaintyMapper
    - NoActionIntelligence
    """
    
    def __init__(self):
        self.multi_path = MultiPathReasoningEngine()
        self.uncertainty = UncertaintyMapper()
        self.no_action = NoActionIntelligence()
    
    async def full_analysis(
        self,
        analysis_id: str,
        problem: str,
        context: Dict[str, Any],
        known_facts: List[str],
        unknown_factors: List[str]
    ) -> Dict[str, Any]:
        """
        Esegue analisi completa con tutti i componenti.
        
        Returns:
            Risultati integrati
        """
        
        # 1. Multi-path reasoning
        path_results = await self.multi_path.analyze(
            analysis_id,
            problem,
            context
        )
        
        # 2. Uncertainty mapping
        confidence_breakdown = {
            path.value: result.confidence
            for path, result in path_results.items()
        }
        
        umap = self.uncertainty.create_map(
            analysis_id,
            known_facts=known_facts,
            unknown_factors=unknown_factors,
            assumptions=list(set(
                a for r in path_results.values() for a in r.assumptions
            )),
            sensitivity_factors=unknown_factors[:3],
            confidence_breakdown=confidence_breakdown
        )
        
        # 3. No-action analysis
        no_action = self.no_action.analyze_no_action(
            analysis_id,
            problem,
            context
        )
        
        # Consensus
        conclusion, conf, caveats = self.multi_path.get_consensus(analysis_id)
        
        return {
            'analysis_id': analysis_id,
            'consensus': {
                'conclusion': conclusion,
                'confidence': conf,
                'caveats': caveats
            },
            'paths': {p.value: {
                'conclusion': r.conclusion,
                'confidence': r.confidence
            } for p, r in path_results.items()},
            'uncertainty': {
                'known': umap.known_facts,
                'unknown': umap.unknown_factors,
                'assumptions': umap.assumptions_made
            },
            'no_action': {
                'viable': no_action.is_viable_option,
                'time_sensitivity': no_action.time_sensitivity,
                'risks': no_action.risks,
                'benefits': no_action.benefits
            }
        }
    
    def format_full_report(self, analysis_id: str) -> str:
        """Formatta report completo"""
        
        return f"""
{self.multi_path.format_analysis(analysis_id)}

---

{self.uncertainty.format_map(analysis_id)}

---

{self.no_action.format_analysis(analysis_id)}
"""


# Singleton
_reasoning_layer: Optional[ReasoningLayer] = None


def get_reasoning_layer() -> ReasoningLayer:
    """Ottiene istanza singleton"""
    global _reasoning_layer
    if _reasoning_layer is None:
        _reasoning_layer = ReasoningLayer()
    return _reasoning_layer
