"""
🔁 REDUNDANT REASONING ENGINE
==============================
Per decisioni critiche, GIDEON:
- Esegue lo stesso reasoning con 2-3 approcci
- Confronta output
- Segnala divergenze

Ispirato a: sistemi aeronautici (triple modular redundancy)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
import logging
import asyncio
from abc import ABC, abstractmethod
import hashlib

logger = logging.getLogger(__name__)


class ReasoningApproach(Enum):
    """Approcci di reasoning disponibili"""
    ANALYTICAL = "analytical"       # Logico-deduttivo
    HEURISTIC = "heuristic"         # Basato su pattern
    CONSERVATIVE = "conservative"   # Risk-averse
    AGGRESSIVE = "aggressive"       # Opportunity-seeking
    BAYESIAN = "bayesian"           # Probabilistico
    RULE_BASED = "rule_based"       # Basato su regole
    ANALOGICAL = "analogical"       # Basato su casi simili


class ConsensusLevel(Enum):
    """Livelli di consenso tra approcci"""
    UNANIMOUS = "unanimous"         # 100% accordo
    STRONG = "strong"               # >80% accordo
    MAJORITY = "majority"           # >50% accordo
    SPLIT = "split"                 # ~50% split
    DIVERGENT = "divergent"         # <50% accordo
    CONTRADICTORY = "contradictory" # Risultati opposti


class ConfidenceLevel(Enum):
    """Livelli di confidenza nel risultato"""
    VERY_HIGH = 5
    HIGH = 4
    MODERATE = 3
    LOW = 2
    VERY_LOW = 1


@dataclass
class ReasoningResult:
    """Risultato di un singolo approccio"""
    approach: ReasoningApproach
    conclusion: str
    confidence: float
    reasoning_path: List[str]
    key_factors: List[str]
    warnings: List[str]
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Divergence:
    """Divergenza tra approcci"""
    approach_a: ReasoningApproach
    approach_b: ReasoningApproach
    divergence_type: str  # 'conclusion', 'confidence', 'factors'
    description: str
    severity: float  # 0-1
    resolution: Optional[str] = None


@dataclass
class RedundantResult:
    """Risultato completo del reasoning ridondante"""
    query: str
    results: List[ReasoningResult]
    consensus_level: ConsensusLevel
    consensus_conclusion: Optional[str]
    confidence: float
    divergences: List[Divergence]
    requires_human_review: bool
    recommendation: str
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.now)


class ReasoningModule(ABC):
    """Modulo di reasoning astratto"""
    
    def __init__(self, approach: ReasoningApproach):
        self.approach = approach
    
    @abstractmethod
    async def reason(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> ReasoningResult:
        pass


class AnalyticalReasoning(ReasoningModule):
    """Reasoning logico-deduttivo"""
    
    def __init__(self):
        super().__init__(ReasoningApproach.ANALYTICAL)
    
    async def reason(self, query: str, context: Dict[str, Any]) -> ReasoningResult:
        import time
        start = time.time()
        
        # Simula reasoning analitico
        path = [
            "1. Identificazione premesse",
            "2. Analisi logica delle relazioni",
            "3. Deduzione conseguenze",
            "4. Validazione conclusione"
        ]
        
        factors = list(context.get('factors', ['dato_1', 'dato_2']))[:5]
        
        # Confidenza basata su completezza dati
        data_quality = context.get('data_quality', 0.7)
        confidence = 0.6 + (data_quality * 0.3)
        
        conclusion = f"Analisi logica: {self._analyze(query, context)}"
        
        return ReasoningResult(
            approach=self.approach,
            conclusion=conclusion,
            confidence=confidence,
            reasoning_path=path,
            key_factors=factors,
            warnings=self._identify_gaps(context),
            execution_time=time.time() - start,
            metadata={'method': 'deductive'}
        )
    
    def _analyze(self, query: str, context: Dict[str, Any]) -> str:
        # Placeholder per analisi reale
        if 'risk' in query.lower():
            return "Rischio valutato come MEDIO-ALTO basato su evidenze"
        return "Conclusione derivata da analisi delle premesse"
    
    def _identify_gaps(self, context: Dict[str, Any]) -> List[str]:
        warnings = []
        if context.get('data_quality', 1.0) < 0.8:
            warnings.append("Dati incompleti potrebbero influenzare conclusione")
        return warnings


class HeuristicReasoning(ReasoningModule):
    """Reasoning basato su pattern ed euristiche"""
    
    def __init__(self):
        super().__init__(ReasoningApproach.HEURISTIC)
    
    async def reason(self, query: str, context: Dict[str, Any]) -> ReasoningResult:
        import time
        start = time.time()
        
        path = [
            "1. Pattern recognition",
            "2. Confronto con casi noti",
            "3. Applicazione euristica",
            "4. Validazione rapida"
        ]
        
        # Euristiche tendono ad essere più veloci ma meno precise
        confidence = 0.5 + (len(context) * 0.05)
        confidence = min(0.85, confidence)
        
        conclusion = f"Pattern match: {self._match_pattern(query, context)}"
        
        return ReasoningResult(
            approach=self.approach,
            conclusion=conclusion,
            confidence=confidence,
            reasoning_path=path,
            key_factors=['pattern_similarity', 'historical_cases'],
            warnings=["Euristica potrebbe non coprire casi edge"],
            execution_time=time.time() - start,
            metadata={'method': 'pattern_based'}
        )
    
    def _match_pattern(self, query: str, context: Dict[str, Any]) -> str:
        if 'urgent' in query.lower() or 'emergency' in str(context).lower():
            return "Situazione richiede azione immediata"
        return "Situazione simile a casi precedenti risolti"


class ConservativeReasoning(ReasoningModule):
    """Reasoning conservativo/risk-averse"""
    
    def __init__(self):
        super().__init__(ReasoningApproach.CONSERVATIVE)
    
    async def reason(self, query: str, context: Dict[str, Any]) -> ReasoningResult:
        import time
        start = time.time()
        
        path = [
            "1. Identificazione rischi",
            "2. Worst-case analysis",
            "3. Valutazione cautela",
            "4. Raccomandazione prudente"
        ]
        
        # Conservative è sempre più cauto
        confidence = context.get('data_quality', 0.5) * 0.8  # Riduce confidenza
        
        conclusion = f"Approccio cauto: {self._conservative_conclusion(query, context)}"
        
        return ReasoningResult(
            approach=self.approach,
            conclusion=conclusion,
            confidence=confidence,
            reasoning_path=path,
            key_factors=['risk_exposure', 'worst_case'],
            warnings=["Potrebbe essere eccessivamente cauto"],
            execution_time=time.time() - start,
            metadata={'method': 'risk_averse', 'bias': 'cautious'}
        )
    
    def _conservative_conclusion(self, query: str, context: Dict[str, Any]) -> str:
        return "Consigliata cautela e verifica aggiuntiva prima di procedere"


class RuleBasedReasoning(ReasoningModule):
    """Reasoning basato su regole predefinite"""
    
    RULES = {
        'high_risk': "IF risk_score > 0.7 THEN require_human_approval",
        'data_quality': "IF data_quality < 0.5 THEN flag_uncertainty",
        'urgency': "IF urgency == 'high' AND confidence > 0.8 THEN proceed",
    }
    
    def __init__(self):
        super().__init__(ReasoningApproach.RULE_BASED)
    
    async def reason(self, query: str, context: Dict[str, Any]) -> ReasoningResult:
        import time
        start = time.time()
        
        # Applica regole
        matched_rules = []
        conclusions = []
        
        risk = context.get('risk_score', 0.5)
        quality = context.get('data_quality', 0.7)
        urgency = context.get('urgency', 'medium')
        confidence_ctx = context.get('confidence', 0.7)
        
        if risk > 0.7:
            matched_rules.append('high_risk')
            conclusions.append("Richiesta approvazione umana")
        if quality < 0.5:
            matched_rules.append('data_quality')
            conclusions.append("Incertezza segnalata")
        if urgency == 'high' and confidence_ctx > 0.8:
            matched_rules.append('urgency')
            conclusions.append("Procedere autorizzato")
        
        path = [f"Regola applicata: {r}" for r in matched_rules] or ["Nessuna regola matchata"]
        
        conclusion = "; ".join(conclusions) or "Nessuna regola attivata - valutazione manuale"
        
        return ReasoningResult(
            approach=self.approach,
            conclusion=conclusion,
            confidence=0.9 if matched_rules else 0.4,  # Alta se regole matchano
            reasoning_path=path,
            key_factors=matched_rules,
            warnings=[] if matched_rules else ["Nessuna regola applicabile"],
            execution_time=time.time() - start,
            metadata={'rules_matched': len(matched_rules)}
        )


class RedundantReasoningEngine:
    """
    Engine per reasoning ridondante.
    
    Per decisioni critiche:
    - Esegue lo stesso reasoning con 2-3 approcci
    - Confronta output
    - Segnala divergenze
    
    Ispirato a: sistemi aeronautici (triple modular redundancy)
    """
    
    # Soglie
    DIVERGENCE_THRESHOLD = 0.3  # 30% differenza = divergenza
    HUMAN_REVIEW_THRESHOLD = 0.5  # 50% divergenza = review umano
    MIN_CONSENSUS_FOR_AUTO = 0.8  # 80% accordo per auto-approve
    
    def __init__(self):
        # Inizializza moduli di default
        self.modules: Dict[ReasoningApproach, ReasoningModule] = {
            ReasoningApproach.ANALYTICAL: AnalyticalReasoning(),
            ReasoningApproach.HEURISTIC: HeuristicReasoning(),
            ReasoningApproach.CONSERVATIVE: ConservativeReasoning(),
            ReasoningApproach.RULE_BASED: RuleBasedReasoning(),
        }
        
        self.history: List[RedundantResult] = []
        self.divergence_log: List[Divergence] = []
    
    def register_module(self, module: ReasoningModule):
        """Registra modulo di reasoning custom"""
        self.modules[module.approach] = module
        logger.info(f"🔁 Modulo registrato: {module.approach.value}")
    
    async def reason(
        self,
        query: str,
        context: Dict[str, Any],
        approaches: Optional[List[ReasoningApproach]] = None,
        min_approaches: int = 2
    ) -> RedundantResult:
        """
        Esegue reasoning ridondante.
        
        Args:
            query: Query da processare
            context: Contesto per il reasoning
            approaches: Approcci specifici da usare (default: tutti)
            min_approaches: Minimo approcci richiesti
        """
        
        import time
        start = time.time()
        
        # Seleziona approcci
        if approaches is None:
            approaches = list(self.modules.keys())[:3]  # Default: primi 3
        
        if len(approaches) < min_approaches:
            logger.warning(f"🔁 Richiesti {min_approaches} approcci, disponibili {len(approaches)}")
        
        # Esegui reasoning in parallelo
        tasks = []
        for approach in approaches:
            if approach in self.modules:
                tasks.append(self._run_reasoning(self.modules[approach], query, context))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtra errori
        valid_results = [r for r in results if isinstance(r, ReasoningResult)]
        
        if not valid_results:
            logger.error("🔁 Nessun risultato valido!")
            return self._empty_result(query, time.time() - start)
        
        # Analizza consenso
        consensus_level, consensus_conclusion = self._analyze_consensus(valid_results)
        
        # Identifica divergenze
        divergences = self._identify_divergences(valid_results)
        
        # Calcola confidenza aggregata
        confidence = self._aggregate_confidence(valid_results, consensus_level)
        
        # Determina se serve review umano
        requires_review = self._needs_human_review(
            consensus_level, divergences, confidence
        )
        
        # Genera raccomandazione
        recommendation = self._generate_recommendation(
            consensus_level, consensus_conclusion, divergences, confidence
        )
        
        result = RedundantResult(
            query=query,
            results=valid_results,
            consensus_level=consensus_level,
            consensus_conclusion=consensus_conclusion,
            confidence=confidence,
            divergences=divergences,
            requires_human_review=requires_review,
            recommendation=recommendation,
            execution_time=time.time() - start
        )
        
        # Log
        self.history.append(result)
        self.divergence_log.extend(divergences)
        
        logger.info(f"🔁 Reasoning completato: {consensus_level.value}, conf={confidence:.2f}")
        
        return result
    
    async def _run_reasoning(
        self,
        module: ReasoningModule,
        query: str,
        context: Dict[str, Any]
    ) -> ReasoningResult:
        """Esegue singolo modulo con error handling"""
        try:
            return await module.reason(query, context)
        except Exception as e:
            logger.error(f"🔁 Errore modulo {module.approach.value}: {e}")
            raise
    
    def _analyze_consensus(
        self,
        results: List[ReasoningResult]
    ) -> Tuple[ConsensusLevel, Optional[str]]:
        """Analizza livello di consenso tra risultati"""
        
        if len(results) == 1:
            return ConsensusLevel.UNANIMOUS, results[0].conclusion
        
        # Calcola similarità conclusioni
        conclusions = [r.conclusion.lower() for r in results]
        
        # Semplice: check parole chiave comuni
        all_words = [set(c.split()) for c in conclusions]
        
        # Intersezione parole
        common = all_words[0]
        for words in all_words[1:]:
            common = common.intersection(words)
        
        # Calcola overlap
        avg_len = sum(len(w) for w in all_words) / len(all_words)
        overlap = len(common) / avg_len if avg_len > 0 else 0
        
        # Determina livello
        if overlap > 0.7:
            level = ConsensusLevel.UNANIMOUS
        elif overlap > 0.5:
            level = ConsensusLevel.STRONG
        elif overlap > 0.3:
            level = ConsensusLevel.MAJORITY
        elif overlap > 0.15:
            level = ConsensusLevel.SPLIT
        else:
            level = ConsensusLevel.DIVERGENT
        
        # Conclusione di consenso (da risultato con confidenza più alta)
        best = max(results, key=lambda r: r.confidence)
        
        return level, best.conclusion if level.value <= 3 else None
    
    def _identify_divergences(
        self,
        results: List[ReasoningResult]
    ) -> List[Divergence]:
        """Identifica divergenze tra risultati"""
        
        divergences = []
        
        for i, r1 in enumerate(results):
            for r2 in results[i+1:]:
                # Check divergenza confidenza
                conf_diff = abs(r1.confidence - r2.confidence)
                if conf_diff > self.DIVERGENCE_THRESHOLD:
                    divergences.append(Divergence(
                        approach_a=r1.approach,
                        approach_b=r2.approach,
                        divergence_type='confidence',
                        description=f"Differenza confidenza: {conf_diff:.2f}",
                        severity=conf_diff,
                        resolution=f"Usare {r1.approach.value if r1.confidence > r2.confidence else r2.approach.value}"
                    ))
                
                # Check divergenza conclusioni
                c1_words = set(r1.conclusion.lower().split())
                c2_words = set(r2.conclusion.lower().split())
                
                overlap = len(c1_words & c2_words) / len(c1_words | c2_words) if c1_words | c2_words else 1
                
                if overlap < 0.3:
                    divergences.append(Divergence(
                        approach_a=r1.approach,
                        approach_b=r2.approach,
                        divergence_type='conclusion',
                        description="Conclusioni significativamente diverse",
                        severity=1 - overlap,
                        resolution="Richiesta valutazione umana"
                    ))
                
                # Check divergenza fattori chiave
                f1 = set(r1.key_factors)
                f2 = set(r2.key_factors)
                factor_overlap = len(f1 & f2) / len(f1 | f2) if f1 | f2 else 1
                
                if factor_overlap < 0.3:
                    divergences.append(Divergence(
                        approach_a=r1.approach,
                        approach_b=r2.approach,
                        divergence_type='factors',
                        description="Fattori chiave considerati molto diversi",
                        severity=1 - factor_overlap
                    ))
        
        return divergences
    
    def _aggregate_confidence(
        self,
        results: List[ReasoningResult],
        consensus: ConsensusLevel
    ) -> float:
        """Aggrega confidenza basata su consenso"""
        
        # Media pesata delle confidenze
        confidences = [r.confidence for r in results]
        avg_conf = sum(confidences) / len(confidences)
        
        # Modifica basata su consenso
        consensus_multiplier = {
            ConsensusLevel.UNANIMOUS: 1.2,
            ConsensusLevel.STRONG: 1.1,
            ConsensusLevel.MAJORITY: 1.0,
            ConsensusLevel.SPLIT: 0.8,
            ConsensusLevel.DIVERGENT: 0.6,
            ConsensusLevel.CONTRADICTORY: 0.4
        }
        
        multiplier = consensus_multiplier.get(consensus, 1.0)
        
        return min(1.0, avg_conf * multiplier)
    
    def _needs_human_review(
        self,
        consensus: ConsensusLevel,
        divergences: List[Divergence],
        confidence: float
    ) -> bool:
        """Determina se serve review umano"""
        
        # Sempre review per divergenze gravi
        severe_divergences = [d for d in divergences if d.severity > self.HUMAN_REVIEW_THRESHOLD]
        if severe_divergences:
            return True
        
        # Review se consenso basso
        if consensus in [ConsensusLevel.SPLIT, ConsensusLevel.DIVERGENT, ConsensusLevel.CONTRADICTORY]:
            return True
        
        # Review se confidenza bassa
        if confidence < 0.5:
            return True
        
        return False
    
    def _generate_recommendation(
        self,
        consensus: ConsensusLevel,
        conclusion: Optional[str],
        divergences: List[Divergence],
        confidence: float
    ) -> str:
        """Genera raccomandazione finale"""
        
        if consensus == ConsensusLevel.UNANIMOUS:
            return f"✅ Consenso unanime (conf: {confidence:.0%}): {conclusion[:100] if conclusion else 'N/A'}"
        
        elif consensus == ConsensusLevel.STRONG:
            return f"✅ Forte consenso (conf: {confidence:.0%}): Procedere con conclusione principale"
        
        elif consensus == ConsensusLevel.MAJORITY:
            return f"⚠️ Consenso maggioritario (conf: {confidence:.0%}): Verificare divergenze prima di procedere"
        
        elif consensus == ConsensusLevel.SPLIT:
            return f"❓ Opinioni divise (conf: {confidence:.0%}): Richiesta valutazione umana"
        
        else:
            return f"🛑 Risultati divergenti (conf: {confidence:.0%}): NON procedere senza review umano"
    
    def _empty_result(self, query: str, exec_time: float) -> RedundantResult:
        """Risultato vuoto per errori"""
        return RedundantResult(
            query=query,
            results=[],
            consensus_level=ConsensusLevel.CONTRADICTORY,
            consensus_conclusion=None,
            confidence=0.0,
            divergences=[],
            requires_human_review=True,
            recommendation="🛑 Nessun risultato valido - review umano richiesto",
            execution_time=exec_time
        )
    
    def format_result(self, result: RedundantResult) -> str:
        """Formatta risultato per visualizzazione"""
        
        consensus_emoji = {
            ConsensusLevel.UNANIMOUS: '✅',
            ConsensusLevel.STRONG: '✅',
            ConsensusLevel.MAJORITY: '⚠️',
            ConsensusLevel.SPLIT: '❓',
            ConsensusLevel.DIVERGENT: '🔴',
            ConsensusLevel.CONTRADICTORY: '🛑'
        }
        
        emoji = consensus_emoji.get(result.consensus_level, '❓')
        
        return f"""
# 🔁 Redundant Reasoning Report

**Query**: {result.query[:100]}

---

## {emoji} Consenso: {result.consensus_level.value}
**Confidenza Aggregata**: {result.confidence:.1%}
**Richiede Review Umano**: {'Sì' if result.requires_human_review else 'No'}

---

## 🎯 Raccomandazione
{result.recommendation}

---

## 📊 Risultati per Approccio ({len(result.results)})
{chr(10).join(f'''
### {r.approach.value.capitalize()}
- **Conclusione**: {r.conclusion[:100]}
- **Confidenza**: {r.confidence:.1%}
- **Fattori chiave**: {', '.join(r.key_factors[:3])}
- **Tempo**: {r.execution_time:.3f}s
''' for r in result.results)}

---

## ⚡ Divergenze ({len(result.divergences)})
{chr(10).join(f"- [{d.divergence_type}] {d.approach_a.value} vs {d.approach_b.value}: {d.description} (sev: {d.severity:.1%})" for d in result.divergences) or 'Nessuna divergenza significativa'}

---

*Tempo totale: {result.execution_time:.3f}s | Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    def get_stats(self) -> Dict[str, Any]:
        """Ritorna statistiche"""
        return {
            'total_reasonings': len(self.history),
            'total_divergences': len(self.divergence_log),
            'human_reviews_required': sum(1 for r in self.history if r.requires_human_review),
            'avg_confidence': sum(r.confidence for r in self.history) / len(self.history) if self.history else 0,
            'consensus_distribution': {
                level.value: sum(1 for r in self.history if r.consensus_level == level)
                for level in ConsensusLevel
            }
        }
