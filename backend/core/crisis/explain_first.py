"""
🧭 EXPLAIN-FIRST PROTOCOL
==========================
In situazioni critiche, GIDEON NON può dare output senza spiegazione.

Ogni risposta deve includere:
- Perché lo pensa
- Cosa NON sa
- Cosa cambierebbe la decisione

Questo riduce: panico, misuse, over-trust nell'AI
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ExplanationLevel(Enum):
    """Livelli di dettaglio della spiegazione"""
    MINIMAL = 1      # Solo conclusione + confidenza
    STANDARD = 2     # + reasoning principale
    DETAILED = 3     # + alternative considerate
    FULL = 4         # + limiti + cosa non sappiamo
    CRITICAL = 5     # Tutto + raccomandazioni di cautela


class UncertaintyType(Enum):
    """Tipi di incertezza"""
    DATA_INCOMPLETE = "dati_incompleti"
    MODEL_LIMITATION = "limitazione_modello"
    CONTEXT_AMBIGUOUS = "contesto_ambiguo"
    NOVEL_SITUATION = "situazione_nuova"
    CONFLICTING_EVIDENCE = "evidenze_contrastanti"
    TIME_SENSITIVE = "sensibile_al_tempo"
    ETHICAL_COMPLEXITY = "complessità_etica"


@dataclass
class ExplanationComponent:
    """Componente di una spiegazione"""
    component_type: str  # 'reasoning', 'uncertainty', 'alternative', 'limitation'
    content: str
    confidence: float = 1.0
    source: Optional[str] = None


@dataclass
class StructuredExplanation:
    """Spiegazione strutturata completa"""
    conclusion: str
    confidence: float
    reasoning_chain: List[str]
    evidence_used: List[Dict[str, Any]]
    uncertainties: List[Dict[str, Any]]
    what_we_dont_know: List[str]
    alternatives_considered: List[Dict[str, Any]]
    what_would_change_decision: List[str]
    limitations: List[str]
    recommendations: List[str]
    level: ExplanationLevel
    timestamp: datetime = field(default_factory=datetime.now)


class ExplainFirstProtocol:
    """
    Protocollo che garantisce spiegazioni complete per ogni output.
    
    Regola dura: In situazioni critiche, NO output senza spiegazione.
    """
    
    # Situazioni che richiedono spiegazione FULL
    CRITICAL_TRIGGERS = [
        'emergency', 'crisis', 'security', 'safety', 'health',
        'financial', 'legal', 'irreversible', 'high_impact'
    ]
    
    def __init__(self):
        self.explanation_history: List[StructuredExplanation] = []
        self.default_level = ExplanationLevel.STANDARD
        self.force_full_explanation = False  # Override per modalità critica
        
    def determine_explanation_level(
        self,
        context: Dict[str, Any],
        confidence: float,
        action_type: Optional[str] = None
    ) -> ExplanationLevel:
        """Determina il livello di spiegazione richiesto"""
        
        # Override manuale
        if self.force_full_explanation:
            return ExplanationLevel.CRITICAL
        
        # Check trigger critici
        context_str = str(context).lower()
        is_critical = any(trigger in context_str for trigger in self.CRITICAL_TRIGGERS)
        
        if is_critical:
            return ExplanationLevel.CRITICAL
        
        # Bassa confidenza richiede più spiegazione
        if confidence < 0.4:
            return ExplanationLevel.CRITICAL
        elif confidence < 0.6:
            return ExplanationLevel.FULL
        elif confidence < 0.75:
            return ExplanationLevel.DETAILED
        
        # Check tipo azione
        if action_type and any(t in action_type.lower() for t in ['execute', 'delete', 'send', 'commit']):
            return ExplanationLevel.DETAILED
        
        return self.default_level
    
    async def generate_explanation(
        self,
        conclusion: str,
        confidence: float,
        context: Dict[str, Any],
        reasoning_steps: Optional[List[str]] = None,
        evidence: Optional[List[Dict[str, Any]]] = None,
        alternatives: Optional[List[Dict[str, Any]]] = None,
        model_info: Optional[Dict[str, Any]] = None
    ) -> StructuredExplanation:
        """
        Genera una spiegazione strutturata completa.
        """
        level = self.determine_explanation_level(context, confidence)
        
        # Costruisci spiegazione
        explanation = StructuredExplanation(
            conclusion=conclusion,
            confidence=confidence,
            reasoning_chain=reasoning_steps or self._infer_reasoning(conclusion, context),
            evidence_used=evidence or [],
            uncertainties=self._identify_uncertainties(context, confidence, evidence),
            what_we_dont_know=self._identify_unknowns(context, evidence),
            alternatives_considered=alternatives or self._generate_alternatives(conclusion, context),
            what_would_change_decision=self._identify_decision_changers(conclusion, context, confidence),
            limitations=self._identify_limitations(model_info, context),
            recommendations=self._generate_recommendations(confidence, level, context),
            level=level
        )
        
        # Valida completezza
        if level.value >= ExplanationLevel.CRITICAL.value:
            self._validate_critical_explanation(explanation)
        
        self.explanation_history.append(explanation)
        
        return explanation
    
    def _infer_reasoning(self, conclusion: str, context: Dict[str, Any]) -> List[str]:
        """Inferisce la catena di ragionamento se non fornita"""
        steps = []
        
        # Step 1: Input analizzato
        if 'input' in context:
            steps.append(f"1. Analisi input: '{str(context['input'])[:100]}...'")
        
        # Step 2: Contesto considerato
        context_keys = [k for k in context.keys() if k not in ['input', 'raw_data']]
        if context_keys:
            steps.append(f"2. Contesto considerato: {', '.join(context_keys[:5])}")
        
        # Step 3: Conclusione
        steps.append(f"3. Conclusione derivata: {conclusion[:100]}")
        
        return steps
    
    def _identify_uncertainties(
        self,
        context: Dict[str, Any],
        confidence: float,
        evidence: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Identifica le incertezze presenti"""
        uncertainties = []
        
        # Incertezza da bassa confidenza
        if confidence < 0.7:
            uncertainties.append({
                'type': UncertaintyType.MODEL_LIMITATION.value,
                'description': f"Confidenza del modello: {confidence:.1%}",
                'impact': 'medio' if confidence > 0.5 else 'alto',
                'mitigation': "Richiedere conferma umana o dati aggiuntivi"
            })
        
        # Incertezza da dati mancanti
        data_quality = context.get('data_quality', 1.0)
        if data_quality < 0.8:
            uncertainties.append({
                'type': UncertaintyType.DATA_INCOMPLETE.value,
                'description': f"Qualità dati: {data_quality:.1%}",
                'impact': 'alto' if data_quality < 0.5 else 'medio',
                'mitigation': "Integrare con fonti aggiuntive"
            })
        
        # Incertezza da evidenze contrastanti
        if evidence and len(evidence) > 1:
            confidences = [e.get('confidence', 0.5) for e in evidence]
            if max(confidences) - min(confidences) > 0.3:
                uncertainties.append({
                    'type': UncertaintyType.CONFLICTING_EVIDENCE.value,
                    'description': "Evidenze con livelli di confidenza discordanti",
                    'impact': 'medio',
                    'mitigation': "Valutare fonte più affidabile"
                })
        
        # Incertezza da contesto ambiguo
        if context.get('ambiguous', False) or 'forse' in str(context).lower():
            uncertainties.append({
                'type': UncertaintyType.CONTEXT_AMBIGUOUS.value,
                'description': "Contesto interpretabile in modi diversi",
                'impact': 'medio',
                'mitigation': "Richiedere chiarimento all'utente"
            })
        
        return uncertainties
    
    def _identify_unknowns(
        self,
        context: Dict[str, Any],
        evidence: Optional[List[Dict[str, Any]]]
    ) -> List[str]:
        """Identifica cosa NON sappiamo"""
        unknowns = []
        
        # Dati temporali
        if 'timestamp' not in context and 'time' not in str(context).lower():
            unknowns.append("Tempistiche precise degli eventi")
        
        # Contesto completo
        if not context.get('full_context'):
            unknowns.append("Contesto completo della situazione")
        
        # Storia precedente
        if 'history' not in context:
            unknowns.append("Eventi precedenti che potrebbero essere rilevanti")
        
        # Impatto futuro
        unknowns.append("Conseguenze a lungo termine di questa decisione")
        
        # Fattori esterni
        unknowns.append("Fattori esterni non monitorati che potrebbero influire")
        
        # Intenzioni umane
        if context.get('user_input'):
            unknowns.append("Intenzione esatta dell'utente dietro la richiesta")
        
        return unknowns[:5]  # Limita a 5 principali
    
    def _generate_alternatives(
        self,
        conclusion: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Genera alternative considerate"""
        alternatives = []
        
        # Alternativa: non fare nulla
        alternatives.append({
            'option': "Non intraprendere azione",
            'pros': ["Nessun rischio di errore attivo", "Più tempo per valutare"],
            'cons': ["Potenziale perdita di opportunità", "Situazione potrebbe evolvere"],
            'confidence': 0.5
        })
        
        # Alternativa: azione parziale
        alternatives.append({
            'option': "Azione parziale/incrementale",
            'pros': ["Rischio ridotto", "Possibilità di correzione"],
            'cons': ["Risultato potenzialmente incompleto"],
            'confidence': 0.6
        })
        
        # Alternativa: escalation
        alternatives.append({
            'option': "Escalare a supervisore umano",
            'pros': ["Decisione condivisa", "Responsabilità distribuita"],
            'cons': ["Tempo aggiuntivo richiesto"],
            'confidence': 0.8
        })
        
        return alternatives
    
    def _identify_decision_changers(
        self,
        conclusion: str,
        context: Dict[str, Any],
        confidence: float
    ) -> List[str]:
        """Identifica cosa cambierebbe la decisione"""
        changers = []
        
        # Nuovi dati
        changers.append("Nuove informazioni che contraddicono le evidenze attuali")
        
        # Cambio contesto
        changers.append("Cambiamento significativo nel contesto o nelle circostanze")
        
        # Feedback umano
        changers.append("Input diretto da un esperto del dominio")
        
        # Confidenza
        if confidence < 0.7:
            changers.append(f"Aumento della confidenza sopra il 70% (attuale: {confidence:.1%})")
        
        # Tempo
        changers.append("Passaggio di tempo che rende obsolete le informazioni attuali")
        
        # Priorità
        changers.append("Cambio nelle priorità o negli obiettivi dell'utente")
        
        return changers
    
    def _identify_limitations(
        self,
        model_info: Optional[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> List[str]:
        """Identifica le limitazioni del modello/sistema"""
        limitations = []
        
        # Limitazioni generali
        limitations.append("Non posso prevedere eventi futuri con certezza")
        limitations.append("La mia conoscenza potrebbe non essere aggiornata")
        limitations.append("Potrei non avere accesso a tutte le informazioni rilevanti")
        
        # Limitazioni specifiche
        if model_info:
            if model_info.get('training_cutoff'):
                limitations.append(f"Dati di training fino a: {model_info['training_cutoff']}")
        
        # Limitazioni contestuali
        if 'external_api' in str(context):
            limitations.append("Dipendenza da servizi esterni che potrebbero non essere disponibili")
        
        return limitations
    
    def _generate_recommendations(
        self,
        confidence: float,
        level: ExplanationLevel,
        context: Dict[str, Any]
    ) -> List[str]:
        """Genera raccomandazioni basate sulla situazione"""
        recommendations = []
        
        if confidence < 0.5:
            recommendations.append("⚠️ Raccomando fortemente revisione umana prima di procedere")
            recommendations.append("Considera di raccogliere più dati prima di decidere")
        elif confidence < 0.7:
            recommendations.append("Procedi con cautela e monitora i risultati")
            recommendations.append("Prepara un piano di rollback")
        
        if level.value >= ExplanationLevel.CRITICAL.value:
            recommendations.append("Documenta la decisione e le motivazioni")
            recommendations.append("Informa le parti interessate")
        
        if context.get('is_irreversible'):
            recommendations.append("⚠️ Azione IRREVERSIBILE - doppia conferma richiesta")
        
        return recommendations
    
    def _validate_critical_explanation(self, explanation: StructuredExplanation):
        """Valida che una spiegazione critica sia completa"""
        required_fields = [
            ('reasoning_chain', 1),
            ('uncertainties', 0),  # Può essere vuoto ma deve esistere
            ('what_we_dont_know', 1),
            ('what_would_change_decision', 1),
            ('recommendations', 1)
        ]
        
        for field, min_items in required_fields:
            value = getattr(explanation, field)
            if len(value) < min_items:
                logger.warning(f"⚠️ Critical explanation missing: {field}")
    
    def format_for_display(
        self,
        explanation: StructuredExplanation,
        format_type: str = 'full'
    ) -> str:
        """Formatta la spiegazione per la visualizzazione"""
        
        if format_type == 'minimal':
            return f"""
**{explanation.conclusion}**
Confidenza: {explanation.confidence:.1%}
"""
        
        elif format_type == 'standard':
            return f"""
## 📊 Analisi

**Conclusione**: {explanation.conclusion}
**Confidenza**: {explanation.confidence:.1%}

### Ragionamento
{chr(10).join(f"- {step}" for step in explanation.reasoning_chain)}

### ⚠️ Incertezze
{chr(10).join(f"- {u['description']}" for u in explanation.uncertainties) or "Nessuna significativa"}
"""
        
        else:  # full
            return f"""
# 📊 Analisi Completa GIDEON

## Conclusione
**{explanation.conclusion}**

📈 **Confidenza**: {explanation.confidence:.1%}
📅 **Timestamp**: {explanation.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

---

## 🧠 Ragionamento
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(explanation.reasoning_chain))}

---

## 📚 Evidenze Utilizzate
{chr(10).join(f"- {e.get('source', 'N/A')}: {e.get('summary', 'N/A')}" for e in explanation.evidence_used) or "Nessuna evidenza specifica registrata"}

---

## ⚠️ Incertezze Identificate
{chr(10).join(f"- **{u['type']}**: {u['description']} (Impatto: {u['impact']})" for u in explanation.uncertainties) or "Nessuna incertezza significativa"}

---

## ❓ Cosa NON Sappiamo
{chr(10).join(f"- {unknown}" for unknown in explanation.what_we_dont_know)}

---

## 🔄 Alternative Considerate
{chr(10).join(f"- **{a['option']}**: Pro: {', '.join(a['pros'][:2])} | Contro: {', '.join(a['cons'][:2])}" for a in explanation.alternatives_considered)}

---

## 🔑 Cosa Cambierebbe la Decisione
{chr(10).join(f"- {changer}" for changer in explanation.what_would_change_decision)}

---

## 📋 Limitazioni
{chr(10).join(f"- {lim}" for lim in explanation.limitations)}

---

## 💡 Raccomandazioni
{chr(10).join(f"- {rec}" for rec in explanation.recommendations)}

---
*Livello spiegazione: {explanation.level.name}*
"""
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Ritorna la storia delle spiegazioni"""
        return [
            {
                'conclusion': e.conclusion[:100],
                'confidence': e.confidence,
                'level': e.level.name,
                'timestamp': e.timestamp.isoformat(),
                'uncertainties_count': len(e.uncertainties)
            }
            for e in self.explanation_history[-limit:]
        ]
