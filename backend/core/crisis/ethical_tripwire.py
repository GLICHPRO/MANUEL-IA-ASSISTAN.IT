"""
⚖️ ETHICAL TRIPWIRE SYSTEM
===========================
Un layer che rileva azioni eticamente ambigue:
- Discriminazione
- Disinformazione
- Violazioni privacy
- Risposte manipolative

Quando scatta: blocca + log + notifica
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Callable
from enum import Enum
import logging
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


class EthicalViolationType(Enum):
    """Tipi di violazioni etiche"""
    DISCRIMINATION = "discrimination"
    PRIVACY_VIOLATION = "privacy_violation"
    DISINFORMATION = "disinformation"
    MANIPULATION = "manipulation"
    HARM_PROMOTION = "harm_promotion"
    ILLEGAL_ACTIVITY = "illegal_activity"
    CONSENT_VIOLATION = "consent_violation"
    EXPLOITATION = "exploitation"
    DECEPTION = "deception"
    BIAS_AMPLIFICATION = "bias_amplification"


class TripwireAction(Enum):
    """Azioni quando scatta un tripwire"""
    ALLOW = "allow"               # Permetti (nessuna violazione)
    FLAG = "flag"                 # Segnala per review
    WARN = "warn"                 # Avvisa ma permetti
    MODIFY = "modify"            # Modifica output
    BLOCK = "block"              # Blocca completamente
    ESCALATE = "escalate"        # Escala a supervisore umano


class SeverityLevel(Enum):
    """Livelli di severità violazione"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EXTREME = 5


@dataclass
class EthicalViolation:
    """Violazione etica rilevata"""
    violation_type: EthicalViolationType
    severity: SeverityLevel
    description: str
    evidence: List[str]
    affected_groups: List[str]
    confidence: float
    mitigation: Optional[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TripwireResult:
    """Risultato analisi tripwire"""
    triggered: bool
    action: TripwireAction
    violations: List[EthicalViolation]
    risk_score: float  # 0-100
    explanation: str
    modifications: Optional[Dict[str, Any]]
    audit_record: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class EthicalTripwireSystem:
    """
    Sistema di tripwire etici per GIDEON.
    
    Rileva azioni eticamente ambigue:
    - Discriminazione
    - Disinformazione  
    - Violazioni privacy
    - Risposte manipolative
    
    Quando scatta: blocca + log + notifica
    """
    
    # Pattern discriminatori (da estendere)
    DISCRIMINATION_PATTERNS = [
        r'\b(tutti i|sempre i|mai i)\s+\w+\s+(sono|hanno|fanno)\b',
        r'\b(razza|etnia|religione)\s+\w+\s+(è|sono)\s+(inferiore|peggiore|stupid)',
        r'\b(non assumere|non fidarti di|evita)\s+\w+\s+(perché|in quanto)\s+\w+\b',
    ]
    
    # Pattern disinformazione
    DISINFORMATION_PATTERNS = [
        r'\b(è dimostrato che|tutti sanno che|la scienza dice che)\b',
        r'\b(cure miracolose?|guarigione garantita|100% sicuro)\b',
        r'\b(complotto|cospirazione|non vogliono che tu sappia)\b',
    ]
    
    # Pattern privacy
    PRIVACY_PATTERNS = [
        r'\b(password|carta di credito|numero\s+di\s+telefono|codice\s+fiscale)\b',
        r'\b(dati personali|informazioni private|senza consenso)\b',
        r'\b(tracciare|monitorare|spiare)\s+\w+\s+(senza)\b',
    ]
    
    # Pattern manipolazione
    MANIPULATION_PATTERNS = [
        r'\b(devi assolutamente|non hai scelta|solo io posso)\b',
        r'\b(paura|terrore|pericolo imminente)\b.*\b(agisci subito|ora|immediatamente)\b',
        r'\b(tutti lo fanno|nessuno ti crederà|solo tu)\b',
    ]
    
    # Gruppi protetti
    PROTECTED_GROUPS = [
        'donne', 'uomini', 'bambini', 'anziani',
        'disabili', 'immigrati', 'rifugiati',
        'lgbt', 'minoranze', 'religioni',
        'etnie', 'nazionalità'
    ]
    
    def __init__(self):
        self.violations_log: List[EthicalViolation] = []
        self.tripwire_history: List[TripwireResult] = []
        self.custom_patterns: Dict[EthicalViolationType, List[str]] = defaultdict(list)
        self.custom_checks: List[Callable[[str, Dict], Optional[EthicalViolation]]] = []
        self.blocked_outputs: int = 0
        self.flagged_outputs: int = 0
        
        # Soglie
        self.risk_threshold_flag = 20
        self.risk_threshold_warn = 40
        self.risk_threshold_block = 60
        self.risk_threshold_escalate = 80
        
        # Compila pattern
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compila i pattern regex"""
        self.compiled_patterns = {
            EthicalViolationType.DISCRIMINATION: [
                re.compile(p, re.IGNORECASE) for p in self.DISCRIMINATION_PATTERNS
            ],
            EthicalViolationType.DISINFORMATION: [
                re.compile(p, re.IGNORECASE) for p in self.DISINFORMATION_PATTERNS
            ],
            EthicalViolationType.PRIVACY_VIOLATION: [
                re.compile(p, re.IGNORECASE) for p in self.PRIVACY_PATTERNS
            ],
            EthicalViolationType.MANIPULATION: [
                re.compile(p, re.IGNORECASE) for p in self.MANIPULATION_PATTERNS
            ]
        }
    
    async def analyze(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        action_type: Optional[str] = None
    ) -> TripwireResult:
        """
        Analizza contenuto per violazioni etiche.
        
        Args:
            content: Testo da analizzare
            context: Contesto dell'azione
            action_type: Tipo di azione
            
        Returns:
            TripwireResult con analisi completa
        """
        
        logger.info("⚖️ Analisi etica in corso...")
        
        violations = []
        
        # Check pattern predefiniti
        for vtype, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(content)
                if matches:
                    violation = self._create_violation(
                        vtype,
                        matches,
                        content,
                        context
                    )
                    violations.append(violation)
        
        # Check pattern custom
        for vtype, patterns in self.custom_patterns.items():
            for pattern_str in patterns:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                matches = pattern.findall(content)
                if matches:
                    violation = self._create_violation(
                        vtype,
                        matches,
                        content,
                        context
                    )
                    violations.append(violation)
        
        # Check custom
        for check_func in self.custom_checks:
            try:
                result = check_func(content, context or {})
                if result:
                    violations.append(result)
            except Exception as e:
                logger.error(f"Custom check failed: {e}")
        
        # Check contenuto sensibile
        sensitive_violations = self._check_sensitive_content(content, context)
        violations.extend(sensitive_violations)
        
        # Check gruppi protetti
        group_violations = self._check_protected_groups(content, context)
        violations.extend(group_violations)
        
        # Calcola risk score
        risk_score = self._calculate_risk_score(violations)
        
        # Determina azione
        action = self._determine_action(risk_score, violations)
        
        # Genera spiegazione
        explanation = self._generate_explanation(violations, action)
        
        # Genera modifiche se necessario
        modifications = self._generate_modifications(content, violations) if action == TripwireAction.MODIFY else None
        
        # Crea record audit
        audit_record = self._create_audit_record(content, violations, action, context)
        
        result = TripwireResult(
            triggered=len(violations) > 0,
            action=action,
            violations=violations,
            risk_score=risk_score,
            explanation=explanation,
            modifications=modifications,
            audit_record=audit_record
        )
        
        # Log
        self.tripwire_history.append(result)
        self.violations_log.extend(violations)
        
        if action == TripwireAction.BLOCK:
            self.blocked_outputs += 1
            logger.warning(f"⚖️ OUTPUT BLOCCATO - Risk: {risk_score:.1f}")
        elif action in [TripwireAction.FLAG, TripwireAction.WARN]:
            self.flagged_outputs += 1
        
        return result
    
    def _create_violation(
        self,
        vtype: EthicalViolationType,
        matches: List[str],
        content: str,
        context: Optional[Dict[str, Any]]
    ) -> EthicalViolation:
        """Crea oggetto violazione"""
        
        # Determina severità
        severity = self._determine_severity(vtype, matches, content)
        
        # Identifica gruppi affetti
        affected = self._identify_affected_groups(content, vtype)
        
        # Calcola confidenza
        confidence = min(0.95, 0.5 + len(matches) * 0.1)
        
        # Genera descrizione
        descriptions = {
            EthicalViolationType.DISCRIMINATION: "Contenuto potenzialmente discriminatorio rilevato",
            EthicalViolationType.DISINFORMATION: "Possibile disinformazione o claim non verificabile",
            EthicalViolationType.PRIVACY_VIOLATION: "Potenziale violazione della privacy",
            EthicalViolationType.MANIPULATION: "Pattern manipolativo rilevato",
            EthicalViolationType.HARM_PROMOTION: "Contenuto potenzialmente dannoso",
            EthicalViolationType.DECEPTION: "Possibile intento ingannevole"
        }
        
        return EthicalViolation(
            violation_type=vtype,
            severity=severity,
            description=descriptions.get(vtype, "Violazione etica rilevata"),
            evidence=matches[:5],  # Max 5 evidenze
            affected_groups=affected,
            confidence=confidence,
            mitigation=self._suggest_mitigation(vtype)
        )
    
    def _determine_severity(
        self,
        vtype: EthicalViolationType,
        matches: List[str],
        content: str
    ) -> SeverityLevel:
        """Determina severità della violazione"""
        
        # Tipi intrinsecamente gravi
        critical_types = [
            EthicalViolationType.ILLEGAL_ACTIVITY,
            EthicalViolationType.HARM_PROMOTION,
            EthicalViolationType.EXPLOITATION
        ]
        
        if vtype in critical_types:
            return SeverityLevel.CRITICAL
        
        # Basato su numero match
        if len(matches) >= 5:
            return SeverityLevel.HIGH
        elif len(matches) >= 3:
            return SeverityLevel.MEDIUM
        elif len(matches) >= 1:
            return SeverityLevel.LOW
        
        return SeverityLevel.LOW
    
    def _identify_affected_groups(
        self,
        content: str,
        vtype: EthicalViolationType
    ) -> List[str]:
        """Identifica gruppi potenzialmente affetti"""
        
        affected = []
        content_lower = content.lower()
        
        for group in self.PROTECTED_GROUPS:
            if group in content_lower:
                affected.append(group)
        
        return affected[:5]  # Max 5
    
    def _suggest_mitigation(self, vtype: EthicalViolationType) -> str:
        """Suggerisce mitigazione per tipo violazione"""
        
        mitigations = {
            EthicalViolationType.DISCRIMINATION: "Riformulare senza generalizzazioni su gruppi specifici",
            EthicalViolationType.DISINFORMATION: "Aggiungere fonti verificabili o qualificare l'informazione",
            EthicalViolationType.PRIVACY_VIOLATION: "Rimuovere o anonimizzare dati personali",
            EthicalViolationType.MANIPULATION: "Riformulare in modo neutro senza pressione emotiva",
            EthicalViolationType.HARM_PROMOTION: "Rimuovere contenuto e consultare linee guida",
            EthicalViolationType.DECEPTION: "Essere trasparenti sull'intento",
            EthicalViolationType.BIAS_AMPLIFICATION: "Bilanciare prospettive diverse"
        }
        
        return mitigations.get(vtype, "Rivedere il contenuto per conformità etica")
    
    def _check_sensitive_content(
        self,
        content: str,
        context: Optional[Dict[str, Any]]
    ) -> List[EthicalViolation]:
        """Check per contenuto sensibile"""
        
        violations = []
        content_lower = content.lower()
        
        # Check armi/violenza
        violence_keywords = ['uccidere', 'ammazzare', 'ferire', 'arma', 'bomba', 'esplosivo']
        if any(k in content_lower for k in violence_keywords):
            # Verifica contesto (potrebbe essere educativo)
            if not context or not context.get('educational_context'):
                violations.append(EthicalViolation(
                    violation_type=EthicalViolationType.HARM_PROMOTION,
                    severity=SeverityLevel.HIGH,
                    description="Contenuto relativo a violenza/armi rilevato",
                    evidence=[k for k in violence_keywords if k in content_lower],
                    affected_groups=['public_safety'],
                    confidence=0.7,
                    mitigation="Verificare contesto e necessità"
                ))
        
        # Check sostanze
        substance_keywords = ['droga', 'stupefacente', 'overdose', 'come farsi']
        if any(k in content_lower for k in substance_keywords):
            violations.append(EthicalViolation(
                violation_type=EthicalViolationType.HARM_PROMOTION,
                severity=SeverityLevel.HIGH,
                description="Contenuto relativo a sostanze illegali",
                evidence=[k for k in substance_keywords if k in content_lower],
                affected_groups=['vulnerabili'],
                confidence=0.7,
                mitigation="Indirizzare a risorse di aiuto appropriate"
            ))
        
        return violations
    
    def _check_protected_groups(
        self,
        content: str,
        context: Optional[Dict[str, Any]]
    ) -> List[EthicalViolation]:
        """Check specifico per gruppi protetti"""
        
        violations = []
        content_lower = content.lower()
        
        # Check generalizzazioni
        generalization_patterns = [
            (r'(tutti|tutte)\s+(?:i|le|gli)\s+(\w+)', 'Generalizzazione su gruppo'),
            (r'(\w+)\s+(sono\s+sempre|non\s+sanno|non\s+possono)', 'Stereotipo'),
        ]
        
        for pattern, desc in generalization_patterns:
            matches = re.findall(pattern, content_lower)
            for match in matches:
                # Verifica se riguarda gruppo protetto
                match_str = ' '.join(match) if isinstance(match, tuple) else match
                for group in self.PROTECTED_GROUPS:
                    if group in match_str:
                        violations.append(EthicalViolation(
                            violation_type=EthicalViolationType.DISCRIMINATION,
                            severity=SeverityLevel.MEDIUM,
                            description=f"{desc} riguardante '{group}'",
                            evidence=[match_str],
                            affected_groups=[group],
                            confidence=0.6,
                            mitigation="Evitare generalizzazioni su gruppi specifici"
                        ))
        
        return violations
    
    def _calculate_risk_score(self, violations: List[EthicalViolation]) -> float:
        """Calcola risk score complessivo"""
        
        if not violations:
            return 0.0
        
        total = 0.0
        
        for v in violations:
            # Base: severità
            base = v.severity.value * 15
            
            # Peso confidenza
            weighted = base * v.confidence
            
            # Bonus tipo critico
            critical_types = [
                EthicalViolationType.HARM_PROMOTION,
                EthicalViolationType.ILLEGAL_ACTIVITY,
                EthicalViolationType.EXPLOITATION
            ]
            if v.violation_type in critical_types:
                weighted *= 1.5
            
            total += weighted
        
        return min(100, total)
    
    def _determine_action(
        self,
        risk_score: float,
        violations: List[EthicalViolation]
    ) -> TripwireAction:
        """Determina azione appropriata"""
        
        # Check violazioni critiche
        critical_violations = [
            v for v in violations 
            if v.severity in [SeverityLevel.CRITICAL, SeverityLevel.EXTREME]
        ]
        if critical_violations:
            return TripwireAction.ESCALATE
        
        # Basato su score
        if risk_score >= self.risk_threshold_escalate:
            return TripwireAction.ESCALATE
        elif risk_score >= self.risk_threshold_block:
            return TripwireAction.BLOCK
        elif risk_score >= self.risk_threshold_warn:
            return TripwireAction.WARN
        elif risk_score >= self.risk_threshold_flag:
            return TripwireAction.FLAG
        
        return TripwireAction.ALLOW
    
    def _generate_explanation(
        self,
        violations: List[EthicalViolation],
        action: TripwireAction
    ) -> str:
        """Genera spiegazione dell'analisi"""
        
        if not violations:
            return "✅ Nessuna violazione etica rilevata."
        
        parts = [f"⚖️ Rilevate {len(violations)} potenziali violazioni etiche:"]
        
        for v in violations[:3]:  # Top 3
            parts.append(f"- {v.violation_type.value}: {v.description} (confidenza: {v.confidence:.1%})")
        
        action_explanations = {
            TripwireAction.ALLOW: "Contenuto permesso con monitoring",
            TripwireAction.FLAG: "Contenuto segnalato per review",
            TripwireAction.WARN: "Utente avvisato, contenuto permesso",
            TripwireAction.MODIFY: "Contenuto modificato per compliance",
            TripwireAction.BLOCK: "Contenuto BLOCCATO",
            TripwireAction.ESCALATE: "ESCALATO a supervisore umano"
        }
        
        parts.append(f"\n📋 Azione: {action_explanations.get(action, action.value)}")
        
        return "\n".join(parts)
    
    def _generate_modifications(
        self,
        content: str,
        violations: List[EthicalViolation]
    ) -> Dict[str, Any]:
        """Genera modifiche suggerite"""
        
        modifications = {
            'original_length': len(content),
            'suggested_changes': [],
            'redacted_content': content
        }
        
        for v in violations:
            for evidence in v.evidence:
                # Suggerisci rimozione/sostituzione
                modifications['suggested_changes'].append({
                    'type': 'redact',
                    'original': evidence,
                    'reason': v.violation_type.value,
                    'suggestion': f"[{v.violation_type.value.upper()} RIMOSSO]"
                })
                
                # Applica redaction
                modifications['redacted_content'] = modifications['redacted_content'].replace(
                    evidence, 
                    f"[REDACTED]"
                )
        
        return modifications
    
    def _create_audit_record(
        self,
        content: str,
        violations: List[EthicalViolation],
        action: TripwireAction,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Crea record per audit trail"""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'content_hash': hash(content),
            'content_preview': content[:100] + '...' if len(content) > 100 else content,
            'violations_count': len(violations),
            'violation_types': [v.violation_type.value for v in violations],
            'max_severity': max((v.severity.value for v in violations), default=0),
            'action_taken': action.value,
            'user_id': context.get('user_id') if context else None,
            'session_id': context.get('session_id') if context else None,
            'reviewed': False,
            'reviewer': None,
            'review_outcome': None
        }
    
    def add_custom_pattern(
        self,
        violation_type: EthicalViolationType,
        pattern: str
    ):
        """Aggiunge pattern custom"""
        self.custom_patterns[violation_type].append(pattern)
        logger.info(f"⚖️ Pattern custom aggiunto per {violation_type.value}")
    
    def add_custom_check(
        self,
        check_func: Callable[[str, Dict], Optional[EthicalViolation]]
    ):
        """Aggiunge check custom"""
        self.custom_checks.append(check_func)
        logger.info("⚖️ Check custom aggiunto")
    
    def get_stats(self) -> Dict[str, Any]:
        """Ritorna statistiche"""
        return {
            'total_analyses': len(self.tripwire_history),
            'total_violations': len(self.violations_log),
            'blocked_outputs': self.blocked_outputs,
            'flagged_outputs': self.flagged_outputs,
            'violations_by_type': self._count_by_type(),
            'recent_violations': [
                {
                    'type': v.violation_type.value,
                    'severity': v.severity.value,
                    'timestamp': v.timestamp.isoformat()
                }
                for v in self.violations_log[-10:]
            ]
        }
    
    def _count_by_type(self) -> Dict[str, int]:
        """Conta violazioni per tipo"""
        counts = defaultdict(int)
        for v in self.violations_log:
            counts[v.violation_type.value] += 1
        return dict(counts)
    
    def format_result(self, result: TripwireResult) -> str:
        """Formatta risultato per visualizzazione"""
        
        action_emoji = {
            TripwireAction.ALLOW: "✅",
            TripwireAction.FLAG: "🏳️",
            TripwireAction.WARN: "⚠️",
            TripwireAction.MODIFY: "✏️",
            TripwireAction.BLOCK: "🛑",
            TripwireAction.ESCALATE: "🚨"
        }
        
        emoji = action_emoji.get(result.action, "❓")
        
        return f"""
# ⚖️ Ethical Tripwire Analysis

{emoji} **Azione**: {result.action.value.upper()}
📊 **Risk Score**: {result.risk_score:.1f}/100

---

## Spiegazione
{result.explanation}

---

## Violazioni Rilevate ({len(result.violations)})
{chr(10).join(f"- [{v.severity.name}] {v.violation_type.value}: {v.description}" for v in result.violations) or "Nessuna"}

---

## Suggerimenti
{chr(10).join(f"- {v.mitigation}" for v in result.violations if v.mitigation) or "Nessuno"}

*Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}*
"""
