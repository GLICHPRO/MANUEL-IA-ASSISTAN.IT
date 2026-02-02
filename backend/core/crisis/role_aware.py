"""
🎭 ROLE-AWARE OUTPUT ADAPTER
=============================
Gli output devono adattarsi al ruolo dell'utente:

- Manager: Summary, KPI, rischi
- Tecnico: Dettagli, log, cause root
- Operatore: Passi chiari, cosa fare ora
- Auditor: Tracciabilità, compliance, cronologia

Stesso dato → linguaggio diverso
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)


class UserRole(Enum):
    """Ruoli utente supportati"""
    MANAGER = "manager"
    TECHNICAL = "technical"
    OPERATOR = "operator"
    AUDITOR = "auditor"
    EXECUTIVE = "executive"
    SECURITY = "security"
    DEVELOPER = "developer"
    END_USER = "end_user"
    UNKNOWN = "unknown"


class CommunicationStyle(Enum):
    """Stili di comunicazione"""
    EXECUTIVE_SUMMARY = "executive"      # Breve, impatto business
    TECHNICAL_DETAIL = "technical"       # Dettagliato, cause root
    OPERATIONAL = "operational"          # Azioni chiare, step-by-step
    COMPLIANCE = "compliance"            # Tracciabile, audit-ready
    CONVERSATIONAL = "conversational"    # Amichevole, semplice


@dataclass
class RoleProfile:
    """Profilo dettagliato di un ruolo"""
    role: UserRole
    preferred_style: CommunicationStyle
    include_technical_details: bool
    include_business_impact: bool
    include_action_items: bool
    include_audit_trail: bool
    max_summary_length: int  # caratteri
    terminology_level: int   # 1=basic, 5=expert
    show_confidence_scores: bool
    show_alternatives: bool
    show_raw_data: bool


# Profili predefiniti per ruolo
ROLE_PROFILES: Dict[UserRole, RoleProfile] = {
    UserRole.MANAGER: RoleProfile(
        role=UserRole.MANAGER,
        preferred_style=CommunicationStyle.EXECUTIVE_SUMMARY,
        include_technical_details=False,
        include_business_impact=True,
        include_action_items=True,
        include_audit_trail=False,
        max_summary_length=500,
        terminology_level=2,
        show_confidence_scores=True,
        show_alternatives=True,
        show_raw_data=False
    ),
    UserRole.TECHNICAL: RoleProfile(
        role=UserRole.TECHNICAL,
        preferred_style=CommunicationStyle.TECHNICAL_DETAIL,
        include_technical_details=True,
        include_business_impact=False,
        include_action_items=True,
        include_audit_trail=True,
        max_summary_length=2000,
        terminology_level=5,
        show_confidence_scores=True,
        show_alternatives=True,
        show_raw_data=True
    ),
    UserRole.OPERATOR: RoleProfile(
        role=UserRole.OPERATOR,
        preferred_style=CommunicationStyle.OPERATIONAL,
        include_technical_details=False,
        include_business_impact=False,
        include_action_items=True,
        include_audit_trail=False,
        max_summary_length=800,
        terminology_level=3,
        show_confidence_scores=False,
        show_alternatives=False,
        show_raw_data=False
    ),
    UserRole.AUDITOR: RoleProfile(
        role=UserRole.AUDITOR,
        preferred_style=CommunicationStyle.COMPLIANCE,
        include_technical_details=True,
        include_business_impact=True,
        include_action_items=True,
        include_audit_trail=True,
        max_summary_length=3000,
        terminology_level=4,
        show_confidence_scores=True,
        show_alternatives=True,
        show_raw_data=True
    ),
    UserRole.EXECUTIVE: RoleProfile(
        role=UserRole.EXECUTIVE,
        preferred_style=CommunicationStyle.EXECUTIVE_SUMMARY,
        include_technical_details=False,
        include_business_impact=True,
        include_action_items=True,
        include_audit_trail=False,
        max_summary_length=300,
        terminology_level=1,
        show_confidence_scores=False,
        show_alternatives=True,
        show_raw_data=False
    ),
    UserRole.SECURITY: RoleProfile(
        role=UserRole.SECURITY,
        preferred_style=CommunicationStyle.TECHNICAL_DETAIL,
        include_technical_details=True,
        include_business_impact=True,
        include_action_items=True,
        include_audit_trail=True,
        max_summary_length=1500,
        terminology_level=5,
        show_confidence_scores=True,
        show_alternatives=True,
        show_raw_data=True
    ),
    UserRole.DEVELOPER: RoleProfile(
        role=UserRole.DEVELOPER,
        preferred_style=CommunicationStyle.TECHNICAL_DETAIL,
        include_technical_details=True,
        include_business_impact=False,
        include_action_items=True,
        include_audit_trail=False,
        max_summary_length=2000,
        terminology_level=5,
        show_confidence_scores=True,
        show_alternatives=True,
        show_raw_data=True
    ),
    UserRole.END_USER: RoleProfile(
        role=UserRole.END_USER,
        preferred_style=CommunicationStyle.CONVERSATIONAL,
        include_technical_details=False,
        include_business_impact=False,
        include_action_items=True,
        include_audit_trail=False,
        max_summary_length=400,
        terminology_level=1,
        show_confidence_scores=False,
        show_alternatives=False,
        show_raw_data=False
    ),
    UserRole.UNKNOWN: RoleProfile(
        role=UserRole.UNKNOWN,
        preferred_style=CommunicationStyle.CONVERSATIONAL,
        include_technical_details=False,
        include_business_impact=True,
        include_action_items=True,
        include_audit_trail=False,
        max_summary_length=600,
        terminology_level=2,
        show_confidence_scores=True,
        show_alternatives=True,
        show_raw_data=False
    )
}


# Dizionario terminologia per livello
TERMINOLOGY_TRANSLATIONS = {
    # Termine tecnico -> {livello: versione semplificata}
    "latency": {1: "tempo di risposta", 2: "ritardo", 3: "latenza", 5: "latency (ms)"},
    "throughput": {1: "velocità", 2: "capacità di elaborazione", 3: "throughput", 5: "throughput (ops/s)"},
    "CPU utilization": {1: "uso del processore", 2: "carico CPU", 5: "CPU utilization"},
    "memory leak": {1: "problema di memoria", 2: "perdita di memoria", 5: "memory leak"},
    "timeout": {1: "tempo scaduto", 2: "timeout", 5: "timeout (TTL exceeded)"},
    "authentication failure": {1: "accesso negato", 2: "errore di accesso", 5: "authentication failure"},
    "SQL injection": {1: "attacco al database", 3: "SQL injection", 5: "SQLi attack vector"},
    "XSS": {1: "codice malevolo", 3: "cross-site scripting", 5: "XSS (reflected/stored)"},
    "DDoS": {1: "sovraccarico server", 2: "attacco traffico", 3: "DDoS", 5: "DDoS (volumetric/L7)"},
    "anomaly": {1: "comportamento strano", 2: "anomalia", 5: "statistical anomaly"},
    "root cause": {1: "causa principale", 3: "root cause", 5: "root cause analysis"},
    "remediation": {1: "soluzione", 2: "correzione", 5: "remediation steps"},
    "mitigation": {1: "protezione", 2: "mitigazione", 5: "mitigation controls"},
}


@dataclass
class AdaptedOutput:
    """Output adattato al ruolo"""
    role: UserRole
    profile: RoleProfile
    summary: str
    details: Optional[str]
    action_items: List[str]
    metrics: Dict[str, Any]
    audit_info: Optional[Dict[str, Any]]
    raw_data: Optional[Dict[str, Any]]
    timestamp: datetime = field(default_factory=datetime.now)


class RoleAwareOutputAdapter:
    """
    Adattatore che trasforma gli output in base al ruolo dell'utente.
    
    Stesso dato, linguaggio diverso per:
    - Manager: Summary, KPI, rischi
    - Tecnico: Dettagli, log, cause root
    - Operatore: Passi chiari, cosa fare ora
    - Auditor: Tracciabilità, compliance, cronologia
    """
    
    def __init__(self):
        self.role_profiles = ROLE_PROFILES.copy()
        self.terminology = TERMINOLOGY_TRANSLATIONS.copy()
        self.current_role: UserRole = UserRole.UNKNOWN
        self.adaptation_history: List[AdaptedOutput] = []
        self.custom_adapters: Dict[UserRole, Callable] = {}
    
    def set_role(self, role: UserRole):
        """Imposta il ruolo corrente"""
        self.current_role = role
        logger.info(f"🎭 Role impostato: {role.value}")
    
    def detect_role_from_context(self, context: Dict[str, Any]) -> UserRole:
        """Rileva automaticamente il ruolo dal contesto"""
        
        # Check esplicito
        if 'user_role' in context:
            role_str = context['user_role'].lower()
            for role in UserRole:
                if role.value in role_str:
                    return role
        
        # Check da query/input
        query = str(context.get('query', '') or context.get('input', '')).lower()
        
        # Pattern per ruoli
        role_patterns = {
            UserRole.EXECUTIVE: ['board', 'executive summary', 'strategic', 'bottom line', 'roi'],
            UserRole.MANAGER: ['team', 'kpi', 'deadline', 'budget', 'resource', 'priority'],
            UserRole.TECHNICAL: ['debug', 'stack trace', 'error log', 'root cause', 'api', 'code'],
            UserRole.OPERATOR: ['how do i', 'step by step', 'what should i do', 'procedure'],
            UserRole.AUDITOR: ['compliance', 'audit', 'policy', 'regulation', 'evidence'],
            UserRole.SECURITY: ['threat', 'vulnerability', 'attack', 'breach', 'cve'],
            UserRole.DEVELOPER: ['implement', 'function', 'class', 'refactor', 'test'],
        }
        
        for role, patterns in role_patterns.items():
            if any(p in query for p in patterns):
                return role
        
        return UserRole.UNKNOWN
    
    def adapt_terminology(self, text: str, level: int) -> str:
        """Adatta la terminologia al livello dell'utente"""
        adapted = text
        
        for term, translations in self.terminology.items():
            if term.lower() in adapted.lower():
                # Trova la traduzione appropriata
                target_level = min(level, max(translations.keys()))
                while target_level not in translations and target_level > 0:
                    target_level -= 1
                
                if target_level in translations:
                    # Sostituisci preservando il case
                    pattern = re.compile(re.escape(term), re.IGNORECASE)
                    adapted = pattern.sub(translations[target_level], adapted)
        
        return adapted
    
    async def adapt_output(
        self,
        raw_output: Dict[str, Any],
        role: Optional[UserRole] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AdaptedOutput:
        """
        Adatta l'output al ruolo specificato.
        """
        # Determina ruolo
        if role is None:
            if context:
                role = self.detect_role_from_context(context)
            else:
                role = self.current_role
        
        profile = self.role_profiles.get(role, self.role_profiles[UserRole.UNKNOWN])
        
        # Usa adapter custom se disponibile
        if role in self.custom_adapters:
            return self.custom_adapters[role](raw_output, profile)
        
        # Costruisci output adattato
        adapted = AdaptedOutput(
            role=role,
            profile=profile,
            summary=self._generate_summary(raw_output, profile),
            details=self._generate_details(raw_output, profile) if profile.include_technical_details else None,
            action_items=self._generate_action_items(raw_output, profile) if profile.include_action_items else [],
            metrics=self._extract_metrics(raw_output, profile),
            audit_info=self._generate_audit_info(raw_output, context) if profile.include_audit_trail else None,
            raw_data=raw_output if profile.show_raw_data else None
        )
        
        self.adaptation_history.append(adapted)
        
        return adapted
    
    def _generate_summary(self, raw_output: Dict[str, Any], profile: RoleProfile) -> str:
        """Genera summary adattato al profilo"""
        
        base_message = raw_output.get('message', raw_output.get('result', str(raw_output)))
        
        # Adatta terminologia
        summary = self.adapt_terminology(str(base_message), profile.terminology_level)
        
        # Tronca se necessario
        if len(summary) > profile.max_summary_length:
            summary = summary[:profile.max_summary_length - 3] + "..."
        
        # Aggiungi contesto business se richiesto
        if profile.include_business_impact:
            impact = raw_output.get('business_impact', raw_output.get('impact'))
            if impact:
                summary += f"\n\n📊 **Impatto Business**: {impact}"
        
        # Aggiungi confidenza se richiesta
        if profile.show_confidence_scores:
            confidence = raw_output.get('confidence', raw_output.get('score'))
            if confidence:
                summary += f"\n🎯 **Confidenza**: {confidence:.1%}" if isinstance(confidence, float) else f"\n🎯 **Confidenza**: {confidence}"
        
        return summary
    
    def _generate_details(self, raw_output: Dict[str, Any], profile: RoleProfile) -> str:
        """Genera dettagli tecnici"""
        
        details_parts = []
        
        # Stack trace o error details
        if 'error' in raw_output or 'exception' in raw_output:
            error = raw_output.get('error', raw_output.get('exception'))
            details_parts.append(f"**Errore**: {error}")
        
        # Log entries
        if 'logs' in raw_output:
            details_parts.append("**Log**:")
            for log in raw_output['logs'][-10:]:  # Ultimi 10
                details_parts.append(f"  - {log}")
        
        # Root cause
        if 'root_cause' in raw_output:
            details_parts.append(f"\n**Root Cause**: {raw_output['root_cause']}")
        
        # Technical metrics
        if 'metrics' in raw_output:
            details_parts.append("\n**Metriche Tecniche**:")
            for k, v in raw_output['metrics'].items():
                details_parts.append(f"  - {k}: {v}")
        
        # Raw analysis
        if 'analysis' in raw_output:
            details_parts.append(f"\n**Analisi**: {raw_output['analysis']}")
        
        return "\n".join(details_parts)
    
    def _generate_action_items(self, raw_output: Dict[str, Any], profile: RoleProfile) -> List[str]:
        """Genera action items adattati"""
        
        actions = []
        
        # Azioni esistenti
        if 'actions' in raw_output:
            actions.extend(raw_output['actions'])
        elif 'recommendations' in raw_output:
            actions.extend(raw_output['recommendations'])
        
        # Adatta al livello
        adapted_actions = []
        for action in actions:
            # Semplifica per operatori
            if profile.preferred_style == CommunicationStyle.OPERATIONAL:
                # Numera e rendi imperativi
                action = action.replace("Si consiglia di", "")
                action = action.replace("Potrebbe essere utile", "")
                action = action.strip()
                if action:
                    adapted_actions.append(action)
            else:
                adapted_actions.append(action)
        
        # Aggiungi priorità per manager
        if profile.role == UserRole.MANAGER:
            adapted_actions = [f"[P{i+1}] {a}" for i, a in enumerate(adapted_actions[:5])]
        
        return adapted_actions
    
    def _extract_metrics(self, raw_output: Dict[str, Any], profile: RoleProfile) -> Dict[str, Any]:
        """Estrae metriche rilevanti per il ruolo"""
        
        metrics = {}
        
        # Metriche business per manager/executive
        if profile.include_business_impact:
            business_metrics = ['revenue_impact', 'cost', 'risk_score', 'priority', 'affected_users']
            for m in business_metrics:
                if m in raw_output:
                    metrics[m] = raw_output[m]
        
        # Metriche tecniche per technical/developer
        if profile.include_technical_details:
            tech_metrics = ['latency', 'throughput', 'error_rate', 'cpu', 'memory', 'response_time']
            for m in tech_metrics:
                if m in raw_output:
                    metrics[m] = raw_output[m]
        
        # Metriche compliance per auditor
        if profile.include_audit_trail:
            compliance_metrics = ['compliance_score', 'policy_violations', 'risk_level']
            for m in compliance_metrics:
                if m in raw_output:
                    metrics[m] = raw_output[m]
        
        return metrics
    
    def _generate_audit_info(
        self,
        raw_output: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Genera informazioni di audit"""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'operation_id': raw_output.get('operation_id', 'N/A'),
            'user_id': context.get('user_id', 'unknown') if context else 'unknown',
            'input_hash': hash(str(context.get('input', ''))) if context else None,
            'output_hash': hash(str(raw_output)),
            'data_sources': raw_output.get('sources', []),
            'processing_time': raw_output.get('processing_time'),
            'model_version': raw_output.get('model_version'),
            'confidence': raw_output.get('confidence'),
            'decision_path': raw_output.get('reasoning', [])
        }
    
    def format_adapted_output(self, adapted: AdaptedOutput) -> str:
        """Formatta l'output adattato per la visualizzazione"""
        
        profile = adapted.profile
        parts = []
        
        # Header basato sullo stile
        if profile.preferred_style == CommunicationStyle.EXECUTIVE_SUMMARY:
            parts.append("# 📊 Executive Summary\n")
        elif profile.preferred_style == CommunicationStyle.TECHNICAL_DETAIL:
            parts.append("# 🔧 Technical Report\n")
        elif profile.preferred_style == CommunicationStyle.OPERATIONAL:
            parts.append("# ✅ Action Guide\n")
        elif profile.preferred_style == CommunicationStyle.COMPLIANCE:
            parts.append("# 📋 Audit Report\n")
        else:
            parts.append("# 📝 Report\n")
        
        # Summary
        parts.append(adapted.summary)
        parts.append("")
        
        # Action items
        if adapted.action_items:
            parts.append("\n## 🎯 Azioni")
            for i, action in enumerate(adapted.action_items, 1):
                parts.append(f"{i}. {action}")
            parts.append("")
        
        # Metrics
        if adapted.metrics:
            parts.append("\n## 📈 Metriche")
            for k, v in adapted.metrics.items():
                parts.append(f"- **{k}**: {v}")
            parts.append("")
        
        # Details (se tecnico)
        if adapted.details:
            parts.append("\n## 🔍 Dettagli Tecnici")
            parts.append(adapted.details)
            parts.append("")
        
        # Audit info
        if adapted.audit_info:
            parts.append("\n## 📋 Audit Trail")
            parts.append(f"- **Timestamp**: {adapted.audit_info['timestamp']}")
            parts.append(f"- **Operation ID**: {adapted.audit_info['operation_id']}")
            if adapted.audit_info.get('confidence'):
                parts.append(f"- **Confidenza**: {adapted.audit_info['confidence']}")
        
        return "\n".join(parts)
    
    def register_custom_adapter(
        self,
        role: UserRole,
        adapter_func: Callable[[Dict[str, Any], RoleProfile], AdaptedOutput]
    ):
        """Registra un adapter custom per un ruolo"""
        self.custom_adapters[role] = adapter_func
        logger.info(f"🎭 Custom adapter registrato per: {role.value}")
    
    def get_profile(self, role: UserRole) -> RoleProfile:
        """Ottiene il profilo per un ruolo"""
        return self.role_profiles.get(role, self.role_profiles[UserRole.UNKNOWN])
    
    def update_profile(self, role: UserRole, **kwargs):
        """Aggiorna un profilo esistente"""
        if role in self.role_profiles:
            profile = self.role_profiles[role]
            for key, value in kwargs.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)
            logger.info(f"🎭 Profilo aggiornato: {role.value}")
