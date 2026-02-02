"""
📝 AUTO-DOCUMENTATION ENGINE
============================
GIDEON documenta automaticamente:
- Decisioni con rationale
- Azioni con contesto
- Cronologia ragionata
- Audit trail completo

"Questa decisione è stata presa perché... [documentato automaticamente]"
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import logging
import json
import hashlib

logger = logging.getLogger(__name__)


class DocumentationType(Enum):
    """Tipi di documentazione"""
    DECISION = "decision"  # Decisione presa
    ACTION = "action"  # Azione eseguita
    REASONING = "reasoning"  # Processo di ragionamento
    CHANGE = "change"  # Modifica effettuata
    ERROR = "error"  # Errore e handling
    USER_INTERACTION = "user_interaction"  # Interazione utente
    SYSTEM_EVENT = "system_event"  # Evento di sistema


class ImportanceLevel(Enum):
    """Livello di importanza"""
    LOW = 1  # Routine, può essere omesso
    MEDIUM = 2  # Standard, include in riassunti
    HIGH = 3  # Importante, sempre visibile
    CRITICAL = 4  # Critico, evidenzia sempre


class AuditCategory(Enum):
    """Categorie per audit"""
    SECURITY = "security"
    DATA = "data"
    CONFIGURATION = "configuration"
    USER_ACTION = "user_action"
    SYSTEM = "system"
    COMPLIANCE = "compliance"


@dataclass
class DocumentationEntry:
    """Entry di documentazione"""
    entry_id: str
    doc_type: DocumentationType
    title: str
    summary: str
    details: Dict[str, Any]
    rationale: Optional[str] = None
    alternatives_considered: Optional[List[str]] = None
    importance: ImportanceLevel = ImportanceLevel.MEDIUM
    audit_category: Optional[AuditCategory] = None
    tags: List[str] = field(default_factory=list)
    related_entries: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    author: str = "GIDEON"
    checksum: Optional[str] = None


@dataclass
class DocumentationSession:
    """Sessione di documentazione"""
    session_id: str
    title: str
    entries: List[str] = field(default_factory=list)  # Entry IDs
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    summary: Optional[str] = None


class AutoDocumentationEngine:
    """
    Engine per documentazione automatica.
    
    Cattura e documenta:
    - Decisioni con rationale
    - Azioni con contesto
    - Ragionamenti con alternative
    - Cronologia completa
    """
    
    def __init__(self):
        self.entries: Dict[str, DocumentationEntry] = {}
        self.sessions: Dict[str, DocumentationSession] = {}
        self.current_session: Optional[str] = None
        
        # Indici per ricerca veloce
        self.by_type: Dict[DocumentationType, List[str]] = {t: [] for t in DocumentationType}
        self.by_tag: Dict[str, List[str]] = {}
        self.by_audit_category: Dict[AuditCategory, List[str]] = {c: [] for c in AuditCategory}
        
        # Template per documentazione
        self.templates: Dict[DocumentationType, str] = self._init_templates()
    
    def _init_templates(self) -> Dict[DocumentationType, str]:
        """Inizializza template di documentazione"""
        
        return {
            DocumentationType.DECISION: """
## 🎯 Decisione: {title}

**Data:** {timestamp}
**Importanza:** {importance}

### Riassunto
{summary}

### Rationale
{rationale}

### Alternative Considerate
{alternatives}

### Dettagli
{details}
""",
            DocumentationType.ACTION: """
## ⚡ Azione: {title}

**Data:** {timestamp}
**Tipo:** {action_type}

### Descrizione
{summary}

### Contesto
{context}

### Risultato
{result}
""",
            DocumentationType.REASONING: """
## 🧠 Ragionamento: {title}

**Data:** {timestamp}

### Processo
{summary}

### Passaggi
{steps}

### Conclusione
{conclusion}
""",
            DocumentationType.ERROR: """
## ⚠️ Errore: {title}

**Data:** {timestamp}
**Severità:** {severity}

### Descrizione
{summary}

### Causa
{cause}

### Handling
{handling}

### Prevenzione
{prevention}
"""
        }
    
    def start_session(self, title: str) -> str:
        """Avvia una nuova sessione di documentazione"""
        
        session_id = f"session_{datetime.now().timestamp()}"
        
        session = DocumentationSession(
            session_id=session_id,
            title=title
        )
        
        self.sessions[session_id] = session
        self.current_session = session_id
        
        logger.info(f"📝 Sessione documentazione avviata: {title}")
        
        return session_id
    
    def end_session(self, summary: Optional[str] = None) -> Optional[DocumentationSession]:
        """Termina sessione corrente"""
        
        if not self.current_session:
            return None
        
        session = self.sessions[self.current_session]
        session.ended_at = datetime.now()
        
        # Genera summary automatico se non fornito
        if not summary:
            summary = self._generate_session_summary(session)
        
        session.summary = summary
        
        logger.info(f"📝 Sessione documentazione terminata: {session.title}")
        
        self.current_session = None
        return session
    
    def document_decision(
        self,
        title: str,
        summary: str,
        rationale: str,
        alternatives: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None,
        importance: ImportanceLevel = ImportanceLevel.MEDIUM,
        tags: Optional[List[str]] = None
    ) -> str:
        """Documenta una decisione"""
        
        entry = self._create_entry(
            doc_type=DocumentationType.DECISION,
            title=title,
            summary=summary,
            details=details or {},
            rationale=rationale,
            alternatives_considered=alternatives,
            importance=importance,
            tags=tags or ['decision']
        )
        
        return entry.entry_id
    
    def document_action(
        self,
        title: str,
        action_type: str,
        summary: str,
        context: Dict[str, Any],
        result: Optional[str] = None,
        importance: ImportanceLevel = ImportanceLevel.MEDIUM,
        audit_category: Optional[AuditCategory] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Documenta un'azione"""
        
        details = {
            'action_type': action_type,
            'context': context,
            'result': result
        }
        
        entry = self._create_entry(
            doc_type=DocumentationType.ACTION,
            title=title,
            summary=summary,
            details=details,
            importance=importance,
            audit_category=audit_category,
            tags=tags or ['action', action_type]
        )
        
        return entry.entry_id
    
    def document_reasoning(
        self,
        title: str,
        process: str,
        steps: List[str],
        conclusion: str,
        importance: ImportanceLevel = ImportanceLevel.MEDIUM,
        tags: Optional[List[str]] = None
    ) -> str:
        """Documenta un processo di ragionamento"""
        
        details = {
            'steps': steps,
            'conclusion': conclusion
        }
        
        entry = self._create_entry(
            doc_type=DocumentationType.REASONING,
            title=title,
            summary=process,
            details=details,
            importance=importance,
            tags=tags or ['reasoning']
        )
        
        return entry.entry_id
    
    def document_error(
        self,
        title: str,
        description: str,
        cause: str,
        handling: str,
        prevention: Optional[str] = None,
        severity: str = "medium",
        audit_category: AuditCategory = AuditCategory.SYSTEM,
        tags: Optional[List[str]] = None
    ) -> str:
        """Documenta un errore"""
        
        details = {
            'cause': cause,
            'handling': handling,
            'prevention': prevention,
            'severity': severity
        }
        
        entry = self._create_entry(
            doc_type=DocumentationType.ERROR,
            title=title,
            summary=description,
            details=details,
            importance=ImportanceLevel.HIGH,
            audit_category=audit_category,
            tags=tags or ['error', severity]
        )
        
        return entry.entry_id
    
    def document_change(
        self,
        title: str,
        what_changed: str,
        before: Any,
        after: Any,
        reason: str,
        audit_category: AuditCategory = AuditCategory.CONFIGURATION,
        tags: Optional[List[str]] = None
    ) -> str:
        """Documenta una modifica"""
        
        details = {
            'before': self._serialize_value(before),
            'after': self._serialize_value(after),
            'reason': reason
        }
        
        entry = self._create_entry(
            doc_type=DocumentationType.CHANGE,
            title=title,
            summary=what_changed,
            details=details,
            rationale=reason,
            importance=ImportanceLevel.MEDIUM,
            audit_category=audit_category,
            tags=tags or ['change']
        )
        
        return entry.entry_id
    
    def _create_entry(
        self,
        doc_type: DocumentationType,
        title: str,
        summary: str,
        details: Dict[str, Any],
        rationale: Optional[str] = None,
        alternatives_considered: Optional[List[str]] = None,
        importance: ImportanceLevel = ImportanceLevel.MEDIUM,
        audit_category: Optional[AuditCategory] = None,
        tags: Optional[List[str]] = None
    ) -> DocumentationEntry:
        """Crea entry di documentazione"""
        
        entry_id = f"doc_{datetime.now().timestamp()}"
        
        entry = DocumentationEntry(
            entry_id=entry_id,
            doc_type=doc_type,
            title=title,
            summary=summary,
            details=details,
            rationale=rationale,
            alternatives_considered=alternatives_considered,
            importance=importance,
            audit_category=audit_category,
            tags=tags or []
        )
        
        # Calcola checksum per integrità
        entry.checksum = self._calculate_checksum(entry)
        
        # Salva
        self.entries[entry_id] = entry
        
        # Aggiorna indici
        self.by_type[doc_type].append(entry_id)
        
        for tag in entry.tags:
            if tag not in self.by_tag:
                self.by_tag[tag] = []
            self.by_tag[tag].append(entry_id)
        
        if audit_category:
            self.by_audit_category[audit_category].append(entry_id)
        
        # Aggiungi a sessione corrente
        if self.current_session:
            self.sessions[self.current_session].entries.append(entry_id)
        
        logger.debug(f"📝 Documentato: [{doc_type.value}] {title}")
        
        return entry
    
    def _calculate_checksum(self, entry: DocumentationEntry) -> str:
        """Calcola checksum per integrità"""
        
        content = json.dumps({
            'id': entry.entry_id,
            'type': entry.doc_type.value,
            'title': entry.title,
            'summary': entry.summary,
            'timestamp': entry.timestamp.isoformat()
        }, sort_keys=True)
        
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _serialize_value(self, value: Any) -> str:
        """Serializza valore per documentazione"""
        
        if isinstance(value, (dict, list)):
            return json.dumps(value, indent=2, default=str)
        return str(value)
    
    def _generate_session_summary(self, session: DocumentationSession) -> str:
        """Genera summary automatico di sessione"""
        
        entries = [self.entries[eid] for eid in session.entries if eid in self.entries]
        
        if not entries:
            return "Nessuna documentazione in questa sessione."
        
        # Conta per tipo
        by_type = {}
        for entry in entries:
            t = entry.doc_type.value
            by_type[t] = by_type.get(t, 0) + 1
        
        # Durata
        duration = (session.ended_at or datetime.now()) - session.started_at
        
        # Decisioni importanti
        important = [e for e in entries if e.importance.value >= ImportanceLevel.HIGH.value]
        
        summary_parts = [
            f"Sessione: {session.title}",
            f"Durata: {duration.total_seconds() / 60:.1f} minuti",
            f"Entries totali: {len(entries)}",
            f"Per tipo: {', '.join(f'{k}: {v}' for k, v in by_type.items())}"
        ]
        
        if important:
            summary_parts.append(f"Decisioni importanti: {len(important)}")
        
        return " | ".join(summary_parts)
    
    def link_entries(self, entry_id1: str, entry_id2: str):
        """Collega due entry"""
        
        if entry_id1 in self.entries and entry_id2 in self.entries:
            self.entries[entry_id1].related_entries.append(entry_id2)
            self.entries[entry_id2].related_entries.append(entry_id1)
    
    def get_entry(self, entry_id: str) -> Optional[DocumentationEntry]:
        """Ottiene entry per ID"""
        return self.entries.get(entry_id)
    
    def search_entries(
        self,
        query: Optional[str] = None,
        doc_type: Optional[DocumentationType] = None,
        tags: Optional[List[str]] = None,
        audit_category: Optional[AuditCategory] = None,
        min_importance: Optional[ImportanceLevel] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        limit: int = 50
    ) -> List[DocumentationEntry]:
        """Cerca entries"""
        
        # Inizia con tutti o filtrati per tipo
        if doc_type:
            entry_ids = set(self.by_type.get(doc_type, []))
        else:
            entry_ids = set(self.entries.keys())
        
        # Filtra per tag
        if tags:
            tag_entries = set()
            for tag in tags:
                tag_entries.update(self.by_tag.get(tag, []))
            entry_ids &= tag_entries
        
        # Filtra per audit category
        if audit_category:
            entry_ids &= set(self.by_audit_category.get(audit_category, []))
        
        # Ottieni entries
        entries = [self.entries[eid] for eid in entry_ids if eid in self.entries]
        
        # Filtra per importanza
        if min_importance:
            entries = [e for e in entries if e.importance.value >= min_importance.value]
        
        # Filtra per data
        if from_date:
            entries = [e for e in entries if e.timestamp >= from_date]
        if to_date:
            entries = [e for e in entries if e.timestamp <= to_date]
        
        # Filtra per query testuale
        if query:
            query_lower = query.lower()
            entries = [
                e for e in entries
                if query_lower in e.title.lower() or query_lower in e.summary.lower()
            ]
        
        # Ordina per timestamp (più recenti prima)
        entries.sort(key=lambda x: x.timestamp, reverse=True)
        
        return entries[:limit]
    
    def generate_audit_report(
        self,
        category: AuditCategory,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> str:
        """Genera report di audit per categoria"""
        
        entries = self.search_entries(
            audit_category=category,
            from_date=from_date,
            to_date=to_date
        )
        
        if not entries:
            return f"Nessuna entry per categoria: {category.value}"
        
        report = f"""
# 📋 Audit Report: {category.value.upper()}

**Periodo:** {from_date or 'Inizio'} - {to_date or 'Ora'}
**Entries totali:** {len(entries)}

---

"""
        
        for entry in entries[:50]:  # Max 50
            report += f"""
## {entry.title}

**ID:** `{entry.entry_id}`
**Data:** {entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
**Tipo:** {entry.doc_type.value}
**Importanza:** {entry.importance.name}

{entry.summary}

"""
            if entry.rationale:
                report += f"**Rationale:** {entry.rationale}\n\n"
            
            report += f"**Checksum:** `{entry.checksum}`\n\n---\n"
        
        return report
    
    def format_entry(self, entry: DocumentationEntry) -> str:
        """Formatta entry per visualizzazione"""
        
        template = self.templates.get(entry.doc_type, """
## {title}

**Data:** {timestamp}

{summary}

{details}
""")
        
        # Prepara valori
        values = {
            'title': entry.title,
            'timestamp': entry.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': entry.summary,
            'importance': entry.importance.name,
            'rationale': entry.rationale or 'N/A',
            'alternatives': '\n'.join(f'- {a}' for a in (entry.alternatives_considered or [])) or 'Nessuna',
            'details': json.dumps(entry.details, indent=2, default=str),
        }
        
        # Aggiungi dettagli specifici
        values.update(entry.details)
        
        # Formatta
        try:
            return template.format(**values)
        except KeyError:
            # Fallback se mancano chiavi
            return f"""
## {entry.title}

**Data:** {entry.timestamp}
**Tipo:** {entry.doc_type.value}

{entry.summary}

**Dettagli:** {json.dumps(entry.details, indent=2, default=str)}
"""
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiche documentazione"""
        
        total = len(self.entries)
        
        by_type = {t.value: len(ids) for t, ids in self.by_type.items()}
        by_importance = {}
        
        for entry in self.entries.values():
            imp = entry.importance.name
            by_importance[imp] = by_importance.get(imp, 0) + 1
        
        return {
            'total_entries': total,
            'by_type': by_type,
            'by_importance': by_importance,
            'total_sessions': len(self.sessions),
            'current_session': self.current_session,
            'unique_tags': len(self.by_tag)
        }
    
    def format_status(self) -> str:
        """Formatta status per visualizzazione"""
        
        stats = self.get_stats()
        
        session_info = "Nessuna sessione attiva"
        if self.current_session and self.current_session in self.sessions:
            s = self.sessions[self.current_session]
            session_info = f"**{s.title}** ({len(s.entries)} entries)"
        
        return f"""
# 📝 Auto-Documentation Engine

## Sessione Corrente
{session_info}

## Statistiche
| Metrica | Valore |
|---------|--------|
| Entries totali | {stats['total_entries']} |
| Sessioni totali | {stats['total_sessions']} |
| Tag unici | {stats['unique_tags']} |

## Per Tipo
{chr(10).join(f"- **{k}**: {v}" for k, v in stats['by_type'].items() if v > 0)}

## Per Importanza
{chr(10).join(f"- **{k}**: {v}" for k, v in stats['by_importance'].items())}

## Ultimi 5 Documenti
{self._format_recent_entries(5)}
"""
    
    def _format_recent_entries(self, limit: int) -> str:
        """Formatta entries recenti"""
        
        recent = sorted(
            self.entries.values(),
            key=lambda x: x.timestamp,
            reverse=True
        )[:limit]
        
        if not recent:
            return "- Nessun documento"
        
        lines = []
        for entry in recent:
            lines.append(
                f"- [{entry.doc_type.value}] **{entry.title}** "
                f"({entry.timestamp.strftime('%H:%M')})"
            )
        
        return '\n'.join(lines)


# Singleton
_doc_engine: Optional[AutoDocumentationEngine] = None


def get_documentation_engine() -> AutoDocumentationEngine:
    """Ottiene istanza singleton"""
    global _doc_engine
    if _doc_engine is None:
        _doc_engine = AutoDocumentationEngine()
    return _doc_engine
