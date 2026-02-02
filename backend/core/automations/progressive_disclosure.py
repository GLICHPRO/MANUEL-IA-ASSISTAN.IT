"""
📊 PROGRESSIVE DISCLOSURE ENGINE
================================
GIDEON rivela informazioni progressivamente:
- Prima il riassunto essenziale
- Poi i dettagli su richiesta
- Layers di complessità adattivi
- Drill-down interattivo

"Ecco il riassunto. Vuoi approfondire i dettagli?"
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DetailLevel(Enum):
    """Livelli di dettaglio"""
    MINIMAL = 1  # Una frase
    SUMMARY = 2  # Paragrafo
    STANDARD = 3  # Risposta completa
    DETAILED = 4  # Con esempi e contesto
    EXHAUSTIVE = 5  # Tutto il possibile


class ContentType(Enum):
    """Tipi di contenuto"""
    TEXT = "text"
    CODE = "code"
    DATA = "data"
    ANALYSIS = "analysis"
    RECOMMENDATION = "recommendation"


@dataclass
class ContentLayer:
    """Layer di contenuto"""
    level: DetailLevel
    content: str
    content_type: ContentType
    metadata: Dict[str, Any] = field(default_factory=dict)
    has_more: bool = False
    next_prompt: Optional[str] = None


@dataclass
class ProgressiveContent:
    """Contenuto con disclosure progressiva"""
    content_id: str
    topic: str
    layers: List[ContentLayer]
    current_level: DetailLevel = DetailLevel.SUMMARY
    user_requests_more: int = 0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class DisclosurePreferences:
    """Preferenze utente per disclosure"""
    default_level: DetailLevel = DetailLevel.SUMMARY
    auto_expand_code: bool = False
    show_examples: bool = True
    max_initial_lines: int = 20
    prefer_bullet_points: bool = True


class ProgressiveDisclosureEngine:
    """
    Engine per rivelazione progressiva delle informazioni.
    
    Principi:
    1. Mai sopraffare l'utente
    2. Risposta iniziale sempre concisa
    3. Dettagli disponibili su richiesta
    4. Adatta al feedback implicito
    """
    
    def __init__(self):
        self.active_content: Dict[str, ProgressiveContent] = {}
        self.preferences = DisclosurePreferences()
        self.expansion_history: List[Dict[str, Any]] = []
        
    def set_preferences(self, prefs: DisclosurePreferences):
        """Imposta preferenze disclosure"""
        self.preferences = prefs
    
    def create_progressive_content(
        self,
        topic: str,
        full_content: str,
        content_type: ContentType = ContentType.TEXT
    ) -> ProgressiveContent:
        """
        Crea contenuto con layers progressivi.
        
        Genera automaticamente i livelli:
        - MINIMAL: Prima frase/punto chiave
        - SUMMARY: Punti principali
        - STANDARD: Contenuto normale
        - DETAILED: Con contesto aggiuntivo
        - EXHAUSTIVE: Tutto
        """
        
        content_id = f"pc_{datetime.now().timestamp()}"
        
        layers = self._generate_layers(full_content, content_type)
        
        progressive = ProgressiveContent(
            content_id=content_id,
            topic=topic,
            layers=layers,
            current_level=self.preferences.default_level
        )
        
        self.active_content[content_id] = progressive
        
        return progressive
    
    def _generate_layers(
        self,
        content: str,
        content_type: ContentType
    ) -> List[ContentLayer]:
        """Genera layers automaticamente"""
        
        layers = []
        
        # Analizza contenuto
        lines = content.strip().split('\n')
        paragraphs = self._split_paragraphs(content)
        
        # Layer MINIMAL - Una frase chiave
        minimal_content = self._extract_key_sentence(content)
        layers.append(ContentLayer(
            level=DetailLevel.MINIMAL,
            content=minimal_content,
            content_type=content_type,
            has_more=True,
            next_prompt="📖 Mostra riassunto"
        ))
        
        # Layer SUMMARY - Punti principali
        summary_content = self._create_summary(content, paragraphs)
        layers.append(ContentLayer(
            level=DetailLevel.SUMMARY,
            content=summary_content,
            content_type=content_type,
            has_more=True,
            next_prompt="🔍 Mostra dettagli completi"
        ))
        
        # Layer STANDARD - Contenuto normale
        standard_content = self._create_standard(content, len(lines))
        layers.append(ContentLayer(
            level=DetailLevel.STANDARD,
            content=standard_content,
            content_type=content_type,
            has_more=True,
            next_prompt="📚 Mostra tutto con esempi"
        ))
        
        # Layer DETAILED - Con contesto
        detailed_content = content
        layers.append(ContentLayer(
            level=DetailLevel.DETAILED,
            content=detailed_content,
            content_type=content_type,
            has_more=True,
            next_prompt="📖 Mostra versione esaustiva"
        ))
        
        # Layer EXHAUSTIVE - Tutto
        exhaustive_content = self._create_exhaustive(content, content_type)
        layers.append(ContentLayer(
            level=DetailLevel.EXHAUSTIVE,
            content=exhaustive_content,
            content_type=content_type,
            has_more=False,
            next_prompt=None
        ))
        
        return layers
    
    def _split_paragraphs(self, content: str) -> List[str]:
        """Divide in paragrafi"""
        return [p.strip() for p in content.split('\n\n') if p.strip()]
    
    def _extract_key_sentence(self, content: str) -> str:
        """Estrae frase chiave"""
        
        # Prima frase che termina con punto
        sentences = content.replace('\n', ' ').split('.')
        if sentences:
            return sentences[0].strip() + '.'
        
        # Fallback: prime 100 caratteri
        return content[:100] + '...' if len(content) > 100 else content
    
    def _create_summary(self, content: str, paragraphs: List[str]) -> str:
        """Crea riassunto"""
        
        if len(paragraphs) <= 2:
            return content
        
        # Prendi primo paragrafo + punti chiave
        summary_parts = [paragraphs[0]]
        
        # Estrai punti da altri paragrafi
        for para in paragraphs[1:4]:
            key_point = self._extract_key_sentence(para)
            if key_point:
                summary_parts.append(f"• {key_point}")
        
        if len(paragraphs) > 4:
            summary_parts.append(f"\n*...e altri {len(paragraphs) - 4} punti*")
        
        return '\n'.join(summary_parts)
    
    def _create_standard(self, content: str, total_lines: int) -> str:
        """Crea versione standard"""
        
        max_lines = self.preferences.max_initial_lines
        lines = content.split('\n')
        
        if len(lines) <= max_lines:
            return content
        
        # Mostra prime N righe
        truncated = '\n'.join(lines[:max_lines])
        remaining = len(lines) - max_lines
        
        return truncated + f"\n\n*...{remaining} righe omesse*"
    
    def _create_exhaustive(self, content: str, content_type: ContentType) -> str:
        """Crea versione esaustiva"""
        
        exhaustive = content
        
        # Aggiungi contesto extra per tipo
        if content_type == ContentType.CODE:
            exhaustive += "\n\n---\n### 📝 Note aggiuntive\n"
            exhaustive += "- Questo codice può essere ottimizzato ulteriormente\n"
            exhaustive += "- Considera l'aggiunta di error handling\n"
            exhaustive += "- Verifica la compatibilità con le versioni target"
        
        elif content_type == ContentType.ANALYSIS:
            exhaustive += "\n\n---\n### ⚠️ Limitazioni dell'analisi\n"
            exhaustive += "- Basata sui dati disponibili al momento\n"
            exhaustive += "- Potrebbero esserci fattori non considerati"
        
        return exhaustive
    
    def get_current_layer(self, content_id: str) -> Optional[ContentLayer]:
        """Ottiene layer corrente per un contenuto"""
        
        progressive = self.active_content.get(content_id)
        if not progressive:
            return None
        
        current_level = progressive.current_level
        
        for layer in progressive.layers:
            if layer.level == current_level:
                return layer
        
        return progressive.layers[0] if progressive.layers else None
    
    def expand_content(self, content_id: str) -> Optional[ContentLayer]:
        """Espande al livello successivo"""
        
        progressive = self.active_content.get(content_id)
        if not progressive:
            return None
        
        current_idx = progressive.current_level.value - 1
        
        if current_idx < len(progressive.layers) - 1:
            progressive.current_level = DetailLevel(current_idx + 2)
            progressive.user_requests_more += 1
            
            # Log espansione
            self.expansion_history.append({
                'content_id': content_id,
                'topic': progressive.topic,
                'from_level': current_idx + 1,
                'to_level': current_idx + 2,
                'timestamp': datetime.now().isoformat()
            })
            
            return self.get_current_layer(content_id)
        
        return None
    
    def collapse_content(self, content_id: str) -> Optional[ContentLayer]:
        """Collassa al livello precedente"""
        
        progressive = self.active_content.get(content_id)
        if not progressive:
            return None
        
        current_idx = progressive.current_level.value - 1
        
        if current_idx > 0:
            progressive.current_level = DetailLevel(current_idx)
            return self.get_current_layer(content_id)
        
        return None
    
    def format_response(
        self,
        topic: str,
        full_content: str,
        content_type: ContentType = ContentType.TEXT,
        initial_level: Optional[DetailLevel] = None
    ) -> Dict[str, Any]:
        """
        Formatta risposta con disclosure progressiva.
        
        Returns:
            Dict con contenuto e opzioni per espansione
        """
        
        # Crea contenuto progressivo
        progressive = self.create_progressive_content(topic, full_content, content_type)
        
        # Usa livello specificato o default
        if initial_level:
            progressive.current_level = initial_level
        
        current_layer = self.get_current_layer(progressive.content_id)
        
        if not current_layer:
            return {
                'content': full_content,
                'has_more': False,
                'content_id': None
            }
        
        return {
            'content': current_layer.content,
            'has_more': current_layer.has_more,
            'next_prompt': current_layer.next_prompt,
            'content_id': progressive.content_id,
            'current_level': current_layer.level.value,
            'total_levels': len(progressive.layers)
        }
    
    def format_with_expandable_sections(
        self,
        sections: Dict[str, str],
        initially_expanded: Optional[List[str]] = None
    ) -> str:
        """
        Formatta con sezioni espandibili.
        
        Args:
            sections: Dict {titolo: contenuto}
            initially_expanded: Titoli delle sezioni inizialmente aperte
        """
        
        initially_expanded = initially_expanded or []
        
        output_parts = []
        
        for title, content in sections.items():
            is_expanded = title in initially_expanded
            
            if is_expanded:
                output_parts.append(f"### ▼ {title}")
                output_parts.append(content)
            else:
                # Mostra preview
                preview = self._extract_key_sentence(content)
                output_parts.append(f"### ▶ {title}")
                output_parts.append(f"*{preview}* [Espandi]")
            
            output_parts.append("")  # Spazio
        
        return '\n'.join(output_parts)
    
    def create_drill_down(
        self,
        data: Dict[str, Any],
        path: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Crea struttura drill-down per dati gerarchici.
        
        Permette navigazione progressiva in strutture complesse.
        """
        
        path = path or []
        
        # Naviga al nodo corrente
        current = data
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                current = data  # Reset se path invalido
                path = []
                break
        
        # Determina cosa mostrare
        if isinstance(current, dict):
            # Mostra chiavi con preview
            items = []
            for key, value in current.items():
                preview = self._get_value_preview(value)
                items.append({
                    'key': key,
                    'preview': preview,
                    'type': type(value).__name__,
                    'expandable': isinstance(value, (dict, list)) and len(str(value)) > 50
                })
            
            return {
                'type': 'object',
                'path': path,
                'items': items,
                'can_go_back': len(path) > 0
            }
        
        elif isinstance(current, list):
            # Mostra elementi con preview
            items = []
            for i, item in enumerate(current[:10]):  # Max 10 items
                preview = self._get_value_preview(item)
                items.append({
                    'index': i,
                    'preview': preview,
                    'type': type(item).__name__,
                    'expandable': isinstance(item, (dict, list))
                })
            
            return {
                'type': 'array',
                'path': path,
                'items': items,
                'total_items': len(current),
                'showing': min(10, len(current)),
                'can_go_back': len(path) > 0
            }
        
        else:
            # Valore primitivo - mostra completo
            return {
                'type': 'value',
                'path': path,
                'value': current,
                'can_go_back': len(path) > 0
            }
    
    def _get_value_preview(self, value: Any) -> str:
        """Ottiene preview di un valore"""
        
        if isinstance(value, dict):
            keys = list(value.keys())[:3]
            return f"{{'{', '.join(keys)}'...}}" if len(value) > 3 else f"{{{', '.join(keys)}}}"
        
        elif isinstance(value, list):
            return f"[{len(value)} items]"
        
        elif isinstance(value, str):
            return value[:50] + '...' if len(value) > 50 else value
        
        else:
            return str(value)[:30]
    
    def analyze_user_disclosure_preference(self) -> Dict[str, Any]:
        """Analizza preferenze utente dalla storia"""
        
        if not self.expansion_history:
            return {
                'sample_size': 0,
                'recommendation': self.preferences.default_level.value
            }
        
        # Analizza pattern di espansione
        expansions_per_content = {}
        for exp in self.expansion_history:
            cid = exp['content_id']
            if cid not in expansions_per_content:
                expansions_per_content[cid] = []
            expansions_per_content[cid].append(exp['to_level'])
        
        # Livello finale medio
        final_levels = [max(levels) for levels in expansions_per_content.values()]
        avg_final_level = sum(final_levels) / len(final_levels) if final_levels else 2
        
        # Percentuale di espansioni
        total_contents = len(expansions_per_content)
        expanded_contents = sum(1 for levels in expansions_per_content.values() if len(levels) > 0)
        expansion_rate = expanded_contents / total_contents if total_contents else 0
        
        # Raccomandazione
        if expansion_rate > 0.7:
            # Espande spesso - suggerisci livello più alto
            recommended = min(avg_final_level + 1, 5)
        elif expansion_rate < 0.3:
            # Raramente espande - suggerisci livello più basso
            recommended = max(avg_final_level - 1, 1)
        else:
            recommended = round(avg_final_level)
        
        return {
            'sample_size': total_contents,
            'avg_final_level': avg_final_level,
            'expansion_rate': expansion_rate,
            'current_default': self.preferences.default_level.value,
            'recommendation': int(recommended),
            'recommendation_name': DetailLevel(int(recommended)).name
        }
    
    def format_status(self) -> str:
        """Formatta status per visualizzazione"""
        
        analysis = self.analyze_user_disclosure_preference()
        active_count = len(self.active_content)
        
        return f"""
# 📊 Progressive Disclosure Engine Status

## Configurazione
| Impostazione | Valore |
|--------------|--------|
| Livello default | {self.preferences.default_level.name} |
| Auto-expand code | {self.preferences.auto_expand_code} |
| Max righe iniziali | {self.preferences.max_initial_lines} |
| Mostra esempi | {self.preferences.show_examples} |

## Statistiche
| Metrica | Valore |
|---------|--------|
| Contenuti attivi | {active_count} |
| Espansioni totali | {len(self.expansion_history)} |
| Tasso espansione | {analysis.get('expansion_rate', 0):.1%} |
| Livello finale medio | {analysis.get('avg_final_level', 2):.1f} |

## Raccomandazione
Basato sul comportamento, il livello default consigliato è: **{analysis.get('recommendation_name', 'SUMMARY')}**
"""


# Singleton
_disclosure_engine: Optional[ProgressiveDisclosureEngine] = None


def get_disclosure_engine() -> ProgressiveDisclosureEngine:
    """Ottiene istanza singleton"""
    global _disclosure_engine
    if _disclosure_engine is None:
        _disclosure_engine = ProgressiveDisclosureEngine()
    return _disclosure_engine
