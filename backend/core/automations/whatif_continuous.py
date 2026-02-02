"""
🔮 WHAT-IF CONTINUOUS SIMULATOR
===============================
GIDEON simula scenari continuamente:
- Background simulation di alternative
- Proactive warnings
- Opportunity detection
- Risk anticipation

"Nel frattempo ho simulato lo scenario B - potrebbe essere interessante"
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Awaitable
from enum import Enum
import logging
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)


class SimulationType(Enum):
    """Tipi di simulazione"""
    ALTERNATIVE_PATH = "alternative"  # Cosa succederebbe con scelta diversa
    RISK_SCENARIO = "risk"  # Cosa potrebbe andare storto
    OPPORTUNITY = "opportunity"  # Opportunità nascoste
    FAILURE_MODE = "failure"  # Modi di fallimento
    OPTIMIZATION = "optimization"  # Possibili ottimizzazioni


class SimulationPriority(Enum):
    """Priorità simulazione"""
    BACKGROUND = 1  # Bassa priorità, quando c'è tempo
    NORMAL = 2  # Priorità normale
    HIGH = 3  # Alta priorità, risultati attesi
    CRITICAL = 4  # Critica, interrompere altro


class SimulationStatus(Enum):
    """Status simulazione"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SimulationResult:
    """Risultato di una simulazione"""
    success: bool
    outcome_description: str
    probability: float  # 0-1
    impact_score: float  # -1 a 1 (negativo = danno, positivo = beneficio)
    key_factors: List[str]
    recommendations: List[str]
    confidence: float


@dataclass
class Simulation:
    """Simulazione what-if"""
    simulation_id: str
    simulation_type: SimulationType
    title: str
    description: str
    context: Dict[str, Any]
    priority: SimulationPriority = SimulationPriority.NORMAL
    status: SimulationStatus = SimulationStatus.QUEUED
    result: Optional[SimulationResult] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    notified: bool = False


@dataclass
class SimulationInsight:
    """Insight emerso da simulazioni"""
    insight_id: str
    title: str
    description: str
    source_simulations: List[str]
    relevance_score: float
    actionable: bool
    action_suggestion: Optional[str] = None
    discovered_at: datetime = field(default_factory=datetime.now)


class WhatIfContinuousSimulator:
    """
    Simulatore continuo di scenari what-if.
    
    Opera in background per:
    - Esplorare alternative
    - Anticipare rischi
    - Scoprire opportunità
    - Preparare raccomandazioni
    """
    
    # Configurazione
    MAX_CONCURRENT_SIMULATIONS = 3
    BACKGROUND_INTERVAL = 30  # secondi
    MAX_QUEUE_SIZE = 50
    INSIGHT_THRESHOLD = 0.7  # Rilevanza minima per insight
    
    def __init__(self):
        self.simulations: Dict[str, Simulation] = {}
        self.queue: List[str] = []  # IDs in coda
        self.insights: List[SimulationInsight] = []
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        # Handlers per tipi di simulazione
        self.simulation_handlers: Dict[SimulationType, Callable] = {}
        
        # Context corrente per simulazioni
        self.current_context: Dict[str, Any] = {}
        
        # Flag per background loop
        self._running = False
        self._background_task: Optional[asyncio.Task] = None
    
    def register_handler(
        self,
        sim_type: SimulationType,
        handler: Callable[[Simulation], Awaitable[SimulationResult]]
    ):
        """Registra handler per tipo di simulazione"""
        self.simulation_handlers[sim_type] = handler
    
    def update_context(self, context: Dict[str, Any]):
        """Aggiorna contesto per simulazioni"""
        self.current_context.update(context)
        
        # Trigger nuove simulazioni basate su contesto
        asyncio.create_task(self._generate_context_simulations())
    
    async def _generate_context_simulations(self):
        """Genera simulazioni basate su contesto corrente"""
        
        context = self.current_context
        
        # Esempio: se c'è un file modificato, simula impatto
        if 'modified_file' in context:
            await self.queue_simulation(
                SimulationType.RISK_SCENARIO,
                f"Impatto modifica {context['modified_file']}",
                "Analizza possibili effetti collaterali della modifica",
                priority=SimulationPriority.NORMAL
            )
        
        # Esempio: se c'è una decisione pendente, simula alternative
        if 'pending_decision' in context:
            await self.queue_simulation(
                SimulationType.ALTERNATIVE_PATH,
                f"Alternative per {context['pending_decision']}",
                "Esplora percorsi alternativi alla decisione corrente",
                priority=SimulationPriority.HIGH
            )
    
    async def queue_simulation(
        self,
        sim_type: SimulationType,
        title: str,
        description: str,
        context: Optional[Dict[str, Any]] = None,
        priority: SimulationPriority = SimulationPriority.NORMAL
    ) -> str:
        """Aggiunge simulazione alla coda"""
        
        # Limita coda
        if len(self.queue) >= self.MAX_QUEUE_SIZE:
            # Rimuovi simulazioni background a bassa priorità
            low_priority = [
                sid for sid in self.queue
                if self.simulations[sid].priority == SimulationPriority.BACKGROUND
            ]
            if low_priority:
                removed_id = low_priority[0]
                self.queue.remove(removed_id)
                del self.simulations[removed_id]
            else:
                logger.warning("Queue piena, simulazione rifiutata")
                return ""
        
        sim_id = f"sim_{datetime.now().timestamp()}"
        
        simulation = Simulation(
            simulation_id=sim_id,
            simulation_type=sim_type,
            title=title,
            description=description,
            context={**self.current_context, **(context or {})},
            priority=priority
        )
        
        self.simulations[sim_id] = simulation
        
        # Inserisci in coda in base a priorità
        insert_idx = 0
        for i, queued_id in enumerate(self.queue):
            if self.simulations[queued_id].priority.value < priority.value:
                insert_idx = i
                break
            insert_idx = i + 1
        
        self.queue.insert(insert_idx, sim_id)
        
        logger.debug(f"🔮 Simulazione in coda: {title} (priorità: {priority.value})")
        
        return sim_id
    
    async def start_background_loop(self):
        """Avvia loop di simulazione background"""
        
        if self._running:
            return
        
        self._running = True
        self._background_task = asyncio.create_task(self._background_loop())
        logger.info("🔮 What-If Simulator avviato")
    
    async def stop_background_loop(self):
        """Ferma loop background"""
        
        self._running = False
        
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        
        # Cancella task in esecuzione
        for task in self.running_tasks.values():
            task.cancel()
        
        logger.info("🔮 What-If Simulator fermato")
    
    async def _background_loop(self):
        """Loop principale di simulazione"""
        
        while self._running:
            try:
                await self._process_queue()
                await asyncio.sleep(self.BACKGROUND_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Errore in background loop: {e}")
                await asyncio.sleep(5)
    
    async def _process_queue(self):
        """Processa simulazioni in coda"""
        
        # Conta simulazioni in esecuzione
        running_count = sum(
            1 for s in self.simulations.values()
            if s.status == SimulationStatus.RUNNING
        )
        
        # Avvia nuove simulazioni se possibile
        while running_count < self.MAX_CONCURRENT_SIMULATIONS and self.queue:
            sim_id = self.queue.pop(0)
            simulation = self.simulations.get(sim_id)
            
            if simulation and simulation.status == SimulationStatus.QUEUED:
                task = asyncio.create_task(self._run_simulation(simulation))
                self.running_tasks[sim_id] = task
                running_count += 1
    
    async def _run_simulation(self, simulation: Simulation):
        """Esegue una simulazione"""
        
        simulation.status = SimulationStatus.RUNNING
        simulation.started_at = datetime.now()
        
        try:
            # Cerca handler
            handler = self.simulation_handlers.get(simulation.simulation_type)
            
            if handler:
                result = await handler(simulation)
            else:
                # Handler di default
                result = await self._default_simulation_handler(simulation)
            
            simulation.result = result
            simulation.status = SimulationStatus.COMPLETED
            simulation.completed_at = datetime.now()
            
            # Genera insight se rilevante
            if result.probability * abs(result.impact_score) >= self.INSIGHT_THRESHOLD:
                await self._generate_insight(simulation)
            
            logger.debug(f"🔮 Simulazione completata: {simulation.title}")
            
        except Exception as e:
            simulation.status = SimulationStatus.FAILED
            simulation.completed_at = datetime.now()
            logger.error(f"Simulazione fallita {simulation.simulation_id}: {e}")
        
        finally:
            # Rimuovi da task running
            if simulation.simulation_id in self.running_tasks:
                del self.running_tasks[simulation.simulation_id]
    
    async def _default_simulation_handler(
        self,
        simulation: Simulation
    ) -> SimulationResult:
        """Handler di default per simulazioni"""
        
        # Simula processing
        await asyncio.sleep(0.5)
        
        # Risultato placeholder basato su tipo
        if simulation.simulation_type == SimulationType.RISK_SCENARIO:
            return SimulationResult(
                success=True,
                outcome_description=f"Analisi rischio per '{simulation.title}'",
                probability=0.3,
                impact_score=-0.4,
                key_factors=["Complessità", "Dipendenze", "Testing"],
                recommendations=["Verifica test coverage", "Review delle dipendenze"],
                confidence=0.6
            )
        
        elif simulation.simulation_type == SimulationType.OPPORTUNITY:
            return SimulationResult(
                success=True,
                outcome_description=f"Opportunità identificata: {simulation.title}",
                probability=0.5,
                impact_score=0.6,
                key_factors=["Timing", "Risorse disponibili"],
                recommendations=["Valuta fattibilità", "Stima effort"],
                confidence=0.5
            )
        
        else:
            return SimulationResult(
                success=True,
                outcome_description=f"Scenario simulato: {simulation.title}",
                probability=0.5,
                impact_score=0.0,
                key_factors=["Da analizzare"],
                recommendations=["Richiede analisi approfondita"],
                confidence=0.4
            )
    
    async def _generate_insight(self, simulation: Simulation):
        """Genera insight da simulazione significativa"""
        
        if not simulation.result:
            return
        
        result = simulation.result
        
        # Determina se è actionable
        actionable = (
            result.impact_score < -0.5 or  # Rischio significativo
            result.impact_score > 0.5  # Opportunità significativa
        )
        
        action = None
        if actionable:
            if result.impact_score < 0:
                action = f"Considera mitigazione: {result.recommendations[0] if result.recommendations else 'verifica manuale'}"
            else:
                action = f"Considera opportunità: {result.recommendations[0] if result.recommendations else 'valuta fattibilità'}"
        
        insight = SimulationInsight(
            insight_id=f"insight_{datetime.now().timestamp()}",
            title=f"Insight: {simulation.title}",
            description=result.outcome_description,
            source_simulations=[simulation.simulation_id],
            relevance_score=result.probability * abs(result.impact_score),
            actionable=actionable,
            action_suggestion=action
        )
        
        self.insights.append(insight)
        
        # Limita insights
        if len(self.insights) > 100:
            self.insights = self.insights[-100:]
    
    def get_pending_insights(self) -> List[SimulationInsight]:
        """Ottiene insights non ancora notificati"""
        
        # Filtra per rilevanza
        relevant = [
            i for i in self.insights
            if i.relevance_score >= self.INSIGHT_THRESHOLD
        ]
        
        # Ordina per rilevanza
        relevant.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return relevant[:10]  # Top 10
    
    def get_simulation_status(self, sim_id: str) -> Optional[Dict[str, Any]]:
        """Ottiene status di una simulazione"""
        
        sim = self.simulations.get(sim_id)
        if not sim:
            return None
        
        return {
            'id': sim.simulation_id,
            'type': sim.simulation_type.value,
            'title': sim.title,
            'status': sim.status.value,
            'priority': sim.priority.value,
            'created_at': sim.created_at.isoformat(),
            'started_at': sim.started_at.isoformat() if sim.started_at else None,
            'completed_at': sim.completed_at.isoformat() if sim.completed_at else None,
            'has_result': sim.result is not None
        }
    
    def get_completed_simulations(
        self,
        sim_type: Optional[SimulationType] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Ottiene simulazioni completate"""
        
        completed = [
            s for s in self.simulations.values()
            if s.status == SimulationStatus.COMPLETED
            and (sim_type is None or s.simulation_type == sim_type)
        ]
        
        # Ordina per completamento
        completed.sort(key=lambda x: x.completed_at or datetime.min, reverse=True)
        
        results = []
        for sim in completed[:limit]:
            result_data = None
            if sim.result:
                result_data = {
                    'outcome': sim.result.outcome_description,
                    'probability': sim.result.probability,
                    'impact': sim.result.impact_score,
                    'confidence': sim.result.confidence,
                    'recommendations': sim.result.recommendations
                }
            
            results.append({
                'id': sim.simulation_id,
                'type': sim.simulation_type.value,
                'title': sim.title,
                'completed_at': sim.completed_at.isoformat() if sim.completed_at else None,
                'result': result_data
            })
        
        return results
    
    def format_insight_notification(self, insight: SimulationInsight) -> str:
        """Formatta notifica per insight"""
        
        emoji = "⚠️" if insight.actionable else "💡"
        
        notification = f"""
{emoji} **{insight.title}**

{insight.description}

**Rilevanza:** {insight.relevance_score:.1%}
"""
        
        if insight.actionable and insight.action_suggestion:
            notification += f"""
**Azione suggerita:** {insight.action_suggestion}
"""
        
        return notification
    
    def get_stats(self) -> Dict[str, Any]:
        """Ottiene statistiche simulatore"""
        
        total = len(self.simulations)
        by_status = defaultdict(int)
        by_type = defaultdict(int)
        
        for sim in self.simulations.values():
            by_status[sim.status.value] += 1
            by_type[sim.simulation_type.value] += 1
        
        return {
            'total_simulations': total,
            'queue_size': len(self.queue),
            'running': by_status.get('running', 0),
            'completed': by_status.get('completed', 0),
            'failed': by_status.get('failed', 0),
            'by_type': dict(by_type),
            'total_insights': len(self.insights),
            'actionable_insights': sum(1 for i in self.insights if i.actionable),
            'is_running': self._running
        }
    
    def format_status(self) -> str:
        """Formatta status per visualizzazione"""
        
        stats = self.get_stats()
        pending_insights = self.get_pending_insights()
        
        insights_section = ""
        if pending_insights:
            insights_section = "\n## 💡 Insights Recenti\n"
            for insight in pending_insights[:5]:
                emoji = "⚠️" if insight.actionable else "💡"
                insights_section += f"- {emoji} **{insight.title}** (rel: {insight.relevance_score:.1%})\n"
        
        return f"""
# 🔮 What-If Continuous Simulator

## Status
| Metrica | Valore |
|---------|--------|
| Loop attivo | {'✅ Sì' if stats['is_running'] else '❌ No'} |
| Simulazioni totali | {stats['total_simulations']} |
| In coda | {stats['queue_size']} |
| In esecuzione | {stats['running']} |
| Completate | {stats['completed']} |
| Fallite | {stats['failed']} |

## Per Tipo
{chr(10).join(f"- **{k}**: {v}" for k, v in stats['by_type'].items()) or '- Nessuna simulazione'}

## Insights
| Metrica | Valore |
|---------|--------|
| Totali | {stats['total_insights']} |
| Actionable | {stats['actionable_insights']} |

{insights_section}
"""


# Singleton
_simulator: Optional[WhatIfContinuousSimulator] = None


def get_whatif_simulator() -> WhatIfContinuousSimulator:
    """Ottiene istanza singleton"""
    global _simulator
    if _simulator is None:
        _simulator = WhatIfContinuousSimulator()
    return _simulator
