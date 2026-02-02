"""
🔮 GIDEON 3.0 - Modulo Predittivo, Analitico e di Simulazione

Questo modulo fornisce:
- Analisi dettagliate
- Previsioni
- Simulazione scenari (Monte Carlo, Bayesian)
- Ranking decisioni (TOPSIS, Pareto)
- Suggerimenti intelligenti
- Costruzione scenari multipli (ScenarioBuilder)
- Simulazioni parallele e predittive (SimulationEngine)
- Calcolo probabilità successo/fallimento (ProbabilityCalculator)
- Analisi rischi e mitigazioni (RiskAnalyzer)
- Analisi storica e apprendimento (HistoricalAnalyzer)
- Ragionamento temporale e impatti futuri (TemporalReasoning)
- Gestione obiettivi e priorità (GoalManagement)
- Meta-cognizione e auto-valutazione (MetaCognition)
- Auto-correzione previsioni (SelfCorrectionEngine)
- Identità e memoria contestuale (IdentityCore)
- Risoluzione intent vs outcome (IntentOutcomeResolver)

⚠️ NON esegue azioni - passa raccomandazioni a Jarvis Core
"""

from .predictor import Predictor
from .analyzer import Analyzer
from .simulator import Simulator, SimulationConfig, MonteCarloResult
from .ranker import DecisionRanker, RankingMethod
from .suggester import SuggestionEngine, Suggestion, SuggestionType
from .scenario_builder import (
    ScenarioBuilder, Scenario, ScenarioType, ScenarioStatus,
    ScenarioVariable, VariableType, ScenarioConstraint, ScenarioOutcome,
    ScenarioComparison
)
from .simulation_engine import (
    SimulationEngine, SimulationMode, DistributionType,
    SimulationResult, SimulationSummary, SensitivityResult, PredictiveResult
)
from .probability_calculator import (
    ProbabilityCalculator, ProbabilityMethod, ConfidenceLevel,
    ProbabilityFactor, ProbabilityResult, ConditionalProbability, BayesianUpdate
)
from .risk_analyzer import (
    RiskAnalyzer, RiskLevel, RiskCategory, MitigationType,
    Risk, Mitigation, RiskAssessment
)
from .historical_analyzer import (
    HistoricalAnalyzer, TrendDirection, PatternType, LearningType,
    HistoricalEvent, Pattern, Learning, HistoricalAnalysis
)
from .temporal_reasoning import (
    TemporalReasoning, TimeHorizon, TemporalRelation, ImpactType,
    TemporalEvent, TemporalImpact, TemporalSequence, FutureProjection
)
from .goal_management import (
    GoalManagement, GoalStatus, GoalPriority, GoalType, AlignmentType,
    Goal, GoalMetric, GoalAlignment, GoalRecommendation
)
from .meta_cognition import (
    MetaCognition, UncertaintyType, LimitationType, ConfidenceCalibration,
    ReflectionType, UncertaintyAssessment, LimitationAwareness, SelfReflection,
    KnowsAbout, MetaCognitiveState
)
from .self_correction import (
    SelfCorrectionEngine, ErrorType, CorrectionStrategy, SeverityLevel,
    PredictionRecord, CorrectionAction, ErrorPattern, CalibrationState
)
from .identity_core import (
    IdentityCore, PersonalityTrait, CommunicationStyle, ContextType, MemoryType,
    ContextualMemory, Context, IdentityProfile, ConsistencyCheck
)
from .intent_outcome_resolver import (
    IntentOutcomeResolver, IntentStatus, MatchQuality, GapType, ResolutionAction,
    Intent, Outcome, Gap, Resolution, ReconciliationResult
)
from .response_generator import (
    GideonResponseGenerator, GideonResponse, ResponseType, ResponseTone
)

__all__ = [
    'Predictor', 
    'Analyzer', 
    'Simulator', 
    'SimulationConfig',
    'MonteCarloResult',
    'DecisionRanker', 
    'RankingMethod',
    'SuggestionEngine',
    'Suggestion',
    'SuggestionType',
    'GideonCore',
    # Scenario Builder
    'ScenarioBuilder',
    'Scenario',
    'ScenarioType',
    'ScenarioStatus',
    'ScenarioVariable',
    'VariableType',
    'ScenarioConstraint',
    'ScenarioOutcome',
    'ScenarioComparison',
    # Simulation Engine
    'SimulationEngine',
    'SimulationMode',
    'DistributionType',
    'SimulationResult',
    'SimulationSummary',
    'SensitivityResult',
    'PredictiveResult',
    # Probability Calculator
    'ProbabilityCalculator',
    'ProbabilityMethod',
    'ConfidenceLevel',
    'ProbabilityFactor',
    'ProbabilityResult',
    'ConditionalProbability',
    'BayesianUpdate',
    # Risk Analyzer
    'RiskAnalyzer',
    'RiskLevel',
    'RiskCategory',
    'MitigationType',
    'Risk',
    'Mitigation',
    'RiskAssessment',
    # Historical Analyzer
    'HistoricalAnalyzer',
    'TrendDirection',
    'PatternType',
    'LearningType',
    'HistoricalEvent',
    'Pattern',
    'Learning',
    'HistoricalAnalysis',
    # Temporal Reasoning
    'TemporalReasoning',
    'TimeHorizon',
    'TemporalRelation',
    'ImpactType',
    'TemporalEvent',
    'TemporalImpact',
    'TemporalSequence',
    'FutureProjection',
    # Goal Management
    'GoalManagement',
    'GoalStatus',
    'GoalPriority',
    'GoalType',
    'AlignmentType',
    'Goal',
    'GoalMetric',
    'GoalAlignment',
    'GoalRecommendation',
    # Meta-Cognition
    'MetaCognition',
    'UncertaintyType',
    'LimitationType',
    'ConfidenceCalibration',
    'ReflectionType',
    'UncertaintyAssessment',
    'LimitationAwareness',
    'SelfReflection',
    'KnowsAbout',
    'MetaCognitiveState',
    # Self-Correction Engine
    'SelfCorrectionEngine',
    'ErrorType',
    'CorrectionStrategy',
    'SeverityLevel',
    'PredictionRecord',
    'CorrectionAction',
    'ErrorPattern',
    'CalibrationState',
    # Identity Core
    'IdentityCore',
    'PersonalityTrait',
    'CommunicationStyle',
    'ContextType',
    'MemoryType',
    'ContextualMemory',
    'Context',
    'IdentityProfile',
    'ConsistencyCheck',
    # Intent-Outcome Resolver
    'IntentOutcomeResolver',
    'IntentStatus',
    'MatchQuality',
    'GapType',
    'ResolutionAction',
    'Intent',
    'Outcome',
    'Gap',
    'Resolution',
    'ReconciliationResult',
    # Response Generator
    'GideonResponseGenerator',
    'GideonResponse',
    'ResponseType',
    'ResponseTone'
]


class GideonCore:
    """
    Core del modulo Gideon 3.0
    Coordina analisi, previsioni, simulazioni, ranking e suggerimenti
    
    Features avanzate:
    - Simulazione Monte Carlo con intervalli di confidenza
    - Analisi Bayesiana con aggiornamento credenze
    - Ranking TOPSIS multi-criterio
    - Ottimizzazione Pareto
    - Suggerimenti intelligenti basati su pattern
    - Costruzione scenari multipli (ScenarioBuilder)
    - Simulazioni parallele e predittive (SimulationEngine)
    - Calcolo probabilità (ProbabilityCalculator)
    - Analisi rischi (RiskAnalyzer)
    - Apprendimento storico (HistoricalAnalyzer)
    - Ragionamento temporale (TemporalReasoning)
    - Gestione obiettivi (GoalManagement)
    - Meta-cognizione (MetaCognition)
    - Auto-correzione (SelfCorrectionEngine)
    - Identità e memoria (IdentityCore)
    - Risoluzione intent-outcome (IntentOutcomeResolver)
    """
    
    def __init__(self, simulation_config: SimulationConfig = None, max_workers: int = 4):
        self.predictor = Predictor()
        self.analyzer = Analyzer()
        self.simulator = Simulator(simulation_config)
        self.ranker = DecisionRanker()
        self.suggester = SuggestionEngine()
        self.scenario_builder = ScenarioBuilder()
        self.simulation_engine = SimulationEngine(max_workers=max_workers)
        self.probability_calculator = ProbabilityCalculator()
        self.risk_analyzer = RiskAnalyzer()
        self.historical_analyzer = HistoricalAnalyzer()
        self.temporal_reasoning = TemporalReasoning()
        self.goal_management = GoalManagement()
        self.meta_cognition = MetaCognition()
        self.self_correction = SelfCorrectionEngine()  # NEW
        self.identity = IdentityCore()  # NEW
        self.intent_resolver = IntentOutcomeResolver()  # NEW
        self.response_generator = GideonResponseGenerator()  # NEW - Output narrativi
        self.is_active = True
        
    async def analyze(self, data: dict) -> dict:
        """Esegue analisi completa dei dati"""
        return await self.analyzer.analyze(data)
    
    async def predict(self, context: dict) -> dict:
        """Genera previsioni basate sul contesto"""
        return await self.predictor.predict(context)
    
    async def simulate(self, scenario: dict) -> dict:
        """Simula uno scenario e restituisce risultati"""
        return await self.simulator.run(scenario)
    
    async def simulate_monte_carlo(self, scenario: dict, iterations: int = 1000) -> dict:
        """Esegue simulazione Monte Carlo"""
        result = await self.simulator.monte_carlo(scenario, iterations)
        return result.to_dict()
    
    async def rank_options(self, options: list, method: str = "auto") -> list:
        """Classifica le opzioni per priorità/efficacia"""
        return await self.ranker.smart_rank(options, method)
    
    async def rank_topsis(self, options: list) -> list:
        """Ranking con algoritmo TOPSIS"""
        return await self.ranker.topsis_rank(options)
    
    async def pareto_analysis(self, options: list, objectives: list = None) -> dict:
        """Trova frontiera di Pareto"""
        return await self.ranker.pareto_rank(options, objectives)
    
    async def get_suggestions(self, context: dict) -> list:
        """Genera suggerimenti intelligenti"""
        suggestions = await self.suggester.generate(context)
        return [s.to_dict() for s in suggestions]
    
    def record_action(self, action: dict):
        """Registra un'azione per pattern learning"""
        self.suggester.record_action(action)
    
    async def get_recommendation(self, query: str, context: dict = None) -> dict:
        """
        Pipeline completa: analizza -> prevede -> simula -> classifica -> suggerisce
        Restituisce raccomandazione finale per Jarvis Core
        """
        # 1. Analizza la situazione attuale
        analysis = await self.analyzer.analyze(context or {})
        
        # 2. Genera previsioni
        predictions = await self.predictor.predict({
            "query": query,
            "analysis": analysis
        })
        
        # 3. Simula scenari possibili
        scenarios = await self.simulator.generate_scenarios(predictions)
        
        # 4. Esegui simulazione Monte Carlo sui top scenari
        monte_carlo_results = {}
        for scenario in scenarios[:3]:  # Top 3
            mc = await self.simulator.monte_carlo(scenario, iterations=500)
            monte_carlo_results[scenario.get("id")] = mc.to_dict()
        
        # 5. Classifica con analisi completa
        ranked = await self.ranker.comprehensive_analysis(scenarios)
        
        # 6. Genera suggerimenti
        suggestions = await self.suggester.generate(context or {})
        
        # 7. Analisi rischio della raccomandazione principale
        risk_analysis = None
        if scenarios:
            risk_analysis = await self.simulator.risk_analysis(scenarios[0])
        
        return {
            "query": query,
            "analysis": analysis,
            "predictions": predictions,
            "scenarios": scenarios,
            "monte_carlo": monte_carlo_results,
            "ranking": ranked,
            "suggestions": [s.to_dict() for s in suggestions[:3]],
            "risk_analysis": risk_analysis,
            "recommendation": ranked.get("weighted_ranking", [[]])[0] if ranked.get("weighted_ranking") else None,
            "confidence": self._calculate_confidence(ranked)
        }
    
    def _calculate_confidence(self, ranked: dict) -> float:
        """Calcola il livello di confidenza della raccomandazione"""
        if not ranked:
            return 0.0
        
        weighted = ranked.get("weighted_ranking", [])
        if not weighted:
            return 0.0
        
        if len(weighted) == 1:
            return weighted[0].get("final_score", 0.5)
        
        # Differenza tra primo e secondo indica confidenza
        diff = weighted[0].get("final_score", 0) - weighted[1].get("final_score", 0)
        
        # Considera anche consensus
        consensus_conf = ranked.get("confidence", 0)
        
        return min((0.5 + diff + consensus_conf) / 2, 1.0)
    
    async def what_if(self, base_scenario: dict, modifications: list) -> dict:
        """Analisi what-if: confronta scenario base con varianti"""
        return await self.simulator.what_if_analysis(base_scenario, modifications)
    
    async def sensitivity_check(self, scenario: dict, parameter: str) -> dict:
        """Analisi di sensitività su un parametro"""
        return await self.simulator.sensitivity_analysis(scenario, parameter)
    
    # === NEW: Scenario Builder Methods ===
    
    def build_scenario(self, name: str, description: str,
                       scenario_type: ScenarioType = ScenarioType.BASELINE,
                       template: str = None, context: dict = None) -> Scenario:
        """Crea nuovo scenario con ScenarioBuilder"""
        return self.scenario_builder.create_scenario(
            name, description, scenario_type, template, context
        )
    
    def derive_scenario(self, base_id: str, name: str,
                        scenario_type: ScenarioType,
                        modifications: dict = None) -> Scenario:
        """Deriva nuovo scenario da uno esistente"""
        return self.scenario_builder.derive_scenario(
            base_id, name, scenario_type, modifications
        )
    
    def create_scenario_variants(self, base_id: str,
                                  optimistic_factor: float = 1.2,
                                  pessimistic_factor: float = 0.8) -> tuple:
        """Crea varianti ottimistica e pessimistica"""
        return self.scenario_builder.create_optimistic_pessimistic(
            base_id, optimistic_factor, pessimistic_factor
        )
    
    def compare_scenarios(self, scenario_ids: list, metrics: list = None) -> ScenarioComparison:
        """Confronta multipli scenari"""
        return self.scenario_builder.compare_scenarios(scenario_ids, metrics)
    
    def get_scenario(self, scenario_id: str) -> dict:
        """Ottiene scenario per ID"""
        return self.scenario_builder.get_scenario(scenario_id)
    
    def list_scenarios(self, scenario_type: ScenarioType = None) -> list:
        """Lista scenari filtrati"""
        return self.scenario_builder.list_scenarios(scenario_type)
    
    # === NEW: Simulation Engine Methods ===
    
    def run_simulation(self, scenario: dict) -> SimulationResult:
        """Esegue singola simulazione"""
        return self.simulation_engine.simulate_single(scenario)
    
    def run_monte_carlo(self, scenario: dict, iterations: int = 1000,
                        parallel: bool = False) -> SimulationSummary:
        """Esegue simulazione Monte Carlo"""
        if parallel:
            return self.simulation_engine.simulate_monte_carlo_parallel(
                scenario, iterations
            )
        return self.simulation_engine.simulate_monte_carlo(scenario, iterations)
    
    def run_sensitivity_analysis(self, scenario: dict, variable: str,
                                  values: list, iterations: int = 100) -> SensitivityResult:
        """Esegue analisi di sensitività"""
        return self.simulation_engine.analyze_sensitivity(
            scenario, variable, values, iterations
        )
    
    def run_predictive_simulation(self, scenario: dict,
                                   hours: int = 24,
                                   steps: int = 24) -> PredictiveResult:
        """Esegue simulazione predittiva temporale"""
        from datetime import timedelta
        return self.simulation_engine.simulate_predictive(
            scenario,
            time_horizon=timedelta(hours=hours),
            time_steps=steps
        )
    
    def simulate_batch(self, scenarios: list,
                       iterations: int = 1000) -> dict:
        """Simula batch di scenari in parallelo"""
        return self.simulation_engine.simulate_batch(
            scenarios, SimulationMode.MONTE_CARLO, iterations
        )
    
    def compare_simulations(self, summaries: list) -> dict:
        """Confronta risultati simulazioni"""
        return self.simulation_engine.compare_simulations(summaries)
    
    # === NEW: Probability Calculator Methods ===
    
    def calculate_probability(self, context: dict,
                               method: ProbabilityMethod = ProbabilityMethod.COMBINED) -> ProbabilityResult:
        """Calcola probabilità di successo per un'azione"""
        return self.probability_calculator.calculate(context, method)
    
    def analyze_scenario_probability(self, scenario: dict) -> dict:
        """Analisi probabilistica completa di uno scenario"""
        return self.probability_calculator.analyze_scenario(scenario)
    
    def chain_events_probability(self, events: list) -> float:
        """Probabilità catena di eventi indipendenti"""
        return self.probability_calculator.chain_probability(events)
    
    def at_least_one_success_probability(self, probabilities: list) -> float:
        """Probabilità almeno un successo"""
        return self.probability_calculator.at_least_one_success(probabilities)
    
    def expected_attempts(self, success_prob: float) -> float:
        """Numero atteso tentativi per successo"""
        return self.probability_calculator.expected_attempts(success_prob)
    
    # === NEW: Risk Analyzer Methods ===
    
    def assess_risk(self, scenario: dict) -> RiskAssessment:
        """Valutazione rischio completa di uno scenario"""
        return self.risk_analyzer.assess(scenario)
    
    def identify_risks(self, scenario: dict) -> list:
        """Identifica rischi potenziali"""
        return self.risk_analyzer.identify_risks(scenario)
    
    def generate_mitigations(self, risk: Risk) -> list:
        """Genera mitigazioni per un rischio"""
        return self.risk_analyzer.generate_mitigations(risk)
    
    def analyze_risk_matrix(self, risks: list) -> dict:
        """Genera matrice di rischio"""
        return self.risk_analyzer.analyze_risk_matrix(risks)
    
    def compare_risk_scenarios(self, scenarios: list) -> dict:
        """Confronta rischi tra scenari"""
        return self.risk_analyzer.compare_scenarios(scenarios)
    
    def get_risk_trend(self, scenario_type: str = None) -> dict:
        """Analizza trend rischi nel tempo"""
        return self.risk_analyzer.trend_analysis(scenario_type)
    
    # === NEW: Historical Analyzer Methods ===
    
    def record_event(self, event_type: str, success: bool,
                     context: dict = None, parameters: dict = None,
                     duration_ms: float = 0.0, tags: list = None) -> HistoricalEvent:
        """Registra evento storico"""
        return self.historical_analyzer.record_event(
            event_type, success, context, parameters, duration_ms, tags=tags
        )
    
    def analyze_history(self, event_type: str = None,
                        time_range: tuple = None,
                        tags: list = None) -> HistoricalAnalysis:
        """Analizza dati storici"""
        return self.historical_analyzer.analyze(event_type, time_range, tags)
    
    def get_success_rate(self, event_type: str, days: int = 30) -> dict:
        """Ottieni success rate per tipo evento"""
        return self.historical_analyzer.get_success_rate(event_type, days)
    
    def get_performance_trend(self, event_type: str = None,
                               granularity: str = "day") -> list:
        """Performance nel tempo"""
        return self.historical_analyzer.get_performance_over_time(event_type, granularity)
    
    def apply_historical_learning(self, event_type: str, context: dict) -> dict:
        """Applica apprendimenti storici"""
        return self.historical_analyzer.apply_learning(event_type, context)
    
    def export_learnings(self) -> dict:
        """Esporta apprendimenti"""
        return self.historical_analyzer.export_learnings()
    
    # === NEW: Temporal Reasoning Methods ===
    
    def create_temporal_event(self, name: str, start_time, duration=None, **kwargs):
        """Crea evento temporale"""
        return self.temporal_reasoning.create_event(name, start_time, duration, **kwargs)
    
    def analyze_temporal_impact(self, action: dict, context: dict = None):
        """Analizza impatto temporale di un'azione"""
        return self.temporal_reasoning.analyze_impact(action, context)
    
    def project_future(self, current_state: dict, planned_actions: list,
                       horizon: TimeHorizon = None):
        """Proietta stato futuro"""
        from .temporal_reasoning import TimeHorizon
        horizon = horizon or TimeHorizon.MEDIUM_TERM
        return self.temporal_reasoning.project_future(current_state, planned_actions, horizon)
    
    def find_temporal_conflicts(self, events: list = None) -> list:
        """Trova conflitti temporali"""
        return self.temporal_reasoning.find_conflicts(events)
    
    def optimize_schedule(self, events: list) -> list:
        """Ottimizza scheduling eventi"""
        return self.temporal_reasoning.optimize_schedule(events)
    
    def suggest_optimal_timing(self, action: dict, context: dict = None) -> dict:
        """Suggerisce timing ottimale per un'azione"""
        return self.temporal_reasoning.suggest_optimal_timing(action, context)
    
    def create_temporal_sequence(self, name: str, event_ids: list):
        """Crea sequenza temporale"""
        return self.temporal_reasoning.create_sequence(name, event_ids)
    
    # === NEW: Goal Management Methods ===
    
    def create_goal(self, name: str, description: str,
                    goal_type: GoalType = None, priority: GoalPriority = None,
                    deadline=None, **kwargs):
        """Crea nuovo obiettivo"""
        from .goal_management import GoalType, GoalPriority
        goal_type = goal_type or GoalType.OUTCOME
        priority = priority or GoalPriority.MEDIUM
        return self.goal_management.create_goal(
            name, description, goal_type, priority, deadline, **kwargs
        )
    
    def create_sub_goal(self, parent_id: str, name: str, description: str, **kwargs):
        """Crea sub-obiettivo"""
        return self.goal_management.create_sub_goal(parent_id, name, description, **kwargs)
    
    def update_goal_status(self, goal_id: str, status) -> bool:
        """Aggiorna stato obiettivo"""
        return self.goal_management.update_status(goal_id, status)
    
    def update_goal_metric(self, goal_id: str, metric_name: str, value: float) -> bool:
        """Aggiorna metrica obiettivo"""
        return self.goal_management.update_metric(goal_id, metric_name, value)
    
    def update_goal_progress(self, goal_id: str, progress: float) -> bool:
        """Aggiorna progresso obiettivo"""
        return self.goal_management.update_progress(goal_id, progress)
    
    def prioritize_goals(self, goals: list = None) -> list:
        """Prioritizza obiettivi"""
        return self.goal_management.prioritize_goals(goals)
    
    def analyze_goal_health(self, goal_id: str) -> dict:
        """Analizza salute obiettivo"""
        return self.goal_management.analyze_goal_health(goal_id)
    
    def get_active_goals(self) -> list:
        """Ottieni obiettivi attivi prioritizzati"""
        return self.goal_management.get_active_goals()
    
    def get_overdue_goals(self) -> list:
        """Ottieni obiettivi in ritardo"""
        return self.goal_management.get_overdue_goals()
    
    def get_blocked_goals(self) -> list:
        """Ottieni obiettivi bloccati"""
        return self.goal_management.get_blocked_goals()
    
    def get_goal_tree(self, root_id: str = None) -> dict:
        """Ottieni albero obiettivi"""
        return self.goal_management.get_goal_tree(root_id)
    
    def add_goal_dependency(self, goal_id: str, depends_on_id: str) -> bool:
        """Aggiunge dipendenza tra obiettivi"""
        return self.goal_management.add_dependency(goal_id, depends_on_id)
    
    # === NEW: Meta-Cognition Methods ===
    
    def assess_uncertainty(self, domain: str, context: dict = None):
        """Valuta incertezza in un dominio"""
        return self.meta_cognition.assess_uncertainty(domain, context)
    
    def check_limitations(self, task: dict) -> list:
        """Verifica limitazioni per un task"""
        return self.meta_cognition.check_limitations(task)
    
    def acknowledge_limitations(self, task: dict) -> dict:
        """Riconosce limitazioni e suggerisce workaround"""
        return self.meta_cognition.acknowledge_limitation(task)
    
    def reflect(self, subject: str, reflection_type, 
                predicted_confidence: float, actual_outcome: float,
                context: dict = None):
        """Esegue riflessione su azione passata"""
        return self.meta_cognition.reflect(
            subject, reflection_type, predicted_confidence, actual_outcome, context
        )
    
    def get_meta_cognitive_state(self):
        """Ottiene stato meta-cognitivo"""
        return self.meta_cognition.get_meta_state()
    
    def what_i_know(self, domain: str = None) -> dict:
        """Cosa il sistema sa di sapere"""
        return self.meta_cognition.what_i_know(domain)
    
    def what_i_dont_know(self) -> dict:
        """Cosa il sistema sa di NON sapere"""
        return self.meta_cognition.what_i_dont_know()
    
    def update_knowledge(self, domain: str, knowledge_level: float,
                         known_unknowns: list = None):
        """Aggiorna conoscenza su un dominio"""
        self.meta_cognition.update_knowledge(domain, knowledge_level, known_unknowns)
    
    # === NEW: Self-Correction Engine Methods ===
    
    def track_prediction(self, domain: str, predicted_value: float,
                         confidence: float = 0.5, context: dict = None,
                         factors: list = None):
        """Registra previsione per tracking"""
        return self.self_correction.record_prediction(
            domain, predicted_value, confidence, context, factors
        )
    
    def resolve_prediction(self, prediction_id: str, actual_value: float) -> dict:
        """Risolve previsione con valore effettivo"""
        return self.self_correction.resolve_prediction(prediction_id, actual_value)
    
    def correct_prediction(self, predicted_value: float, domain: str,
                           context: dict = None) -> dict:
        """Applica correzioni note a previsione"""
        return self.self_correction.correct_prediction(predicted_value, domain, context)
    
    def calibrate_domain(self, domain: str, force: bool = False) -> dict:
        """Calibra previsioni per un dominio"""
        return self.self_correction.calibrate_domain(domain, force)
    
    def suggest_corrections(self, domain: str = None) -> list:
        """Suggerisce correzioni basate su pattern"""
        return self.self_correction.suggest_corrections(domain)
    
    def get_accuracy_metrics(self, domain: str = None) -> dict:
        """Ottiene metriche accuratezza"""
        return self.self_correction.get_accuracy_metrics(domain)
    
    # === NEW: Identity Core Methods ===
    
    def get_identity(self) -> dict:
        """Ottiene identità sistema"""
        return self.identity.get_identity()
    
    def introduce(self) -> str:
        """Auto-presentazione"""
        return self.identity.introduce()
    
    def remember(self, content: dict, memory_type=None, tags: list = None,
                 persistent: bool = False):
        """Memorizza informazione"""
        from .identity_core import MemoryType
        memory_type = memory_type or MemoryType.SHORT_TERM
        return self.identity.remember(content, memory_type, tags, persistent)
    
    def recall(self, query: dict = None, memory_type=None, tags: list = None,
               limit: int = 10) -> list:
        """Recupera memorie rilevanti"""
        return self.identity.recall(query, memory_type, tags, limit)
    
    def push_context(self, context_type, data: dict = None):
        """Attiva nuovo contesto"""
        return self.identity.push_context(context_type, data)
    
    def pop_context(self):
        """Rimuove contesto attuale"""
        return self.identity.pop_context()
    
    def get_full_context(self) -> dict:
        """Ottiene contesto completo"""
        return self.identity.get_full_context()
    
    def check_consistency(self, statement: dict):
        """Verifica coerenza affermazione"""
        return self.identity.check_consistency(statement)
    
    def consolidate_memory(self) -> dict:
        """Consolida memoria"""
        return self.identity.consolidate_memory()
    
    def record_interaction(self, interaction: dict):
        """Registra interazione utente"""
        self.identity.record_interaction(interaction)
    
    # === NEW: Intent-Outcome Resolver Methods ===
    
    def register_intent(self, description: str, expected_outcomes: list = None,
                        success_criteria: list = None, priority: int = 5,
                        deadline=None, context: dict = None):
        """Registra nuovo intent"""
        return self.intent_resolver.register_intent(
            description, expected_outcomes, success_criteria, None,
            priority, deadline, context
        )
    
    def record_outcome(self, intent_id: str, description: str,
                       achieved_results: list = None, metrics: dict = None,
                       side_effects: list = None, errors: list = None):
        """Registra outcome per un intent"""
        return self.intent_resolver.record_outcome(
            intent_id, description, achieved_results, metrics, side_effects, errors
        )
    
    def resolve_intent_outcome(self, intent_id: str, outcome_id: str):
        """Risolve confronto intent-outcome"""
        return self.intent_resolver.resolve(intent_id, outcome_id)
    
    def reconcile_intent(self, intent_id: str, outcome_id: str):
        """Riconciliazione completa intent-outcome"""
        return self.intent_resolver.reconcile(intent_id, outcome_id)
    
    def get_pending_intents(self) -> list:
        """Ottiene intents pendenti"""
        return self.intent_resolver.get_pending_intents()
    
    def get_gap_statistics(self) -> dict:
        """Statistiche sui gap intent-outcome"""
        return self.intent_resolver.get_gap_statistics()
    
    # === NEW: Response Generator Methods ===
    
    def respond_scenario_optimal(self, probability: float, risk_level: str,
                                  scenario_name: str = None,
                                  details: dict = None) -> 'GideonResponse':
        """Genera risposta per scenario ottimale identificato
        
        Es: "Scenario ottimale identificato con probabilità 0.91, rischio minimo."
        """
        return self.response_generator.scenario_optimal(
            probability, risk_level, scenario_name, details
        )
    
    def respond_simulation_completed(self, outcome: str,
                                      iterations: int = None,
                                      confidence: float = 0.85,
                                      suggestions: list = None) -> 'GideonResponse':
        """Genera risposta per simulazione completata
        
        Es: "Simulazione completata, suggerisco modifiche al flusso operativo."
        """
        return self.response_generator.simulation_completed(
            outcome, iterations, confidence, suggestions
        )
    
    def respond_ranking_ready(self, scenarios: list,
                              top_n: int = 5,
                              criterion: str = "sicurezza") -> 'GideonResponse':
        """Genera risposta per classifica pronta
        
        Es: "Classifica dei cinque scenari più sicuri pronta per Jarvis."
        """
        return self.response_generator.ranking_ready(scenarios, top_n, criterion)
    
    def respond_risk_assessment(self, risk_level: str, risk_score: float,
                                 factors: list = None,
                                 mitigations: list = None) -> 'GideonResponse':
        """Genera risposta per valutazione rischio"""
        return self.response_generator.risk_assessment(
            risk_level, risk_score, factors, mitigations
        )
    
    def respond_prediction(self, metric: str, value, confidence: float,
                           trend: str = "neutral",
                           context: dict = None) -> 'GideonResponse':
        """Genera risposta per previsione"""
        return self.response_generator.prediction(
            metric, value, confidence, trend, context
        )
    
    def respond_recommendation(self, action: str, probability: float,
                                reasoning: str = None,
                                alternatives: list = None) -> 'GideonResponse':
        """Genera risposta per raccomandazione"""
        return self.response_generator.recommendation(
            action, probability, reasoning, alternatives
        )
    
    def respond_wait(self, reason: str, estimated_wait: str = None) -> 'GideonResponse':
        """Genera risposta per suggerimento attesa"""
        return self.response_generator.suggest_wait(reason, estimated_wait)
    
    def respond_warning(self, message: str, severity: str = "medium",
                        action_required: bool = False) -> 'GideonResponse':
        """Genera risposta di warning"""
        return self.response_generator.warning(message, severity, action_required)
    
    def respond_status(self, status: str, progress: float = None,
                       eta: str = None) -> 'GideonResponse':
        """Genera aggiornamento stato"""
        return self.response_generator.status_update(status, progress, eta)
    
    def respond_full_analysis(self, scenario: dict, simulation_result: dict,
                               risk: dict, recommendations: list) -> 'GideonResponse':
        """Genera report analisi completo"""
        return self.response_generator.full_analysis_report(
            scenario, simulation_result, risk, recommendations
        )
    
    def respond_compare_scenarios(self, scenarios: list) -> 'GideonResponse':
        """Genera confronto tra scenari"""
        return self.response_generator.compare_scenarios(scenarios)
    
    def get_response_history(self, limit: int = 10) -> list:
        """Ottiene storia risposte recenti"""
        return [r.to_dict() for r in self.response_generator.get_history(limit)]

    def get_statistics(self) -> dict:
        """Statistiche complete del modulo Gideon"""
        return {
            "simulator": self.simulator.get_statistics(),
            "ranker": self.ranker.get_statistics(),
            "suggester": self.suggester.get_statistics(),
            "scenario_builder": self.scenario_builder.get_status(),
            "simulation_engine": self.simulation_engine.get_status(),
            "probability_calculator": self.probability_calculator.get_status(),
            "risk_analyzer": self.risk_analyzer.get_status(),
            "historical_analyzer": self.historical_analyzer.get_status(),
            "temporal_reasoning": self.temporal_reasoning.get_status(),
            "goal_management": self.goal_management.get_status(),
            "meta_cognition": self.meta_cognition.get_status(),
            "self_correction": self.self_correction.get_status(),
            "identity": self.identity.get_status(),
            "intent_resolver": self.intent_resolver.get_status(),
            "response_generator": self.response_generator.get_status(),
            "is_active": self.is_active
        }