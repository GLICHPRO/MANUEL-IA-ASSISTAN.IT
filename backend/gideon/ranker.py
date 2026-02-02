"""
🏆 GIDEON 3.0 - Advanced Decision Ranker
Sistema di ranking e prioritizzazione decisioni con algoritmi avanzati
TOPSIS, Pareto Optimization, Multi-Criteria Decision Making
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import math
from dataclasses import dataclass
from enum import Enum


class RankingMethod(Enum):
    """Metodi di ranking disponibili"""
    WEIGHTED_SUM = "weighted_sum"
    TOPSIS = "topsis"
    PARETO = "pareto"
    AHP = "ahp"  # Analytic Hierarchy Process
    ELECTRE = "electre"


@dataclass
class RankingResult:
    """Risultato di ranking"""
    rank: int
    score: float
    option: dict
    method: str
    breakdown: dict
    
    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "score": round(self.score, 4),
            "option": self.option,
            "method": self.method,
            "breakdown": self.breakdown
        }


class DecisionRanker:
    """
    Sistema di ranking decisionale avanzato di Gideon 3.0
    
    Features:
    - Ranking multi-criterio (MCDM)
    - TOPSIS (Technique for Order Preference)
    - Pareto Optimization
    - Analisi trade-off
    - Spiegazioni dettagliate
    """
    
    def __init__(self):
        self.weights = {
            "benefit": 0.35,
            "risk": -0.25,
            "probability": 0.20,
            "urgency": 0.15,
            "effort": -0.05
        }
        self.history = []
        self.criteria_types = {
            "benefit": "maximize",
            "probability": "maximize",
            "urgency": "maximize",
            "risk": "minimize",
            "effort": "minimize"
        }
        
    # ============================================
    # TOPSIS RANKING
    # ============================================
    
    async def topsis_rank(self, options: list, weights: dict = None) -> list:
        """
        Ranking usando TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
        
        1. Normalizza la matrice decisionale
        2. Calcola matrice pesata
        3. Determina soluzioni ideali positive e negative
        4. Calcola distanze dalle soluzioni ideali
        5. Calcola closeness coefficient
        """
        if not options or len(options) < 2:
            return await self.rank(options)
        
        weights = weights or self.weights
        criteria = list(self.criteria_types.keys())
        
        # Costruisci matrice decisionale
        matrix = []
        for option in options:
            row = [option.get(c, 0.5) for c in criteria]
            matrix.append(row)
        
        # 1. Normalizza (vettoriale)
        normalized = self._normalize_matrix(matrix)
        
        # 2. Applica pesi
        weighted = self._apply_weights(normalized, weights, criteria)
        
        # 3. Trova soluzioni ideali
        ideal_best, ideal_worst = self._find_ideal_solutions(weighted, criteria)
        
        # 4. Calcola distanze
        distances = []
        for row in weighted:
            d_best = math.sqrt(sum((row[i] - ideal_best[i])**2 for i in range(len(row))))
            d_worst = math.sqrt(sum((row[i] - ideal_worst[i])**2 for i in range(len(row))))
            distances.append((d_best, d_worst))
        
        # 5. Calcola closeness coefficient
        results = []
        for i, option in enumerate(options):
            d_best, d_worst = distances[i]
            closeness = d_worst / (d_best + d_worst) if (d_best + d_worst) > 0 else 0.5
            
            scored_option = option.copy()
            scored_option["topsis_score"] = round(closeness, 4)
            scored_option["distance_from_ideal"] = round(d_best, 4)
            scored_option["distance_from_anti_ideal"] = round(d_worst, 4)
            results.append(scored_option)
        
        # Ordina per closeness (più alto = migliore)
        results.sort(key=lambda x: x["topsis_score"], reverse=True)
        
        for i, opt in enumerate(results):
            opt["rank"] = i + 1
            opt["method"] = "TOPSIS"
        
        return results
    
    def _normalize_matrix(self, matrix: List[List[float]]) -> List[List[float]]:
        """Normalizzazione vettoriale della matrice"""
        n_criteria = len(matrix[0])
        normalized = []
        
        # Calcola norme per ogni criterio
        norms = []
        for j in range(n_criteria):
            col_sum = sum(matrix[i][j]**2 for i in range(len(matrix)))
            norms.append(math.sqrt(col_sum) if col_sum > 0 else 1)
        
        # Normalizza
        for row in matrix:
            normalized_row = [row[j] / norms[j] for j in range(n_criteria)]
            normalized.append(normalized_row)
        
        return normalized
    
    def _apply_weights(self, matrix: List[List[float]], 
                       weights: dict, criteria: list) -> List[List[float]]:
        """Applica pesi alla matrice normalizzata"""
        weighted = []
        for row in matrix:
            weighted_row = [row[j] * abs(weights.get(criteria[j], 0.2)) 
                          for j in range(len(row))]
            weighted.append(weighted_row)
        return weighted
    
    def _find_ideal_solutions(self, weighted: List[List[float]], 
                              criteria: list) -> Tuple[List[float], List[float]]:
        """Trova soluzione ideale positiva e negativa"""
        n_criteria = len(criteria)
        ideal_best = []
        ideal_worst = []
        
        for j in range(n_criteria):
            col = [weighted[i][j] for i in range(len(weighted))]
            criterion = criteria[j]
            
            if self.criteria_types.get(criterion) == "maximize":
                ideal_best.append(max(col))
                ideal_worst.append(min(col))
            else:
                ideal_best.append(min(col))
                ideal_worst.append(max(col))
        
        return ideal_best, ideal_worst
    
    # ============================================
    # PARETO OPTIMIZATION
    # ============================================
    
    async def pareto_rank(self, options: list, 
                          objectives: List[str] = None) -> dict:
        """
        Trova la frontiera di Pareto (soluzioni non dominate)
        
        Un'opzione A domina B se:
        - A è migliore o uguale a B in tutti gli obiettivi
        - A è strettamente migliore in almeno un obiettivo
        """
        if not options:
            return {"pareto_front": [], "dominated": []}
        
        objectives = objectives or ["benefit", "probability"]
        
        # Determina direzione ottimizzazione per ogni obiettivo
        maximize = {obj: self.criteria_types.get(obj, "maximize") == "maximize" 
                   for obj in objectives}
        
        pareto_front = []
        dominated = []
        
        for i, opt_a in enumerate(options):
            is_dominated = False
            
            for j, opt_b in enumerate(options):
                if i == j:
                    continue
                
                # Controlla se B domina A
                b_dominates_a = True
                strictly_better = False
                
                for obj in objectives:
                    val_a = opt_a.get(obj, 0.5)
                    val_b = opt_b.get(obj, 0.5)
                    
                    if maximize[obj]:
                        if val_b < val_a:
                            b_dominates_a = False
                            break
                        if val_b > val_a:
                            strictly_better = True
                    else:
                        if val_b > val_a:
                            b_dominates_a = False
                            break
                        if val_b < val_a:
                            strictly_better = True
                
                if b_dominates_a and strictly_better:
                    is_dominated = True
                    break
            
            opt_copy = opt_a.copy()
            opt_copy["is_pareto_optimal"] = not is_dominated
            
            if not is_dominated:
                pareto_front.append(opt_copy)
            else:
                dominated.append(opt_copy)
        
        return {
            "pareto_front": pareto_front,
            "dominated": dominated,
            "objectives": objectives,
            "front_size": len(pareto_front),
            "total_options": len(options),
            "efficiency": len(pareto_front) / len(options) if options else 0
        }
    
    # ============================================
    # TRADE-OFF ANALYSIS
    # ============================================
    
    async def trade_off_analysis(self, options: list, 
                                  criteria_a: str = "benefit",
                                  criteria_b: str = "risk") -> dict:
        """
        Analizza trade-off tra due criteri
        """
        if len(options) < 2:
            return {"error": "Servono almeno 2 opzioni"}
        
        # Ordina per criterio A
        sorted_by_a = sorted(options, 
                            key=lambda x: x.get(criteria_a, 0), 
                            reverse=True)
        
        trade_offs = []
        
        for i in range(len(sorted_by_a) - 1):
            opt_1 = sorted_by_a[i]
            opt_2 = sorted_by_a[i + 1]
            
            delta_a = opt_1.get(criteria_a, 0) - opt_2.get(criteria_a, 0)
            delta_b = opt_1.get(criteria_b, 0) - opt_2.get(criteria_b, 0)
            
            # Trade-off ratio
            if delta_a != 0:
                trade_off_ratio = delta_b / delta_a
            else:
                trade_off_ratio = 0
            
            trade_offs.append({
                "from": opt_1.get("name", f"Option {i}"),
                "to": opt_2.get("name", f"Option {i+1}"),
                f"gain_{criteria_b}": round(delta_b, 4),
                f"cost_{criteria_a}": round(-delta_a, 4),
                "trade_off_ratio": round(trade_off_ratio, 4)
            })
        
        # Trova miglior trade-off
        best_trade_off = max(trade_offs, key=lambda x: x["trade_off_ratio"]) if trade_offs else None
        
        return {
            "criteria_analyzed": [criteria_a, criteria_b],
            "trade_offs": trade_offs,
            "best_trade_off": best_trade_off,
            "recommendation": f"Miglior compromesso: passare da {best_trade_off['from']} a {best_trade_off['to']}" if best_trade_off else None
        }
        
    async def rank(self, options: list) -> list:
        """
        Classifica una lista di opzioni
        
        Args:
            options: Lista di opzioni/scenari da classificare
            
        Returns:
            Lista ordinata per score (dal migliore al peggiore)
        """
        if not options:
            return []
        
        ranked = []
        
        for option in options:
            scored_option = await self._score_option(option)
            ranked.append(scored_option)
        
        # Ordina per score decrescente
        ranked.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        
        # Aggiungi rank position
        for i, option in enumerate(ranked):
            option["rank"] = i + 1
            option["is_recommended"] = i == 0
        
        # Salva nella history
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "options_count": len(options),
            "winner": ranked[0] if ranked else None
        })
        
        return ranked
    
    async def _score_option(self, option: dict) -> dict:
        """Calcola lo score complessivo di un'opzione"""
        scored = option.copy()
        
        # Estrai metriche
        benefit = option.get("benefit", option.get("score", 0.5))
        risk = option.get("risk", 0.2)
        probability = option.get("probability", 0.7)
        urgency = option.get("urgency", 0.5)
        effort = option.get("effort", 0.3)
        
        # Calcola score pesato
        weighted_score = (
            benefit * self.weights["benefit"] +
            risk * self.weights["risk"] +
            probability * self.weights["probability"] +
            urgency * self.weights["urgency"] +
            effort * self.weights["effort"]
        )
        
        # Normalizza tra 0 e 1
        final_score = max(0, min(1, weighted_score + 0.5))
        
        scored["final_score"] = round(final_score, 3)
        scored["score_breakdown"] = {
            "benefit_contribution": round(benefit * self.weights["benefit"], 3),
            "risk_contribution": round(risk * self.weights["risk"], 3),
            "probability_contribution": round(probability * self.weights["probability"], 3),
            "urgency_contribution": round(urgency * self.weights["urgency"], 3),
            "effort_contribution": round(effort * self.weights["effort"], 3)
        }
        
        return scored
    
    async def rank_by_criteria(self, options: list, criteria: str) -> list:
        """
        Classifica opzioni per un criterio specifico
        
        Args:
            options: Lista di opzioni
            criteria: Criterio di ordinamento (benefit, risk, probability, etc.)
            
        Returns:
            Lista ordinata per il criterio specificato
        """
        reverse = criteria not in ["risk", "effort"]  # Risk e effort: minore è meglio
        
        sorted_options = sorted(
            options,
            key=lambda x: x.get(criteria, 0),
            reverse=reverse
        )
        
        for i, option in enumerate(sorted_options):
            option[f"rank_by_{criteria}"] = i + 1
            
        return sorted_options
    
    async def multi_criteria_rank(self, options: list, criteria_weights: dict) -> list:
        """
        Ranking multi-criterio con pesi personalizzati
        
        Args:
            options: Lista di opzioni
            criteria_weights: Dizionario {criterio: peso}
            
        Returns:
            Lista ordinata per score multi-criterio
        """
        # Temporaneamente sostituisci i pesi
        original_weights = self.weights.copy()
        self.weights.update(criteria_weights)
        
        result = await self.rank(options)
        
        # Ripristina pesi originali
        self.weights = original_weights
        
        return result
    
    async def get_top_n(self, options: list, n: int = 3) -> list:
        """Restituisce le top N opzioni"""
        ranked = await self.rank(options)
        return ranked[:n]
    
    async def filter_viable(self, options: list, min_score: float = 0.3) -> list:
        """Filtra solo le opzioni con score minimo"""
        ranked = await self.rank(options)
        return [o for o in ranked if o.get("final_score", 0) >= min_score]
    
    async def compare_pair(self, option_a: dict, option_b: dict) -> dict:
        """Confronta due opzioni direttamente"""
        scored_a = await self._score_option(option_a)
        scored_b = await self._score_option(option_b)
        
        score_diff = scored_a["final_score"] - scored_b["final_score"]
        
        if abs(score_diff) < 0.05:
            result = "equivalent"
            winner = None
        elif score_diff > 0:
            result = "option_a_better"
            winner = scored_a
        else:
            result = "option_b_better"
            winner = scored_b
            
        return {
            "option_a": scored_a,
            "option_b": scored_b,
            "score_difference": round(abs(score_diff), 3),
            "result": result,
            "winner": winner,
            "confidence": min(abs(score_diff) * 5, 1.0)  # Confidenza basata sulla differenza
        }
    
    async def explain_ranking(self, ranked_options: list) -> str:
        """Genera spiegazione testuale del ranking"""
        if not ranked_options:
            return "Nessuna opzione da valutare."
        
        explanation = []
        
        # Top choice
        top = ranked_options[0]
        explanation.append(
            f"🥇 **Raccomandazione principale**: {top.get('name', 'Opzione 1')}\n"
            f"   Score: {top.get('final_score', 0):.2f}/1.00\n"
            f"   Motivo: Miglior rapporto beneficio/rischio"
        )
        
        # Runner up
        if len(ranked_options) > 1:
            second = ranked_options[1]
            explanation.append(
                f"\n🥈 **Alternativa**: {second.get('name', 'Opzione 2')}\n"
                f"   Score: {second.get('final_score', 0):.2f}/1.00"
            )
        
        # Opzione sconsigliata
        if len(ranked_options) > 2:
            last = ranked_options[-1]
            explanation.append(
                f"\n⚠️ **Sconsigliata**: {last.get('name', 'Ultima opzione')}\n"
                f"   Score: {last.get('final_score', 0):.2f}/1.00"
            )
        
        return "\n".join(explanation)
    
    def update_weights(self, new_weights: dict):
        """Aggiorna i pesi di ranking"""
        for key, value in new_weights.items():
            if key in self.weights:
                self.weights[key] = value
    
    def get_weights(self) -> dict:
        """Restituisce i pesi attuali"""
        return self.weights.copy()
    
    # ============================================
    # UTILITY FUNCTIONS
    # ============================================
    
    async def smart_rank(self, options: list, method: str = "auto") -> list:
        """
        Seleziona automaticamente il metodo di ranking migliore
        """
        if not options:
            return []
        
        n = len(options)
        
        if method == "auto":
            # Seleziona metodo in base al numero di opzioni
            if n <= 3:
                method = "weighted_sum"
            elif n <= 10:
                method = "topsis"
            else:
                method = "pareto"
        
        if method == "topsis":
            return await self.topsis_rank(options)
        elif method == "pareto":
            pareto_result = await self.pareto_rank(options)
            # Ordina frontiera di Pareto per score
            return await self.rank(pareto_result["pareto_front"])
        else:
            return await self.rank(options)
    
    async def comprehensive_analysis(self, options: list) -> dict:
        """
        Analisi completa con tutti i metodi di ranking
        """
        if not options:
            return {"error": "Nessuna opzione da analizzare"}
        
        # Esegui tutti i metodi
        weighted_result = await self.rank(options)
        topsis_result = await self.topsis_rank(options)
        pareto_result = await self.pareto_rank(options)
        trade_off = await self.trade_off_analysis(options)
        
        # Trova consensus (opzione che appare in cima in più metodi)
        top_votes = {}
        for result in [weighted_result, topsis_result]:
            if result:
                top_id = result[0].get("id", result[0].get("name"))
                top_votes[top_id] = top_votes.get(top_id, 0) + 1
        
        consensus = max(top_votes, key=top_votes.get) if top_votes else None
        
        return {
            "weighted_ranking": weighted_result[:5],
            "topsis_ranking": topsis_result[:5],
            "pareto_analysis": pareto_result,
            "trade_off_analysis": trade_off,
            "consensus_winner": consensus,
            "confidence": top_votes.get(consensus, 0) / 2 if consensus else 0,
            "recommendation": f"Tutte le analisi convergono su '{consensus}'" if top_votes.get(consensus, 0) == 2 else f"Raccomando '{weighted_result[0].get('name')}' basandosi sul ranking pesato"
        }
    
    def get_ranking_history(self, limit: int = 10) -> List[dict]:
        """Restituisce storico dei ranking"""
        return self.history[-limit:]
    
    def get_statistics(self) -> dict:
        """Statistiche sul sistema di ranking"""
        return {
            "total_rankings": len(self.history),
            "current_weights": self.weights,
            "criteria_types": self.criteria_types,
            "available_methods": [m.value for m in RankingMethod]
        }
