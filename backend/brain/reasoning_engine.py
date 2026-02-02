"""
Gideon Reasoning Engine - Autonomous Logical Brain
Handles autonomous reasoning, analysis, and decision-making
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
import json
import psutil
import platform


class ReasoningEngine:
    """Autonomous reasoning and decision-making engine"""
    
    def __init__(self):
        self.reasoning_history: List[Dict[str, Any]] = []
        self.knowledge_base: Dict[str, Any] = {}
        self.active_thoughts: List[str] = []
        self.conclusions: List[Dict[str, Any]] = []
        
    async def initialize(self):
        """Initialize reasoning engine"""
        logger.info("🧠 Initializing Reasoning Engine...")
        
        # Load base knowledge
        self.knowledge_base = {
            "system_facts": await self._gather_system_facts(),
            "capabilities": self._get_capabilities(),
            "rules": self._load_reasoning_rules()
        }
        
        logger.info("✅ Reasoning Engine ready")
    
    async def autonomous_think(
        self,
        topic: str,
        context: Optional[Dict[str, Any]] = None,
        depth: int = 3
    ) -> Dict[str, Any]:
        """
        Autonomous thinking process
        
        Args:
            topic: What to think about
            context: Additional context
            depth: How deep to think (1-5)
            
        Returns:
            Thought process with conclusions
        """
        logger.info(f"🤔 Starting autonomous thinking on: {topic}")
        
        thought_chain = []
        start_time = datetime.now()
        
        # Step 1: Understand the topic
        understanding = await self._understand_topic(topic, context)
        thought_chain.append({
            "step": 1,
            "action": "understanding",
            "thought": understanding["summary"],
            "confidence": understanding["confidence"]
        })
        
        # Step 2: Gather relevant information
        info = await self._gather_information(topic, context)
        thought_chain.append({
            "step": 2,
            "action": "information_gathering",
            "thought": f"Raccolte {len(info.get('sources', []))} fonti di informazione",
            "data": info
        })
        
        # Step 3: Analyze patterns and relationships
        analysis = await self._analyze_patterns(topic, info)
        thought_chain.append({
            "step": 3,
            "action": "pattern_analysis",
            "thought": analysis["insight"],
            "patterns_found": len(analysis.get("patterns", []))
        })
        
        # Step 4: Apply logical reasoning
        reasoning = await self._apply_logic(topic, info, analysis, depth)
        thought_chain.append({
            "step": 4,
            "action": "logical_reasoning",
            "thought": reasoning["reasoning"],
            "logical_steps": reasoning.get("steps", [])
        })
        
        # Step 5: Draw conclusions
        conclusion = await self._draw_conclusion(topic, thought_chain)
        thought_chain.append({
            "step": 5,
            "action": "conclusion",
            "thought": conclusion["statement"],
            "confidence": conclusion["confidence"],
            "recommendations": conclusion.get("recommendations", [])
        })
        
        # Calculate thinking time
        duration = (datetime.now() - start_time).total_seconds()
        
        result = {
            "topic": topic,
            "thought_chain": thought_chain,
            "conclusion": conclusion,
            "thinking_time_seconds": duration,
            "depth_used": depth,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to history
        self.reasoning_history.append(result)
        self.conclusions.append(conclusion)
        
        logger.info(f"✅ Thinking completed in {duration:.2f}s: {conclusion['statement'][:100]}")
        
        return result
    
    async def _understand_topic(
        self,
        topic: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Understand what the topic is about"""
        
        # Extract keywords
        keywords = self._extract_keywords(topic)
        
        # Classify topic type
        topic_type = self._classify_topic(topic, keywords)
        
        # Determine complexity
        complexity = self._assess_complexity(topic, context)
        
        return {
            "summary": f"Il topic riguarda {topic_type} con {len(keywords)} concetti chiave",
            "keywords": keywords,
            "type": topic_type,
            "complexity": complexity,
            "confidence": 0.85
        }
    
    async def _gather_information(
        self,
        topic: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Gather relevant information"""
        
        sources = []
        
        # System information
        if any(word in topic.lower() for word in ['sistema', 'cpu', 'memoria', 'disco', 'performance']):
            system_info = await self._get_system_information()
            sources.append({
                "type": "system_metrics",
                "data": system_info,
                "relevance": 0.9
            })
        
        # Knowledge base
        relevant_knowledge = self._search_knowledge_base(topic)
        if relevant_knowledge:
            sources.append({
                "type": "knowledge_base",
                "data": relevant_knowledge,
                "relevance": 0.7
            })
        
        # Context data
        if context:
            sources.append({
                "type": "provided_context",
                "data": context,
                "relevance": 0.8
            })
        
        return {
            "sources": sources,
            "total_data_points": sum(len(str(s["data"])) for s in sources),
            "confidence": 0.8
        }
    
    async def _analyze_patterns(
        self,
        topic: str,
        info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze patterns in the information"""
        
        patterns = []
        
        # Analyze system patterns if available
        for source in info.get("sources", []):
            if source["type"] == "system_metrics":
                data = source["data"]
                
                # CPU pattern
                cpu = data.get("cpu", 0)
                if cpu > 80:
                    patterns.append({
                        "type": "high_cpu_usage",
                        "value": cpu,
                        "severity": "warning",
                        "implication": "Possibile sovraccarico del processore"
                    })
                elif cpu < 20:
                    patterns.append({
                        "type": "low_cpu_usage",
                        "value": cpu,
                        "severity": "info",
                        "implication": "Sistema sottoutilizzato"
                    })
                
                # Memory pattern
                memory = data.get("memory", 0)
                if memory > 85:
                    patterns.append({
                        "type": "high_memory_usage",
                        "value": memory,
                        "severity": "warning",
                        "implication": "Memoria quasi esaurita"
                    })
                
                # Disk pattern
                disk = data.get("disk", 0)
                if disk > 90:
                    patterns.append({
                        "type": "disk_critical",
                        "value": disk,
                        "severity": "critical",
                        "implication": "Spazio disco critico"
                    })
        
        insight = self._generate_pattern_insight(patterns)
        
        return {
            "patterns": patterns,
            "insight": insight,
            "pattern_count": len(patterns)
        }
    
    async def _apply_logic(
        self,
        topic: str,
        info: Dict[str, Any],
        analysis: Dict[str, Any],
        depth: int
    ) -> Dict[str, Any]:
        """Apply logical reasoning"""
        
        steps = []
        
        # Step 1: Premise identification
        premises = self._identify_premises(info, analysis)
        steps.append({
            "type": "premises",
            "content": f"Identificate {len(premises)} premesse",
            "premises": premises
        })
        
        # Step 2: Deductive reasoning
        deductions = self._apply_deduction(premises)
        steps.append({
            "type": "deduction",
            "content": f"Dedotte {len(deductions)} conclusioni logiche",
            "deductions": deductions
        })
        
        # Step 3: Inductive reasoning (if depth > 2)
        if depth > 2:
            inductions = self._apply_induction(analysis.get("patterns", []))
            steps.append({
                "type": "induction",
                "content": f"Indotte {len(inductions)} generalizzazioni",
                "inductions": inductions
            })
        
        # Step 4: Abductive reasoning (if depth > 3)
        if depth > 3:
            explanations = self._apply_abduction(premises, deductions)
            steps.append({
                "type": "abduction",
                "content": f"Trovate {len(explanations)} possibili spiegazioni",
                "explanations": explanations
            })
        
        reasoning_summary = self._summarize_reasoning(steps)
        
        return {
            "reasoning": reasoning_summary,
            "steps": steps,
            "logical_validity": self._validate_logic(steps)
        }
    
    async def _draw_conclusion(
        self,
        topic: str,
        thought_chain: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Draw final conclusion from thought chain"""
        
        # Extract key insights from each step
        insights = []
        for step in thought_chain:
            if step.get("thought"):
                insights.append(step["thought"])
        
        # Synthesize conclusion
        if "sistema" in topic.lower() or "performance" in topic.lower():
            conclusion = self._conclude_system_analysis(thought_chain)
        elif "ottimizzazione" in topic.lower() or "migliorare" in topic.lower():
            conclusion = self._conclude_optimization(thought_chain)
        else:
            conclusion = self._conclude_general(thought_chain)
        
        # Add recommendations
        recommendations = self._generate_recommendations(thought_chain)
        
        return {
            "statement": conclusion,
            "confidence": self._calculate_confidence(thought_chain),
            "recommendations": recommendations,
            "key_insights": insights[:3],  # Top 3
            "reasoning_quality": self._assess_reasoning_quality(thought_chain)
        }
    
    # Helper methods
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction
        stop_words = {'il', 'la', 'di', 'da', 'in', 'con', 'su', 'per', 'a', 'e', 'che', 'è'}
        words = text.lower().split()
        keywords = [w for w in words if len(w) > 3 and w not in stop_words]
        return keywords[:5]  # Top 5
    
    def _classify_topic(self, topic: str, keywords: List[str]) -> str:
        """Classify the type of topic"""
        topic_lower = topic.lower()
        
        if any(w in topic_lower for w in ['sistema', 'cpu', 'memoria', 'disco']):
            return "analisi_sistema"
        elif any(w in topic_lower for w in ['ottimizzazione', 'migliorare', 'performance']):
            return "ottimizzazione"
        elif any(w in topic_lower for w in ['analizza', 'studia', 'esamina']):
            return "analisi"
        elif any(w in topic_lower for w in ['calcola', 'conta', 'somma']):
            return "calcolo"
        else:
            return "generale"
    
    def _assess_complexity(self, topic: str, context: Optional[Dict]) -> str:
        """Assess topic complexity"""
        word_count = len(topic.split())
        has_context = context is not None
        
        if word_count > 10 or has_context:
            return "alta"
        elif word_count > 5:
            return "media"
        else:
            return "bassa"
    
    async def _get_system_information(self) -> Dict[str, Any]:
        """Get current system information"""
        return {
            "cpu": psutil.cpu_percent(interval=0.5),
            "memory": psutil.virtual_memory().percent,
            "disk": psutil.disk_usage('/').percent,
            "processes": len(psutil.pids()),
            "platform": platform.system(),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _gather_system_facts(self) -> Dict[str, Any]:
        """Gather system facts for knowledge base"""
        return {
            "os": platform.system(),
            "os_version": platform.version(),
            "processor": platform.processor(),
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2)
        }
    
    def _get_capabilities(self) -> List[str]:
        """Get Gideon's capabilities"""
        return [
            "analisi_sistema",
            "ottimizzazione",
            "ragionamento_logico",
            "riconoscimento_pattern",
            "conclusioni_autonome",
            "calcoli_matematici",
            "analisi_dati"
        ]
    
    def _load_reasoning_rules(self) -> List[Dict[str, Any]]:
        """Load reasoning rules"""
        return [
            {"rule": "Se CPU > 80% allora sistema_sovraccarico", "type": "conditional"},
            {"rule": "Se memoria > 90% allora rischio_crash", "type": "conditional"},
            {"rule": "Se disco > 95% allora azione_urgente", "type": "conditional"},
            {"rule": "Efficienza = 100 - max(CPU%, MEM%, DISK%)", "type": "formula"}
        ]
    
    def _search_knowledge_base(self, topic: str) -> Optional[Dict[str, Any]]:
        """Search knowledge base for relevant info"""
        topic_lower = topic.lower()
        
        for key, value in self.knowledge_base.items():
            if key in topic_lower:
                return {key: value}
        
        return None
    
    def _generate_pattern_insight(self, patterns: List[Dict]) -> str:
        """Generate insight from patterns"""
        if not patterns:
            return "Nessun pattern anomalo rilevato"
        
        critical = [p for p in patterns if p.get("severity") == "critical"]
        warnings = [p for p in patterns if p.get("severity") == "warning"]
        
        if critical:
            return f"Rilevati {len(critical)} pattern critici che richiedono attenzione immediata"
        elif warnings:
            return f"Identificati {len(warnings)} pattern di warning da monitorare"
        else:
            return f"Sistema in stato normale con {len(patterns)} pattern informativi"
    
    def _identify_premises(self, info: Dict, analysis: Dict) -> List[str]:
        """Identify logical premises"""
        premises = []
        
        for source in info.get("sources", []):
            if source["type"] == "system_metrics":
                data = source["data"]
                premises.append(f"CPU attualmente a {data.get('cpu', 0):.1f}%")
                premises.append(f"Memoria utilizzata al {data.get('memory', 0):.1f}%")
                premises.append(f"Disco occupato al {data.get('disk', 0):.1f}%")
        
        return premises
    
    def _apply_deduction(self, premises: List[str]) -> List[str]:
        """Apply deductive reasoning"""
        deductions = []
        
        for premise in premises:
            if "CPU" in premise:
                try:
                    value = float(premise.split()[3].replace('%', ''))
                    if value > 80:
                        deductions.append("Il sistema è sotto stress computazionale")
                    elif value < 20:
                        deductions.append("Il sistema ha capacità di elaborazione disponibile")
                except:
                    pass
        
        return deductions
    
    def _apply_induction(self, patterns: List[Dict]) -> List[str]:
        """Apply inductive reasoning"""
        inductions = []
        
        if len(patterns) > 2:
            inductions.append("Tendenza generale verso un comportamento specifico del sistema")
        
        warning_count = sum(1 for p in patterns if p.get("severity") == "warning")
        if warning_count > 0:
            inductions.append(f"Pattern ricorrente: {warning_count} situazioni di warning")
        
        return inductions
    
    def _apply_abduction(self, premises: List[str], deductions: List[str]) -> List[str]:
        """Apply abductive reasoning (best explanation)"""
        explanations = []
        
        if any("stress" in d for d in deductions):
            explanations.append("Possibile causa: processi in background o applicazioni intensive")
        
        if any("disponibile" in d for d in deductions):
            explanations.append("Sistema ottimizzato o carico di lavoro ridotto")
        
        return explanations
    
    def _summarize_reasoning(self, steps: List[Dict]) -> str:
        """Summarize reasoning process"""
        total_conclusions = sum(len(step.get("deductions", [])) + 
                               len(step.get("inductions", [])) + 
                               len(step.get("explanations", [])) 
                               for step in steps)
        
        return f"Processo di ragionamento completato con {len(steps)} passaggi logici e {total_conclusions} conclusioni derivate"
    
    def _validate_logic(self, steps: List[Dict]) -> float:
        """Validate logical consistency"""
        # Simple validation based on completeness
        has_premises = any(s.get("type") == "premises" for s in steps)
        has_deduction = any(s.get("type") == "deduction" for s in steps)
        
        score = 0.5
        if has_premises:
            score += 0.25
        if has_deduction:
            score += 0.25
        
        return score
    
    def _conclude_system_analysis(self, thought_chain: List[Dict]) -> str:
        """Conclude system analysis"""
        # Extract system data from thought chain
        for step in thought_chain:
            if step.get("action") == "information_gathering":
                for source in step.get("data", {}).get("sources", []):
                    if source["type"] == "system_metrics":
                        data = source["data"]
                        cpu = data.get("cpu", 0)
                        memory = data.get("memory", 0)
                        disk = data.get("disk", 0)
                        
                        avg = (cpu + memory + disk) / 3
                        
                        if avg > 80:
                            return f"Analisi completata: Il sistema è sotto stress elevato ({avg:.1f}% di utilizzo medio). Raccomando ottimizzazione immediata."
                        elif avg > 60:
                            return f"Analisi completata: Il sistema opera con carico moderato ({avg:.1f}% di utilizzo medio). Performance accettabili."
                        else:
                            return f"Analisi completata: Il sistema opera in modo ottimale ({avg:.1f}% di utilizzo medio). Nessuna azione richiesta."
        
        return "Analisi completata: Dati di sistema elaborati con successo."
    
    def _conclude_optimization(self, thought_chain: List[Dict]) -> str:
        """Conclude optimization analysis"""
        patterns = []
        for step in thought_chain:
            if step.get("action") == "pattern_analysis":
                patterns = step.get("patterns_found", 0)
                break
        
        if patterns > 3:
            return f"Identificate {patterns} aree di ottimizzazione. Implementazione delle raccomandazioni porterà a miglioramento significativo delle performance."
        elif patterns > 0:
            return f"Trovate {patterns} opportunità di ottimizzazione. Miglioramenti moderati previsti."
        else:
            return "Sistema già ottimizzato. Nessun intervento necessario al momento."
    
    def _conclude_general(self, thought_chain: List[Dict]) -> str:
        """General conclusion"""
        depth = len(thought_chain)
        return f"Analisi completata attraverso {depth} livelli di ragionamento. Conclusioni elaborate autonomamente basate sui dati disponibili."
    
    def _generate_recommendations(self, thought_chain: List[Dict]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Extract patterns
        for step in thought_chain:
            if step.get("action") == "pattern_analysis":
                data = step.get("data", {})
                patterns = data.get("patterns", [])
                
                for pattern in patterns:
                    if pattern.get("severity") == "critical":
                        recommendations.append(f"URGENTE: {pattern.get('implication')}")
                    elif pattern.get("severity") == "warning":
                        recommendations.append(f"Attenzione: {pattern.get('implication')}")
        
        if not recommendations:
            recommendations.append("Continua il monitoraggio regolare del sistema")
        
        return recommendations[:5]  # Top 5
    
    def _calculate_confidence(self, thought_chain: List[Dict]) -> float:
        """Calculate confidence in conclusion"""
        # Base confidence on depth and data availability
        data_steps = sum(1 for step in thought_chain if step.get("data"))
        total_steps = len(thought_chain)
        
        if total_steps == 0:
            return 0.5
        
        confidence = (data_steps / total_steps) * 0.6 + 0.4
        return min(confidence, 0.95)
    
    def _assess_reasoning_quality(self, thought_chain: List[Dict]) -> str:
        """Assess quality of reasoning"""
        depth = len(thought_chain)
        
        if depth >= 5:
            return "eccellente"
        elif depth >= 4:
            return "buona"
        elif depth >= 3:
            return "sufficiente"
        else:
            return "base"
    
    async def shutdown(self):
        """Cleanup reasoning engine"""
        logger.info("🧠 Reasoning Engine shutting down...")
        # Save reasoning history if needed
        logger.info(f"📊 Completed {len(self.reasoning_history)} reasoning sessions")
