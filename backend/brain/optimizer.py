"""
Optimizer Engine - System Analysis and Optimization Suggestions
Analyzes performance and provides intelligent optimization recommendations
"""

from typing import Dict, List, Any
from loguru import logger
import asyncio
import psutil
import random
from datetime import datetime


class OptimizerEngine:
    """Engine for analyzing and optimizing systems"""
    
    def __init__(self):
        self.analysis_cache = {}
        self.optimization_rules = []
        
    async def initialize(self):
        """Initialize optimizer"""
        logger.info("⚙️ Initializing Optimizer Engine...")
        await self._load_optimization_rules()
        logger.info("✅ Optimizer ready")
        
    async def shutdown(self):
        """Shutdown optimizer"""
        pass
        
    async def _load_optimization_rules(self):
        """Load optimization rules"""
        self.optimization_rules = [
            {
                "id": "cpu_high",
                "condition": lambda metrics: metrics.get("cpu_percent", 0) > 80,
                "description": "Ridurre il carico CPU eliminando processi non necessari",
                "impact_percent": 25.0,
                "priority": "high"
            },
            {
                "id": "memory_high",
                "condition": lambda metrics: metrics.get("memory_percent", 0) > 80,
                "description": "Ottimizzare l'uso della memoria con caching più efficiente",
                "impact_percent": 30.0,
                "priority": "high"
            },
            {
                "id": "disk_high",
                "condition": lambda metrics: metrics.get("disk_percent", 0) > 90,
                "description": "Liberare spazio su disco rimuovendo file temporanei",
                "impact_percent": 15.0,
                "priority": "medium"
            },
            {
                "id": "response_time",
                "condition": lambda metrics: metrics.get("avg_response_time", 0) > 1000,
                "description": "Migliorare i tempi di risposta con query optimization",
                "impact_percent": 40.0,
                "priority": "high"
            },
            {
                "id": "connection_pool",
                "condition": lambda metrics: metrics.get("db_connections", 0) > 80,
                "description": "Aumentare il pool di connessioni al database",
                "impact_percent": 20.0,
                "priority": "medium"
            }
        ]
        
    async def analyze(self, target: str) -> Dict[str, Any]:
        """
        Analyze a target system/component
        
        Args:
            target: Target to analyze (system, database, network, etc.)
            
        Returns:
            Analysis results with issues and optimizations
        """
        logger.info(f"🔍 Analyzing: {target}")
        
        # Collect metrics based on target
        metrics = await self._collect_metrics(target)
        
        # Identify issues
        issues = await self._identify_issues(metrics)
        
        # Calculate efficiency score
        score = await self._calculate_efficiency_score(metrics, issues)
        
        # Get optimizations
        optimizations = await self._get_applicable_optimizations(metrics)
        
        analysis = {
            "target": target,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "score": score,
            "issues": issues,
            "optimizations": optimizations,
            "summary": self._generate_summary(score, issues, optimizations)
        }
        
        # Cache result
        self.analysis_cache[target] = analysis
        
        return analysis
        
    async def _collect_metrics(self, target: str) -> Dict[str, Any]:
        """Collect metrics for target"""
        metrics = {}
        
        if target in ["system", "server"]:
            # System metrics
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics.update({
                "cpu_percent": cpu,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2)
            })
            
        elif target == "database":
            # Simulated database metrics
            metrics.update({
                "db_connections": random.randint(50, 95),
                "query_time_avg_ms": random.randint(50, 1500),
                "cache_hit_rate": random.uniform(0.6, 0.95),
                "index_efficiency": random.uniform(0.7, 0.98)
            })
            
        elif target == "network":
            # Network metrics
            net_io = psutil.net_io_counters()
            metrics.update({
                "bytes_sent_mb": round(net_io.bytes_sent / (1024**2), 2),
                "bytes_recv_mb": round(net_io.bytes_recv / (1024**2), 2),
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
                "errors": net_io.errin + net_io.errout
            })
        
        # Add common metrics
        metrics["avg_response_time"] = random.randint(100, 2000)
        
        return metrics
        
    async def _identify_issues(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify issues from metrics"""
        issues = []
        
        # CPU issues
        if metrics.get("cpu_percent", 0) > 80:
            issues.append({
                "type": "performance",
                "severity": "high",
                "component": "CPU",
                "message": f"Utilizzo CPU elevato: {metrics['cpu_percent']:.1f}%",
                "value": metrics["cpu_percent"]
            })
        
        # Memory issues
        if metrics.get("memory_percent", 0) > 80:
            issues.append({
                "type": "resource",
                "severity": "high",
                "component": "Memory",
                "message": f"Memoria quasi esaurita: {metrics['memory_percent']:.1f}%",
                "value": metrics["memory_percent"]
            })
        
        # Disk issues
        if metrics.get("disk_percent", 0) > 90:
            issues.append({
                "type": "storage",
                "severity": "critical",
                "component": "Disk",
                "message": f"Spazio disco critico: {metrics['disk_percent']:.1f}%",
                "value": metrics["disk_percent"]
            })
        
        # Response time issues
        if metrics.get("avg_response_time", 0) > 1000:
            issues.append({
                "type": "performance",
                "severity": "medium",
                "component": "Response Time",
                "message": f"Tempi di risposta elevati: {metrics['avg_response_time']}ms",
                "value": metrics["avg_response_time"]
            })
        
        return issues
        
    async def _calculate_efficiency_score(
        self,
        metrics: Dict[str, Any],
        issues: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall efficiency score (0-1)"""
        
        # Start with perfect score
        score = 1.0
        
        # Deduct points for issues
        severity_weights = {
            "critical": 0.25,
            "high": 0.15,
            "medium": 0.10,
            "low": 0.05
        }
        
        for issue in issues:
            severity = issue.get("severity", "low")
            score -= severity_weights.get(severity, 0.05)
        
        # Consider specific metrics
        cpu = metrics.get("cpu_percent", 0)
        memory = metrics.get("memory_percent", 0)
        
        if cpu > 70:
            score -= (cpu - 70) * 0.002
        if memory > 70:
            score -= (memory - 70) * 0.002
        
        return max(0.0, min(1.0, score))
        
    async def _get_applicable_optimizations(
        self,
        metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get applicable optimizations based on metrics"""
        applicable = []
        
        for rule in self.optimization_rules:
            try:
                if rule["condition"](metrics):
                    applicable.append({
                        "id": rule["id"],
                        "description": rule["description"],
                        "impact_percent": rule["impact_percent"],
                        "priority": rule["priority"],
                        "estimated_improvement": f"+{rule['impact_percent']:.1f}%"
                    })
            except Exception as e:
                logger.error(f"Error evaluating rule {rule['id']}: {e}")
        
        # Sort by impact
        applicable.sort(key=lambda x: x["impact_percent"], reverse=True)
        
        return applicable
        
    def _generate_summary(
        self,
        score: float,
        issues: List[Dict],
        optimizations: List[Dict]
    ) -> str:
        """Generate human-readable summary"""
        
        score_pct = score * 100
        
        if score >= 0.9:
            status = "eccellente"
        elif score >= 0.7:
            status = "buono"
        elif score >= 0.5:
            status = "accettabile"
        else:
            status = "critico"
        
        summary = f"Stato {status} con efficienza al {score_pct:.1f}%. "
        
        if issues:
            summary += f"Rilevati {len(issues)} problemi. "
        
        if optimizations:
            total_impact = sum(opt["impact_percent"] for opt in optimizations[:3])
            summary += f"Possibile miglioramento fino al +{total_impact:.1f}% con le ottimizzazioni suggerite."
        else:
            summary += "Nessuna ottimizzazione necessaria al momento."
        
        return summary
        
    async def get_optimizations(self, target: str) -> List[Dict[str, Any]]:
        """Get optimizations for target"""
        
        # Check cache first
        if target in self.analysis_cache:
            return self.analysis_cache[target].get("optimizations", [])
        
        # Run new analysis
        analysis = await self.analyze(target)
        return analysis.get("optimizations", [])
        
    async def comprehensive_analysis(self, target: str) -> Dict[str, Any]:
        """Perform comprehensive multi-aspect analysis"""
        
        results = {
            "target": target,
            "timestamp": datetime.now().isoformat(),
            "analyses": {}
        }
        
        # Analyze different aspects
        aspects = ["system", "performance", "security", "optimization"]
        
        for aspect in aspects:
            results["analyses"][aspect] = await self.analyze(f"{target}_{aspect}")
        
        # Aggregate scores
        scores = [a["score"] for a in results["analyses"].values()]
        results["overall_score"] = sum(scores) / len(scores) if scores else 0.0
        
        # Aggregate issues
        all_issues = []
        for analysis in results["analyses"].values():
            all_issues.extend(analysis.get("issues", []))
        results["total_issues"] = len(all_issues)
        
        # Get top optimizations
        all_opts = []
        for analysis in results["analyses"].values():
            all_opts.extend(analysis.get("optimizations", []))
        results["top_optimizations"] = sorted(
            all_opts,
            key=lambda x: x["impact_percent"],
            reverse=True
        )[:5]
        
        return results
