"""
📊 GIDEON 3.0 - Analyzer
Sistema di analisi dati e situazioni
"""

import asyncio
import psutil
from datetime import datetime
from typing import Dict, List, Any, Optional


class Analyzer:
    """
    Motore analitico di Gideon 3.0
    Analizza dati, situazioni e contesti per fornire insight
    """
    
    def __init__(self):
        self.analysis_history = []
        self.thresholds = {
            "cpu_warning": 70,
            "cpu_critical": 90,
            "memory_warning": 75,
            "memory_critical": 90,
            "disk_warning": 80,
            "disk_critical": 95
        }
        
    async def analyze(self, data: dict) -> dict:
        """
        Esegue analisi completa dei dati forniti
        
        Args:
            data: Dizionario con dati da analizzare
            
        Returns:
            Dizionario con risultati analisi
        """
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "system": await self.analyze_system(),
            "context": await self.analyze_context(data),
            "anomalies": [],
            "insights": [],
            "recommendations": []
        }
        
        # Trova anomalie
        analysis["anomalies"] = self._detect_anomalies(analysis["system"])
        
        # Genera insight
        analysis["insights"] = self._generate_insights(analysis)
        
        # Genera raccomandazioni
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        # Salva nella history
        self.analysis_history.append(analysis)
        
        return analysis
    
    async def analyze_system(self) -> dict:
        """Analizza lo stato del sistema"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Calcola metriche derivate
            cpu_status = self._get_status(cpu_percent, "cpu")
            memory_status = self._get_status(memory.percent, "memory")
            disk_status = self._get_status(disk.percent, "disk")
            
            return {
                "cpu": {
                    "percent": cpu_percent,
                    "status": cpu_status,
                    "cores": psutil.cpu_count(),
                    "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else 0
                },
                "memory": {
                    "percent": memory.percent,
                    "status": memory_status,
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2)
                },
                "disk": {
                    "percent": disk.percent,
                    "status": disk_status,
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "used_gb": round(disk.used / (1024**3), 2)
                },
                "overall_health": self._calculate_health(cpu_percent, memory.percent, disk.percent),
                "efficiency_score": self._calculate_efficiency(cpu_percent, memory.percent, disk.percent)
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def analyze_context(self, data: dict) -> dict:
        """Analizza il contesto fornito"""
        context = {
            "has_query": "query" in data,
            "query_type": None,
            "complexity": "low",
            "requires_action": False,
            "entities": []
        }
        
        if "query" in data:
            query = data["query"].lower()
            
            # Determina tipo di query
            if any(word in query for word in ["apri", "esegui", "avvia", "chiudi"]):
                context["query_type"] = "action"
                context["requires_action"] = True
                context["complexity"] = "medium"
            elif any(word in query for word in ["analizza", "controlla", "verifica", "stato"]):
                context["query_type"] = "analysis"
                context["complexity"] = "medium"
            elif any(word in query for word in ["cerca", "trova", "dove"]):
                context["query_type"] = "search"
                context["complexity"] = "medium"
            elif any(word in query for word in ["calcola", "quanto", "percentuale"]):
                context["query_type"] = "calculation"
                context["complexity"] = "low"
            elif any(word in query for word in ["prevedi", "stima", "futuro", "probabile"]):
                context["query_type"] = "prediction"
                context["complexity"] = "high"
            else:
                context["query_type"] = "information"
                context["complexity"] = "low"
                
        return context
    
    async def analyze_trends(self, metric: str, period: str = "hour") -> dict:
        """Analizza trend di una metrica nel tempo"""
        # Filtra history per periodo
        relevant_data = self._filter_by_period(period)
        
        if not relevant_data:
            return {"trend": "unknown", "data_points": 0}
        
        # Estrai valori della metrica
        values = []
        for entry in relevant_data:
            if "system" in entry and metric in entry["system"]:
                if isinstance(entry["system"][metric], dict):
                    values.append(entry["system"][metric].get("percent", 0))
                else:
                    values.append(entry["system"][metric])
        
        if len(values) < 2:
            return {"trend": "insufficient_data", "data_points": len(values)}
        
        # Calcola trend
        avg_first_half = sum(values[:len(values)//2]) / (len(values)//2)
        avg_second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
        
        diff = avg_second_half - avg_first_half
        
        if diff > 5:
            trend = "increasing"
        elif diff < -5:
            trend = "decreasing"
        else:
            trend = "stable"
            
        return {
            "trend": trend,
            "change": round(diff, 2),
            "data_points": len(values),
            "average": round(sum(values) / len(values), 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2)
        }
    
    def _get_status(self, value: float, metric_type: str) -> str:
        """Determina lo status basato sui threshold"""
        warning = self.thresholds.get(f"{metric_type}_warning", 70)
        critical = self.thresholds.get(f"{metric_type}_critical", 90)
        
        if value >= critical:
            return "critical"
        elif value >= warning:
            return "warning"
        else:
            return "normal"
    
    def _calculate_health(self, cpu: float, memory: float, disk: float) -> dict:
        """Calcola la salute complessiva del sistema"""
        # Media pesata
        weighted = (cpu * 0.4) + (memory * 0.35) + (disk * 0.25)
        health_score = 100 - weighted
        
        if health_score >= 80:
            status = "excellent"
        elif health_score >= 60:
            status = "good"
        elif health_score >= 40:
            status = "fair"
        else:
            status = "poor"
            
        return {
            "score": round(health_score, 1),
            "status": status
        }
    
    def _calculate_efficiency(self, cpu: float, memory: float, disk: float) -> float:
        """Calcola l'efficienza del sistema"""
        return round(100 - max(cpu, memory, disk), 1)
    
    def _detect_anomalies(self, system_data: dict) -> list:
        """Rileva anomalie nei dati di sistema"""
        anomalies = []
        
        if "error" in system_data:
            return anomalies
        
        # CPU anomalies
        if system_data["cpu"]["status"] == "critical":
            anomalies.append({
                "type": "cpu_overload",
                "severity": "high",
                "value": system_data["cpu"]["percent"],
                "message": f"CPU al {system_data['cpu']['percent']}% - Carico critico"
            })
        elif system_data["cpu"]["status"] == "warning":
            anomalies.append({
                "type": "cpu_high",
                "severity": "medium",
                "value": system_data["cpu"]["percent"],
                "message": f"CPU al {system_data['cpu']['percent']}% - Carico elevato"
            })
            
        # Memory anomalies
        if system_data["memory"]["status"] == "critical":
            anomalies.append({
                "type": "memory_critical",
                "severity": "high",
                "value": system_data["memory"]["percent"],
                "message": f"Memoria al {system_data['memory']['percent']}% - Critico"
            })
        elif system_data["memory"]["status"] == "warning":
            anomalies.append({
                "type": "memory_high",
                "severity": "medium",
                "value": system_data["memory"]["percent"],
                "message": f"Memoria al {system_data['memory']['percent']}% - Elevata"
            })
            
        # Disk anomalies
        if system_data["disk"]["status"] == "critical":
            anomalies.append({
                "type": "disk_full",
                "severity": "high",
                "value": system_data["disk"]["percent"],
                "message": f"Disco al {system_data['disk']['percent']}% - Quasi pieno"
            })
            
        return anomalies
    
    def _generate_insights(self, analysis: dict) -> list:
        """Genera insight dall'analisi"""
        insights = []
        system = analysis.get("system", {})
        
        if "error" in system:
            return insights
        
        # Insight sulla salute del sistema
        health = system.get("overall_health", {})
        if health.get("status") == "excellent":
            insights.append({
                "type": "positive",
                "message": "Il sistema è in ottime condizioni",
                "confidence": 0.9
            })
        elif health.get("status") == "poor":
            insights.append({
                "type": "negative",
                "message": "Il sistema richiede attenzione immediata",
                "confidence": 0.9
            })
            
        # Insight su risorse
        if system.get("memory", {}).get("available_gb", 0) < 2:
            insights.append({
                "type": "warning",
                "message": "Memoria disponibile limitata, considera di chiudere alcune applicazioni",
                "confidence": 0.8
            })
            
        if system.get("disk", {}).get("free_gb", 0) < 10:
            insights.append({
                "type": "warning",
                "message": "Spazio disco in esaurimento",
                "confidence": 0.9
            })
            
        return insights
    
    def _generate_recommendations(self, analysis: dict) -> list:
        """Genera raccomandazioni basate sull'analisi"""
        recommendations = []
        
        for anomaly in analysis.get("anomalies", []):
            if anomaly["type"] == "cpu_overload":
                recommendations.append({
                    "action": "reduce_cpu_load",
                    "priority": "high",
                    "description": "Chiudi applicazioni non necessarie o processi pesanti",
                    "automated": False
                })
            elif anomaly["type"] == "memory_critical":
                recommendations.append({
                    "action": "free_memory",
                    "priority": "high",
                    "description": "Libera memoria chiudendo applicazioni o svuotando cache",
                    "automated": True
                })
            elif anomaly["type"] == "disk_full":
                recommendations.append({
                    "action": "clean_disk",
                    "priority": "medium",
                    "description": "Elimina file temporanei e svuota cestino",
                    "automated": True
                })
                
        return recommendations
    
    def _filter_by_period(self, period: str) -> list:
        """Filtra history per periodo"""
        # Implementazione semplificata
        return self.analysis_history[-100:] if period == "hour" else self.analysis_history
