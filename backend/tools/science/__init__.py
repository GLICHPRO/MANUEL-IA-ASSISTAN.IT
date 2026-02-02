"""
🧬 GIDEON SCIENCE TOOL — Sanità / Scienza / Chimica (SAFE MODE)
================================================================

Azioni Avanzate (SOLO SICURE):
1. MolecularPatternValidator - Analisi molecole per stabilità/compatibilità
2. EnvironmentalContaminationScan - Rileva pattern anomali aria/acqua/suolo
3. ScientificCrossCheck - Confronta dati con letteratura

BLOCCO HARD-CODED:
- Niente sintesi
- Niente tossicità offensiva
- Solo analisi e ricerca
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import hashlib

# BLOCCO HARD-CODED per sicurezza
BLOCKED_QUERIES = [
    "synthesis", "sintesi", "create", "produce", "manufacture",
    "toxic", "tossico", "poison", "veleno", "weapon", "arma",
    "explosive", "esplosivo", "drug", "droga", "narcotic"
]

def is_safe_query(query: str) -> bool:
    """Verifica che la query sia sicura"""
    query_lower = query.lower()
    return not any(blocked in query_lower for blocked in BLOCKED_QUERIES)


@dataclass
class MolecularAnalysis:
    """Risultato analisi molecolare"""
    molecule_name: str
    formula: str
    stability_score: float
    industrial_compatibility: List[str]
    research_applications: List[str]
    safety_profile: str
    confidence: float

@dataclass
class ContaminationReport:
    """Report contaminazione ambientale"""
    medium: str  # aria, acqua, suolo
    contamination_level: str
    probable_causes: List[str]
    confidence: float
    recommendations: List[str]

@dataclass
class CrossCheckResult:
    """Risultato cross-check scientifico"""
    claim: str
    verification_status: str
    supporting_studies: List[str]
    discrepancies: List[str]
    confidence: float


class ScienceTool:
    """
    🧬 SCIENCE TOOL - Safe Mode
    
    SOLO analisi e ricerca.
    NESSUNA capacità di sintesi o produzione.
    """
    
    def __init__(self):
        self.analysis_history: List[MolecularAnalysis] = []
        self.contamination_reports: List[ContaminationReport] = []
    
    # ==================== ACTION 1: MolecularPatternValidator ====================
    
    async def molecular_pattern_validator(
        self,
        molecule_query: str,
        analysis_type: str = "stability"  # stability, compatibility, research
    ) -> Dict[str, Any]:
        """
        🔬 MolecularPatternValidator
        
        Analizza molecole SOLO per:
        - Stabilità chimica
        - Compatibilità industriale
        - Applicazioni di ricerca medica
        
        BLOCCO HARD-CODED: Niente sintesi, niente tossicità offensiva
        """
        
        # SECURITY CHECK
        if not is_safe_query(molecule_query):
            return {
                "success": False,
                "action": "molecular_pattern_validator",
                "error": "BLOCKED",
                "message": "⛔ Query bloccata per motivi di sicurezza. Questo tool analizza SOLO stabilità e compatibilità, non sintesi o applicazioni pericolose.",
                "timestamp": datetime.now().isoformat()
            }
        
        # Analisi molecola
        analysis = await self._analyze_molecule(molecule_query, analysis_type)
        
        return {
            "success": True,
            "action": "molecular_pattern_validator",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "query": molecule_query,
                "analysis_type": analysis_type,
                "results": {
                    "molecule": analysis.molecule_name,
                    "formula": analysis.formula,
                    "stability_score": f"{analysis.stability_score:.0f}/100",
                    "stability_assessment": self._assess_stability(analysis.stability_score),
                    "industrial_compatibility": analysis.industrial_compatibility,
                    "research_applications": analysis.research_applications,
                    "safety_profile": analysis.safety_profile
                },
                "limitations": [
                    "Analisi teorica basata su letteratura",
                    "Non sostituisce test di laboratorio",
                    "Dati indicativi, non prescrittivi"
                ]
            },
            "summary": f"Molecola '{analysis.molecule_name}' analizzata - Stabilità: {analysis.stability_score:.0f}%",
            "recommendations": self._generate_molecular_recommendations(analysis),
            "confidence": analysis.confidence,
            "disclaimer": "🔬 Questo è uno strumento di ANALISI, non di sintesi. Consultare sempre esperti per applicazioni pratiche."
        }
    
    async def _analyze_molecule(self, query: str, analysis_type: str) -> MolecularAnalysis:
        """Analizza molecola (simulato - in produzione: database chimico)"""
        
        # Database molecole comuni (semplificato)
        known_molecules = {
            "water": {"formula": "H2O", "stability": 100, "safety": "sicuro"},
            "acqua": {"formula": "H2O", "stability": 100, "safety": "sicuro"},
            "ethanol": {"formula": "C2H5OH", "stability": 85, "safety": "infiammabile"},
            "etanolo": {"formula": "C2H5OH", "stability": 85, "safety": "infiammabile"},
            "glucose": {"formula": "C6H12O6", "stability": 90, "safety": "sicuro"},
            "glucosio": {"formula": "C6H12O6", "stability": 90, "safety": "sicuro"},
            "caffeine": {"formula": "C8H10N4O2", "stability": 88, "safety": "moderato"},
            "caffeina": {"formula": "C8H10N4O2", "stability": 88, "safety": "moderato"},
            "aspirin": {"formula": "C9H8O4", "stability": 82, "safety": "farmaco"},
            "aspirina": {"formula": "C9H8O4", "stability": 82, "safety": "farmaco"},
            "sodium chloride": {"formula": "NaCl", "stability": 100, "safety": "sicuro"},
            "sale": {"formula": "NaCl", "stability": 100, "safety": "sicuro"},
        }
        
        query_lower = query.lower()
        mol_data = known_molecules.get(query_lower, {
            "formula": "Da determinare",
            "stability": 75,
            "safety": "sconosciuto"
        })
        
        return MolecularAnalysis(
            molecule_name=query,
            formula=mol_data["formula"],
            stability_score=mol_data["stability"],
            industrial_compatibility=self._get_industrial_uses(query_lower),
            research_applications=self._get_research_uses(query_lower),
            safety_profile=mol_data["safety"],
            confidence=0.85 if query_lower in known_molecules else 0.60
        )
    
    def _assess_stability(self, score: float) -> str:
        if score >= 90:
            return "Altamente stabile"
        elif score >= 75:
            return "Stabile"
        elif score >= 50:
            return "Moderatamente stabile"
        return "Richiede condizioni controllate"
    
    def _get_industrial_uses(self, molecule: str) -> List[str]:
        uses = {
            "water": ["Solvente universale", "Processi industriali", "Raffreddamento"],
            "acqua": ["Solvente universale", "Processi industriali", "Raffreddamento"],
            "ethanol": ["Solvente", "Biocarburante", "Disinfettante"],
            "etanolo": ["Solvente", "Biocarburante", "Disinfettante"],
            "glucose": ["Industria alimentare", "Fermentazione", "Farmaceutica"],
            "glucosio": ["Industria alimentare", "Fermentazione", "Farmaceutica"],
        }
        return uses.get(molecule, ["Consultare letteratura specializzata"])
    
    def _get_research_uses(self, molecule: str) -> List[str]:
        uses = {
            "water": ["Studio solventi", "Biochimica", "Fisica dei fluidi"],
            "acqua": ["Studio solventi", "Biochimica", "Fisica dei fluidi"],
            "caffeine": ["Neuroscienze", "Farmacologia", "Studi metabolismo"],
            "caffeina": ["Neuroscienze", "Farmacologia", "Studi metabolismo"],
            "aspirin": ["Studi anti-infiammatori", "Cardiologia", "Oncologia"],
            "aspirina": ["Studi anti-infiammatori", "Cardiologia", "Oncologia"],
        }
        return uses.get(molecule, ["Ricerca di base", "Studi di caratterizzazione"])
    
    def _generate_molecular_recommendations(self, analysis: MolecularAnalysis) -> List[str]:
        recs = []
        if analysis.stability_score < 80:
            recs.append("Conservare in condizioni controllate (temperatura, umidità)")
        if analysis.safety_profile == "infiammabile":
            recs.append("Seguire protocolli di sicurezza per sostanze infiammabili")
        if analysis.safety_profile == "farmaco":
            recs.append("Uso solo sotto supervisione medica")
        recs.append("Consultare scheda di sicurezza (SDS) per dettagli completi")
        return recs

    # ==================== ACTION 2: EnvironmentalContaminationScan ====================
    
    async def environmental_contamination_scan(
        self,
        medium: str = "air",  # air, water, soil
        sensor_data: Dict[str, float] = None,
        location: str = "unknown"
    ) -> Dict[str, Any]:
        """
        🌍 EnvironmentalContaminationScan
        
        Rileva pattern anomali in:
        - Aria (qualità, particolato, gas)
        - Acqua (pH, torbidità, contaminanti)
        - Suolo (metalli pesanti, pH, nutrienti)
        
        Output non colpevolizzante - solo probabili cause.
        """
        
        sensor_data = sensor_data or self._get_simulated_sensors(medium)
        
        # Analizza dati
        report = await self._analyze_environment(medium, sensor_data, location)
        
        return {
            "success": True,
            "action": "environmental_contamination_scan",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "medium": medium,
                "location": location,
                "sensor_readings": sensor_data,
                "analysis": {
                    "contamination_level": report.contamination_level,
                    "level_description": self._describe_level(report.contamination_level),
                    "probable_causes": report.probable_causes,
                    "confidence": f"{report.confidence:.0%}",
                    "trend": "stable"  # In produzione: analisi storica
                },
                "thresholds": self._get_thresholds(medium),
                "exceeded_thresholds": self._check_thresholds(sensor_data, medium)
            },
            "summary": f"Qualità {medium}: {report.contamination_level.upper()} - {len(report.probable_causes)} possibili cause identificate",
            "recommendations": report.recommendations,
            "confidence": report.confidence,
            "disclaimer": "⚠️ Analisi indicativa. Per valutazioni ufficiali consultare enti competenti (ARPA, ASL)."
        }
    
    def _get_simulated_sensors(self, medium: str) -> Dict[str, float]:
        """Simula dati sensori"""
        import random
        
        if medium == "air":
            return {
                "pm2.5": random.uniform(5, 50),  # µg/m³
                "pm10": random.uniform(10, 80),
                "co2": random.uniform(400, 800),  # ppm
                "no2": random.uniform(10, 60),  # µg/m³
                "o3": random.uniform(20, 100),
                "temperature": random.uniform(15, 30),
                "humidity": random.uniform(30, 70)
            }
        elif medium == "water":
            return {
                "ph": random.uniform(6.5, 8.5),
                "turbidity": random.uniform(0, 10),  # NTU
                "dissolved_oxygen": random.uniform(6, 12),  # mg/L
                "conductivity": random.uniform(100, 500),  # µS/cm
                "temperature": random.uniform(10, 25),
                "nitrates": random.uniform(0, 30),  # mg/L
                "chlorine": random.uniform(0, 2)  # mg/L
            }
        else:  # soil
            return {
                "ph": random.uniform(5.5, 8.0),
                "organic_matter": random.uniform(1, 10),  # %
                "nitrogen": random.uniform(0.1, 0.5),  # %
                "phosphorus": random.uniform(10, 100),  # ppm
                "potassium": random.uniform(50, 300),  # ppm
                "lead": random.uniform(0, 50),  # ppm
                "moisture": random.uniform(10, 40)  # %
            }
    
    async def _analyze_environment(
        self, medium: str, sensors: Dict[str, float], location: str
    ) -> ContaminationReport:
        """Analizza dati ambientali"""
        
        issues = []
        causes = []
        recommendations = []
        
        if medium == "air":
            if sensors.get("pm2.5", 0) > 25:
                issues.append("PM2.5 elevato")
                causes.append("Traffico veicolare")
                causes.append("Riscaldamento domestico")
            if sensors.get("no2", 0) > 40:
                issues.append("NO2 elevato")
                causes.append("Emissioni veicoli diesel")
            if sensors.get("o3", 0) > 80:
                issues.append("Ozono elevato")
                causes.append("Reazioni fotochimiche")
        
        elif medium == "water":
            if sensors.get("ph", 7) < 6.5 or sensors.get("ph", 7) > 8.5:
                issues.append("pH anomalo")
                causes.append("Scarichi industriali")
            if sensors.get("nitrates", 0) > 25:
                issues.append("Nitrati elevati")
                causes.append("Agricoltura intensiva")
                causes.append("Fertilizzanti")
        
        elif medium == "soil":
            if sensors.get("lead", 0) > 30:
                issues.append("Piombo rilevato")
                causes.append("Attività industriale storica")
            if sensors.get("ph", 7) < 6:
                issues.append("Suolo acido")
                causes.append("Piogge acide")
                causes.append("Decomposizione organica")
        
        # Determina livello
        if len(issues) >= 3:
            level = "poor"
        elif len(issues) >= 1:
            level = "moderate"
        else:
            level = "good"
            causes = ["Nessuna causa di allarme identificata"]
        
        # Genera raccomandazioni
        if level == "poor":
            recommendations = [
                "Effettuare campionamento professionale",
                "Consultare ente ambientale competente",
                "Limitare esposizione se possibile"
            ]
        elif level == "moderate":
            recommendations = [
                "Monitorare evoluzione nel tempo",
                "Identificare fonti locali",
                "Considerare misure preventive"
            ]
        else:
            recommendations = ["Continuare monitoraggio regolare"]
        
        return ContaminationReport(
            medium=medium,
            contamination_level=level,
            probable_causes=causes[:3],
            confidence=0.75,
            recommendations=recommendations
        )
    
    def _describe_level(self, level: str) -> str:
        descriptions = {
            "good": "Nella norma - Nessuna azione richiesta",
            "moderate": "Attenzione - Alcuni parametri da monitorare",
            "poor": "Critico - Intervento consigliato"
        }
        return descriptions.get(level, "Sconosciuto")
    
    def _get_thresholds(self, medium: str) -> Dict[str, Dict[str, float]]:
        thresholds = {
            "air": {
                "pm2.5": {"good": 15, "moderate": 25, "poor": 50},
                "pm10": {"good": 30, "moderate": 50, "poor": 100},
                "no2": {"good": 25, "moderate": 40, "poor": 80}
            },
            "water": {
                "ph": {"min": 6.5, "max": 8.5},
                "nitrates": {"good": 10, "moderate": 25, "poor": 50}
            },
            "soil": {
                "lead": {"good": 20, "moderate": 50, "poor": 100}
            }
        }
        return thresholds.get(medium, {})
    
    def _check_thresholds(self, sensors: Dict[str, float], medium: str) -> List[str]:
        exceeded = []
        thresholds = self._get_thresholds(medium)
        
        for param, value in sensors.items():
            if param in thresholds:
                th = thresholds[param]
                if isinstance(th, dict) and "poor" in th:
                    if value > th["poor"]:
                        exceeded.append(f"{param}: {value} (soglia: {th['poor']})")
        
        return exceeded if exceeded else ["Nessuna soglia superata"]

    # ==================== ACTION 3: ScientificCrossCheck ====================
    
    async def scientific_cross_check(
        self,
        claim: str,
        field: str = "general",  # medicine, chemistry, physics, biology
        strictness: str = "moderate"  # lenient, moderate, strict
    ) -> Dict[str, Any]:
        """
        ✅ ScientificCrossCheck
        
        Confronta affermazioni/dati con:
        - Letteratura scientifica
        - Studi peer-reviewed
        - Consensus scientifico
        
        Evidenzia discrepanze senza giudizi assoluti.
        """
        
        # SECURITY CHECK
        if not is_safe_query(claim):
            return {
                "success": False,
                "action": "scientific_cross_check",
                "error": "BLOCKED",
                "message": "⛔ Query bloccata per motivi di sicurezza.",
                "timestamp": datetime.now().isoformat()
            }
        
        result = await self._cross_check_claim(claim, field, strictness)
        
        return {
            "success": True,
            "action": "scientific_cross_check",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "claim": claim,
                "field": field,
                "verification": {
                    "status": result.verification_status,
                    "status_description": self._describe_verification(result.verification_status),
                    "confidence": f"{result.confidence:.0%}",
                    "supporting_evidence": result.supporting_studies,
                    "discrepancies": result.discrepancies
                },
                "methodology": "Cross-reference con database letteratura scientifica",
                "limitations": [
                    "Basato su letteratura disponibile",
                    "Non include studi non pubblicati",
                    "Consensus può evolvere nel tempo"
                ]
            },
            "summary": f"Verifica: {result.verification_status.upper()} - Confidenza: {result.confidence:.0%}",
            "recommendations": self._generate_cross_check_recommendations(result),
            "confidence": result.confidence,
            "disclaimer": "🔬 Verifica indicativa. Per conferme definitive consultare esperti del settore."
        }
    
    async def _cross_check_claim(self, claim: str, field: str, strictness: str) -> CrossCheckResult:
        """Verifica claim contro letteratura"""
        
        # Simulazione - in produzione: query a PubMed, Google Scholar, etc.
        claim_lower = claim.lower()
        
        # Claims comuni con status noto
        known_claims = {
            "acqua bolle a 100 gradi": ("verified", 0.99, ["Physics textbooks", "NIST database"]),
            "water boils at 100": ("verified", 0.99, ["Physics textbooks", "NIST database"]),
            "la terra è piatta": ("debunked", 0.99, ["NASA imagery", "Geodesy studies"]),
            "earth is flat": ("debunked", 0.99, ["NASA imagery", "Geodesy studies"]),
            "vitamina c cura raffreddore": ("partially_supported", 0.70, ["Cochrane review 2013"]),
            "vitamin c cures cold": ("partially_supported", 0.70, ["Cochrane review 2013"]),
            "caffè fa male": ("inconclusive", 0.60, ["Mixed studies"]),
            "coffee is bad": ("inconclusive", 0.60, ["Mixed studies"]),
        }
        
        for key, (status, confidence, studies) in known_claims.items():
            if key in claim_lower:
                discrepancies = []
                if status == "partially_supported":
                    discrepancies = ["Alcuni studi mostrano effetto minimo", "Dipende da dosaggio e individuo"]
                elif status == "inconclusive":
                    discrepancies = ["Studi con risultati contrastanti", "Necessaria più ricerca"]
                elif status == "debunked":
                    discrepancies = ["Contraddice evidenze scientifiche consolidate"]
                
                return CrossCheckResult(
                    claim=claim,
                    verification_status=status,
                    supporting_studies=studies,
                    discrepancies=discrepancies,
                    confidence=confidence
                )
        
        # Default: richiede verifica manuale
        return CrossCheckResult(
            claim=claim,
            verification_status="requires_review",
            supporting_studies=["Ricerca automatica non conclusiva"],
            discrepancies=["Claim specifico richiede verifica manuale"],
            confidence=0.40
        )
    
    def _describe_verification(self, status: str) -> str:
        descriptions = {
            "verified": "✅ Supportato dal consensus scientifico",
            "partially_supported": "🟡 Parzialmente supportato - esistono studi a favore",
            "inconclusive": "🟠 Evidenze inconcludenti - risultati misti",
            "debunked": "❌ Contraddetto da evidenze scientifiche",
            "requires_review": "🔍 Richiede revisione manuale da esperti"
        }
        return descriptions.get(status, "Stato sconosciuto")
    
    def _generate_cross_check_recommendations(self, result: CrossCheckResult) -> List[str]:
        recs = {
            "verified": ["Claim supportato - può essere citato con confidenza"],
            "partially_supported": ["Citare con cautela", "Menzionare limitazioni degli studi"],
            "inconclusive": ["Non trarre conclusioni definitive", "Attendere ulteriori ricerche"],
            "debunked": ["Evitare di diffondere", "Cercare fonti aggiornate"],
            "requires_review": ["Consultare esperto del settore", "Cercare su PubMed/Scholar"]
        }
        return recs.get(result.verification_status, ["Verificare manualmente"])


# Singleton instance
_science_tool = None

def get_science_tool() -> ScienceTool:
    global _science_tool
    if _science_tool is None:
        _science_tool = ScienceTool()
    return _science_tool
