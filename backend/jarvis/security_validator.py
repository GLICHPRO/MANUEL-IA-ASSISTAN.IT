"""
🛡️ JARVIS CORE - Security Validator (Avanzato)

Validatore di sicurezza che:
- Verifica permessi utente per ogni azione
- Valuta rischi con scoring multi-livello
- Blocca azioni pericolose automaticamente
- Richiede conferme per azioni sensibili
- Mantiene audit log completo

Features:
- Risk assessment multi-fattore
- Blacklist/Whitelist per azioni
- Rate limiting per abuse prevention
- Context-aware security (mode, time, location)
- Escalation automatica per azioni critiche
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import asyncio
import re


# ========== ENUMS ==========

class ThreatLevel(Enum):
    """Livelli di minaccia"""
    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    BLOCKED = 5


class PermissionLevel(Enum):
    """Livelli di permesso"""
    PUBLIC = 0          # Tutti possono eseguire
    AUTHENTICATED = 1   # Richiede autenticazione
    CONFIRMED = 2       # Richiede conferma vocale
    ELEVATED = 3        # Richiede conferma esplicita
    ADMIN = 4           # Richiede PIN
    CRITICAL = 5        # Doppia conferma + timeout


class SecurityAction(Enum):
    """Azioni di sicurezza"""
    ALLOW = "allow"
    ALLOW_MONITORED = "allow_monitored"
    REQUIRE_CONFIRMATION = "require_confirmation"
    REQUIRE_PIN = "require_pin"
    RATE_LIMIT = "rate_limit"
    BLOCK = "block"
    ESCALATE = "escalate"


class ViolationType(Enum):
    """Tipi di violazione"""
    PERMISSION_DENIED = "permission_denied"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    BLACKLISTED_ACTION = "blacklisted_action"
    DANGEROUS_PARAMS = "dangerous_params"
    CONTEXT_VIOLATION = "context_violation"
    SUSPICIOUS_PATTERN = "suspicious_pattern"


# ========== DATACLASSES ==========

@dataclass
class RiskAssessment:
    """Risultato valutazione rischio"""
    threat_level: ThreatLevel
    risk_score: float  # 0.0 - 1.0
    
    # Dettagli
    risk_factors: List[str] = field(default_factory=list)
    mitigations: List[str] = field(default_factory=list)
    
    # Raccomandazione
    recommended_action: SecurityAction = SecurityAction.ALLOW
    requires_confirmation: bool = False
    confirmation_type: str = "vocal"  # vocal, explicit, pin
    
    # Metadata
    assessed_at: datetime = field(default_factory=datetime.now)
    context_flags: Dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "threat_level": self.threat_level.name,
            "risk_score": round(self.risk_score, 3),
            "risk_factors": self.risk_factors,
            "mitigations": self.mitigations,
            "recommended_action": self.recommended_action.value,
            "requires_confirmation": self.requires_confirmation,
            "confirmation_type": self.confirmation_type
        }


@dataclass
class SecurityViolation:
    """Registra una violazione di sicurezza"""
    id: str
    violation_type: ViolationType
    action: str
    params: Dict
    severity: ThreatLevel
    
    # Context
    user: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    blocked: bool = True
    
    # Details
    reason: str = ""
    evidence: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.violation_type.value,
            "action": self.action,
            "severity": self.severity.name,
            "timestamp": self.timestamp.isoformat(),
            "blocked": self.blocked,
            "reason": self.reason
        }


@dataclass
class PendingConfirmation:
    """Conferma in attesa"""
    id: str
    action: str
    params: Dict
    confirmation_type: str
    
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = None
    confirmed: bool = False
    
    risk_assessment: Optional[RiskAssessment] = None
    context: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.expires_at:
            # Default 2 minuti
            self.expires_at = self.created_at + timedelta(minutes=2)


# ========== SECURITY VALIDATOR ==========

class SecurityValidator:
    """
    🛡️ Validatore di Sicurezza Avanzato
    
    Verifica permessi, valuta rischi, blocca azioni pericolose.
    
    Pipeline:
    1. validate_action(action) → RiskAssessment
    2. check_permission(action) → allowed/denied
    3. assess_risk(action) → threat level
    4. get_recommendation(action) → SecurityAction
    """
    
    def __init__(self):
        # Auth state
        self._authenticated = False
        self._current_user: Optional[str] = None
        self._session_start: Optional[datetime] = None
        self._session_timeout = timedelta(hours=8)
        
        # PIN hash (default: 1234)
        self._admin_pin_hash = self._hash("1234")
        
        # Permission mappings
        self._action_permissions = self._build_permission_map()
        
        # Risk factors
        self._risk_factors = self._build_risk_factors()
        
        # Blacklist/Whitelist
        self._blacklisted_actions: Set[str] = set()
        self._whitelisted_actions: Set[str] = {"time", "date", "help", "notify"}
        
        # Dangerous patterns (regex)
        self._dangerous_patterns = self._build_dangerous_patterns()
        
        # Rate limiting
        self._rate_limits: Dict[str, int] = {
            "shutdown": 1,      # 1 per session
            "restart": 1,
            "delete_file": 10,  # 10 per hour
            "send_email": 20,
            "run_command": 30,
            "default": 100      # 100 per hour
        }
        self._action_counts: Dict[str, List[datetime]] = {}
        
        # Pending confirmations
        self._pending_confirmations: Dict[str, PendingConfirmation] = {}
        
        # Violations log
        self._violations: List[SecurityViolation] = []
        self._max_violations = 500
        
        # Audit log
        self._audit_log: List[Dict] = []
        self._max_audit = 1000
        
        # Callbacks
        self._confirmation_callback: Optional[Callable] = None
        self._violation_callback: Optional[Callable] = None
        
        # Stats
        self._stats = {
            "total_validations": 0,
            "blocked": 0,
            "confirmed": 0,
            "violations": 0
        }
    
    # ========== BUILD METHODS ==========
    
    def _build_permission_map(self) -> Dict[str, PermissionLevel]:
        """Costruisce mappa permessi"""
        return {
            # PUBLIC - sempre permesso
            "notify": PermissionLevel.PUBLIC,
            "respond": PermissionLevel.PUBLIC,
            "help": PermissionLevel.PUBLIC,
            "time": PermissionLevel.PUBLIC,
            "date": PermissionLevel.PUBLIC,
            "weather": PermissionLevel.PUBLIC,
            "calculate": PermissionLevel.PUBLIC,
            
            # AUTHENTICATED - richiede login
            "search_web": PermissionLevel.AUTHENTICATED,
            "open_url": PermissionLevel.AUTHENTICATED,
            "open_app": PermissionLevel.AUTHENTICATED,
            "open_file": PermissionLevel.AUTHENTICATED,
            "set_volume": PermissionLevel.AUTHENTICATED,
            "mute": PermissionLevel.AUTHENTICATED,
            "play_music": PermissionLevel.AUTHENTICATED,
            "screenshot": PermissionLevel.AUTHENTICATED,
            
            # CONFIRMED - richiede conferma vocale
            "create_file": PermissionLevel.CONFIRMED,
            "copy_file": PermissionLevel.CONFIRMED,
            "move_file": PermissionLevel.CONFIRMED,
            "close_app": PermissionLevel.CONFIRMED,
            "send_notification": PermissionLevel.CONFIRMED,
            
            # ELEVATED - conferma esplicita
            "run_command": PermissionLevel.ELEVATED,
            "run_script": PermissionLevel.ELEVATED,
            "send_email": PermissionLevel.ELEVATED,
            "modify_settings": PermissionLevel.ELEVATED,
            
            # ADMIN - richiede PIN
            "delete_file": PermissionLevel.ADMIN,
            "delete_folder": PermissionLevel.ADMIN,
            "lock": PermissionLevel.ADMIN,
            "uninstall": PermissionLevel.ADMIN,
            "kill_process": PermissionLevel.ADMIN,
            
            # CRITICAL - doppia conferma
            "shutdown": PermissionLevel.CRITICAL,
            "restart": PermissionLevel.CRITICAL,
            "sleep": PermissionLevel.CRITICAL,
            "format": PermissionLevel.CRITICAL,
            "modify_registry": PermissionLevel.CRITICAL,
            "delete_system": PermissionLevel.CRITICAL
        }
    
    def _build_risk_factors(self) -> Dict[str, Dict]:
        """Costruisce fattori di rischio per azioni"""
        return {
            # System critical
            "shutdown": {"base_risk": 0.9, "reversible": False, "impact": "system"},
            "restart": {"base_risk": 0.85, "reversible": False, "impact": "system"},
            "format": {"base_risk": 1.0, "reversible": False, "impact": "data_loss"},
            "delete_system": {"base_risk": 1.0, "reversible": False, "impact": "system"},
            
            # Data destruction
            "delete_file": {"base_risk": 0.7, "reversible": False, "impact": "data"},
            "delete_folder": {"base_risk": 0.8, "reversible": False, "impact": "data"},
            
            # System modification
            "modify_registry": {"base_risk": 0.9, "reversible": True, "impact": "system"},
            "uninstall": {"base_risk": 0.75, "reversible": True, "impact": "software"},
            "kill_process": {"base_risk": 0.6, "reversible": False, "impact": "process"},
            
            # Execution
            "run_command": {"base_risk": 0.6, "reversible": True, "impact": "variable"},
            "run_script": {"base_risk": 0.7, "reversible": True, "impact": "variable"},
            
            # Communication
            "send_email": {"base_risk": 0.5, "reversible": False, "impact": "communication"},
            
            # File operations
            "create_file": {"base_risk": 0.2, "reversible": True, "impact": "data"},
            "copy_file": {"base_risk": 0.15, "reversible": True, "impact": "data"},
            "move_file": {"base_risk": 0.3, "reversible": True, "impact": "data"},
            
            # Safe operations
            "open_app": {"base_risk": 0.1, "reversible": True, "impact": "none"},
            "open_file": {"base_risk": 0.1, "reversible": True, "impact": "none"},
            "open_url": {"base_risk": 0.15, "reversible": True, "impact": "none"},
            "search_web": {"base_risk": 0.05, "reversible": True, "impact": "none"},
            
            # Info only
            "time": {"base_risk": 0.0, "reversible": True, "impact": "none"},
            "date": {"base_risk": 0.0, "reversible": True, "impact": "none"},
            "help": {"base_risk": 0.0, "reversible": True, "impact": "none"}
        }
    
    def _build_dangerous_patterns(self) -> List[Dict]:
        """Pattern pericolosi nei parametri"""
        return [
            # System paths
            {
                "pattern": r"(C:\\Windows|/etc|/usr|System32|\\system\\)",
                "reason": "Path di sistema critico",
                "severity": ThreatLevel.HIGH
            },
            # Shell injection
            {
                "pattern": r"[;&|`$]|\$\(|`.*`|\|\|",
                "reason": "Possibile shell injection",
                "severity": ThreatLevel.CRITICAL
            },
            # Dangerous extensions
            {
                "pattern": r"\.(exe|bat|cmd|ps1|vbs|sh|dll)$",
                "reason": "File eseguibile",
                "severity": ThreatLevel.MEDIUM
            },
            # Registry
            {
                "pattern": r"(HKEY_|regedit|reg\s+add|reg\s+delete)",
                "reason": "Operazione su registro",
                "severity": ThreatLevel.HIGH
            },
            # Network/External
            {
                "pattern": r"(curl|wget|nc\s|netcat|ssh\s|ftp\s|telnet)",
                "reason": "Comando di rete esterno",
                "severity": ThreatLevel.MEDIUM
            },
            # Privilege escalation
            {
                "pattern": r"(sudo|runas|admin|Administrator|root)",
                "reason": "Escalation privilegi",
                "severity": ThreatLevel.HIGH
            },
            # Process manipulation
            {
                "pattern": r"(taskkill|kill\s+-9|pkill|killall)",
                "reason": "Terminazione forzata processo",
                "severity": ThreatLevel.MEDIUM
            }
        ]
    
    # ========== HASH UTILS ==========
    
    def _hash(self, value: str) -> str:
        """Hash sicuro"""
        return hashlib.sha256(value.encode()).hexdigest()
    
    # ========== AUTH ==========
    
    def authenticate(self, user: str = "default") -> bool:
        """Autentica utente"""
        self._authenticated = True
        self._current_user = user
        self._session_start = datetime.now()
        self._log_audit("authenticate", {"user": user}, True)
        return True
    
    def logout(self):
        """Logout"""
        self._log_audit("logout", {"user": self._current_user}, True)
        self._authenticated = False
        self._current_user = None
        self._session_start = None
    
    def is_session_valid(self) -> bool:
        """Verifica sessione"""
        if not self._authenticated or not self._session_start:
            return False
        return datetime.now() - self._session_start < self._session_timeout
    
    def verify_pin(self, pin: str) -> bool:
        """Verifica PIN admin"""
        return self._hash(pin) == self._admin_pin_hash
    
    def set_pin(self, old_pin: str, new_pin: str) -> bool:
        """Cambia PIN"""
        if not self.verify_pin(old_pin):
            return False
        self._admin_pin_hash = self._hash(new_pin)
        self._log_audit("change_pin", {}, True)
        return True
    
    # ========== MAIN VALIDATION ==========
    
    async def validate_action(self, action: Dict, context: Dict = None) -> Dict:
        """
        🎯 Validazione principale di un'azione
        
        Returns:
            {
                "allowed": bool,
                "action": SecurityAction,
                "risk_assessment": RiskAssessment,
                "reason": str,
                "confirmation_id": str (se richiede conferma)
            }
        """
        context = context or {}
        self._stats["total_validations"] += 1
        
        action_type = action.get("action", "unknown")
        params = action.get("params", {})
        
        result = {
            "allowed": False,
            "action": SecurityAction.BLOCK,
            "risk_assessment": None,
            "reason": "",
            "confirmation_id": None
        }
        
        # 1. Check blacklist
        if action_type in self._blacklisted_actions:
            result["reason"] = "Azione in blacklist"
            self._record_violation(
                ViolationType.BLACKLISTED_ACTION,
                action_type, params,
                ThreatLevel.HIGH,
                "Tentativo azione blacklistata"
            )
            return result
        
        # 2. Check whitelist (bypass)
        if action_type in self._whitelisted_actions:
            result["allowed"] = True
            result["action"] = SecurityAction.ALLOW
            result["reason"] = "Azione in whitelist"
            return result
        
        # 3. Risk assessment
        risk = await self.assess_risk(action_type, params, context)
        result["risk_assessment"] = risk
        
        # 4. Check threat level
        if risk.threat_level == ThreatLevel.BLOCKED:
            self._stats["blocked"] += 1
            result["reason"] = f"Bloccato: {', '.join(risk.risk_factors)}"
            self._record_violation(
                ViolationType.DANGEROUS_PARAMS,
                action_type, params,
                ThreatLevel.CRITICAL,
                result["reason"]
            )
            return result
        
        # 5. Check permission level
        permission_result = await self._check_permission_level(action_type, params, context)
        
        if not permission_result["passed"]:
            result["reason"] = permission_result["reason"]
            
            if permission_result.get("requires_confirmation"):
                # Crea pending confirmation
                conf_id = await self._create_confirmation(
                    action_type, params,
                    permission_result.get("confirmation_type", "vocal"),
                    risk
                )
                result["confirmation_id"] = conf_id
                result["action"] = SecurityAction.REQUIRE_CONFIRMATION
                result["reason"] = f"Richiede conferma: {permission_result['reason']}"
            else:
                self._record_violation(
                    ViolationType.PERMISSION_DENIED,
                    action_type, params,
                    ThreatLevel.MEDIUM,
                    permission_result["reason"]
                )
            return result
        
        # 6. Check rate limit
        if not self._check_rate_limit(action_type):
            result["action"] = SecurityAction.RATE_LIMIT
            result["reason"] = "Rate limit superato"
            self._record_violation(
                ViolationType.RATE_LIMIT_EXCEEDED,
                action_type, params,
                ThreatLevel.LOW,
                "Rate limit"
            )
            return result
        
        # 7. Check dangerous patterns
        pattern_check = self._check_dangerous_patterns(params)
        if pattern_check["dangerous"]:
            if pattern_check["severity"] == ThreatLevel.CRITICAL:
                result["reason"] = f"Pattern pericoloso: {pattern_check['reason']}"
                self._record_violation(
                    ViolationType.DANGEROUS_PARAMS,
                    action_type, params,
                    ThreatLevel.CRITICAL,
                    pattern_check["reason"]
                )
                return result
            else:
                # Richiedi conferma per pattern sospetti
                conf_id = await self._create_confirmation(
                    action_type, params, "explicit", risk
                )
                result["confirmation_id"] = conf_id
                result["action"] = SecurityAction.REQUIRE_CONFIRMATION
                result["reason"] = f"Pattern sospetto: {pattern_check['reason']}"
                return result
        
        # 8. All checks passed
        result["allowed"] = True
        result["action"] = risk.recommended_action
        result["reason"] = "Validazione superata"
        
        # Log audit
        self._log_audit("validate_action", {
            "action": action_type,
            "allowed": True
        }, True)
        
        return result
    
    # ========== RISK ASSESSMENT ==========
    
    async def assess_risk(self, action_type: str, params: Dict, 
                          context: Dict = None) -> RiskAssessment:
        """Valuta rischio di un'azione"""
        context = context or {}
        
        assessment = RiskAssessment(
            threat_level=ThreatLevel.SAFE,
            risk_score=0.0,
            recommended_action=SecurityAction.ALLOW
        )
        
        # 1. Base risk from action type
        risk_info = self._risk_factors.get(action_type, {
            "base_risk": 0.3,
            "reversible": True,
            "impact": "unknown"
        })
        
        assessment.risk_score = risk_info["base_risk"]
        
        if not risk_info["reversible"]:
            assessment.risk_factors.append("Azione non reversibile")
            assessment.risk_score += 0.1
        
        # 2. Analyze params
        param_risk = self._analyze_params_risk(params)
        assessment.risk_score += param_risk["additional_risk"]
        assessment.risk_factors.extend(param_risk["factors"])
        
        # 3. Context modifiers
        if context.get("mode") == "passive":
            assessment.risk_score += 0.1
            assessment.risk_factors.append("Modalità passiva attiva")
        
        if context.get("time_sensitive"):
            assessment.risk_score -= 0.05
            assessment.mitigations.append("Operazione time-sensitive")
        
        # 4. Determine threat level
        if assessment.risk_score >= 0.9:
            assessment.threat_level = ThreatLevel.CRITICAL
        elif assessment.risk_score >= 0.7:
            assessment.threat_level = ThreatLevel.HIGH
        elif assessment.risk_score >= 0.5:
            assessment.threat_level = ThreatLevel.MEDIUM
        elif assessment.risk_score >= 0.2:
            assessment.threat_level = ThreatLevel.LOW
        else:
            assessment.threat_level = ThreatLevel.SAFE
        
        # 5. Determine recommended action
        if assessment.threat_level == ThreatLevel.CRITICAL:
            assessment.recommended_action = SecurityAction.REQUIRE_PIN
            assessment.requires_confirmation = True
            assessment.confirmation_type = "pin"
        elif assessment.threat_level == ThreatLevel.HIGH:
            assessment.recommended_action = SecurityAction.REQUIRE_CONFIRMATION
            assessment.requires_confirmation = True
            assessment.confirmation_type = "explicit"
        elif assessment.threat_level == ThreatLevel.MEDIUM:
            assessment.recommended_action = SecurityAction.ALLOW_MONITORED
            assessment.requires_confirmation = True
            assessment.confirmation_type = "vocal"
        else:
            assessment.recommended_action = SecurityAction.ALLOW
        
        # 6. Add mitigations
        if risk_info["reversible"]:
            assessment.mitigations.append("Azione reversibile")
        if assessment.threat_level.value <= ThreatLevel.LOW.value:
            assessment.mitigations.append("Basso impatto sul sistema")
        
        return assessment
    
    def _analyze_params_risk(self, params: Dict) -> Dict:
        """Analizza rischio nei parametri"""
        result = {
            "additional_risk": 0.0,
            "factors": []
        }
        
        params_str = str(params).lower()
        
        # Check for sensitive paths
        if any(p in params_str for p in ["windows", "system32", "program files", "/etc", "/usr"]):
            result["additional_risk"] += 0.2
            result["factors"].append("Path di sistema nei parametri")
        
        # Check for wildcards
        if "*" in params_str or "?" in params_str:
            result["additional_risk"] += 0.15
            result["factors"].append("Wildcards nei parametri")
        
        # Check for recursive flags
        if "-r" in params_str or "/s" in params_str or "recursive" in params_str:
            result["additional_risk"] += 0.1
            result["factors"].append("Operazione ricorsiva")
        
        # Check for force flags
        if "-f" in params_str or "/f" in params_str or "force" in params_str:
            result["additional_risk"] += 0.1
            result["factors"].append("Flag force attivo")
        
        return result
    
    # ========== PERMISSION CHECK ==========
    
    async def _check_permission_level(self, action_type: str, params: Dict,
                                       context: Dict) -> Dict:
        """Verifica livello permessi richiesto"""
        required = self._action_permissions.get(action_type, PermissionLevel.CONFIRMED)
        
        result = {
            "passed": False,
            "reason": "",
            "requires_confirmation": False,
            "confirmation_type": None
        }
        
        # PUBLIC: sempre OK
        if required == PermissionLevel.PUBLIC:
            result["passed"] = True
            return result
        
        # AUTHENTICATED e superiori: richiede sessione
        if not self.is_session_valid():
            result["reason"] = "Sessione non valida"
            return result
        
        if required == PermissionLevel.AUTHENTICATED:
            result["passed"] = True
            return result
        
        # CONFIRMED: richiede conferma vocale
        if required == PermissionLevel.CONFIRMED:
            result["requires_confirmation"] = True
            result["confirmation_type"] = "vocal"
            result["reason"] = "Richiede conferma vocale"
            return result
        
        # ELEVATED: conferma esplicita
        if required == PermissionLevel.ELEVATED:
            result["requires_confirmation"] = True
            result["confirmation_type"] = "explicit"
            result["reason"] = "Richiede conferma esplicita"
            return result
        
        # ADMIN: PIN
        if required == PermissionLevel.ADMIN:
            result["requires_confirmation"] = True
            result["confirmation_type"] = "pin"
            result["reason"] = "Richiede PIN amministratore"
            return result
        
        # CRITICAL: doppia conferma
        if required == PermissionLevel.CRITICAL:
            result["requires_confirmation"] = True
            result["confirmation_type"] = "critical"
            result["reason"] = "Azione critica - richiede doppia conferma"
            return result
        
        return result
    
    # ========== RATE LIMITING ==========
    
    def _check_rate_limit(self, action_type: str) -> bool:
        """Verifica rate limit"""
        limit = self._rate_limits.get(action_type, self._rate_limits["default"])
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        # Pulisci vecchi record
        if action_type in self._action_counts:
            self._action_counts[action_type] = [
                t for t in self._action_counts[action_type]
                if t > hour_ago
            ]
        else:
            self._action_counts[action_type] = []
        
        # Check limit
        if len(self._action_counts[action_type]) >= limit:
            return False
        
        # Record action
        self._action_counts[action_type].append(now)
        return True
    
    # ========== PATTERN CHECK ==========
    
    def _check_dangerous_patterns(self, params: Dict) -> Dict:
        """Verifica pattern pericolosi"""
        params_str = str(params)
        
        for pattern_info in self._dangerous_patterns:
            if re.search(pattern_info["pattern"], params_str, re.IGNORECASE):
                return {
                    "dangerous": True,
                    "reason": pattern_info["reason"],
                    "severity": pattern_info["severity"]
                }
        
        return {"dangerous": False}
    
    # ========== CONFIRMATION ==========
    
    async def _create_confirmation(self, action_type: str, params: Dict,
                                    conf_type: str, risk: RiskAssessment) -> str:
        """Crea richiesta di conferma"""
        conf_id = f"conf_{datetime.now().timestamp()}_{action_type}"
        
        # Timeout basato su tipo
        timeout_minutes = {
            "vocal": 2,
            "explicit": 3,
            "pin": 1,
            "critical": 5
        }.get(conf_type, 2)
        
        confirmation = PendingConfirmation(
            id=conf_id,
            action=action_type,
            params=params,
            confirmation_type=conf_type,
            expires_at=datetime.now() + timedelta(minutes=timeout_minutes),
            risk_assessment=risk
        )
        
        self._pending_confirmations[conf_id] = confirmation
        
        # Notify callback
        if self._confirmation_callback:
            await self._confirmation_callback(conf_id, action_type, conf_type)
        
        return conf_id
    
    async def confirm(self, confirmation_id: str, pin: str = None) -> Dict:
        """Conferma un'azione in attesa"""
        if confirmation_id not in self._pending_confirmations:
            return {"success": False, "error": "Conferma non trovata"}
        
        conf = self._pending_confirmations[confirmation_id]
        
        # Check expiry
        if datetime.now() > conf.expires_at:
            del self._pending_confirmations[confirmation_id]
            return {"success": False, "error": "Conferma scaduta"}
        
        # Check PIN se richiesto
        if conf.confirmation_type in ["pin", "critical"]:
            if not pin or not self.verify_pin(pin):
                return {"success": False, "error": "PIN non valido"}
        
        conf.confirmed = True
        self._stats["confirmed"] += 1
        
        self._log_audit("confirm_action", {
            "action": conf.action,
            "confirmation_id": confirmation_id
        }, True)
        
        # Remove from pending
        del self._pending_confirmations[confirmation_id]
        
        return {
            "success": True,
            "action": conf.action,
            "params": conf.params
        }
    
    def cancel_confirmation(self, confirmation_id: str) -> bool:
        """Cancella conferma"""
        if confirmation_id in self._pending_confirmations:
            del self._pending_confirmations[confirmation_id]
            return True
        return False
    
    def get_pending_confirmations(self) -> List[Dict]:
        """Ottiene conferme in attesa"""
        now = datetime.now()
        
        # Cleanup expired
        expired = [k for k, v in self._pending_confirmations.items()
                   if now > v.expires_at]
        for k in expired:
            del self._pending_confirmations[k]
        
        return [
            {
                "id": conf.id,
                "action": conf.action,
                "type": conf.confirmation_type,
                "expires_in": (conf.expires_at - now).total_seconds()
            }
            for conf in self._pending_confirmations.values()
        ]
    
    # ========== VIOLATIONS ==========
    
    def _record_violation(self, vtype: ViolationType, action: str,
                          params: Dict, severity: ThreatLevel, reason: str):
        """Registra violazione"""
        violation = SecurityViolation(
            id=f"viol_{datetime.now().timestamp()}",
            violation_type=vtype,
            action=action,
            params=params,
            severity=severity,
            user=self._current_user,
            reason=reason
        )
        
        self._violations.append(violation)
        self._stats["violations"] += 1
        
        if len(self._violations) > self._max_violations:
            self._violations = self._violations[-self._max_violations:]
        
        # Callback
        if self._violation_callback:
            asyncio.create_task(self._violation_callback(violation))
    
    def get_violations(self, limit: int = 50) -> List[Dict]:
        """Ottiene violazioni recenti"""
        return [v.to_dict() for v in self._violations[-limit:]]
    
    # ========== AUDIT LOG ==========
    
    def _log_audit(self, action: str, details: Dict, success: bool):
        """Log audit"""
        self._audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "user": self._current_user,
            "action": action,
            "details": details,
            "success": success
        })
        
        if len(self._audit_log) > self._max_audit:
            self._audit_log = self._audit_log[-self._max_audit:]
    
    def get_audit_log(self, limit: int = 100) -> List[Dict]:
        """Ottiene audit log"""
        return self._audit_log[-limit:]
    
    # ========== BLACKLIST/WHITELIST ==========
    
    def blacklist_action(self, action: str):
        """Aggiunge azione a blacklist"""
        self._blacklisted_actions.add(action)
        self._whitelisted_actions.discard(action)
    
    def whitelist_action(self, action: str):
        """Aggiunge azione a whitelist"""
        self._whitelisted_actions.add(action)
        self._blacklisted_actions.discard(action)
    
    def unlist_action(self, action: str):
        """Rimuove azione da entrambe le liste"""
        self._blacklisted_actions.discard(action)
        self._whitelisted_actions.discard(action)
    
    # ========== CALLBACKS ==========
    
    def set_confirmation_callback(self, callback: Callable):
        """Imposta callback per conferme"""
        self._confirmation_callback = callback
    
    def set_violation_callback(self, callback: Callable):
        """Imposta callback per violazioni"""
        self._violation_callback = callback
    
    # ========== PUBLIC API ==========
    
    def get_stats(self) -> Dict:
        """Statistiche sicurezza"""
        return {
            **self._stats,
            "pending_confirmations": len(self._pending_confirmations),
            "blacklisted_actions": len(self._blacklisted_actions),
            "whitelisted_actions": len(self._whitelisted_actions)
        }
    
    def get_status(self) -> Dict:
        """Stato completo"""
        return {
            "authenticated": self._authenticated,
            "user": self._current_user,
            "session_valid": self.is_session_valid(),
            "total_validations": self._stats["total_validations"],
            "blocked": self._stats["blocked"],
            "violations": self._stats["violations"],
            "pending_confirmations": len(self._pending_confirmations)
        }
    
    def set_permission(self, action: str, level: PermissionLevel):
        """Imposta livello permesso per azione"""
        self._action_permissions[action] = level
    
    def get_permission(self, action: str) -> PermissionLevel:
        """Ottiene livello permesso per azione"""
        return self._action_permissions.get(action, PermissionLevel.CONFIRMED)
