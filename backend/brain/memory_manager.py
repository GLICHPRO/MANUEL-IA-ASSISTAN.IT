"""
Memory Manager - Handles conversational memory, learning, and contextualization
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from loguru import logger
import json
from collections import defaultdict, Counter


class MemoryManager:
    """Manages assistant memory, learning, and context"""
    
    def __init__(self):
        self.db = None
        self.short_term_memory: List[Dict[str, Any]] = []
        self.session_context: Dict[str, Any] = {}
        self.user_preferences: Dict[str, Any] = {}
        # Struttura corretta per learned_patterns:
        # - keywords: Dict[intent, List[str]] - parole chiave per ogni intent
        # - topic_frequency: Dict[intent, int] - frequenza di ogni topic
        # - time_patterns: List[Dict] - pattern temporali
        self.learned_patterns: Dict[str, Any] = {
            "keywords": defaultdict(list),
            "topic_frequency": {},
            "time_patterns": []
        }
        self.conversation_history: List[Dict[str, Any]] = []
        self.max_short_term = 20  # Keep last 20 interactions in RAM
    
    async def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation history for AI context"""
        # Return last N interactions from conversation history
        history = self.conversation_history[-limit:] if self.conversation_history else []
        
        # Format for AI consumption
        formatted = []
        for item in history:
            formatted.append({
                "role": "user",
                "content": item.get("query", "")
            })
            formatted.append({
                "role": "assistant", 
                "content": item.get("response", "")
            })
        
        return formatted
        
    async def initialize(self):
        """Initialize memory manager and load learned data"""
        from database.database import init_db
        
        logger.info("🧠 Initializing Memory Manager...")
        
        # Initialize database
        self.db = await init_db()
        
        # Load learned patterns
        await self._load_learned_patterns()
        
        # Load user preferences
        await self._load_user_preferences()
        
        logger.info("✅ Memory Manager ready")
    
    async def save_interaction(
        self,
        query: str,
        response: str,
        intent: str,
        confidence: float,
        mode: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """Save interaction to memory"""
        timestamp = datetime.now().isoformat()
        
        interaction = {
            "timestamp": timestamp,
            "query": query,
            "response": response,
            "intent": intent,
            "confidence": confidence,
            "mode": mode,
            "context": context or {}
        }
        
        # Add to short-term memory
        self.short_term_memory.append(interaction)
        self.conversation_history.append(interaction)
        
        # Keep only recent interactions in RAM
        if len(self.short_term_memory) > self.max_short_term:
            self.short_term_memory.pop(0)
        
        # Save to database
        if self.db and self.db.connection:
            try:
                async with self.db.connection.cursor() as cursor:
                    await cursor.execute("""
                        INSERT INTO interactions 
                        (timestamp, query, response, intent, confidence, mode, context)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        timestamp,
                        query,
                        response,
                        intent,
                        confidence,
                        mode,
                        json.dumps(context) if context else None
                    ))
                    await self.db.connection.commit()
            except Exception as e:
                logger.error(f"Error saving interaction: {e}")
        
        # Learn from interaction
        await self._learn_from_interaction(interaction)
        
        # Update conversation context for multi-turn
        await self._update_conversation_context(interaction)
    
    async def _update_conversation_context(self, interaction: Dict[str, Any]):
        """Update multi-turn conversation context"""
        # Track conversation flow
        if "conversation_flow" not in self.session_context:
            self.session_context["conversation_flow"] = []
        
        self.session_context["conversation_flow"].append({
            "turn": len(self.session_context["conversation_flow"]) + 1,
            "intent": interaction["intent"],
            "topic": self._extract_topic(interaction["query"]),
            "timestamp": interaction["timestamp"]
        })
        
        # Keep only last 10 turns
        if len(self.session_context["conversation_flow"]) > 10:
            self.session_context["conversation_flow"] = self.session_context["conversation_flow"][-10:]
        
        # Track active topic
        self.session_context["active_topic"] = self._extract_topic(interaction["query"])
        self.session_context["last_intent"] = interaction["intent"]
        self.session_context["last_query"] = interaction["query"]
        self.session_context["last_response"] = interaction["response"]
    
    def _extract_topic(self, text: str) -> str:
        """Extract main topic from text"""
        text_lower = text.lower()
        
        topics = {
            "sistema": ["sistema", "cpu", "memoria", "disco", "performance", "stato"],
            "tempo": ["ora", "data", "giorno", "oggi", "ieri"],
            "calcolo": ["calcola", "percentuale", "quanto", "efficienza"],
            "analisi": ["analizza", "verifica", "controlla", "esamina"],
            "ragionamento": ["ragiona", "pensa", "considera", "valuta"],
            "ottimizzazione": ["ottimizza", "migliora", "suggerisci", "consiglia"],
            "informazione": ["cos'è", "cosa", "come", "perché", "quando"]
        }
        
        for topic, keywords in topics.items():
            if any(kw in text_lower for kw in keywords):
                return topic
        
        return "generale"
    
    async def get_relevant_context(self, text: str) -> Optional[Dict[str, Any]]:
        """Get relevant context for current query"""
        context = {
            "recent_topics": self._extract_recent_topics(),
            "current_intent_pattern": self._detect_intent_pattern(text),
            "user_preferences": self.user_preferences,
            "conversation_history": self.short_term_memory[-5:],  # Last 5 interactions
            "learned_patterns": await self._get_matching_patterns(text),
            # Multi-turn context
            "multi_turn": {
                "active_topic": self.session_context.get("active_topic"),
                "last_intent": self.session_context.get("last_intent"),
                "last_query": self.session_context.get("last_query"),
                "conversation_flow": self.session_context.get("conversation_flow", []),
                "turn_count": len(self.session_context.get("conversation_flow", [])),
                "is_follow_up": self._is_follow_up_query(text),
                "referenced_context": self._detect_context_references(text)
            }
        }
        
        return context
    
    def _is_follow_up_query(self, text: str) -> bool:
        """Detect if current query is a follow-up to previous conversation"""
        follow_up_indicators = [
            "e poi", "ancora", "altro", "di più", "continua",
            "quindi", "allora", "poi", "inoltre", "anche",
            "quello", "quella", "questo", "questa", "lo stesso",
            "come prima", "di nuovo", "ripeti", "stessa cosa"
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in follow_up_indicators)
    
    def _detect_context_references(self, text: str) -> Dict[str, Any]:
        """Detect references to previous context in the query"""
        references = {
            "has_pronoun_reference": False,
            "refers_to_previous": False,
            "comparison_request": False,
            "continuation_request": False
        }
        
        text_lower = text.lower()
        
        # Check for pronouns referring to previous context
        pronoun_refs = ["lo", "la", "li", "le", "ne", "ci", "quello", "quella", "questi", "queste"]
        if any(pron in text_lower.split() for pron in pronoun_refs):
            references["has_pronoun_reference"] = True
        
        # Check for comparison requests
        comparison_words = ["rispetto", "confronta", "differenza", "meglio", "peggio", "prima", "dopo"]
        if any(word in text_lower for word in comparison_words):
            references["comparison_request"] = True
        
        # Check for continuation
        continuation_words = ["continua", "prosegui", "vai avanti", "di più", "approfondisci"]
        if any(word in text_lower for word in continuation_words):
            references["continuation_request"] = True
        
        # Check for explicit previous reference
        previous_refs = ["come prima", "di nuovo", "ancora", "ripeti", "stesso"]
        if any(ref in text_lower for ref in previous_refs):
            references["refers_to_previous"] = True
        
        return references
    
    async def _learn_from_interaction(self, interaction: Dict[str, Any]):
        """Learn patterns from user interactions"""
        intent = interaction["intent"]
        query = interaction["query"].lower()
        timestamp = datetime.fromisoformat(interaction["timestamp"])
        
        # Learn time patterns (when user typically asks certain things)
        hour = timestamp.hour
        time_pattern_key = f"{intent}_{hour}"
        self.learned_patterns["time_patterns"].append({
            "intent": intent,
            "hour": hour,
            "timestamp": timestamp
        })
        
        # Learn command preferences (how user phrases things)
        words = query.split()
        for word in words:
            if len(word) > 3:  # Ignore short words
                self.learned_patterns["keywords"][intent].append(word)
        
        # Learn topic interests
        if intent not in ["time", "status"]:  # Skip trivial intents
            self.learned_patterns["topic_frequency"][intent] = \
                self.learned_patterns["topic_frequency"].get(intent, 0) + 1
        
        # Persist important patterns to database
        await self._persist_patterns()
    
    async def _persist_patterns(self):
        """Save learned patterns to database"""
        if not self.db or not self.db.connection:
            return
        
        try:
            # Save topic frequency patterns
            for intent, frequency in self.learned_patterns["topic_frequency"].items():
                pattern_data = {
                    "intent": intent,
                    "frequency": frequency,
                    "learned_at": datetime.now().isoformat()
                }
                
                async with self.db.connection.cursor() as cursor:
                    # Check if pattern exists
                    await cursor.execute("""
                        SELECT id, frequency FROM learned_patterns 
                        WHERE pattern_type = 'topic_interest' 
                        AND json_extract(pattern_data, '$.intent') = ?
                    """, (intent,))
                    
                    existing = await cursor.fetchone()
                    
                    if existing:
                        # Update frequency
                        new_frequency = existing[1] + 1
                        await cursor.execute("""
                            UPDATE learned_patterns 
                            SET frequency = ?, last_seen = ?
                            WHERE id = ?
                        """, (new_frequency, datetime.now().isoformat(), existing[0]))
                    else:
                        # Insert new pattern
                        await cursor.execute("""
                            INSERT INTO learned_patterns 
                            (pattern_type, pattern_data, frequency, last_seen, confidence_score)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            "topic_interest",
                            json.dumps(pattern_data),
                            1,
                            datetime.now().isoformat(),
                            0.7
                        ))
                    
                    await self.db.connection.commit()
        except Exception as e:
            logger.error(f"Error persisting patterns: {e}")
    
    async def _load_learned_patterns(self):
        """Load learned patterns from database"""
        if not self.db or not self.db.connection:
            return
        
        try:
            async with self.db.connection.cursor() as cursor:
                await cursor.execute("""
                    SELECT pattern_type, pattern_data, frequency, confidence_score
                    FROM learned_patterns
                    ORDER BY frequency DESC
                    LIMIT 100
                """)
                
                rows = await cursor.fetchall()
                
                for row in rows:
                    pattern_type = row[0]
                    pattern_data = json.loads(row[1])
                    frequency = row[2]
                    
                    if pattern_type == "topic_interest":
                        intent = pattern_data.get("intent")
                        if intent:
                            self.learned_patterns["topic_frequency"][intent] = frequency
                
                logger.info(f"Loaded {len(rows)} learned patterns")
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
    
    async def _load_user_preferences(self):
        """Load user preferences from interaction history"""
        if not self.db or not self.db.connection:
            return
        
        try:
            async with self.db.connection.cursor() as cursor:
                # Get most common intents
                await cursor.execute("""
                    SELECT intent, COUNT(*) as count
                    FROM interactions
                    GROUP BY intent
                    ORDER BY count DESC
                    LIMIT 10
                """)
                
                rows = await cursor.fetchall()
                
                self.user_preferences["favorite_commands"] = [
                    {"intent": row[0], "usage_count": row[1]}
                    for row in rows
                ]
                
                # Get typical usage time
                await cursor.execute("""
                    SELECT timestamp FROM interactions
                    ORDER BY timestamp DESC
                    LIMIT 100
                """)
                
                timestamps = await cursor.fetchall()
                hours = []
                for ts in timestamps:
                    try:
                        dt = datetime.fromisoformat(ts[0])
                        hours.append(dt.hour)
                    except:
                        pass
                
                if hours:
                    hour_counter = Counter(hours)
                    most_common_hour = hour_counter.most_common(1)[0][0]
                    self.user_preferences["typical_usage_hour"] = most_common_hour
                
                logger.info(f"Loaded user preferences: {len(self.user_preferences)} items")
        except Exception as e:
            logger.error(f"Error loading preferences: {e}")
    
    def _extract_recent_topics(self) -> List[str]:
        """Extract topics from recent conversations"""
        topics = []
        for interaction in self.short_term_memory[-5:]:
            topics.append(interaction.get("intent", "unknown"))
        return list(set(topics))  # Unique topics
    
    def _detect_intent_pattern(self, text: str) -> Optional[str]:
        """Detect if current query matches a known pattern"""
        text_lower = text.lower()
        
        # Check keyword patterns
        for intent, keywords in self.learned_patterns.get("keywords", {}).items():
            keyword_counter = Counter(keywords)
            most_common = keyword_counter.most_common(5)
            
            for keyword, _ in most_common:
                if keyword in text_lower:
                    return intent
        
        return None
    
    async def _get_matching_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Get learned patterns that match current query"""
        matching = []
        
        # Check topic frequency
        detected_intent = self._detect_intent_pattern(text)
        if detected_intent and detected_intent in self.learned_patterns.get("topic_frequency", {}):
            matching.append({
                "type": "frequent_topic",
                "intent": detected_intent,
                "frequency": self.learned_patterns["topic_frequency"][detected_intent]
            })
        
        return matching
    
    async def get_conversation_summary(self, last_n: int = 10) -> str:
        """Get a summary of recent conversations"""
        recent = self.conversation_history[-last_n:]
        
        if not recent:
            return "Nessuna conversazione precedente."
        
        summary_parts = []
        for i, interaction in enumerate(recent, 1):
            summary_parts.append(
                f"{i}. {interaction['intent']}: {interaction['query'][:50]}..."
            )
        
        return "\n".join(summary_parts)
    
    async def suggest_next_actions(self) -> List[str]:
        """Suggest next actions based on learned patterns"""
        suggestions = []
        
        # Based on favorite commands
        if "favorite_commands" in self.user_preferences:
            top_commands = self.user_preferences["favorite_commands"][:3]
            for cmd in top_commands:
                suggestions.append(f"Vuoi controllare {cmd['intent']}?")
        
        # Based on time patterns
        current_hour = datetime.now().hour
        if "typical_usage_hour" in self.user_preferences:
            typical_hour = self.user_preferences["typical_usage_hour"]
            if abs(current_hour - typical_hour) < 2:
                suggestions.append("È il tuo orario abituale. Posso aiutarti?")
        
        return suggestions[:3]  # Max 3 suggestions
    
    async def analyze_history_for_improvements(self) -> Dict[str, Any]:
        """Analyze interaction history to improve suggestions and accuracy"""
        analysis = {
            "total_interactions": 0,
            "success_rate": 0.0,
            "most_used_commands": [],
            "peak_usage_hours": [],
            "improvement_suggestions": [],
            "accuracy_trends": []
        }
        
        if not self.db or not self.db.connection:
            return analysis
        
        try:
            async with self.db.connection.cursor() as cursor:
                # Total interactions
                await cursor.execute("SELECT COUNT(*) FROM interactions")
                result = await cursor.fetchone()
                analysis["total_interactions"] = result[0] if result else 0
                
                # Success rate (based on confidence > 0.7)
                await cursor.execute("""
                    SELECT 
                        COUNT(CASE WHEN confidence > 0.7 THEN 1 END) as high_conf,
                        COUNT(*) as total
                    FROM interactions
                """)
                result = await cursor.fetchone()
                if result and result[1] > 0:
                    analysis["success_rate"] = round((result[0] / result[1]) * 100, 1)
                
                # Most used commands
                await cursor.execute("""
                    SELECT intent, COUNT(*) as count, AVG(confidence) as avg_conf
                    FROM interactions
                    GROUP BY intent
                    ORDER BY count DESC
                    LIMIT 5
                """)
                rows = await cursor.fetchall()
                analysis["most_used_commands"] = [
                    {"intent": row[0], "count": row[1], "avg_confidence": round(row[2], 2)}
                    for row in rows
                ]
                
                # Peak usage hours
                await cursor.execute("""
                    SELECT 
                        CAST(substr(timestamp, 12, 2) AS INTEGER) as hour,
                        COUNT(*) as count
                    FROM interactions
                    GROUP BY hour
                    ORDER BY count DESC
                    LIMIT 3
                """)
                rows = await cursor.fetchall()
                analysis["peak_usage_hours"] = [
                    {"hour": row[0], "usage_count": row[1]}
                    for row in rows
                ]
                
                # Accuracy trends (last 50 vs overall)
                await cursor.execute("""
                    SELECT AVG(confidence) FROM interactions
                    ORDER BY timestamp DESC LIMIT 50
                """)
                recent_avg = await cursor.fetchone()
                
                await cursor.execute("SELECT AVG(confidence) FROM interactions")
                overall_avg = await cursor.fetchone()
                
                if recent_avg and overall_avg and recent_avg[0] and overall_avg[0]:
                    trend = "improving" if recent_avg[0] > overall_avg[0] else "declining"
                    analysis["accuracy_trends"] = {
                        "recent_accuracy": round(recent_avg[0] * 100, 1),
                        "overall_accuracy": round(overall_avg[0] * 100, 1),
                        "trend": trend
                    }
                
                # Generate improvement suggestions
                analysis["improvement_suggestions"] = await self._generate_improvement_suggestions(analysis)
                
        except Exception as e:
            logger.error(f"Error analyzing history: {e}")
        
        return analysis
    
    async def _generate_improvement_suggestions(self, analysis: Dict) -> List[str]:
        """Generate improvement suggestions based on analysis"""
        suggestions = []
        
        # Low success rate suggestion
        if analysis["success_rate"] < 70:
            suggestions.append("Considera di riformulare i comandi per maggiore chiarezza")
        
        # Based on most used commands
        if analysis["most_used_commands"]:
            top_cmd = analysis["most_used_commands"][0]
            suggestions.append(f"Il comando '{top_cmd['intent']}' è il più usato. Ottimizzalo!")
            
            # Check for low confidence commands
            for cmd in analysis["most_used_commands"]:
                if cmd["avg_confidence"] < 0.6:
                    suggestions.append(f"Il comando '{cmd['intent']}' ha bassa accuratezza. Verifica i pattern.")
        
        # Peak hours suggestion
        if analysis["peak_usage_hours"]:
            peak = analysis["peak_usage_hours"][0]
            suggestions.append(f"Picco utilizzo alle ore {peak['hour']}:00 - considera automazioni")
        
        return suggestions[:5]
    
    async def get_scenario_history(self, scenario_type: str = None, limit: int = 20) -> List[Dict]:
        """Get past scenarios and their outcomes"""
        scenarios = []
        
        if not self.db or not self.db.connection:
            return scenarios
        
        try:
            async with self.db.connection.cursor() as cursor:
                if scenario_type:
                    await cursor.execute("""
                        SELECT timestamp, query, response, intent, confidence, context
                        FROM interactions
                        WHERE intent = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """, (scenario_type, limit))
                else:
                    await cursor.execute("""
                        SELECT timestamp, query, response, intent, confidence, context
                        FROM interactions
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """, (limit,))
                
                rows = await cursor.fetchall()
                
                for row in rows:
                    scenarios.append({
                        "timestamp": row[0],
                        "query": row[1],
                        "response": row[2],
                        "intent": row[3],
                        "confidence": row[4],
                        "context": json.loads(row[5]) if row[5] else {}
                    })
        except Exception as e:
            logger.error(f"Error getting scenario history: {e}")
        
        return scenarios
    
    async def calculate_improved_confidence(self, intent: str, base_confidence: float) -> float:
        """Improve confidence based on historical success with this intent"""
        if not self.db or not self.db.connection:
            return base_confidence
        
        try:
            async with self.db.connection.cursor() as cursor:
                # Get historical success rate for this intent
                await cursor.execute("""
                    SELECT AVG(confidence), COUNT(*) 
                    FROM interactions 
                    WHERE intent = ?
                """, (intent,))
                
                result = await cursor.fetchone()
                
                if result and result[1] > 5:  # At least 5 historical samples
                    historical_avg = result[0]
                    sample_count = result[1]
                    
                    # Weight: more samples = more trust in historical data
                    weight = min(sample_count / 50, 0.5)  # Max 50% weight from history
                    
                    improved = (base_confidence * (1 - weight)) + (historical_avg * weight)
                    return round(improved, 3)
                    
        except Exception as e:
            logger.error(f"Error calculating improved confidence: {e}")
        
        return base_confidence
    
    async def close(self):
        """Close memory manager and database"""
        if self.db:
            await self.db.close()
        logger.info("Memory Manager closed")

