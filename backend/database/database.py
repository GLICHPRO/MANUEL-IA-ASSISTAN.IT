"""
Database initialization and connection management
"""

import aiosqlite
from pathlib import Path
from loguru import logger
from datetime import datetime
import json


class Database:
    """Async SQLite database manager"""
    
    def __init__(self, db_path: str = "gideon_memory.db"):
        self.db_path = Path(db_path)
        self.connection = None
        
    async def connect(self):
        """Connect to database"""
        self.connection = await aiosqlite.connect(self.db_path)
        self.connection.row_factory = aiosqlite.Row
        await self.init_tables()
        logger.info(f"✅ Database connected: {self.db_path}")
        
    async def close(self):
        """Close database connection"""
        if self.connection:
            await self.connection.close()
            logger.info("Database connection closed")
    
    async def init_tables(self):
        """Initialize database tables"""
        async with self.connection.cursor() as cursor:
            # Interactions table
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    intent TEXT,
                    confidence REAL,
                    mode TEXT,
                    context TEXT,
                    user_feedback TEXT
                )
            """)
            
            # Analysis results table
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    target TEXT,
                    findings TEXT,
                    recommendations TEXT,
                    priority TEXT
                )
            """)
            
            # Learned patterns table
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS learned_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    pattern_data TEXT NOT NULL,
                    frequency INTEGER DEFAULT 1,
                    last_seen TEXT NOT NULL,
                    confidence_score REAL
                )
            """)
            
            # Context memory table
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS context_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    context_type TEXT,
                    context_data TEXT,
                    expiry TEXT
                )
            """)
            
            # Create indexes
            await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions(timestamp)"
            )
            await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_patterns_type ON learned_patterns(pattern_type)"
            )
            
            await self.connection.commit()
            logger.info("✅ Database tables initialized")


async def init_db() -> Database:
    """Initialize and return database instance"""
    db = Database()
    await db.connect()
    return db

