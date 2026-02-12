"""
Database connection pooling for production deployment.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool
import logging
from typing import Optional

logger = logging.getLogger(__name__)

Base = declarative_base()

class Database:
    """Database connection manager."""
    
    def __init__(self, url: str, pool_size: int = 20, max_overflow: int = 30):
        self.url = url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.engine: Optional[create_engine] = None
        self.SessionLocal: Optional[sessionmaker] = None
    
    def initialize(self):
        """Initialize database connection pool."""
        try:
            self.engine = create_engine(
                self.url,
                poolclass=QueuePool,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
            
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            logger.info(
                f"Database connection pool initialized: "
                f"pool_size={self.pool_size}, max_overflow={self.max_overflow}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def get_session(self):
        """Get database session."""
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized")
        return self.SessionLocal()
    
    def create_tables(self):
        """Create all tables."""
        if self.engine:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created")


# Global database instance
db: Optional[Database] = None


def init_database(config: dict) -> Database:
    """Initialize database from config."""
    global db
    
    db_config = config.get('database', {})
    db_url = db_config.get('url', 'sqlite:///./aads_ulora.db')
    
    db = Database(
        url=db_url,
        pool_size=db_config.get('pool_size', 20),
        max_overflow=db_config.get('max_overflow', 30)
    )
    db.initialize()
    
    return db


def get_db():
    """Dependency for FastAPI endpoints."""
    if not db:
        raise RuntimeError("Database not initialized")
    session = db.get_session()
    try:
        yield session
    finally:
        session.close()