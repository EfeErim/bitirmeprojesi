"""
Graceful shutdown handling for production deployment.
"""
import asyncio
import signal
import sys
import logging
from typing import List

logger = logging.getLogger(__name__)


class GracefulShutdown:
    """Handles graceful shutdown of the API server."""
    
    def __init__(self, app):
        self.app = app
        self.shutdown = False
        self.tasks: List[asyncio.Task] = []
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)
    
    def handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown = True
    
    async def shutdown_handler(self):
        """Handle async shutdown tasks."""
        logger.info("Starting graceful shutdown...")
        
        # Wait for in-flight requests
        await asyncio.sleep(2)
        
        # Close database connections
        try:
            from api.database import db
            if db and db.engine:
                db.engine.dispose()
                logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database: {e}")
        
        # Close Redis connections
        try:
            from api.middleware.caching import cache
            if cache and cache._client:
                await cache._client.close()
                logger.info("Redis connections closed")
        except Exception as e:
            logger.error(f"Error closing Redis: {e}")
        
        # Cancel background tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        logger.info("Graceful shutdown complete")
    
    def add_task(self, task: asyncio.Task):
        """Add background task to track."""
        self.tasks.append(task)


def setup_shutdown_handlers(app):
    """Setup shutdown handlers for FastAPI app."""
    shutdown_handler = GracefulShutdown(app)
    
    @app.on_event("shutdown")
    async def shutdown_event():
        await shutdown_handler.shutdown_handler()
    
    return shutdown_handler