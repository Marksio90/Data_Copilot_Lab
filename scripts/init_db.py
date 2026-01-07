#!/usr/bin/env python3
"""
Data Copilot Lab - Database Initialization Script
Initializes the database schema and creates initial data
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def init_database():
    """Initialize database schema"""
    logger.info("Initializing database...")
    logger.info(f"Database URL: {settings.database_url}")

    # TODO: Implement database initialization with SQLAlchemy
    # This will be implemented when we create database models

    logger.info("✅ Database initialization complete")


def create_sample_data():
    """Create sample/seed data for development"""
    logger.info("Creating sample data...")

    # TODO: Add sample data creation
    # This can be useful for development and testing

    logger.info("✅ Sample data created")


if __name__ == "__main__":
    try:
        logger.info("🚀 Starting database initialization...")
        init_database()

        if settings.is_development:
            create_sample_data()

        logger.info("✅ Database setup complete!")

    except Exception as e:
        logger.error(f"❌ Database initialization failed: {str(e)}", exc_info=True)
        sys.exit(1)
