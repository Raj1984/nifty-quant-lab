"""
NIFTY Quant Lab - Database Connection Manager
================================================
Async SQLAlchemy engine + session factory for MySQL.
Drivers: aiomysql (async) / PyMySQL (sync/Alembic).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from nifty_quant_lab.config.settings import settings
from nifty_quant_lab.database.models import Base

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# ASYNC ENGINE  (FastAPI / background tasks)
# ─────────────────────────────────────────────────────────────

async_engine = create_async_engine(
    settings.database.url,
    pool_size=settings.database.pool_size,
    max_overflow=settings.database.max_overflow,
    pool_recycle=3600,   # MySQL closes idle connections after wait_timeout
    pool_pre_ping=True,  # verify connection alive before use
    echo=settings.debug,
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ─────────────────────────────────────────────────────────────
# SYNC ENGINE  (Alembic / pandas / scripts)
# ─────────────────────────────────────────────────────────────

sync_engine = create_engine(
    settings.database.sync_url,
    pool_size=settings.database.pool_size,
    max_overflow=settings.database.max_overflow,
    pool_recycle=3600,
    pool_pre_ping=True,
    echo=settings.debug,
    future=True,
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


@contextmanager
def get_sync_session() -> Generator[Session, None, None]:
    session = SyncSessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ─────────────────────────────────────────────────────────────
# SCHEMA MANAGEMENT
# ─────────────────────────────────────────────────────────────

async def create_all_tables() -> None:
    """Create all tables if they don't exist."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("All database tables created/verified.")


async def drop_all_tables() -> None:
    """Drop all tables — DESTRUCTIVE, only for testing."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.warning("All database tables dropped.")


async def check_connection() -> bool:
    try:
        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False
