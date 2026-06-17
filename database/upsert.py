"""
NIFTY Quant Lab - MySQL Upsert Helper
========================================
MySQL uses INSERT ... ON DUPLICATE KEY UPDATE instead of
PostgreSQL's INSERT ... ON CONFLICT DO UPDATE.

SQLAlchemy provides this via:
    from sqlalchemy.dialects.mysql import insert as mysql_insert

Usage:
    stmt = mysql_upsert(Symbol).values(rows)
    stmt = stmt.on_duplicate_key_update(name=stmt.inserted.name)
    await session.execute(stmt)

This module wraps the pattern so callers don't import the dialect directly.
"""

from sqlalchemy.dialects.mysql import insert as _mysql_insert


def mysql_upsert(table):
    """Return a MySQL INSERT statement builder for the given ORM model."""
    return _mysql_insert(table)
