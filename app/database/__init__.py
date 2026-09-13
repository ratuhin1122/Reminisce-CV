"""
app.database — SQLite Persistent Memory Storage
=================================================

Public API
----------
.. code-block:: python

    from app.database import DatabaseManager, Memory, VisualReference, EntityType

    db = DatabaseManager(config.db_path)
    db.initialize()

    mem_id = db.insert_memory(Memory(
        entity_type=EntityType.OBJECT,
        name="Grandmother's Watch",
        giver_name="Grandmother",
        occasion="Birthday",
        year="2015",
        narrative="A silver pocket watch she gave me on my 18th birthday.",
    ))

    memory = db.get_memory(mem_id)
    db.close()
"""

from app.database.models import EntityType, Memory, VisualReference
from app.database.manager import DatabaseManager

__all__ = [
    "DatabaseManager",
    "EntityType",
    "Memory",
    "VisualReference",
]
