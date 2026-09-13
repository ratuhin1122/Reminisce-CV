"""
app.memory — Memory Registration and Retrieval
================================================

Public API
----------
.. code-block:: python

    from app.memory import MemoryService

    service = MemoryService(db_path=config.db_path)
    service.start()
    mem_id = service.register_object(name="Watch", giver_name="Grandmother")
    service.stop()

Architecture
~~~~~~~~~~~~
::

    MemoryService      →  business logic, validation
        ↓
    MemoryRepository   →  typed data access (no SQL)
        ↓
    DatabaseManager    →  raw SQL, connection handling
"""

from app.memory.repository import MemoryRepository
from app.memory.service import MemoryService, MemoryWithReferences

__all__ = [
    "MemoryRepository",
    "MemoryService",
    "MemoryWithReferences",
]
