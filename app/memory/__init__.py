"""
app.memory — Memory Registration and Retrieval
================================================

Public API
----------
.. code-block:: python

    from app.memory import (
        MemoryService,
        MemoryRepository,
        MemoryWithReferences,
        ObjectRegistrationService,
        RegisteredObject,
        RegisteredReference,
    )

    # High-level object registration with CLIP embeddings & storage
    reg_service = ObjectRegistrationService(db_path=":memory:")
    reg_service.start()
    obj = reg_service.register_object(
        name="Pocket Watch",
        giver="Grandfather",
        reference_images=["path/to/watch.jpg"],
    )
    reg_service.stop()

Architecture
~~~~~~~~~~~~
::

    ObjectRegistrationService  →  image validation, CLIP embeddings, disk I/O
        ↓
    MemoryService              →  business logic, validation
        ↓
    MemoryRepository           →  typed data access (no SQL)
        ↓
    DatabaseManager            →  raw SQL, connection handling
"""

from app.memory.registration import (
    ObjectRegistrationService,
    RegisteredObject,
    RegisteredReference,
)
from app.memory.repository import MemoryRepository
from app.memory.service import MemoryService, MemoryWithReferences

__all__ = [
    "MemoryRepository",
    "MemoryService",
    "MemoryWithReferences",
    "ObjectRegistrationService",
    "RegisteredObject",
    "RegisteredReference",
]
