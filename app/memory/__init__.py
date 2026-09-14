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
        PersonRegistrationService,
        RegisteredPerson,
        RegisteredFaceReference,
    )
"""

from app.memory.person_registration import (
    PersonRegistrationService,
    RegisteredFaceReference,
    RegisteredPerson,
)
from app.memory.registration import (
    ObjectRegistrationService,
    RegisteredObject,
    RegisteredReference,
)
from app.memory.repository import MemoryRepository
from app.memory.retrieval import (
    MemoryRetrievalService,
    StructuredMemoryResponse,
)
from app.memory.service import MemoryService, MemoryWithReferences

__all__ = [
    "MemoryRepository",
    "MemoryRetrievalService",
    "MemoryService",
    "MemoryWithReferences",
    "ObjectRegistrationService",
    "PersonRegistrationService",
    "RegisteredFaceReference",
    "RegisteredObject",
    "RegisteredPerson",
    "RegisteredReference",
    "StructuredMemoryResponse",
]

