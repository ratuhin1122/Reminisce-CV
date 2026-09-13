"""
app.database.models — Data Transfer Objects for the Memory Database
====================================================================

Plain dataclasses that represent rows in the database.  These are
*not* ORM models — they are simple containers that travel between
the database layer and the rest of the application.

Entity types
------------
- ``Memory``           — a personal memory (object or person)
- ``VisualReference``  — a registered image + embedding for a memory
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class EntityType(str, Enum):
    """Discriminator for the kind of entity a memory represents.

    Using str as a base allows direct storage in SQLite TEXT columns
    and JSON serialization without custom converters.
    """

    PERSON = "person"
    OBJECT = "object"


@dataclass
class Memory:
    """A single personal memory entry.

    Attributes
    ----------
    id : int or None
        Auto-assigned by SQLite on INSERT.
    entity_type : EntityType
        Whether this memory is about a person or an object.
    name : str
        The name or title of the entity (e.g. "Grandmother's Watch").
    giver_name : str or None
        Who gave the item or the person's full name.
    occasion : str or None
        The occasion associated with the memory (e.g. "Birthday 2019").
    year : str or None
        Year or date string — kept as text for flexibility.
    narrative : str or None
        A personal story or description associated with the memory.
    is_active : bool
        Soft-delete flag.  Inactive memories are excluded from
        recognition but preserved in the database.
    created_at : datetime or None
        Set automatically on INSERT.
    updated_at : datetime or None
        Set automatically on INSERT and UPDATE.
    """

    id: Optional[int] = None
    entity_type: EntityType = EntityType.OBJECT
    name: str = ""
    giver_name: Optional[str] = None
    occasion: Optional[str] = None
    year: Optional[str] = None
    narrative: Optional[str] = None
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class VisualReference:
    """A registered visual reference (image + embedding) for a memory.

    Images are stored on disk, not inside SQLite.  This record holds
    the *path* to the image file and the *path* to the serialized
    embedding file (e.g. a .npy file).

    Attributes
    ----------
    id : int or None
        Auto-assigned by SQLite on INSERT.
    memory_id : int
        Foreign key → ``memories.id``.
    image_path : str
        Filesystem path to the reference image.
    embedding_path : str or None
        Filesystem path to the serialized embedding (e.g. .npy).
    created_at : datetime or None
        Set automatically on INSERT.
    """

    id: Optional[int] = None
    memory_id: int = 0
    image_path: str = ""
    embedding_path: Optional[str] = None
    created_at: Optional[datetime] = None
