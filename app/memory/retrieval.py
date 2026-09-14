"""
app.memory.retrieval — Personal Memory Retrieval for Recognized Entities
========================================================================

Retrieves stored personal memories from the SQLite database when the recognition
pipeline produces a stable entity ID:
1. Queries database for the verified entity ID.
2. Extracts stored fields (title/name, giver/person, occasion, year/date, narrative).
3. Produces a typed, structured ``StructuredMemoryResponse``.
4. Handles active/inactive status and missing metadata fields cleanly.

Integrity Guarantees:
---------------------
- ZERO LLM inference or hallucinated content.
- Strictly returns user-authored personal memories stored in the database.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from app.config import config
from app.database.models import Memory
from app.memory.service import MemoryService

logger = logging.getLogger(__name__)


@dataclass
class StructuredMemoryResponse:
    """Structured personal memory response for a recognized entity.

    Attributes
    ----------
    found : bool
        True if an existing, active memory was located in the database.
    entity_id : int or None
        Database ID of the memory.
    entity_type : str or None
        Discriminator: ``"object"`` or ``"person"``.
    title : str
        Name or title of the personal entity (e.g. "Grandmother's Gold Watch").
    giver : str or None
        Giver or associated person name (e.g. "Grandmother Elena").
    occasion : str or None
        Event or occasion associated with the memory (e.g. "High School Graduation").
    year : str or None
        Year or date of the memory (e.g. "1988").
    narrative : str or None
        User-authored personal story or narrative.
    is_active : bool
        Whether the memory record is active (not soft-deleted).
    raw_memory : Memory or None
        The underlying database model if located.
    """

    found: bool = False
    entity_id: Optional[int] = None
    entity_type: Optional[str] = None
    title: str = ""
    giver: Optional[str] = None
    occasion: Optional[str] = None
    year: Optional[str] = None
    narrative: Optional[str] = None
    is_active: bool = False
    raw_memory: Optional[Memory] = None

    # ── Convenience Aliases matching diverse domain terminologies ─────────────

    @property
    def name(self) -> str:
        """Alias for title."""
        return self.title

    @property
    def person(self) -> Optional[str]:
        """Alias for giver / person associated with the memory."""
        return self.giver

    @property
    def date(self) -> Optional[str]:
        """Alias for year / date."""
        return self.year

    @property
    def has_giver(self) -> bool:
        """Whether a giver/person name was provided."""
        return bool(self.giver and self.giver.strip())

    @property
    def has_occasion(self) -> bool:
        """Whether an occasion was provided."""
        return bool(self.occasion and self.occasion.strip())

    @property
    def has_year(self) -> bool:
        """Whether a year/date was provided."""
        return bool(self.year and self.year.strip())

    @property
    def has_narrative(self) -> bool:
        """Whether a narrative description was provided."""
        return bool(self.narrative and self.narrative.strip())

    @property
    def missing_fields(self) -> List[str]:
        """List of optional memory fields that were left empty."""
        missing: List[str] = []
        if not self.has_giver:
            missing.append("giver")
        if not self.has_occasion:
            missing.append("occasion")
        if not self.has_year:
            missing.append("year")
        if not self.has_narrative:
            missing.append("narrative")
        return missing

    def to_dict(self) -> Dict[str, Any]:
        """Convert structured response to a serializable dictionary."""
        return {
            "found": self.found,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "title": self.title,
            "name": self.title,
            "giver": self.giver,
            "person": self.giver,
            "occasion": self.occasion,
            "year": self.year,
            "date": self.year,
            "narrative": self.narrative,
            "is_active": self.is_active,
            "missing_fields": self.missing_fields,
        }

    def formatted_summary(self) -> str:
        """Produce a clean human-readable text summary of stored fields without LLMs."""
        if not self.found:
            return "No personal memory found."

        parts = [f"Memory: {self.title}"]
        if self.has_giver:
            parts.append(f"Giver/Person: {self.giver}")
        if self.has_occasion:
            parts.append(f"Occasion: {self.occasion}")
        if self.has_year:
            parts.append(f"Date: {self.year}")
        if self.has_narrative:
            parts.append(f"Story: {self.narrative}")

        return " | ".join(parts)


class MemoryRetrievalService:
    """Service that queries the SQLite database for personal memories of recognized entities.

    Parameters
    ----------
    memory_service : MemoryService, optional
        Underlying memory persistence service. If None, instantiates with db_path.
    db_path : str or Path, optional
        Database file path. Defaults to ``config.db_path``.
    active_only : bool, optional
        Whether to require memories to be active by default. Default is True.
    """

    def __init__(
        self,
        memory_service: Optional[MemoryService] = None,
        db_path: Optional[Union[str, Path]] = None,
        active_only: bool = True,
    ) -> None:
        if memory_service is not None:
            self._mem_service = memory_service
        else:
            path = db_path or config.db_path
            self._mem_service = MemoryService(db_path=path)

        self.active_only: bool = active_only

    @property
    def memory_service(self) -> MemoryService:
        """Direct access to underlying memory service."""
        return self._mem_service

    def retrieve_memory(
        self,
        entity_id: Optional[int],
        active_only: Optional[bool] = None,
    ) -> StructuredMemoryResponse:
        """Retrieve the stored personal memory for an entity ID.

        Parameters
        ----------
        entity_id : int or None
            Database memory ID to fetch.
        active_only : bool, optional
            If True, returns found=False if memory is inactive.
            Defaults to self.active_only (True).

        Returns
        -------
        StructuredMemoryResponse
            Structured response containing all stored personal memory fields.
        """
        if entity_id is None or entity_id <= 0:
            return StructuredMemoryResponse(
                found=False,
                entity_id=entity_id,
                title="",
                is_active=False,
            )

        require_active = active_only if active_only is not None else self.active_only

        try:
            memory: Optional[Memory] = self._mem_service.get_memory(entity_id)
        except RuntimeError:
            # Auto-start connection if uninitialized
            self._mem_service.start()
            memory = self._mem_service.get_memory(entity_id)
        except Exception as exc:
            logger.warning("Error querying memory for entity ID %s: %s", entity_id, exc)
            return StructuredMemoryResponse(
                found=False,
                entity_id=entity_id,
                title="",
                is_active=False,
            )

        # Entity not found in database
        if memory is None:
            logger.debug("Memory ID %d not found in database.", entity_id)
            return StructuredMemoryResponse(
                found=False,
                entity_id=entity_id,
                title="",
                is_active=False,
            )

        # Handle inactive memory
        if not memory.is_active:
            logger.debug("Memory ID %d is marked inactive.", entity_id)
            if require_active:
                return StructuredMemoryResponse(
                    found=False,
                    entity_id=memory.id,
                    entity_type=(
                        memory.entity_type.value
                        if hasattr(memory.entity_type, "value")
                        else str(memory.entity_type)
                    ),
                    title=memory.name or "",
                    giver=memory.giver_name,
                    occasion=memory.occasion,
                    year=memory.year,
                    narrative=memory.narrative,
                    is_active=False,
                    raw_memory=memory,
                )

        # Active memory found
        entity_type_str = (
            memory.entity_type.value
            if hasattr(memory.entity_type, "value")
            else str(memory.entity_type)
        )

        return StructuredMemoryResponse(
            found=True,
            entity_id=memory.id,
            entity_type=entity_type_str,
            title=memory.name or "",
            giver=memory.giver_name,
            occasion=memory.occasion,
            year=memory.year,
            narrative=memory.narrative,
            is_active=memory.is_active,
            raw_memory=memory,
        )

    def retrieve_from_candidate(
        self, candidate: Any, active_only: Optional[bool] = None
    ) -> StructuredMemoryResponse:
        """Retrieve memory from a TrackedEntity or candidate object."""
        if candidate is None:
            return StructuredMemoryResponse(found=False)

        entity_id = getattr(candidate, "entity_id", None)
        return self.retrieve_memory(entity_id=entity_id, active_only=active_only)
