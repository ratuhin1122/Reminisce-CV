"""
app.recognition — Recognition Engine
====================================

Public API
----------
.. code-block:: python

    from app.recognition import (
        ObjectRecognitionService,
        RecognitionResult,
        PersonRecognitionService,
        PersonRecognitionResult,
    )
"""

from app.recognition.person_service import (
    PersonRecognitionResult,
    PersonRecognitionService,
)
from app.recognition.service import (
    ObjectRecognitionService,
    RecognitionResult,
)

__all__ = [
    "ObjectRecognitionService",
    "PersonRecognitionResult",
    "PersonRecognitionService",
    "RecognitionResult",
]
