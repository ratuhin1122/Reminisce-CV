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
from app.recognition.pipeline import (
    FrameRecognitionResult,
    PipelineTiming,
    RealTimeRecognitionPipeline,
)
from app.recognition.region_proposal import (
    CandidateRegion,
    CandidateRegionExtractor,
)
from app.recognition.service import (
    ObjectRecognitionService,
    RecognitionResult,
)

__all__ = [
    "CandidateRegion",
    "CandidateRegionExtractor",
    "FrameRecognitionResult",
    "ObjectRecognitionService",
    "PersonRecognitionResult",
    "PersonRecognitionService",
    "PipelineTiming",
    "RealTimeRecognitionPipeline",
    "RecognitionResult",
]

