"""
app.recognition — Recognition Engine
====================================

Public API
----------
.. code-block:: python

    from app.recognition import ObjectRecognitionService, RecognitionResult

    service = ObjectRecognitionService()
    result = service.recognize("query_crop.jpg")

    if result.matched:
        print(f"Recognized: {result.name} (Similarity: {result.similarity:.2f})")
    else:
        print(f"No match found. Top score: {result.similarity:.2f} < {result.threshold:.2f}")
"""

from app.recognition.service import (
    ObjectRecognitionService,
    RecognitionResult,
)

__all__ = [
    "ObjectRecognitionService",
    "RecognitionResult",
]
