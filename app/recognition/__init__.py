"""
app.recognition — Real-Time Recognition Engine
================================================

Responsibilities:
    - Orchestrate face + object recognition per frame
    - Temporal stability / debouncing (require N consecutive matches)
    - Per-entity speech cooldown tracking
    - Coordinate between vision, database, and speech modules

Implementation added in Stage 5.
"""
