"""PseuDRAGON: Schema-Centric Compliance NLP for Closed-Network
Financial-Data Pseudonymization.

An authority-constrained, auditable pipeline for schema-level PII detection,
HITL knowledge accumulation, and executable pseudonymization policy generation.
"""

from .pipeline import PseuDRAGONPipeline

Pipeline = PseuDRAGONPipeline
PseuDRAGON = PseuDRAGONPipeline

__all__ = ["Pipeline", "PseuDRAGON", "PseuDRAGONPipeline"]
