"""PseuDRAGON - PseuDonymizing RAG is ON for Sensitive Data

A RAG-enhanced multi-stage automated framework for privacy-preserving data processing.
"""

from .pipeline import PseuDRAGONPipeline

Pipeline = PseuDRAGONPipeline
PseuDRAGON = PseuDRAGONPipeline

__all__ = ["Pipeline", "PseuDRAGON", "PseuDRAGONPipeline"]
