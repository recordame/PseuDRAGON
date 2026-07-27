"""Stage modules for PseuDRAGON pipeline."""

from .stage1_pii_detection import Stage1PIIDetection
from .stage2_policy_synthesis import Stage2PolicySynthesis
from .stage3_hitl_refinement import Stage3HITLRefinement
from .stage4_code_generation import Stage4CodeGeneration

# Aliases for backward compatibility
Stage1PIIDetection = Stage1PIIDetection
Stage2PolicySynthesis = Stage2PolicySynthesis
Stage3HITLRefinement = Stage3HITLRefinement
Stage4CodeGeneration = Stage4CodeGeneration

__all__ = ["Stage1PIIDetection", "Stage2PolicySynthesis", "Stage3HITLRefinement", "Stage4CodeGeneration"]
