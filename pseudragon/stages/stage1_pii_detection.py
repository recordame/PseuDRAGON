"""
Optimized Stage 1: PII Detection with Universal Caching

Supports both OpenAI (prompt caching) and Ollama (KV cache).
"""

# Standard library imports
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

# Project-specific imports
from config.config import HeuristicConfig, Settings
from pseudragon.llm.json_parser import safe_parse_json
from pseudragon.llm.session_manager import UniversalSessionManager


STAGE1_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "stage1_pii_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "is_pii": {"type": "boolean"},
                "pii_type": {"type": "string", "enum": ["PII", "Non-PII"]},
                "confidence": {
                    "type": "string",
                    "enum": ["High", "Medium", "Low"],
                },
                "reasoning": {"type": "string"},
                "evidence_source": {"type": "string"},
                "chain_of_thought": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "input_analysis": {"type": "string"},
                        "key_features": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "decision_steps": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "legal_references": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "final_justification": {"type": "string"},
                    },
                    "required": [
                        "input_analysis",
                        "key_features",
                        "decision_steps",
                        "legal_references",
                        "final_justification",
                    ],
                },
            },
            "required": [
                "is_pii",
                "pii_type",
                "confidence",
                "reasoning",
                "evidence_source",
                "chain_of_thought",
            ],
        },
    },
}

STAGE1_COMPACT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "stage1_pii_compact_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "is_pii": {"type": "boolean"},
                "pii_type": {"type": "string", "enum": ["PII", "Non-PII"]},
                "confidence": {
                    "type": "string",
                    "enum": ["High", "Medium", "Low"],
                },
                "reasoning": {"type": "string", "maxLength": 20},
                "evidence_source": {"type": "string", "maxLength": 20},
            },
            "required": [
                "is_pii",
                "pii_type",
                "confidence",
                "reasoning",
                "evidence_source",
            ],
        },
    },
}


class Stage1LLMResponseError(RuntimeError):
    """Raised when Stage 1 cannot obtain a complete, valid JSON decision."""


def load_prompt(stage_name: str, prompt_type: str = "system") -> str:
    """
    Load prompt template from resources/prompts/ based on PROMPT_MODE
    
    Args:
        stage_name: Name of the stage (e.g., "stage1_pii_identification")
        prompt_type: Type of prompt ("system" or "user")
    
    Returns:
        Prompt text content
    """
    prompt_dir = os.path.join(os.path.dirname(__file__), "..", "..")

    # Extract stage number (stage1 or stage2) from stage_name
    if "stage1" in stage_name:
        stage = "stage1"
    elif "stage2" in stage_name:
        stage = "stage2"
    else:
        # Fallback to old behavior for other stages
        filename = f"{stage_name}_{prompt_type}.txt"
        filepath = os.path.join(prompt_dir, "resources", "prompts", filename)
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    # Use Settings.get_prompt_path() for stage1 and stage2
    prompt_path = Settings.get_prompt_path(stage, prompt_type)
    filepath = os.path.join(prompt_dir, prompt_path)

    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


class Stage1PIIDetection:
    """
    Optimized Stage 1: PII Identification with Universal Session Management
    
    Key features:
    - Auto-detects LLM provider (OpenAI, Ollama, etc.)
    - Applies provider-specific caching optimizations
    - System prompt sent only once per table
    - Estimated 70-90% reduction in system prompt tokens
    
    Provider-specific optimizations:
    - OpenAI: Automatic prompt caching
    - Ollama: KV cache with keep_alive
    - Others: Manual conversation history
    """

    def __init__(self, rag_system, llm_client, provider: str = "auto", base_url: Optional[str] = None, expert_preference_manager=None):
        """
        Initialize optimized Stage 1 with provider detection.

        Args:
            rag_system: RAG system for context retrieval
            llm_client: LLM client
            provider: LLM provider ("auto", "openai", "ollama", "other")
            base_url: Base URL for API (used for auto-detection)
            expert_preference_manager: Optional user preference manager for learned PII classifications
        """
        self.rag = rag_system
        self.client = llm_client
        self.expert_preference_manager = expert_preference_manager
        self.heuristics_enabled = getattr(Settings, "HEURISTICS_ENABLED", True)

        # Get base_url from Settings if not provided
        if base_url is None:
            base_url = getattr(Settings, 'STAGE_1_OPENAI_BASE_URL', None)

        # Initialize universal session manager with provider detection
        self.session_manager = UniversalSessionManager(llm_client, provider=provider, base_url=base_url)

        # Load system prompt once
        self.system_prompt = load_prompt("stage1_pii_identification", "system")

        # Load heuristic patterns from HeuristicManager
        from pseudragon.heuristics.heuristic_manager import HeuristicManager
        self.heuristic_manager = HeuristicManager()

        # Load environment-based patterns as fallback
        self.heuristic_patterns = HeuristicConfig.get_heuristic_patterns()
        self.pii_type_keywords = HeuristicConfig.get_pii_type_keywords()

        # Load Non-PII patterns
        self.non_pii_patterns = HeuristicConfig.get_non_pii_patterns()

        # Initialize audit logger (optional, for COT logging)
        self.audit_logger = None

    def identify_pii(
            self,
            schema: Dict[str, Any],
            table_name: str,
            log_callback: Optional[Callable] = None,
            semantic_similarity_weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Identify PII columns using session-based approach with provider detection.

        Now supports batch parallel processing for LLM verification.

        Args:
            schema: Enhanced schema with column metadata
            table_name: Name of the table
            log_callback: Optional logging callback
            semantic_similarity_weight: Optional per-run alpha override

        Returns:
            Dictionary mapping column names to PII analysis results
        """
        results = {}

        # Create or get session for this table
        # CRITICAL: max_history=0 prevents token overflow
        # Each column verification is independent, so no conversation history is needed
        # This keeps only the system prompt and discards previous user/assistant messages
        session_id = f"stage1_{table_name}"
        session = self.session_manager.get_or_create_session(
            session_id=session_id,
            system_prompt=self.system_prompt,
            ttl_minutes=60,
            max_history=0  # No history - each column is independent
        )

        self._log(f"[Search] [Stage 1] PII Identification for '{table_name}'", log_callback)
        self._log(f"Provider: {session.provider} | Session ID: {session_id}", log_callback)

        # Statistics
        pii_heuristic_count = 0
        non_pii_heuristic_count = 0
        llm_verification_count = 0
        learned_classification_count = 0

        # Columns that need LLM verification (will be processed in parallel batches)
        llm_verification_columns: List[Tuple[str, Dict[str, Any]]] = []

        # Learned heuristic classification count
        learned_heuristic_count = 0

        # First pass: Apply heuristics and learned classifications (fast, sequential)
        for column, metadata in schema.items():
            data_type = metadata.get("type", "")
            column_comment = metadata.get("comment", "")

            # Step 0: Check learned PII classification (HIGHEST PRIORITY)
            # If user has previously corrected this column's classification, use that
            # Now supports RAG-based similarity search for similar columns
            learned_result = None
            if self.heuristics_enabled and self.expert_preference_manager:
                learned_result = self.expert_preference_manager.get_learned_pii_classification(
                    column_name=column,
                    column_comment=column_comment,
                    data_type=data_type
                )

            if learned_result:
                learned_classification_count += 1
                classification = learned_result["classification"]
                match_type = learned_result.get("match_type", "exact")
                similar_column = learned_result.get("similar_column", column)
                similarity = learned_result.get("similarity", 1.0)
                rationale = learned_result.get("rationale", "")
                similar_comment = learned_result.get("similar_column_comment", "")

                is_pii = (classification == "PII")

                # Determine evidence source based on match type
                if match_type == "exact":
                    evidence_source = "EXPERT FEEDBACK (Learned)"
                    reasoning_prefix = f"User previously classified {column} as"
                else:
                    evidence_source = f"EXPERT FEEDBACK (RAG Similar: {similar_column}, similarity={similarity:.2f})"
                    reasoning_prefix = f"Similar column {similar_column} was classified as"

                if is_pii:
                    self._log(f"   [OK] Learned PII: {column} ({match_type} match" + (f", similar to {similar_column}" if match_type == "rag_similar" else "") + ")", log_callback)
                    if match_type == "rag_similar" and similar_comment:
                        self._log(f"     Similar column comment: {similar_comment}", log_callback)
                    if rationale:
                        self._log(f"     Rationale: {rationale}", log_callback)

                    results[column] = {
                        "is_pii": True,
                        "pii_type": self._infer_pii_type(column, column_comment),
                        "reasoning": f"{reasoning_prefix} PII" + (f" (rationale: {rationale})" if rationale else ""),
                        "evidence_source": evidence_source,
                        "detection_method": "learned",
                        "confidence": "Very High" if match_type == "exact" else "High",
                        "rationale": rationale or f"User correction from previous session",
                        "column_comment": column_comment,
                    }
                else:
                    self._log(f"   [X] Learned Non-PII: {column} ({match_type} match" + (f", similar to {similar_column}" if match_type == "rag_similar" else "") + ")", log_callback)
                    if match_type == "rag_similar" and similar_comment:
                        self._log(f"     Similar column comment: {similar_comment}", log_callback)

                    results[column] = {
                        "is_pii": False,
                        "pii_type": "Non-PII",
                        "reasoning": f"{reasoning_prefix} Non-PII",
                        "evidence_source": evidence_source,
                        "detection_method": "learned",
                        "confidence": "Very High" if match_type == "exact" else "High",
                        "rationale": rationale or f"User correction from previous session",
                        "column_comment": column_comment,
                    }
                continue

            # Step 0.5: Check learned heuristic patterns (second highest priority)
            # Fast pattern matching using regex patterns learned from expert feedback
            # This is faster than RAG similarity search (< 1ms vs ~100ms)
            heuristic_result = None
            if self.heuristics_enabled and self.expert_preference_manager:
                heuristic_result = self.expert_preference_manager.classify_by_heuristics(
                    column_name=column,
                    column_comment=column_comment
                )

            if heuristic_result:
                learned_heuristic_count += 1
                classification = heuristic_result["classification"]
                match_type = heuristic_result.get("match_type", "learned_heuristic")
                pattern = heuristic_result.get("pattern", "")
                confidence = heuristic_result.get("confidence", 0.8)
                samples = heuristic_result.get("samples", [])

                is_pii = (classification == "PII")

                evidence_source = f"LEARNED HEURISTIC ({match_type}: {pattern})"

                if is_pii:
                    self._log(f"   [OK] Learned Heuristic PII: {column} (pattern: {pattern}, conf: {confidence:.2f})", log_callback)
                    if samples:
                        self._log(f"     Similar columns: {', '.join(samples[:3])}", log_callback)

                    results[column] = {
                        "is_pii": True,
                        "pii_type": self._infer_pii_type(column, column_comment),
                        "reasoning": f"Matched learned heuristic pattern: {pattern}",
                        "evidence_source": evidence_source,
                        "detection_method": "learned_heuristic",
                        "confidence": "High" if confidence >= 0.9 else "Medium",
                        "rationale": f"Pattern learned from expert feedback (similar columns: {', '.join(samples[:3])})" if samples else f"Pattern learned from expert feedback",
                        "column_comment": column_comment,
                    }
                else:
                    self._log(f"   [X] Learned Heuristic Non-PII: {column} (pattern: {pattern}, conf: {confidence:.2f})", log_callback)

                    results[column] = {
                        "is_pii": False,
                        "pii_type": "Non-PII",
                        "reasoning": f"Matched learned heuristic pattern: {pattern}",
                        "evidence_source": evidence_source,
                        "detection_method": "learned_heuristic",
                        "confidence": "High" if confidence >= 0.9 else "Medium",
                        "rationale": f"Pattern learned from expert feedback",
                        "column_comment": column_comment,
                    }
                continue

            # Step 1: Check PII heuristic (ALWAYS checked first)
            # This ensures PII columns are caught before Non-PII patterns
            # Example: 'phone_number' matches PII pattern first, never reaches Non-PII check
            if self.heuristics_enabled and self._heuristic_pii(column, column_comment):
                pii_heuristic_count += 1
                self._log(f"   [OK] Heuristic PII: {column}", log_callback)

                results[column] = {
                    "is_pii": True,
                    "pii_type": self._infer_pii_type(column, column_comment),
                    "reasoning": f"Matched PII heuristic pattern for {column}" + (f" (Comment: {column_comment})" if column_comment else ""),
                    "evidence_source": "Column Naming Heuristic" + (" + Column Comment Heuristic" if column_comment else ""),
                    "detection_method": "heuristic",
                    "confidence": "High",
                    "rationale": f"Column name matches known PII pattern",
                    "column_comment": column_comment,
                }

            # Step 2: Check Non-PII heuristic (only if NOT PII)
            # Safe to use broad patterns (e.g., 'number') because PII already filtered out
            elif self.heuristics_enabled and self._heuristic_non_pii(column, column_comment):
                non_pii_heuristic_count += 1
                self._log(f"   [X] Heuristic Non-PII: {column} (LLM skipped)", log_callback)

                results[column] = {
                    "is_pii": False,
                    "pii_type": "Non-PII",
                    "reasoning": f"Matched Non-PII heuristic pattern (system/metadata column)" + (f" (Comment: {column_comment})" if column_comment else ""),
                    "evidence_source": "Non-PII Column Heuristic",
                    "detection_method": "heuristic",
                    "confidence": "High",
                    "rationale": f"Column identified as system or metadata field",
                    "column_comment": column_comment,
                }

            # Step 3: Uncertain cases - queue for LLM verification
            else:
                llm_verification_columns.append((column, metadata))

        # Second pass: Process LLM verification columns in parallel batches
        if llm_verification_columns:
            llm_failures = []
            batch_size = Settings.LLM_BATCH_SIZE
            max_workers = Settings.MAX_CONCURRENT_LLM_REQUESTS
            total_llm_columns = len(llm_verification_columns)

            self._log(f"\n   [Parallel] Parallel LLM verification: {total_llm_columns} columns (batch_size={batch_size}, max_workers={max_workers})", log_callback)

            # Process in batches
            for batch_start in range(0, total_llm_columns, batch_size):
                batch_end = min(batch_start + batch_size, total_llm_columns)
                batch = llm_verification_columns[batch_start:batch_end]
                batch_num = (batch_start // batch_size) + 1
                total_batches = (total_llm_columns + batch_size - 1) // batch_size

                self._log(f"   [Batch] Processing batch {batch_num}/{total_batches} ({len(batch)} columns)", log_callback)

                # Process batch in parallel using ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks in this batch
                    future_to_column = {}
                    for column, metadata in batch:
                        future = executor.submit(
                            self._process_single_column_llm,
                            session=session,
                            column=column,
                            metadata=metadata,
                            table_name=table_name,
                            log_callback=log_callback,
                            semantic_similarity_weight=semantic_similarity_weight,
                        )
                        future_to_column[future] = column

                    # Collect results as they complete
                    for future in as_completed(future_to_column):
                        column = future_to_column[future]
                        try:
                            result = future.result()
                            results[column] = result
                            llm_verification_count += 1
                        except Exception as e:
                            self._log(f"   [ERROR] Error processing {column}: {str(e)}", log_callback)
                            llm_failures.append((column, str(e)))

                if llm_failures:
                    failed_columns = ", ".join(
                        column for column, _ in llm_failures
                    )
                    failure_details = "; ".join(
                        f"{column}: {error}"
                        for column, error in llm_failures
                    )
                    raise Stage1LLMResponseError(
                        "Stage 1 stopped without emitting fallback "
                        f"classifications. Failed columns: {failed_columns}. "
                        f"Details: {failure_details}"
                    )

        # Get session stats
        stats = session.get_stats()

        # Enhanced logging with optimization stats
        total_columns = len(schema)
        llm_skipped = learned_classification_count + learned_heuristic_count + pii_heuristic_count + non_pii_heuristic_count
        optimization_rate = (llm_skipped / total_columns * 100) if total_columns > 0 else 0

        self._log(f"\n[Stats] Stage 1 Summary:", log_callback)
        self._log(f"   Total columns: {total_columns}", log_callback)
        if learned_classification_count > 0:
            self._log(f"   Learned (EXPERT FEEDBACK): {learned_classification_count} ", log_callback)
        if learned_heuristic_count > 0:
            self._log(f"   Learned Heuristic: {learned_heuristic_count} [Update]", log_callback)
        self._log(f"   PII (heuristic): {pii_heuristic_count}", log_callback)
        self._log(f"   Non-PII (heuristic): {non_pii_heuristic_count}", log_callback)
        self._log(f"   LLM verification: {llm_verification_count}", log_callback)
        self._log(f"   LLM calls avoided: {llm_skipped} ({optimization_rate:.1f}%)", log_callback)
        self._log(f"   Tokens saved: ~{stats['estimated_tokens_saved']}", log_callback)

        return results

    def _process_single_column_llm(
            self,
            session,
            column: str,
            metadata: Dict[str, Any],
            table_name: str,
            log_callback: Optional[Callable] = None,
            semantic_similarity_weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Process a single column with LLM verification (used for parallel processing).

        Args:
            session: LLM session
            column: Column name
            metadata: Column metadata
            table_name: Table name
            log_callback: Optional logging callback

        Returns:
            PII analysis result for the column
        """
        data_type = metadata.get("type", "")
        column_comment = metadata.get("comment", "")

        self._log(f"   -> LLM verification: {column}", log_callback)

        llm_result = self._llm_verify_pii_with_session(
            session=session,
            column=column,
            table_name=table_name,
            data_type=data_type,
            column_comment=column_comment,
            log_callback=log_callback,
            semantic_similarity_weight=semantic_similarity_weight,
        )

        # Extract LLM decision for comparison
        llm_is_pii = llm_result.get("is_pii", False)
        llm_pii_type = llm_result.get("pii_type", "Unknown")

        if llm_is_pii:
            pii_type = llm_pii_type

            # If LLM returns "Unknown", use fallback inference
            if pii_type == "Unknown":
                pii_type = self._infer_pii_type(column, column_comment)
                self._log(f"   [WARN]️  [PII Type Override]: LLM returned 'Unknown', using fallback inference: {pii_type}", log_callback)

            result = {
                "is_pii": True,
                "pii_type": pii_type,
                "evidence": llm_result.get("evidence", ""),
                "reasoning": llm_result.get("reasoning", ""),
                "evidence_source": llm_result.get("evidence_source", "LLM Verification"),
                "column_comment": column_comment,
                "detection_method": "llm",
                "confidence": llm_result.get("confidence", "Medium")
            }
            self._log(f"    [OK] {column}: PII ({pii_type})", log_callback)
            self._log(f"   [OK] [Final Classification]: is_pii=True, pii_type={pii_type} (matches LLM decision)", log_callback)
        else:
            result = {
                "is_pii": False,
                "pii_type": "Non-PII",
                "evidence": llm_result.get("evidence", ""),
                "reasoning": llm_result.get("reasoning", ""),
                "evidence_source": llm_result.get("evidence_source", "LLM Verification"),
                "column_comment": column_comment,
                "detection_method": "llm",
                "confidence": llm_result.get("confidence", "Medium")
            }
            self._log(f"    [X] {column}: Not PII", log_callback)
            self._log(f"   [OK] [Final Classification]: is_pii=False (matches LLM decision)", log_callback)

        return result

    def _heuristic_pii(self, column_name: str, column_comment: str = "") -> bool:
        """
        Fast heuristic check using regex patterns on column name and comment.

        Implements HEURISTIC_PII() from Algorithm 1 with COMMENT support.

        Args:
            column_name: Name of the column
            column_comment: Column description/comment from database schema
        """
        column_lower = column_name.lower()
        comment_lower = column_comment.lower() if column_comment else ""

        # Check environment-based patterns first (for backwards compatibility)
        for pii_type, pattern in self.heuristic_patterns.items():
            if re.search(pattern, column_lower) or re.search(pattern, comment_lower):
                return True

        # Check HeuristicManager patterns
        matched_heuristic = self.heuristic_manager.match_column(column_name)
        if matched_heuristic and matched_heuristic.get("pii_type") != "Non-PII":
            return True

        # Also check comment against heuristics if available
        if column_comment:
            matched_comment_heuristic = self.heuristic_manager.match_column(column_comment)
            if matched_comment_heuristic and matched_comment_heuristic.get("pii_type") != "Non-PII":
                return True

        return False

    def _infer_pii_type(self, column_name: str, column_comment: str = "") -> str:
        """
        Infer PII type from column name and comment

        CRITICAL: This function is only called for columns that matched PII heuristics.
        Therefore, it should NEVER return "Non-PII".

        Args:
            column_name: Name of the column
            column_comment: Column description/comment from database schema

        Returns:
            PII type: "PII" (NEVER returns "Non-PII")
        """
        # First check HeuristicManager for more specific matches
        matched_heuristic = self.heuristic_manager.match_column(column_name)
        if matched_heuristic:
            pii_type = matched_heuristic.get("pii_type")
            # CRITICAL: Only accept valid PII types, reject "Non-PII"
            # This prevents the bug where is_pii=True but pii_type="Non-PII"
            if pii_type and pii_type == "PII":
                return pii_type

        # Check comment if available
        if column_comment:
            matched_comment_heuristic = self.heuristic_manager.match_column(column_comment)
            if matched_comment_heuristic:
                pii_type = matched_comment_heuristic.get("pii_type")
                # CRITICAL: Only accept valid PII types, reject "Non-PII"
                if pii_type and pii_type == "PII":
                    return pii_type

        # Default: All PII columns are classified as PII
        return "PII"

    def _heuristic_non_pii(self, column_name: str, column_comment: str = "") -> bool:
        """
        Fast heuristic check for clearly Non-PII columns
        
        Identifies system columns, metadata, timestamps, counters, etc.
        that are clearly not personal information.
        
        Args:
            column_name: Name of the column
            column_comment: Column description/comment from database schema
        
        Returns:
            True if column is clearly Non-PII (skip LLM verification)
        """
        column_lower = column_name.lower()
        comment_lower = column_comment.lower() if column_comment else ""

        # Check environment-based patterns first
        for pattern_name, pattern in self.non_pii_patterns.items():
            if re.search(pattern, column_lower) or re.search(pattern, comment_lower):
                return True

        # Check HeuristicManager patterns for Non-PII
        matched_heuristic = self.heuristic_manager.match_column(column_name)
        if matched_heuristic and matched_heuristic.get("pii_type") == "Non-PII":
            return True

        # Check comment against heuristics if available
        if column_comment:
            matched_comment_heuristic = self.heuristic_manager.match_column(column_comment)
            if matched_comment_heuristic and matched_comment_heuristic.get("pii_type") == "Non-PII":
                return True

        return False

    def _llm_verify_pii_with_session(
            self,
            session,
            column: str,
            table_name: str,
            data_type: str = "",
            column_comment: str = "",
            log_callback: Optional[Callable] = None,
            semantic_similarity_weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        LLM-based PII verification using universal session.
        
        Automatically uses provider-specific optimizations:
        - OpenAI: Prompt caching + Streaming (if enabled)
        - Ollama: KV cache with keep_alive
        """
        # Build column description with structured format
        column_desc = f"Column Name: {column}"
        column_desc += f"\nData Type: {data_type if data_type else 'Not specified'}"
        column_desc += f"\nDescription: {column_comment if column_comment else 'Not provided'}"

        # Retrieve context from RAG with detailed information (include data type in query)
        data_type = data_type if data_type else ""
        query = (
            f"Does the schema column '{column}' ({data_type}) with the description "
            f"'{column_comment}' constitute Personal Identifiable Information (PII)?"
        )

        # Use retrieve_with_details if available, otherwise fall back to retrieve
        # Prioritize 'status_change' feedback for PII identification
        #
        # NOTE: The RAG system now performs DUAL-MODEL SEARCH:
        # 1. Legal documents: configured legal-domain embedding model
        # 2. Feedback documents: SBERT for semantic similarity (finds SIMILAR columns, not exact match)
        #
        # metadata_filter is passed for logging purposes but NOT used for strict filtering
        metadata_filter = {'table': table_name, 'column': column}

        # Check if RAG is available (not None)
        if self.rag is not None:
            if hasattr(self.rag, 'retrieve_with_details'):
                context, source_docs, doc_details = self.rag.retrieve_with_details(
                    query,
                    exclude_feedback=False,
                    allowed_feedback_types=['status_change'],
                    metadata_filter=metadata_filter,
                    semantic_similarity_weight=semantic_similarity_weight,
                )
            else:
                context, source_docs = self.rag.retrieve(
                    query,
                    exclude_feedback=False,
                    allowed_feedback_types=['status_change'],
                    metadata_filter=metadata_filter,
                    semantic_similarity_weight=semantic_similarity_weight,
                )
                doc_details = []

            # Log RAG context retrieval for debugging (summary only)
            self._log(f"\n   [RAG] [RAG Context for '{column}' ({data_type or 'unknown type'})]:", log_callback)
            self._log(f"      Query: {query}", log_callback)
            self._log(f"      Retrieved {len(doc_details)} document(s)", log_callback)
            # Note: Document details are not logged to avoid excessive output
        else:
            # RAG is disabled - use empty context
            context = ""
            source_docs = []
            doc_details = []
            self._log(f"\n   [LLM-ONLY] RAG disabled, using LLM knowledge only for '{column}'", log_callback)

        # Load user prompt template
        user_prompt_template = load_prompt("stage1_pii_identification", "user")
        user_prompt = user_prompt_template.format(
            context=context,
            table_name=table_name,
            column_desc=column_desc,
            data_type=data_type,
        )

        # Prepare streaming callback for real-time COT logging
        stream_enabled = Settings.ENABLE_STREAMING and session.supports_streaming()
        stream_callback = None

        if stream_enabled and Settings.STREAMING_LOG_CHUNKS:
            # Accumulate chunks for logging (but don't log individual chunks)
            accumulated_chunks = []

            def log_stream_chunk(chunk: str):
                """Real-time logging callback for streaming chunks."""
                accumulated_chunks.append(chunk)
                # Note: Individual chunks are not logged to avoid excessive output

            stream_callback = log_stream_chunk

            # Note: Streaming header removed to reduce log verbosity

        result = None
        parse_errors = []
        max_attempts = Settings.LLM_PARSE_MAX_ATTEMPTS
        compact_response = (
            os.getenv("STAGE1_COMPACT_RESPONSE", "false").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        response_format = (
            STAGE1_COMPACT_RESPONSE_FORMAT
            if compact_response
            else STAGE1_RESPONSE_FORMAT
        )
        max_tokens = (
            min(Settings.STAGE_1_MAX_TOKENS, 256)
            if compact_response
            else Settings.STAGE_1_MAX_TOKENS
        )

        for attempt in range(1, max_attempts + 1):
            # Use session.chat() with streaming support. The user and system
            # prompts remain unchanged across retries.
            response = session.chat(
                user_prompt=user_prompt,
                model=Settings.LLM_STAGE_1,
                temperature=Settings.LLM_TEMPERATURE,
                response_format=response_format,
                stream=stream_enabled,
                stream_callback=stream_callback,
                max_tokens=max_tokens,
            )

            content = response.choices[0].message.content
            candidate = safe_parse_json(content)
            validation_error = self._validate_stage1_llm_result(
                candidate,
                require_chain_of_thought=not compact_response,
            )

            if validation_error is None:
                result = candidate
                if attempt > 1:
                    self._log(
                        f"   [OK] Valid JSON recovered for '{column}' "
                        f"on attempt {attempt}/{max_attempts}",
                        log_callback,
                    )
                break

            finish_reason = getattr(response.choices[0], "finish_reason", None)
            parse_errors.append(
                f"attempt {attempt}: {validation_error}"
                + (f" (finish_reason={finish_reason})" if finish_reason else "")
            )
            self._log(
                f"   [WARN] Invalid Stage 1 JSON for '{column}' "
                f"on attempt {attempt}/{max_attempts}: {validation_error}",
                log_callback,
            )

        if result is None:
            raise Stage1LLMResponseError(
                f"Stage 1 could not obtain valid JSON for "
                f"{table_name}.{column} after {max_attempts} attempts: "
                + "; ".join(parse_errors)
            )

        # Log LLM's decision for transparency
        llm_is_pii = result.get("is_pii", False)
        llm_pii_type = result.get("pii_type", "Unknown")
        llm_reasoning = result.get("reasoning", result.get("evidence", "No reasoning provided"))

        # Truncate reasoning for readability
        reasoning_display = str(llm_reasoning)[:200] + "..." if len(str(llm_reasoning)) > 200 else str(llm_reasoning)

        self._log(f"   [LLM] [LLM Decision for '{column}']:", log_callback)
        self._log(f"      Is PII: {llm_is_pii}", log_callback)
        self._log(f"      PII Type: {llm_pii_type}", log_callback)
        self._log(f"      Reasoning: {reasoning_display}", log_callback)

        # Log Chain-of-Thought reasoning if available
        if "chain_of_thought" in result:
            self._log_cot_reasoning(column, result["chain_of_thought"], log_callback)

            # Also log to audit logger if available
            if self.audit_logger:
                self.audit_logger.log_stage1_cot_reasoning(
                    table=table_name,
                    column=column,
                    chain_of_thought=result["chain_of_thought"],
                    is_pii=result.get("is_pii", False),
                    pii_type=result.get("pii_type", "Unknown")
                )

        # Add evidence source from RAG
        if doc_details:
            sources = [doc.get('source', 'Unknown') for doc in doc_details]
            # Remove duplicates while preserving order
            unique_sources = list(dict.fromkeys(sources))
            result["evidence_source"] = ", ".join(unique_sources)
        else:
            result["evidence_source"] = "LLM Knowledge (No RAG documents retrieved)"

        return result

    @staticmethod
    def _validate_stage1_llm_result(
        result: Dict[str, Any],
        require_chain_of_thought: bool = True,
    ) -> Optional[str]:
        """Validate the complete JSON contract required by the frozen prompt."""
        if not isinstance(result, dict):
            return "response is not a JSON object"

        if result.get("error"):
            return str(result["error"])

        is_pii = result.get("is_pii")
        if not isinstance(is_pii, bool):
            return "'is_pii' must be a boolean"

        pii_type = result.get("pii_type")
        if pii_type not in {"PII", "Non-PII"}:
            return "'pii_type' must be either 'PII' or 'Non-PII'"
        if is_pii != (pii_type == "PII"):
            return "'is_pii' and 'pii_type' are inconsistent"

        if result.get("confidence") not in {"High", "Medium", "Low"}:
            return "'confidence' must be High, Medium, or Low"

        for field in ("reasoning", "evidence_source"):
            if not isinstance(result.get(field), str) or not result[field].strip():
                return f"'{field}' must be a non-empty string"

        chain_of_thought = result.get("chain_of_thought")
        if not require_chain_of_thought:
            return None
        if not isinstance(chain_of_thought, dict):
            return "'chain_of_thought' must be an object"

        for field in ("input_analysis", "final_justification"):
            value = chain_of_thought.get(field)
            if not isinstance(value, str) or not value.strip():
                return (
                    f"'chain_of_thought.{field}' must be a non-empty string"
                )

        for field in ("key_features", "decision_steps", "legal_references"):
            value = chain_of_thought.get(field)
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                return f"'chain_of_thought.{field}' must be a string array"

        return None

    def _log_cot_reasoning(self, column: str, cot: Dict[str, Any], log_callback: Optional[Callable] = None):
        """
        Log chain-of-thought reasoning from LLM response.

        Args:
            column: Column name
            cot: Chain-of-thought dictionary from LLM response
            log_callback: Optional callback for logging
        """
        self._log(f"\n   [CoT] [Chain-of-Thought] {column}:", log_callback)

        # Input Analysis
        if "input_analysis" in cot:
            self._log(f"      [Stats] Input: {cot['input_analysis']}", log_callback)

        # Key Features
        if "key_features" in cot and cot["key_features"]:
            self._log(f"      [Key] Key Features:", log_callback)
            for feature in cot["key_features"]:
                self._log(f"         - {feature}", log_callback)

        # Decision Steps
        if "decision_steps" in cot and cot["decision_steps"]:
            self._log(f"      [Info] Decision Steps:", log_callback)
            for i, step in enumerate(cot["decision_steps"], 1):
                self._log(f"         {i}. {step}", log_callback)

        # Legal References
        if "legal_references" in cot and cot["legal_references"]:
            self._log(f"        Legal References:", log_callback)
            for ref in cot["legal_references"]:
                self._log(f"         - {ref}", log_callback)

        # Final Justification
        if "final_justification" in cot:
            self._log(f"      [OK] Justification: {cot['final_justification']}", log_callback)

        self._log("", log_callback)  # Empty line for readability

    def set_audit_logger(self, audit_logger):
        """
        Set audit logger for COT logging.
        
        Args:
            audit_logger: AuditLogger instance
        """
        self.audit_logger = audit_logger

    def _log(self, message: str, callback: Optional[Callable] = None):
        """Helper for logging"""
        if callback:
            callback(message)
        else:
            print(message)

    def cleanup_sessions(self, log_callback: Optional[Callable] = None):
        """Clean up all sessions (call at end of pipeline)."""
        stats = self.session_manager.get_all_stats()
        self._log(f"[Stage 1] Session Stats: {json.dumps(stats, indent=2)}", log_callback)

        # Close all sessions
        for session_id in list(self.session_manager.sessions.keys()):
            self.session_manager.close_session(session_id)
