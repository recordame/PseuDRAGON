"""Expert Feedback Manager for PseuDRAGON.

Tracks and applies expert's method preferences across sessions.
Uses RAG-based semantic search to find similar columns and apply learned preferences.
Also triggers automatic heuristic pattern learning from expert feedback.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Configure logger for this module
logger = logging.getLogger(__name__)


class ExpertPreferenceManager:
    """Manages expert feedback for pseudonymization methods.

    Learns from expert selections in Stage 3 and adjusts method rankings
    for similar columns in future runs.
    """

    # Similarity threshold for RAG-based column matching

    SIMILARITY_THRESHOLD = 0.75

    def __init__(self, preference_file: str = "resources/expert_feedback.json", rag_system=None):
        """Initialize preference manager.

        Args:
            preference_file: Path to expert feedback JSON file
            rag_system: Optional RAGSystem instance for semantic similarity search
        """
        # Allow isolated ablation runs to provide an empty or dedicated store
        # without mutating the production feedback artifact.
        self.preference_file = os.environ.get("EXPERT_FEEDBACK_FILE", preference_file)
        self.preferences = self._load_preferences()
        self.rag_system = rag_system

        # Embeddings cache for column descriptions

        self._column_embeddings: Dict[str, np.ndarray] = {}
        self._embedding_generator = None

        # Initialize heuristic pattern learner

        self._heuristic_learner = None
        self._initialize_heuristic_learner()

        # Ensure PII classification section exists

        if "pii_classification" not in self.preferences:
            self.preferences["pii_classification"] = {}

        # Initialize RAG-based feedback embeddings if we have feedback data

        self._initialize_feedback_embeddings()

    def _initialize_heuristic_learner(self):
        """Initialize heuristic pattern learner."""
        try:
            from pseudragon.rag.heuristic_pattern_learner import HeuristicPatternLearner

            self._heuristic_learner = HeuristicPatternLearner(
                feedback_file=self.preference_file,
                # Keep ablation runs isolated.  When an experiment supplies
                # HEURISTICS_FILE, the pattern learner must use that same
                # store instead of silently falling back to the repository's
                # live HITL artifact.
                heuristics_file=os.environ.get("HEURISTICS_FILE", "resources/heuristics.json"),
            )
            logger.info("Initialized heuristic pattern learner")
        except Exception as e:
            logger.warning(f"Failed to initialize heuristic pattern learner: {e}")
            self._heuristic_learner = None

    def _load_preferences(self) -> Dict[str, Any]:
        """Load expert feedback from file.

        Returns:
            Dictionary of expert feedback
        """
        if not os.path.exists(self.preference_file):
            logger.info(f"No existing preferences file found at {self.preference_file}")
            return {
                "version": "1.0",
                "column_patterns": {},  # Pattern-based preferences
                "column_exact": {},  # Exact column name matches
                "statistics": {"total_selections": 0, "last_updated": None},
            }

        try:
            with open(self.preference_file, "r", encoding="utf-8") as f:
                preferences = json.load(f)
                logger.info(
                    f"Loaded {len(preferences.get('column_exact', {}))} expert feedback entries"
                )
                return preferences
        except Exception as e:
            logger.warning(f"Failed to load expert feedback: {e}")
            return {
                "version": "1.0",
                "column_patterns": {},
                "column_exact": {},
                "statistics": {"total_selections": 0, "last_updated": None},
            }

    def _save_preferences(self):
        """Save expert feedback to file."""
        try:
            # Create directory if it doesn't exist
            # Only create directory if the path contains a directory component
            dir_name = os.path.dirname(self.preference_file)
            if dir_name:  # Only create if there's actually a directory path
                os.makedirs(dir_name, exist_ok=True)

            with open(self.preference_file, "w", encoding="utf-8") as f:
                json.dump(self.preferences, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved expert feedback to {self.preference_file}")
        except Exception as e:
            logger.error(f"Failed to save expert feedback: {e}")

    def _initialize_feedback_embeddings(self):
        """Initialize embeddings for existing feedback entries to enable semantic search."""
        has_column_exact = bool(self.preferences.get("column_exact"))
        has_pii_classification = bool(self.preferences.get("pii_classification"))

        # Need to initialize embedding generator if we have any feedback data

        if not has_column_exact and not has_pii_classification:
            return

        try:
            # Lazy import to avoid circular dependencies
            from pseudragon.rag.retriever import EmbeddingGenerator
            from config.config import DEFAULT_LLM_CLIENT, Settings

            # Check if RAG is enabled

            if not getattr(Settings, "RAG_ENABLED", True):
                logger.info(
                    "RAG disabled - skipping embedding initialization for expert preferences"
                )
                self._column_embeddings = {}
                self._embedding_generator = None
                return

            # Use FEEDBACK_EMBEDDING_MODEL for column similarity search (general-purpose model)
            # instead of LEGAL_EMBEDDING_MODEL (legal-domain document RAG)

            # Use a separate embedding model for feedback and legal documents.
            pii_model = Settings.FEEDBACK_EMBEDDING_MODEL
            self._embedding_generator = EmbeddingGenerator(DEFAULT_LLM_CLIENT, pii_model)
            logger.info(
                f"Initialized embedding generator for PII classification RAG search (model: {pii_model})"
            )

            # Generate embeddings for each stored column in column_exact

            if has_column_exact:
                for key, entry in self.preferences["column_exact"].items():
                    column_name = entry.get("column_name", "")
                    pii_type = entry.get("pii_type", "")

                    # Create description text for embedding

                    description = self._create_column_description(column_name, pii_type, entry)

                    # Generate and cache embedding
                    embedding = self._embedding_generator._generate_local_embeddings([description])
                    self._column_embeddings[key] = embedding[0]

                logger.info(
                    f"Initialized {len(self._column_embeddings)} feedback embeddings for RAG search"
                )

        except Exception as e:
            logger.warning(f"Failed to initialize feedback embeddings: {e}")
            import traceback

            logger.warning(traceback.format_exc())
            # Continue without RAG - will fall back to exact matching
            self._column_embeddings = {}
            self._embedding_generator = None

    def _create_column_description(
        self, column_name: str, pii_type: str, entry: Dict[str, Any]
    ) -> str:
        """Create a descriptive text for a column to generate meaningful embeddings.

        Args:
            column_name: Column name
            pii_type: PII type
            entry: Feedback entry data

        Returns:
            Description text for embedding
        """
        # Get the most used method
        method_counts = entry.get("method_counts", {})
        top_method = max(method_counts.items(), key=lambda x: x[1])[0] if method_counts else "KEEP"

        # Build description
        parts = [
            f"Column: {column_name}",
            f"Type: {pii_type}",
            f"Preferred Method: {top_method}",
        ]

        # Add stored column_comment if available (highest priority for semantic search)

        column_comment = entry.get("column_comment", "")
        if column_comment:
            parts.append(f"Description: {column_comment}")

        # Add semantic hints based on column name patterns

        column_lower = column_name.lower()
        if any(x in column_lower for x in ["balance", "blnc", "amt", "amount", "money", "price"]):
            parts.append("Semantic: Financial amount, balance, monetary value")
        elif any(x in column_lower for x in ["phone", "mobile", "tel", "contact"]):
            parts.append("Semantic: Phone number, contact information")
        elif any(x in column_lower for x in ["name", "nm", "firstname", "lastname"]):
            parts.append("Semantic: Person name, identifier")
        elif any(x in column_lower for x in ["id", "no", "num", "seq", "code"]):
            parts.append("Semantic: Identifier, sequence number, code")
        elif any(x in column_lower for x in ["date", "dt", "time", "dtime"]):
            parts.append("Semantic: Date, time, timestamp")
        elif any(x in column_lower for x in ["addr", "address", "loc", "location"]):
            parts.append("Semantic: Address, location information")
        elif any(x in column_lower for x in ["email", "mail"]):
            parts.append("Semantic: Email address")

        return " | ".join(parts)

    def _update_column_embedding(self, key: str, entry: Dict[str, Any]):
        """Update embedding for a single column after recording new feedback.

        Args:
            key: Column key (column_name:pii_type)
            entry: Feedback entry data
        """
        if self._embedding_generator is None:
            return

        try:
            column_name = entry.get("column_name", "")
            pii_type = entry.get("pii_type", "")
            description = self._create_column_description(column_name, pii_type, entry)

            embedding = self._embedding_generator._generate_local_embeddings([description])[0]
            self._column_embeddings[key] = embedding

            logger.info(f"[RAG] Updated embedding for column {column_name}")
        except Exception as e:
            logger.warning(f"Failed to update column embedding: {e}")

    def find_similar_column(
        self, column_name: str, pii_type: str, column_comment: str = "", data_type: str = ""
    ) -> Optional[Tuple[str, str, float]]:
        """Find a similar column in the feedback database using RAG semantic search.

        Args:
            column_name: New column name to search for
            pii_type: PII type of the column
            column_comment: Optional column comment/description
            data_type: Optional data type

        Returns:
            Tuple of (similar_column_key, similar_column_name, similarity_score) or None
        """
        if not self._column_embeddings or self._embedding_generator is None:
            return None

        try:
            from sklearn.metrics.pairwise import cosine_similarity

            # Create description for the new column

            query_parts = [f"Column: {column_name}", f"Type: {pii_type}"]
            if column_comment:
                query_parts.append(f"Description: {column_comment}")
            if data_type:
                query_parts.append(f"Data Type: {data_type}")

            # Add semantic hints
            column_lower = column_name.lower()
            if any(
                x in column_lower for x in ["balance", "blnc", "amt", "amount", "money", "price"]
            ):
                query_parts.append("Semantic: Financial amount, balance, monetary value")
            elif any(x in column_lower for x in ["phone", "mobile", "tel", "contact"]):
                query_parts.append("Semantic: Phone number, contact information")
            elif any(x in column_lower for x in ["name", "nm", "firstname", "lastname"]):
                query_parts.append("Semantic: Person name, identifier")
            elif any(x in column_lower for x in ["id", "no", "num", "seq", "code"]):
                query_parts.append("Semantic: Identifier, sequence number, code")
            elif any(x in column_lower for x in ["date", "dt", "time", "dtime"]):
                query_parts.append("Semantic: Date, time, timestamp")
            elif any(x in column_lower for x in ["addr", "address", "loc", "location"]):
                query_parts.append("Semantic: Address, location information")
            elif any(x in column_lower for x in ["email", "mail"]):
                query_parts.append("Semantic: Email address")

            query_description = " | ".join(query_parts)

            # Generate embedding for query
            query_embedding = self._embedding_generator._generate_local_embeddings(
                [query_description]
            )[0]

            # Compare with all stored embeddings

            best_match = None
            best_similarity = 0.0

            for key, stored_embedding in self._column_embeddings.items():
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1), stored_embedding.reshape(1, -1)
                )[0][0]

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = key

            # Return if above threshold
            if best_match and best_similarity >= self.SIMILARITY_THRESHOLD:
                entry = self.preferences["column_exact"].get(best_match, {})
                similar_column_name = entry.get("column_name", "")
                similar_column_comment = entry.get("column_comment", "")
                logger.info(
                    f"[RAG] Found similar column: {column_name} ≈ {similar_column_name} (similarity: {best_similarity:.3f})"
                )
                if similar_column_comment:
                    logger.info(f"[RAG] Similar column description: {similar_column_comment}")
                if column_comment:
                    logger.info(f"[RAG] Query column description: {column_comment}")
                return (best_match, similar_column_name, best_similarity)

            return None

        except Exception as e:
            logger.warning(f"RAG similarity search failed: {e}")
            return None

    def get_preference_from_similar(
        self, column_name: str, pii_type: str, column_comment: str = "", data_type: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Get preference (method and code snippet) from a similar column.

        Args:
            column_name: New column name
            pii_type: PII type
            column_comment: Optional column comment
            data_type: Optional data type

        Returns:
            Dict with 'method', 'code_snippet', 'similar_column', 'similarity' or None
        """
        # First try exact match

        exact_key = f"{column_name.lower()}:{pii_type}"
        if exact_key in self.preferences.get("column_exact", {}):
            entry = self.preferences["column_exact"][exact_key]
            method_counts = entry.get("method_counts", {})
            if method_counts:
                preferred_method = max(method_counts.items(), key=lambda x: x[1])[0]
                custom_snippet = entry.get("custom_code_snippets", {}).get(preferred_method)
                custom_description = entry.get("custom_descriptions", {}).get(preferred_method)
                stored_comment = entry.get("column_comment", "")
                return {
                    "method": preferred_method,
                    "code_snippet": custom_snippet,
                    "description": custom_description,
                    "similar_column": column_name,
                    "similar_column_comment": stored_comment,
                    "similarity": 1.0,
                    "match_type": "exact",
                }

        # Then try RAG-based similarity search

        similar = self.find_similar_column(column_name, pii_type, column_comment, data_type)
        if similar:
            similar_key, similar_column_name, similarity = similar
            entry = self.preferences["column_exact"].get(similar_key, {})
            method_counts = entry.get("method_counts", {})

            if method_counts:
                preferred_method = max(method_counts.items(), key=lambda x: x[1])[0]
                custom_snippet = entry.get("custom_code_snippets", {}).get(preferred_method)
                custom_description = entry.get("custom_descriptions", {}).get(preferred_method)
                stored_comment = entry.get("column_comment", "")

                logger.info(
                    f"[RAG] Applying preference from similar column {similar_column_name} -> {column_name}"
                )
                if stored_comment:
                    logger.info(f"[RAG] Similar column description: {stored_comment}")
                return {
                    "method": preferred_method,
                    "code_snippet": custom_snippet,
                    "description": custom_description,
                    "similar_column": similar_column_name,
                    "similar_column_comment": stored_comment,
                    "similarity": similarity,
                    "match_type": "rag_similar",
                }

        return None

    def record_selection(
        self,
        table_name: str,
        column_name: str,
        pii_type: str,
        selected_method: str,
        available_methods: List[str],
        selected_index: int,
        code_snippet: Optional[str] = None,
        code_modified: bool = False,
        description: Optional[str] = None,
        description_modified: bool = False,
        column_comment: Optional[str] = None,
    ):
        """Record expert's method selection.

        Args:
            table_name: Table name
            column_name: Column name
            pii_type: PII type (PII or Non-PII)
            selected_method: Method chosen by expert
            available_methods: All available methods in order
            selected_index: Index of selected method (0 = default, >0 = expert changed)
            code_snippet: The code snippet (original or edited by user)
            code_modified: Whether the user edited the code snippet
            description: The method description (original or edited by user)
            description_modified: Whether the user edited the description
            column_comment: Column comment/description from database schema
        """
        # Record every selection, including the default option.

        # Create preference key based on column name and PII type
        exact_key = f"{column_name.lower()}:{pii_type}"

        # Record exact match
        if exact_key not in self.preferences["column_exact"]:
            self.preferences["column_exact"][exact_key] = {
                "column_name": column_name,
                "pii_type": pii_type,
                "method_counts": {},
                "total_selections": 0,
            }

        # Increment selection count
        entry = self.preferences["column_exact"][exact_key]
        entry["method_counts"][selected_method] = entry["method_counts"].get(selected_method, 0) + 1
        entry["total_selections"] += 1

        # Save column_comment if provided (for RAG-based similarity search)

        if column_comment:
            entry["column_comment"] = column_comment

        # Save user-edited code snippet if modified

        if code_modified and code_snippet:
            if "custom_code_snippets" not in entry:
                entry["custom_code_snippets"] = {}
            entry["custom_code_snippets"][selected_method] = code_snippet
            logger.info(f"Saved custom code snippet for {column_name} ({selected_method})")

        # Save user-edited description if modified

        if description_modified and description:
            if "custom_descriptions" not in entry:
                entry["custom_descriptions"] = {}
            entry["custom_descriptions"][selected_method] = description
            logger.info(f"Saved custom description for {column_name} ({selected_method})")

        # Update statistics
        self.preferences["statistics"]["total_selections"] += 1
        from datetime import datetime

        self.preferences["statistics"]["last_updated"] = datetime.now().isoformat()

        # Save preferences
        self._save_preferences()

        # Update RAG embedding for this column (for similarity search)

        self._update_column_embedding(exact_key, entry)

        # Learn patterns when Stage 3 completes.

        logger.info(f"Recorded expert feedback: {column_name} ({pii_type}) -> {selected_method}")
        logger.info(f"Total selections for {column_name}: {entry['total_selections']}")

    def get_preferred_method(self, column_name: str, pii_type: str) -> Optional[str]:
        """Get expert's preferred method for a column.

        Args:
            column_name: Column name
            pii_type: PII type

        Returns:
            Preferred method name or None if no feedback exists
        """
        exact_key = f"{column_name.lower()}:{pii_type}"

        # Check exact match
        if exact_key in self.preferences["column_exact"]:
            entry = self.preferences["column_exact"][exact_key]
            method_counts = entry["method_counts"]

            if method_counts:
                # Return method with highest count
                preferred_method = max(method_counts.items(), key=lambda x: x[1])[0]
                count = method_counts[preferred_method]
                total = entry["total_selections"]

                logger.info(
                    f"Found expert feedback for {column_name}: {preferred_method} ({count}/{total} times)"
                )
                return preferred_method

        # Check pattern-based match (future enhancement)

        return None

    def get_custom_code_snippet(
        self, column_name: str, pii_type: str, method: str
    ) -> Optional[str]:
        """Get user's custom code snippet for a column and method.

        Args:
            column_name: Column name
            pii_type: PII type
            method: The method name (e.g., ROUND, HASH, etc.)

        Returns:
            Custom code snippet or None if no custom snippet exists
        """
        exact_key = f"{column_name.lower()}:{pii_type}"

        if exact_key in self.preferences["column_exact"]:
            entry = self.preferences["column_exact"][exact_key]
            custom_snippets = entry.get("custom_code_snippets", {})

            if method in custom_snippets:
                snippet = custom_snippets[method]
                logger.info(f"Found custom code snippet for {column_name} ({method})")
                return snippet

        return None

    def reorder_methods(
        self, column_name: str, pii_type: str, methods: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Reorder methods based on expert feedback.

        Args:
            column_name: Column name
            pii_type: PII type
            methods: List of method dictionaries

        Returns:
            Reordered list of methods (preferred method first)
        """
        preferred_method = self.get_preferred_method(column_name, pii_type)

        if not preferred_method:
            # No preference found, return original order
            return methods

        # Find the preferred method in the methods list
        preferred_index = None
        for i, method in enumerate(methods):
            if method.get("method", "").upper() == preferred_method.upper():
                preferred_index = i
                break

        if preferred_index is None:
            logger.warning(f"Preferred method {preferred_method} not found in methods")
            return methods

        if preferred_index == 0:
            # Already at the top
            logger.info(f"Preferred method {preferred_method} already at top")
            return methods

        # Move preferred method to the top
        reordered = methods.copy()
        preferred_method_obj = reordered.pop(preferred_index)
        reordered.insert(0, preferred_method_obj)

        logger.info(
            f"Reordered methods for {column_name}: {preferred_method} moved to top (was at index {preferred_index})"
        )

        return reordered

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics.

        Returns:
            Dictionary with statistics
        """
        stats = self.preferences["statistics"].copy()
        stats["unique_columns"] = len(self.preferences["column_exact"])
        stats["pii_classifications"] = len(self.preferences.get("pii_classification", {}))
        return stats

    def record_pii_classification_change(
        self,
        column_name: str,
        old_classification: str,
        new_classification: str,
        rationale: str = "",
        column_comment: str = "",
    ):
        """Record user's PII classification change.

        This allows the system to learn from user corrections:
        - If user changes Non-PII -> PII, remember this for future runs
        - If user changes PII -> Non-PII, remember this as well

        Args:
            column_name: Column name
            old_classification: Original classification (PII or Non-PII)
            new_classification: New classification (PII or Non-PII)
            rationale: User's rationale for the change
            column_comment: Column comment/description from database schema
        """
        # Only record if there was an actual change

        if old_classification == new_classification:
            return

        # Create key for this column

        col_key = column_name.lower()

        # Initialize entry if not exists

        if col_key not in self.preferences["pii_classification"]:
            self.preferences["pii_classification"][col_key] = {
                "column_name": column_name,
                "classification": new_classification,
                "change_count": 0,
                "rationale": rationale,
                "history": [],
            }

        entry = self.preferences["pii_classification"][col_key]

        # Update classification

        entry["classification"] = new_classification
        entry["change_count"] += 1
        if rationale:
            entry["rationale"] = rationale

        # Save column_comment if provided (for RAG-based similarity search)

        if column_comment:
            entry["column_comment"] = column_comment

        # Record change in history

        from datetime import datetime

        entry["history"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "old": old_classification,
                "new": new_classification,
                "rationale": rationale,
            }
        )

        # Update global statistics

        self.preferences["statistics"]["total_selections"] += 1
        self.preferences["statistics"]["last_updated"] = datetime.now().isoformat()

        # Save preferences

        self._save_preferences()

        # Learn patterns when Stage 3 completes.

        logger.info(
            f"Recorded PII classification change: {column_name} ({old_classification} -> {new_classification})"
        )
        if rationale:
            logger.info(f"Rationale: {rationale}")

    def get_learned_pii_classification(
        self, column_name: str, column_comment: str = "", data_type: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Get learned PII classification for a column with RAG-based similarity search.

        Args:
            column_name: Column name
            column_comment: Column comment/description for similarity search
            data_type: Column data type

        Returns:
            Dict with classification info if learned, None otherwise
            Returns: {"classification": "PII"/"Non-PII", "match_type": "exact"/"rag_similar",
                      "similar_column": str, "similarity": float, "rationale": str}
        """
        col_key = column_name.lower()

        # First try exact match

        if col_key in self.preferences["pii_classification"]:
            entry = self.preferences["pii_classification"][col_key]
            classification = entry["classification"]
            change_count = entry["change_count"]
            rationale = entry.get("rationale", "")
            stored_comment = entry.get("column_comment", "")

            logger.info(
                f"Found learned PII classification for {column_name}: {classification} ({change_count} change(s))"
            )
            if rationale:
                logger.info(f"Rationale: {rationale}")

            return {
                "classification": classification,
                "match_type": "exact",
                "similar_column": column_name,
                "similar_column_comment": stored_comment,
                "similarity": 1.0,
                "rationale": rationale,
                "change_count": change_count,
            }

        # RAG-based similarity search using general-purpose embedding model

        similar_result = self._find_similar_pii_classification(
            column_name, column_comment, data_type
        )
        if similar_result:
            return similar_result

        return None

    def _find_similar_pii_classification(
        self, column_name: str, column_comment: str = "", data_type: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Find similar column in pii_classification using pure embedding-based semantic search.

        Strategy:
        1. If column_comment exists, compute direct embedding similarity between comments
           (This is the most reliable signal - same description = same semantics)
        2. Also compute full context similarity (column_name + comment + data_type)
        3. Use the higher of the two similarities with adaptive thresholds

        Args:
            column_name: New column name to search for
            column_comment: Column comment/description
            data_type: Column data type

        Returns:
            Dict with classification info if similar column found, None otherwise
        """
        if not self.preferences.get("pii_classification"):
            logger.info("[RAG-PII] No pii_classification data in preferences, skipping RAG search")
            return None

        if self._embedding_generator is None:
            logger.warning(
                f"[RAG-PII] Embedding generator not initialized, skipping RAG search for {column_name}"
            )
            return None

        logger.info(
            f"[RAG-PII] Starting similarity search for column: {column_name}, comment: '{column_comment}'"
        )

        try:
            from sklearn.metrics.pairwise import cosine_similarity

            best_match_key = None
            best_similarity = 0.0
            best_match_type = "rag_similar"

            # Generate query embeddings

            query_comment_embedding = None
            if column_comment:
                query_comment_embedding = self._embedding_generator._generate_local_embeddings(
                    [column_comment]
                )[0]

            # Full context embedding (column_name + comment + data_type)
            query_full_parts = [column_name]
            if column_comment:
                query_full_parts.append(column_comment)
            if data_type:
                query_full_parts.append(data_type)
            query_full_description = " | ".join(query_full_parts)
            query_full_embedding = self._embedding_generator._generate_local_embeddings(
                [query_full_description]
            )[0]

            for key, entry in self.preferences["pii_classification"].items():
                stored_column_name = entry.get("column_name", "")
                stored_comment = entry.get("column_comment", "")
                stored_rationale = entry.get("rationale", "")

                # Strategy 1: Direct comment-to-comment similarity (highest priority)

                comment_similarity = 0.0
                if query_comment_embedding is not None and stored_comment:
                    stored_comment_embedding = self._embedding_generator._generate_local_embeddings(
                        [stored_comment]
                    )[0]
                    comment_similarity = cosine_similarity(
                        query_comment_embedding.reshape(1, -1),
                        stored_comment_embedding.reshape(1, -1),
                    )[0][0]

                # Strategy 2: Full context similarity

                stored_full_parts = [stored_column_name]
                if stored_comment:
                    stored_full_parts.append(stored_comment)
                if stored_rationale:
                    stored_full_parts.append(stored_rationale)
                stored_full_description = " | ".join(stored_full_parts)
                stored_full_embedding = self._embedding_generator._generate_local_embeddings(
                    [stored_full_description]
                )[0]

                full_similarity = cosine_similarity(
                    query_full_embedding.reshape(1, -1), stored_full_embedding.reshape(1, -1)
                )[0][0]

                # Use the higher similarity

                if comment_similarity > full_similarity:
                    similarity = comment_similarity
                    match_type = "comment_similar"
                else:
                    similarity = full_similarity
                    match_type = "context_similar"

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_key = key
                    best_match_type = match_type

            # Adaptive threshold based on match type

            # - Comment similarity: 0.92 threshold - only very similar descriptions should match
            # - Full context similarity: 0.88 threshold - must be highly related
            # Note: Raised thresholds significantly to prevent false positives
            #       (e.g., STD_DT should NOT match CMS_NO)
            if best_match_type == "comment_similar":
                threshold = 0.92  # Very high threshold - only truly similar descriptions
            else:
                threshold = 0.88  # High threshold for full context to prevent unrelated matches

            # Log similarity scores for debugging

            if best_match_key:
                entry = self.preferences["pii_classification"][best_match_key]
                similar_column_name = entry.get("column_name", "")
                logger.info(
                    f"[RAG-PII] Best match: {column_name} ≈ {similar_column_name} "
                    f"(similarity: {best_similarity:.3f}, type: {best_match_type}, threshold: {threshold})"
                )

            if best_match_key and best_similarity >= threshold:
                entry = self.preferences["pii_classification"][best_match_key]
                similar_column_name = entry.get("column_name", "")
                stored_comment = entry.get("column_comment", "")
                classification = entry["classification"]
                rationale = entry.get("rationale", "")
                change_count = entry.get("change_count", 0)

                logger.info(
                    f"[RAG-PII] [OK] Match accepted! Applying classification: {classification}"
                )
                if stored_comment:
                    logger.info(f"[RAG-PII] Similar column description: {stored_comment}")
                if column_comment:
                    logger.info(f"[RAG-PII] Query column description: {column_comment}")
                if rationale:
                    logger.info(f"[RAG-PII] Rationale: {rationale}")

                return {
                    "classification": classification,
                    "match_type": best_match_type,
                    "similar_column": similar_column_name,
                    "similar_column_comment": stored_comment,
                    "similarity": best_similarity,
                    "rationale": rationale,
                    "change_count": change_count,
                }
            elif best_match_key:
                logger.info(
                    f"[RAG-PII] [X] Match rejected: similarity {best_similarity:.3f} < threshold {threshold}"
                )

            return None

        except Exception as e:
            logger.warning(f"RAG PII classification search failed: {e}")
            return None

    def trigger_pattern_learning(self):
        """Trigger heuristic pattern learning from current feedback data.

        This method should be called at Stage 3 completion to learn patterns
        from all accumulated expert feedback. It's a fast operation (typically < 100ms).

        Returns:
            Dict with learning statistics or None if learner not initialized
        """
        if self._heuristic_learner is None:
            logger.warning("[Heuristic] Pattern learner not initialized, skipping")
            return None

        try:
            # First, learn patterns from new feedback (if any)

            result = self._heuristic_learner.learn_patterns_from_feedback()

            # Always run refinement to merge exact patterns and extract more specific patterns

            # This ensures existing sample_columns are analyzed even without new columns

            self._heuristic_learner.refine_heuristics()

            stats = self._heuristic_learner.get_statistics()
            logger.info(
                f"[Heuristic] Pattern learning completed: {stats['total_patterns']} patterns learned"
            )
            return stats
        except Exception as e:
            logger.warning(f"Failed to trigger pattern learning: {e}")
            return None

    def classify_by_heuristics(
        self, column_name: str, column_comment: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Classify a column using learned heuristic patterns (fast, no LLM).

        This method provides fast classification (< 1ms) using regex patterns
        learned from expert feedback. Should be called before RAG similarity search.

        Processing hierarchy:

        Args:
            column_name: Column name to classify
            column_comment: Column comment/description

        Returns:
            Dict with classification result or None if no pattern matches
            {"classification": "PII"/"Non-PII", "match_type": str, "pattern": str, "confidence": float}
        """
        if self._heuristic_learner is None:
            return None

        return self._heuristic_learner.classify_column(column_name, column_comment)

    def get_heuristic_statistics(self) -> Optional[Dict[str, Any]]:
        """Get statistics about learned heuristic patterns.

        Returns:
            Dictionary with pattern statistics or None if learner not initialized
        """
        if self._heuristic_learner is None:
            return None

        return self._heuristic_learner.get_statistics()
