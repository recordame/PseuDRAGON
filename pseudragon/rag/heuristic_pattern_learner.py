"""Heuristic Pattern Learner for PseuDRAGON.

Automatically extracts regex patterns from expert feedback to enable
fast PII classification without LLM calls. Integrates with the unified
heuristics.json file used by HeuristicManager.

Processing hierarchy:
"""

import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class HeuristicPatternLearner:
    """Learns regex patterns from expert feedback and saves to unified heuristics.json.

    Analyzes column_exact and pii_classification data to extract:
    - Suffix patterns (e.g., _ACNO$ -> PII, _DT$ -> Non-PII)
    - Prefix patterns (e.g., ^USR_ -> PII)
    - Comment keywords (e.g., "account number" -> PII)

    Auto-learned patterns are marked with source="auto_learned" and can be
    edited or deleted by experts through the web interface.
    """

    # Minimum occurrences for a pattern to be considered reliable

    MIN_PATTERN_OCCURRENCES = 2

    # Minimum confidence threshold for a pattern to be saved

    MIN_CONFIDENCE = 0.8

    # Patterns with more than 10% contradictory labels are not promoted into
    # the deterministic rule library.
    MAX_CONFLICT_RATE = float(os.getenv("MAX_PATTERN_CONFLICT_RATE", "0.10"))

    def __init__(
        self,
        feedback_file: str = "resources/expert_feedback.json",
        heuristics_file: str = "resources/heuristics.json",
    ):
        """Initialize the pattern learner.

        Args:
            feedback_file: Path to expert feedback JSON file
            heuristics_file: Path to unified heuristics JSON file (same as HeuristicManager)
        """
        self.feedback_file = feedback_file
        self.heuristics_file = heuristics_file

    def _load_heuristics(self) -> Dict[str, Any]:
        """Load existing heuristics from unified file.

        Returns:
            Dictionary of heuristics
        """
        if not os.path.exists(self.heuristics_file):
            logger.info(f"No existing heuristics file found at {self.heuristics_file}")
            return self._create_empty_heuristics()

        try:
            with open(self.heuristics_file, "r", encoding="utf-8") as f:
                heuristics = json.load(f)
                # Migrate from v1.0 to v2.0 format if needed
                if heuristics.get("version") == "1.0":
                    heuristics = self._migrate_v1_to_v2(heuristics)
                logger.info(f"Loaded heuristics from {self.heuristics_file}")
                return heuristics
        except Exception as e:
            logger.warning(f"Failed to load heuristics: {e}")
            return self._create_empty_heuristics()

    def _migrate_v1_to_v2(self, old_heuristics: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate v1.0 heuristics format to v2.0.

        Args:
            old_heuristics: Old format heuristics

        Returns:
            New format heuristics
        """
        new_heuristics = self._create_empty_heuristics()

        # Migrate existing manual heuristics
        for h in old_heuristics.get("heuristics", []):
            if "source" not in h:
                h["source"] = "manual"
            new_heuristics["heuristics"].append(h)

        return new_heuristics

    def _create_empty_heuristics(self) -> Dict[str, Any]:
        """Create empty heuristics structure (v2.0 format).

        Returns:
            Empty heuristics dictionary
        """
        return {
            "version": "2.0",
            "description": "Unified heuristic patterns for PII detection (manual + auto-learned)",
            "heuristics": [],
            "statistics": {
                "total_patterns": 0,
                "manual_patterns": 0,
                "auto_learned_patterns": 0,
                "last_updated": None,
                "last_auto_learn": None,
            },
        }

    def _save_heuristics(self, heuristics: Dict[str, Any]):
        """Save heuristics to unified file."""
        try:
            # Create directory if it doesn't exist
            dir_name = os.path.dirname(self.heuristics_file)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            # Update statistics
            now = datetime.now().isoformat()
            heuristics["statistics"]["last_updated"] = now
            heuristics["statistics"]["last_auto_learn"] = now

            # Count patterns by source
            manual_count = sum(1 for h in heuristics["heuristics"] if h.get("source") == "manual")
            auto_count = sum(
                1 for h in heuristics["heuristics"] if h.get("source") == "auto_learned"
            )
            heuristics["statistics"]["total_patterns"] = len(heuristics["heuristics"])
            heuristics["statistics"]["manual_patterns"] = manual_count
            heuristics["statistics"]["auto_learned_patterns"] = auto_count

            with open(self.heuristics_file, "w", encoding="utf-8") as f:
                json.dump(heuristics, f, indent=2, ensure_ascii=False)

            logger.info(
                f"Saved {len(heuristics['heuristics'])} heuristic patterns to {self.heuristics_file}"
            )
            logger.info(f"  - Manual: {manual_count}, Auto-learned: {auto_count}")
        except Exception as e:
            logger.error(f"Failed to save heuristics: {e}")

    def _generate_id(self, existing_ids: List[str], prefix: str = "auto") -> str:
        """Generate unique heuristic ID.

        Args:
            existing_ids: List of existing IDs
            prefix: ID prefix ("auto" for auto-learned, "h" for manual)

        Returns:
            New unique ID
        """
        # Extract numeric part from existing IDs
        numbers = []
        for id_str in existing_ids:
            if id_str.startswith(prefix):
                try:
                    numbers.append(int(id_str[len(prefix) :]))
                except ValueError:
                    continue

        # Find next available number
        next_num = max(numbers) + 1 if numbers else 1
        return f"{prefix}{next_num:03d}"

    def learn_patterns_from_feedback(self) -> Dict[str, Any]:
        """Main method to analyze expert feedback and extract patterns.

        Returns:
            Dictionary of learned patterns (unified heuristics)
        """
        logger.info("Starting pattern learning from expert feedback...")

        # Load expert feedback
        if not os.path.exists(self.feedback_file):
            logger.warning(f"Expert feedback file not found: {self.feedback_file}")
            return self._load_heuristics()

        try:
            with open(self.feedback_file, "r", encoding="utf-8") as f:
                feedback = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load expert feedback: {e}")
            return self._load_heuristics()

        # Load existing heuristics (preserve manual patterns)
        heuristics = self._load_heuristics()

        # Remove old auto-learned patterns (will be replaced with new ones)
        heuristics["heuristics"] = [
            h for h in heuristics["heuristics"] if h.get("source") != "auto_learned"
        ]

        # Separate PII and Non-PII entries
        pii_columns = []
        non_pii_columns = []

        # Process column_exact entries
        for key, entry in feedback.get("column_exact", {}).items():
            column_name = entry.get("column_name", "")
            pii_type = entry.get("pii_type", "")
            column_comment = entry.get("column_comment", "")
            total_selections = entry.get("total_selections", 0)

            if pii_type == "PII":
                pii_columns.append(
                    {
                        "column_name": column_name,
                        "column_comment": column_comment,
                        "total_selections": total_selections,
                    }
                )
            elif pii_type == "Non-PII":
                non_pii_columns.append(
                    {
                        "column_name": column_name,
                        "column_comment": column_comment,
                        "total_selections": total_selections,
                    }
                )

        # Process pii_classification entries (user corrections)
        for key, entry in feedback.get("pii_classification", {}).items():
            column_name = entry.get("column_name", "")
            classification = entry.get("classification", "")
            column_comment = entry.get("column_comment", "")
            change_count = entry.get("change_count", 0)

            # User corrections have high weight
            weight = change_count * 5  # Give 5x weight to user corrections

            if classification == "PII":
                pii_columns.append(
                    {
                        "column_name": column_name,
                        "column_comment": column_comment,
                        "total_selections": weight,
                    }
                )
            elif classification == "Non-PII":
                non_pii_columns.append(
                    {
                        "column_name": column_name,
                        "column_comment": column_comment,
                        "total_selections": weight,
                    }
                )

        logger.info(
            f"Found {len(pii_columns)} PII columns and {len(non_pii_columns)} Non-PII columns"
        )

        # Extract patterns
        pii_patterns = self._extract_patterns(pii_columns, "PII")
        non_pii_patterns = self._extract_patterns(non_pii_columns, "Non-PII")

        # Check candidates against all matching feedback.
        pii_patterns, non_pii_patterns = self._resolve_conflicts(
            pii_patterns, non_pii_patterns, pii_columns, non_pii_columns
        )

        # Convert patterns to heuristic entries
        existing_ids = [h.get("id", "") for h in heuristics["heuristics"]]
        now = datetime.now().isoformat()

        # Build a map of existing regex patterns to their heuristic entries for sample_columns update

        existing_pattern_map = {}
        for h in heuristics["heuristics"]:
            regex = h.get("regex", "")
            if regex:
                existing_pattern_map[regex] = h

        # Helper to add columns to existing pattern's sample_columns if matched
        # Returns list of unmatched columns that need new patterns
        def add_columns_to_existing_pattern(sample_columns: List[str], pii_type: str) -> List[str]:
            """Check if columns match existing patterns and add to sample_columns if not already present.
            Returns list of columns that didn't match any existing pattern.
            """
            unmatched_columns = []
            for col in sample_columns:
                col_matched = False
                col_upper = col.upper()
                for regex, h_entry in existing_pattern_map.items():
                    # Only match same PII type
                    if h_entry.get("pii_type") != pii_type:
                        continue
                    try:
                        if re.search(regex, col_upper, re.IGNORECASE):
                            col_matched = True
                            # Add column to sample_columns if not already present

                            existing_samples = h_entry.get("sample_columns", [])
                            if col not in existing_samples:
                                existing_samples.append(col)
                                h_entry["sample_columns"] = existing_samples
                                h_entry["updated_at"] = now
                                logger.debug(f"Added column {col} to existing pattern {regex}")
                            break
                    except re.error:
                        continue
                if not col_matched:
                    unmatched_columns.append(col)
            return unmatched_columns

        # Helper to create exact column name pattern
        def create_exact_column_pattern(column_name: str, pii_type: str):
            """Create a new heuristic pattern for exact column name match.
            Will be refined later through refinement process.
            """
            col_upper = column_name.upper()
            # Create pattern that matches the exact column name as suffix
            # e.g., "FINTECH_USE_NUM" -> "_FINTECH_USE_NUM$" or "^FINTECH_USE_NUM$"
            if "_" in col_upper:
                # Use suffix pattern format for columns with underscore
                pattern = (
                    f"_{col_upper.split('_', 1)[1]}$"
                    if col_upper.count("_") > 0
                    else f"^{col_upper}$"
                )
                # Actually, use the full column name as suffix for better refinement
                pattern = f"^{col_upper}$"
            else:
                pattern = f"^{col_upper}$"

            if (
                self._pattern_conflict_rate(pattern, pii_type, pii_columns, non_pii_columns)
                > self.MAX_CONFLICT_RATE
            ):
                return

            # Check if pattern already exists
            if pattern in existing_pattern_map or pattern in newly_added_patterns:
                return

            new_id = self._generate_id(existing_ids, "auto")
            existing_ids.append(new_id)
            newly_added_patterns.append(pattern)

            new_entry = {
                "id": new_id,
                "name": f"Auto: exact {pattern}",
                "regex": pattern,
                "pii_type": pii_type,
                "rationale": f"Auto-learned from exact column: {column_name}",
                "priority": 50,  # Lower priority, will be refined later
                "enabled": True,
                "source": "auto_learned",
                "pattern_type": "exact",
                "confidence": 0.8,
                "sample_columns": [column_name],
                "created_at": now,
                "updated_at": now,
            }
            heuristics["heuristics"].append(new_entry)
            existing_pattern_map[pattern] = new_entry
            logger.info(f"Created exact pattern for unmatched column: {column_name} -> {pattern}")

        # Track newly added patterns for deduplication within this run
        newly_added_patterns = []

        # Add PII patterns
        for pattern_type, patterns in [
            ("suffix", pii_patterns.get("suffix", {})),
            ("prefix", pii_patterns.get("prefix", {})),
            ("keyword", pii_patterns.get("keyword", {})),
        ]:
            for pattern, data in patterns.items():
                # Skip if pattern already exists

                if pattern in existing_pattern_map:
                    # Add new columns to existing pattern's sample_columns
                    h_entry = existing_pattern_map[pattern]
                    existing_samples = h_entry.get("sample_columns", [])
                    for col in data["samples"]:
                        if col not in existing_samples:
                            existing_samples.append(col)
                    h_entry["sample_columns"] = existing_samples
                    h_entry["updated_at"] = now
                    logger.debug(f"Updated existing pattern {pattern} with new samples")
                    continue

                if pattern in newly_added_patterns:
                    logger.debug(f"Skipping duplicate pattern: {pattern}")
                    continue

                # Check which sample columns already match existing patterns

                unmatched = add_columns_to_existing_pattern(data["samples"], "PII")

                # If all columns matched existing patterns, skip this pattern
                if not unmatched:
                    logger.debug(
                        f"Skipping pattern {pattern} - samples already covered by existing patterns"
                    )
                    continue

                new_id = self._generate_id(existing_ids, "auto")
                existing_ids.append(new_id)
                newly_added_patterns.append(pattern)

                new_entry = {
                    "id": new_id,
                    "name": f"Auto: {pattern_type} {pattern}",
                    "regex": pattern,
                    "pii_type": "PII",
                    "rationale": f"Auto-learned from {data['count']} columns: {', '.join(data['samples'][:3])}",
                    "priority": int(data["confidence"] * 100),
                    "enabled": True,
                    "source": "auto_learned",
                    "pattern_type": pattern_type,
                    "confidence": data["confidence"],
                    "sample_columns": data["samples"],
                    "created_at": now,
                    "updated_at": now,
                }
                heuristics["heuristics"].append(new_entry)
                existing_pattern_map[pattern] = new_entry

        # Process all PII columns that didn't match any pattern - create exact patterns

        for col_entry in pii_columns:
            col_name = col_entry["column_name"]
            col_upper = col_name.upper()
            # Check if this column matches any existing pattern
            matched = False
            for regex in existing_pattern_map.keys():
                try:
                    if re.search(regex, col_upper, re.IGNORECASE):
                        matched = True
                        break
                except re.error:
                    continue
            if not matched:
                create_exact_column_pattern(col_name, "PII")

        # Add Non-PII patterns
        for pattern_type, patterns in [
            ("suffix", non_pii_patterns.get("suffix", {})),
            ("prefix", non_pii_patterns.get("prefix", {})),
            ("keyword", non_pii_patterns.get("keyword", {})),
        ]:
            for pattern, data in patterns.items():
                # Skip if pattern already exists (but update sample_columns)
                if pattern in existing_pattern_map:
                    h_entry = existing_pattern_map[pattern]
                    existing_samples = h_entry.get("sample_columns", [])
                    for col in data["samples"]:
                        if col not in existing_samples:
                            existing_samples.append(col)
                    h_entry["sample_columns"] = existing_samples
                    h_entry["updated_at"] = now
                    logger.debug(f"Updated existing pattern {pattern} with new samples")
                    continue

                if pattern in newly_added_patterns:
                    logger.debug(f"Skipping duplicate pattern: {pattern}")
                    continue

                # Check which sample columns already match existing patterns
                unmatched = add_columns_to_existing_pattern(data["samples"], "Non-PII")

                # If all columns matched existing patterns, skip this pattern
                if not unmatched:
                    logger.debug(
                        f"Skipping pattern {pattern} - samples already covered by existing patterns"
                    )
                    continue

                new_id = self._generate_id(existing_ids, "auto")
                existing_ids.append(new_id)
                newly_added_patterns.append(pattern)

                new_entry = {
                    "id": new_id,
                    "name": f"Auto: {pattern_type} {pattern}",
                    "regex": pattern,
                    "pii_type": "Non-PII",
                    "rationale": f"Auto-learned from {data['count']} columns: {', '.join(data['samples'][:3])}",
                    "priority": int(data["confidence"] * 100),
                    "enabled": True,
                    "source": "auto_learned",
                    "pattern_type": pattern_type,
                    "confidence": data["confidence"],
                    "sample_columns": data["samples"],
                    "created_at": now,
                    "updated_at": now,
                }
                heuristics["heuristics"].append(new_entry)
                existing_pattern_map[pattern] = new_entry

        # Process all Non-PII columns that didn't match any pattern - create exact patterns

        for col_entry in non_pii_columns:
            col_name = col_entry["column_name"]
            col_upper = col_name.upper()
            # Check if this column matches any existing pattern
            matched = False
            for regex in existing_pattern_map.keys():
                try:
                    if re.search(regex, col_upper, re.IGNORECASE):
                        matched = True
                        break
                except re.error:
                    continue
            if not matched:
                create_exact_column_pattern(col_name, "Non-PII")

        # Save to file
        self._save_heuristics(heuristics)

        # Run refinement to extract more specific patterns from sample_columns

        heuristics = self.refine_heuristics()

        return heuristics

    def _extract_patterns(
        self, columns: List[Dict[str, Any]], classification: str
    ) -> Dict[str, Dict[str, Any]]:
        """Extract suffix, prefix, and keyword patterns from column list.

        Args:
            columns: List of column entries
            classification: "PII" or "Non-PII"

        Returns:
            Dictionary of extracted patterns by type
        """
        patterns = {"suffix": {}, "prefix": {}, "keyword": {}}

        # Collect suffixes and prefixes with weights
        suffix_counts = defaultdict(lambda: {"count": 0, "samples": [], "weight": 0})
        prefix_counts = defaultdict(lambda: {"count": 0, "samples": [], "weight": 0})
        keyword_counts = defaultdict(lambda: {"count": 0, "samples": [], "weight": 0})

        for col in columns:
            column_name = col["column_name"].upper()
            column_comment = col.get("column_comment", "")
            weight = col.get("total_selections", 1)

            # Extract suffixes using longest match from the end

            if "_" in column_name:
                parts = column_name.split("_")
                if len(parts) >= 2:
                    # Generate all possible suffixes from longest to shortest
                    # e.g., "REQ_CLIENT_ACCOUNT_NUM" ->
                    #   "_CLIENT_ACCOUNT_NUM$", "_ACCOUNT_NUM$", "_NUM$"
                    for i in range(1, len(parts)):
                        suffix = "_" + "_".join(parts[i:]) + "$"
                        suffix_counts[suffix]["count"] += 1
                        suffix_counts[suffix]["weight"] += weight
                        if column_name not in suffix_counts[suffix]["samples"]:
                            suffix_counts[suffix]["samples"].append(column_name)

                    # Extract prefix (first part before underscore)
                    prefix = f"^{parts[0]}_"
                    prefix_counts[prefix]["count"] += 1
                    prefix_counts[prefix]["weight"] += weight
                    if column_name not in prefix_counts[prefix]["samples"]:
                        prefix_counts[prefix]["samples"].append(column_name)

            # Extract keyword patterns from column names.
            # Description words need not occur in the names being matched.

        # Filter and create final patterns
        total_columns = len(columns) if columns else 1

        # Prefer the longest supported suffix.

        suffix_candidates = {}
        for suffix, data in suffix_counts.items():
            if data["count"] >= self.MIN_PATTERN_OCCURRENCES:
                confidence = min(1.0, data["count"] / total_columns + 0.1 * len(data["samples"]))
                if confidence >= self.MIN_CONFIDENCE:
                    suffix_candidates[suffix] = {
                        "confidence": round(confidence, 3),
                        "count": data["count"],
                        "samples": data["samples"][:5],
                    }

        # Discard shorter suffixes whose support is covered by a longer suffix.

        suffixes_to_remove = set()
        suffix_list = sorted(suffix_candidates.keys(), key=len, reverse=True)

        for i, longer_suffix in enumerate(suffix_list):
            longer_samples = set(suffix_candidates[longer_suffix]["samples"])
            for shorter_suffix in suffix_list[i + 1 :]:
                if shorter_suffix in suffixes_to_remove:
                    continue
                shorter_samples = set(suffix_candidates[shorter_suffix]["samples"])

                if shorter_samples <= longer_samples:
                    suffixes_to_remove.add(shorter_suffix)

        for suffix, data in suffix_candidates.items():
            if suffix not in suffixes_to_remove:
                patterns["suffix"][suffix] = data

        # Process prefixes
        for prefix, data in prefix_counts.items():
            if data["count"] >= self.MIN_PATTERN_OCCURRENCES:
                confidence = min(1.0, data["count"] / total_columns + 0.1 * len(data["samples"]))
                if confidence >= self.MIN_CONFIDENCE:
                    patterns["prefix"][prefix] = {
                        "confidence": round(confidence, 3),
                        "count": data["count"],
                        "samples": data["samples"][:5],
                    }

        # Process keywords
        for keyword, data in keyword_counts.items():
            if data["count"] >= self.MIN_PATTERN_OCCURRENCES:
                confidence = min(1.0, data["count"] / total_columns + 0.1 * len(data["samples"]))
                if confidence >= self.MIN_CONFIDENCE:
                    patterns["keyword"][keyword] = {
                        "confidence": round(confidence, 3),
                        "count": data["count"],
                        "samples": data["samples"][:5],
                    }

        logger.info(
            f"Extracted patterns for {classification}: "
            f"{len(patterns['suffix'])} suffixes, "
            f"{len(patterns['prefix'])} prefixes, "
            f"{len(patterns['keyword'])} keywords"
        )

        return patterns

    def _pattern_conflict_rate(
        self,
        pattern: str,
        label: str,
        pii_columns: List[Dict[str, Any]],
        non_pii_columns: List[Dict[str, Any]],
    ) -> float:
        """Count opposite labels among all feedback entries matched by a rule."""
        pii_count = sum(
            bool(re.search(pattern, column["column_name"], re.IGNORECASE)) for column in pii_columns
        )
        non_pii_count = sum(
            bool(re.search(pattern, column["column_name"], re.IGNORECASE))
            for column in non_pii_columns
        )
        total_count = pii_count + non_pii_count
        opposite_count = non_pii_count if label == "PII" else pii_count
        return opposite_count / total_count if total_count else 1.0

    def _resolve_conflicts(
        self,
        pii_patterns: Dict[str, Dict[str, Any]],
        non_pii_patterns: Dict[str, Dict[str, Any]],
        pii_columns: List[Dict[str, Any]],
        non_pii_columns: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """Check candidate rules against all matching feedback, including rare labels."""
        for pattern_type in ("suffix", "prefix", "keyword"):
            positive = pii_patterns.get(pattern_type, {})
            negative = non_pii_patterns.get(pattern_type, {})
            for pattern in sorted(set(positive) | set(negative)):
                # Rare counterexamples count even without an opposite-class candidate.
                for candidates, label in ((positive, "PII"), (negative, "Non-PII")):
                    if pattern not in candidates:
                        continue
                    conflict_rate = self._pattern_conflict_rate(
                        pattern, label, pii_columns, non_pii_columns
                    )
                    if conflict_rate > self.MAX_CONFLICT_RATE:
                        del candidates[pattern]
                        logger.info(
                            "Conflict filter removed %s '%s' (rate %.3f > %.3f)",
                            pattern_type,
                            pattern,
                            conflict_rate,
                            self.MAX_CONFLICT_RATE,
                        )
                    else:
                        candidates[pattern]["conflict_rate"] = round(conflict_rate, 3)

        return pii_patterns, non_pii_patterns

    def _deduplicate_suffix_patterns_by_samples(
        self, heuristics: Dict[str, Any], now: str
    ) -> Dict[str, Any]:
        """Remove redundant suffix patterns that have identical or subset sample_columns.

        When two suffix patterns have the same sample_columns, keep only the longer (more specific) pattern.

        e.g., _CLIENT_NAME$ and _NAME$ both with [recv_client_name, req_client_name]
        -> Keep only _CLIENT_NAME$ (longer/more specific)

        Also handles subsets: if _NAME$'s samples are a subset of _CLIENT_NAME$'s samples,
        remove _NAME$ only if they are identical.

        Args:
            heuristics: Current heuristics dictionary
            now: Current timestamp

        Returns:
            Updated heuristics dictionary with redundant patterns removed
        """
        # Group auto-learned suffix patterns by pii_type
        suffix_patterns_by_pii = {"PII": [], "Non-PII": []}

        for h in heuristics["heuristics"]:
            if (
                h.get("pattern_type") == "suffix"
                and h.get("source") == "auto_learned"
                and h.get("enabled", True)
            ):
                pii_type = h.get("pii_type", "PII")
                suffix_patterns_by_pii[pii_type].append(h)

        patterns_to_remove = set()

        for pii_type, patterns in suffix_patterns_by_pii.items():
            if len(patterns) < 2:
                continue

            # Sort by regex length (longest first) for longest match priority
            patterns.sort(key=lambda h: len(h.get("regex", "")), reverse=True)

            # Compare each pair of patterns
            for i, longer_pattern in enumerate(patterns):
                longer_regex = longer_pattern.get("regex", "")
                longer_samples = set(c.upper() for c in longer_pattern.get("sample_columns", []))
                longer_id = longer_pattern.get("id")

                if longer_id in patterns_to_remove:
                    continue

                for shorter_pattern in patterns[i + 1 :]:
                    shorter_regex = shorter_pattern.get("regex", "")
                    shorter_samples = set(
                        c.upper() for c in shorter_pattern.get("sample_columns", [])
                    )
                    shorter_id = shorter_pattern.get("id")

                    if shorter_id in patterns_to_remove:
                        continue

                    # Check if samples are identical (case-insensitive)
                    if shorter_samples == longer_samples:
                        # Same samples -> remove the shorter (less specific) pattern
                        patterns_to_remove.add(shorter_id)
                        logger.info(
                            f"Dedup: Removed {shorter_regex} (same samples as {longer_regex}, "
                            f"samples: {list(shorter_samples)[:3]})"
                        )
                    # Check if shorter pattern's samples are strict subset of longer pattern's samples
                    # AND shorter pattern matches all columns that longer pattern matches
                    # In this case, shorter pattern is redundant
                    elif shorter_samples < longer_samples:
                        # Shorter pattern's samples are subset of longer pattern
                        # But only remove if the shorter regex would match all longer samples too
                        # (i.e., longer regex ends with shorter regex pattern)
                        shorter_suffix = shorter_regex.rstrip("$")
                        longer_suffix = longer_regex.rstrip("$")
                        if longer_suffix.endswith(shorter_suffix):
                            patterns_to_remove.add(shorter_id)
                            logger.info(
                                f"Dedup: Removed {shorter_regex} (subset of {longer_regex}, "
                                f"{len(shorter_samples)} vs {len(longer_samples)} samples)"
                            )

        # Remove the redundant patterns
        if patterns_to_remove:
            heuristics["heuristics"] = [
                h for h in heuristics["heuristics"] if h.get("id") not in patterns_to_remove
            ]
            logger.info(
                f"Removed {len(patterns_to_remove)} redundant suffix patterns via deduplication"
            )

        return heuristics

    def refine_heuristics(self) -> Dict[str, Any]:
        """Refine existing heuristics by extracting more specific patterns from sample_columns.

        For example, if "_NUM$" has samples ["CNTR_ACCOUNT_NUM", "FINTECH_USE_NUM", "RECV_CLIENT_ACCOUNT_NUM"],
        this method will create more specific patterns like "_ACCOUNT_NUM$" and "_USE_NUM$".

        Returns:
            Updated heuristics dictionary
        """
        logger.info("Starting heuristics refinement...")
        heuristics = self._load_heuristics()
        now = datetime.now().isoformat()

        try:
            with open(self.feedback_file, "r", encoding="utf-8") as stream:
                feedback = json.load(stream)
        except (OSError, ValueError):
            logger.warning("Skipping refinement: feedback cannot be checked.")
            return heuristics

        feedback_columns = {"PII": [], "Non-PII": []}
        for section, label_key in (
            ("column_exact", "pii_type"),
            ("pii_classification", "classification"),
        ):
            for entry in feedback.get(section, {}).values():
                label = entry.get(label_key)
                if label in feedback_columns and entry.get("column_name"):
                    feedback_columns[label].append(entry)

        def accepts_refinement(pattern: str, label: str) -> bool:
            return (
                self._pattern_conflict_rate(
                    pattern, label, feedback_columns["PII"], feedback_columns["Non-PII"]
                )
                <= self.MAX_CONFLICT_RATE
            )

        # Get existing IDs for new pattern generation
        existing_ids = [h.get("id", "") for h in heuristics["heuristics"]]

        # Track new patterns to add
        new_patterns_to_add = []

        # First, collect all exact patterns and try to merge them into suffix patterns

        exact_patterns_by_pii_type = {"PII": [], "Non-PII": []}
        # Map: column -> exact pattern entry (for removing merged columns)
        column_to_exact_pattern = {}

        for h in heuristics["heuristics"]:
            if (
                h.get("pattern_type") == "exact"
                and h.get("enabled", True)
                and h.get("source") == "auto_learned"
            ):
                pii_type = h.get("pii_type", "PII")
                sample_cols = h.get("sample_columns", [])
                if sample_cols:
                    exact_patterns_by_pii_type[pii_type].extend(sample_cols)
                    for col in sample_cols:
                        column_to_exact_pattern[col.upper()] = h

        # Track columns that were merged into new suffix patterns

        merged_columns = set()

        # Process exact patterns to find common suffixes
        for pii_type, columns in exact_patterns_by_pii_type.items():
            if len(columns) < self.MIN_PATTERN_OCCURRENCES:
                continue

            # Extract all possible suffixes from exact columns
            suffix_groups = defaultdict(list)
            for col in columns:
                col_upper = col.upper()
                if "_" not in col_upper:
                    continue
                parts = col_upper.split("_")
                for i in range(1, len(parts)):
                    suffix = "_" + "_".join(parts[i:]) + "$"
                    suffix_groups[suffix].append(col)

            # Find suffixes with MIN_PATTERN_OCCURRENCES or more columns
            for suffix, matched_cols in suffix_groups.items():
                if len(matched_cols) >= self.MIN_PATTERN_OCCURRENCES:
                    if not accepts_refinement(suffix, pii_type):
                        continue
                    # Check if this suffix already exists
                    existing_pattern = next(
                        (
                            h2
                            for h2 in heuristics["heuristics"]
                            if h2.get("regex") == suffix
                            and h2.get("pii_type") == pii_type
                            and h2.get("source") == "auto_learned"
                        ),
                        None,
                    )
                    if existing_pattern:
                        # Add columns to existing pattern
                        existing_samples = existing_pattern.get("sample_columns", [])
                        for col in matched_cols:
                            if col not in existing_samples:
                                existing_samples.append(col)
                            merged_columns.add(col.upper())
                        existing_pattern["sample_columns"] = existing_samples
                        existing_pattern["updated_at"] = now
                    elif suffix not in [p["regex"] for p in new_patterns_to_add]:
                        # Confidence for exact pattern merging: based on matched count only

                        # Require the configured minimum number of supporting observations.
                        confidence = min(1.0, 0.7 + 0.1 * len(matched_cols))
                        if len(matched_cols) >= self.MIN_PATTERN_OCCURRENCES:
                            new_id = self._generate_id(existing_ids, "auto")
                            existing_ids.append(new_id)

                            new_entry = {
                                "id": new_id,
                                "name": f"Auto: suffix {suffix}",
                                "regex": suffix,
                                "pii_type": pii_type,
                                "rationale": f"Merged from exact patterns: {', '.join(matched_cols[:3])}",
                                "priority": int(confidence * 100),
                                "enabled": True,
                                "source": "auto_learned",
                                "pattern_type": "suffix",
                                "confidence": round(confidence, 3),
                                "sample_columns": matched_cols,
                                "created_at": now,
                                "updated_at": now,
                            }
                            new_patterns_to_add.append(new_entry)
                            # Track merged columns
                            for col in matched_cols:
                                merged_columns.add(col.upper())
                            logger.info(
                                f"Merged exact patterns into suffix: {suffix} "
                                f"({len(matched_cols)} columns: {', '.join(matched_cols[:3])})"
                            )

        # Remove merged columns from their original exact patterns

        exact_patterns_to_remove = []
        for col_upper in merged_columns:
            if col_upper in column_to_exact_pattern:
                exact_entry = column_to_exact_pattern[col_upper]
                existing_samples = exact_entry.get("sample_columns", [])
                # Remove the merged column (case-insensitive)
                exact_entry["sample_columns"] = [
                    c for c in existing_samples if c.upper() != col_upper
                ]
                exact_entry["updated_at"] = now
                # If no samples remain, mark for removal
                if not exact_entry["sample_columns"]:
                    exact_patterns_to_remove.append(exact_entry.get("id"))

        # Remove exact patterns with no remaining samples

        if exact_patterns_to_remove:
            heuristics["heuristics"] = [
                h for h in heuristics["heuristics"] if h.get("id") not in exact_patterns_to_remove
            ]
            logger.info(
                f"Removed {len(exact_patterns_to_remove)} empty exact patterns after merging"
            )

        # Process each suffix heuristic for refinement
        for h in heuristics["heuristics"]:
            if h.get("source") != "auto_learned":
                continue
            if h.get("pattern_type") not in ("suffix", "exact"):
                continue
            if not h.get("enabled", True):
                continue

            current_regex = h.get("regex", "")
            sample_columns = h.get("sample_columns", [])
            pii_type = h.get("pii_type", "PII")

            if len(sample_columns) < self.MIN_PATTERN_OCCURRENCES:
                continue

            # Extract all possible longer suffixes from sample columns

            suffix_groups = defaultdict(list)

            for col in sample_columns:
                col_upper = col.upper()
                if "_" not in col_upper:
                    continue

                parts = col_upper.split("_")
                # Generate suffixes longer than current pattern

                for i in range(1, len(parts)):
                    suffix = "_" + "_".join(parts[i:]) + "$"
                    # Only consider suffixes longer than current
                    if len(suffix) > len(current_regex):
                        suffix_groups[suffix].append(col)

            # Find suffixes that can form independent patterns

            refined_patterns = {}
            for suffix, columns in suffix_groups.items():
                if len(columns) >= self.MIN_PATTERN_OCCURRENCES:
                    if not accepts_refinement(suffix, pii_type):
                        continue
                    # Check if this suffix already exists
                    existing_pattern = next(
                        (
                            h2
                            for h2 in heuristics["heuristics"]
                            if h2.get("regex") == suffix
                            and h2.get("pii_type") == pii_type
                            and h2.get("source") == "auto_learned"
                        ),
                        None,
                    )
                    if existing_pattern:
                        # Add new columns to existing pattern
                        existing_samples = existing_pattern.get("sample_columns", [])
                        updated = False
                        for col in columns:
                            if col not in existing_samples:
                                existing_samples.append(col)
                                updated = True
                        if updated:
                            existing_pattern["sample_columns"] = existing_samples
                            existing_pattern["updated_at"] = now
                        continue

                    # Check if already planned to add
                    if suffix in [p["regex"] for p in new_patterns_to_add]:
                        continue

                    refined_patterns[suffix] = columns

            # Select the best refined patterns using longest match principle

            if refined_patterns:
                # Sort by length (longest first)
                sorted_suffixes = sorted(refined_patterns.keys(), key=len, reverse=True)

                # Remove shorter patterns whose samples are subset of longer patterns
                suffixes_to_keep = []
                for suffix in sorted_suffixes:
                    suffix_samples = set(refined_patterns[suffix])
                    is_subset = False
                    for kept_suffix in suffixes_to_keep:
                        kept_samples = set(refined_patterns[kept_suffix])
                        if suffix_samples <= kept_samples:
                            is_subset = True
                            break
                    if not is_subset:
                        suffixes_to_keep.append(suffix)

                # Create new heuristic entries for refined patterns
                for suffix in suffixes_to_keep:
                    columns = refined_patterns[suffix]
                    confidence = min(1.0, len(columns) / len(sample_columns) + 0.1 * len(columns))

                    if confidence >= self.MIN_CONFIDENCE:
                        new_id = self._generate_id(existing_ids, "auto")
                        existing_ids.append(new_id)

                        new_entry = {
                            "id": new_id,
                            "name": f"Auto: suffix {suffix}",
                            "regex": suffix,
                            "pii_type": pii_type,
                            "rationale": f"Refined from {current_regex}: {', '.join(columns[:3])}",
                            "priority": int(confidence * 100) + 10,  # Higher priority than parent
                            "enabled": True,
                            "source": "auto_learned",
                            "pattern_type": "suffix",
                            "confidence": round(confidence, 3),
                            "sample_columns": columns,
                            "parent_pattern": current_regex,
                            "created_at": now,
                            "updated_at": now,
                        }
                        new_patterns_to_add.append(new_entry)

                        logger.info(
                            f"Refined pattern: {current_regex} -> {suffix} "
                            f"({len(columns)} columns: {', '.join(columns[:3])})"
                        )

                # Remove refined columns from the parent pattern.

                refined_columns = set()
                for suffix in suffixes_to_keep:
                    refined_columns.update(refined_patterns[suffix])

                remaining_columns = [c for c in sample_columns if c not in refined_columns]
                if remaining_columns:
                    h["sample_columns"] = remaining_columns
                    h["updated_at"] = now
                # If all columns moved to refined patterns, we keep the parent but with empty samples
                # (it may still match future columns that don't fit refined patterns)

        # Add all new refined patterns
        for new_entry in new_patterns_to_add:
            heuristics["heuristics"].append(new_entry)

        # Apply longest match deduplication for suffix patterns with same sample_columns

        # e.g., _CLIENT_NAME$ and _NAME$ with same samples -> keep only _CLIENT_NAME$
        heuristics = self._deduplicate_suffix_patterns_by_samples(heuristics, now)

        # Save updated heuristics
        if new_patterns_to_add:
            self._save_heuristics(heuristics)
            logger.info(f"Added {len(new_patterns_to_add)} refined patterns")
        else:
            # Even if no new patterns, save if deduplication occurred
            self._save_heuristics(heuristics)
            logger.info("No refinement opportunities found")

        return heuristics

    def classify_column(
        self, column_name: str, column_comment: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Classify a column using learned heuristic patterns.

        This method provides fast classification (< 1ms) using regex patterns
        learned from expert feedback.

        Args:
            column_name: Column name to classify
            column_comment: Column comment/description

        Returns:
            Dict with classification result or None if no pattern matches
            {"classification": "PII"/"Non-PII", "match_type": str, "pattern": str,
             "confidence": float, "samples": List[str]}
        """
        heuristics = self._load_heuristics()

        # Get all enabled heuristics sorted by priority
        enabled_heuristics = [h for h in heuristics.get("heuristics", []) if h.get("enabled", True)]
        enabled_heuristics.sort(key=lambda h: h.get("priority", 0), reverse=True)

        column_upper = column_name.upper()

        for h in enabled_heuristics:
            try:
                pattern = re.compile(h["regex"], re.IGNORECASE)

                # Check column name only (NOT column_comment)

                if pattern.search(column_upper):
                    return {
                        "classification": h.get("pii_type", "PII"),
                        "match_type": h.get("pattern_type", "unknown"),
                        "pattern": h.get("regex", ""),
                        "confidence": h.get("confidence", 0.8),
                        "samples": h.get("sample_columns", []),
                        "source": h.get("source", "unknown"),
                        "heuristic_id": h.get("id", ""),
                    }
            except re.error:
                # Skip invalid regex patterns
                continue

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about learned patterns.

        Returns:
            Dictionary with pattern statistics
        """
        heuristics = self._load_heuristics()

        auto_patterns = [h for h in heuristics["heuristics"] if h.get("source") == "auto_learned"]
        manual_patterns = [h for h in heuristics["heuristics"] if h.get("source") == "manual"]

        pii_auto = [h for h in auto_patterns if h.get("pii_type") == "PII"]
        non_pii_auto = [h for h in auto_patterns if h.get("pii_type") == "Non-PII"]

        return {
            "total_patterns": len(heuristics["heuristics"]),
            "manual_patterns": len(manual_patterns),
            "auto_learned_patterns": len(auto_patterns),
            "pii_patterns": {
                "suffixes": len([h for h in pii_auto if h.get("pattern_type") == "suffix"]),
                "prefixes": len([h for h in pii_auto if h.get("pattern_type") == "prefix"]),
                "keywords": len([h for h in pii_auto if h.get("pattern_type") == "keyword"]),
                "exact": len([h for h in pii_auto if h.get("pattern_type") == "exact"]),
            },
            "non_pii_patterns": {
                "suffixes": len([h for h in non_pii_auto if h.get("pattern_type") == "suffix"]),
                "prefixes": len([h for h in non_pii_auto if h.get("pattern_type") == "prefix"]),
                "keywords": len([h for h in non_pii_auto if h.get("pattern_type") == "keyword"]),
                "exact": len([h for h in non_pii_auto if h.get("pattern_type") == "exact"]),
            },
            "last_updated": heuristics.get("statistics", {}).get("last_updated"),
            "last_auto_learn": heuristics.get("statistics", {}).get("last_auto_learn"),
        }
