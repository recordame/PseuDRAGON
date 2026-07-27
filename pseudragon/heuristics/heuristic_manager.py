"""
Heuristic Manager

Manages heuristic rules for PII detection.

This module provides:
- CRUD operations for heuristic patterns
- Regex validation
- Priority-based pattern matching
- Integration with Stage 1 PII detection
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class HeuristicManager:
    """
    Manager for heuristic rules
    
    Handles loading, saving, and querying heuristic patterns.
    """

    def __init__(self, heuristics_file: Optional[str] = None):
        """
        Initialize heuristic manager
        
        Args:
            heuristics_file: Path to heuristics.json file
        """
        if heuristics_file is None:
            # Experiments can point at an isolated store so learned rules never
            # leak between conditions or modify the application state.
            configured_file = os.environ.get("HEURISTICS_FILE")
            if configured_file:
                heuristics_file = configured_file
            else:
                # Default to resources/heuristics.json
                base_dir = Path(__file__).parent.parent.parent
                heuristics_file = base_dir / "resources" / "heuristics.json"

        self.heuristics_file = Path(heuristics_file)
        self.heuristics: Dict[str, Any] = self._create_empty_heuristics()

        # Load existing heuristics
        if self.heuristics_file.exists():
            self.load()
        else:
            # Create default heuristics file
            self.heuristics_file.parent.mkdir(parents=True, exist_ok=True)
            self.save()

    def _create_empty_heuristics(self) -> Dict[str, Any]:
        """
        Create empty heuristics structure (v2.0 format)

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
                "last_auto_learn": None
            }
        }

    def load(self) -> None:
        """
        Load heuristics from file
        """
        try:
            with open(self.heuristics_file, 'r', encoding='utf-8') as f:
                self.heuristics = json.load(f)
                # Migrate v1.0 to v2.0 format if needed
                if self.heuristics.get("version") == "1.0":
                    self.heuristics = self._migrate_v1_to_v2(self.heuristics)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading heuristics: {e}")
            self.heuristics = self._create_empty_heuristics()

    def _migrate_v1_to_v2(self, old_heuristics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate v1.0 heuristics format to v2.0

        Args:
            old_heuristics: Old format heuristics

        Returns:
            New format heuristics
        """
        new_heuristics = self._create_empty_heuristics()

        # Migrate existing heuristics, marking them as manual if no source specified
        for h in old_heuristics.get("heuristics", []):
            if "source" not in h:
                h["source"] = "manual"
            new_heuristics["heuristics"].append(h)

        return new_heuristics

    def save(self) -> None:
        """
        Save heuristics to file
        """
        now = datetime.now().isoformat()

        # Update statistics
        if "statistics" not in self.heuristics:
            self.heuristics["statistics"] = {}

        manual_count = sum(1 for h in self.heuristics.get("heuristics", []) if h.get("source") == "manual")
        auto_count = sum(1 for h in self.heuristics.get("heuristics", []) if h.get("source") == "auto_learned")

        self.heuristics["statistics"]["total_patterns"] = len(self.heuristics.get("heuristics", []))
        self.heuristics["statistics"]["manual_patterns"] = manual_count
        self.heuristics["statistics"]["auto_learned_patterns"] = auto_count
        self.heuristics["statistics"]["last_updated"] = now

        with open(self.heuristics_file, 'w', encoding='utf-8') as f:
            json.dump(self.heuristics, f, indent=2, ensure_ascii=False)

    def get_all(self, enabled_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get all heuristics
        
        Args:
            enabled_only: Only return enabled heuristics
        
        Returns:
            List of heuristic dictionaries
        """
        heuristics = self.heuristics.get("heuristics", [])

        if enabled_only:
            heuristics = [h for h in heuristics if h.get("enabled", True)]

        # Sort by priority (descending)
        return sorted(heuristics, key=lambda h: h.get("priority", 0), reverse=True)

    def get_by_id(self, heuristic_id: str) -> Optional[Dict[str, Any]]:
        """
        Get heuristic by ID
        
        Args:
            heuristic_id: Heuristic ID
        
        Returns:
            Heuristic dictionary or None
        """
        for h in self.heuristics.get("heuristics", []):
            if h.get("id") == heuristic_id:
                return h
        return None

    def add(
            self, name: str, regex: str, pii_type: str, rationale: str,
            priority: int = 50, enabled: bool = True, source: str = "manual",
            pattern_type: str = ""
    ) -> Dict[str, Any]:
        """
        Add new heuristic

        Args:
            name: Heuristic name
            regex: Regular expression pattern
            pii_type: PII type (PII or Non-PII)
            rationale: Reason for this heuristic
            priority: Priority (higher = more important)
            enabled: Whether heuristic is enabled
            source: Source of heuristic ("manual" or "auto_learned")
            pattern_type: Type of pattern ("suffix", "prefix", "keyword", "custom")

        Returns:
            Created heuristic dictionary

        Raises:
            ValueError: If regex is invalid
        """
        # Validate regex
        if not self.validate_regex(regex):
            raise ValueError(f"Invalid regex pattern: {regex}")

        # Generate new ID (use different prefix for manual vs auto)
        existing_ids = [h.get("id", "") for h in self.heuristics.get("heuristics", [])]
        new_id = self._generate_id(existing_ids)

        # Create heuristic
        now = datetime.now().isoformat()
        heuristic = {
            "id": new_id,
            "name": name,
            "regex": regex,
            "pii_type": pii_type,
            "rationale": rationale,
            "priority": priority,
            "enabled": enabled,
            "source": source,
            "created_at": now,
            "updated_at": now
        }

        # Add pattern_type if provided
        if pattern_type:
            heuristic["pattern_type"] = pattern_type

        self.heuristics.setdefault("heuristics", []).append(heuristic)
        self.save()

        return heuristic

    def update(self, heuristic_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Update existing heuristic
        
        Args:
            heuristic_id: Heuristic ID to update
            **kwargs: Fields to update
        
        Returns:
            Updated heuristic or None if not found
        
        Raises:
            ValueError: If regex is invalid
        """
        heuristic = self.get_by_id(heuristic_id)
        if not heuristic:
            return None

        # Validate regex if updating
        if "regex" in kwargs:
            if not self.validate_regex(kwargs["regex"]):
                raise ValueError(f"Invalid regex pattern: {kwargs['regex']}")

        # Update fields
        allowed_fields = ["name", "regex", "pii_type", "rationale", "priority", "enabled", "source", "pattern_type"]
        for field in allowed_fields:
            if field in kwargs:
                heuristic[field] = kwargs[field]

        heuristic["updated_at"] = datetime.now().isoformat()
        self.save()

        return heuristic

    def delete(self, heuristic_id: str) -> bool:
        """
        Delete heuristic
        
        Args:
            heuristic_id: Heuristic ID to delete
        
        Returns:
            True if deleted, False if not found
        """
        heuristics = self.heuristics.get("heuristics", [])
        original_length = len(heuristics)

        self.heuristics["heuristics"] = [
            h for h in heuristics if h.get("id") != heuristic_id
        ]

        if len(self.heuristics["heuristics"]) < original_length:
            self.save()
            return True

        return False

    def test_pattern(self, regex: str, test_strings: List[str]) -> Dict[str, Any]:
        """
        Test regex pattern against sample strings
        
        Args:
            regex: Regular expression pattern to test
            test_strings: List of strings to test against
        
        Returns:
            Dictionary with test results
        """
        if not self.validate_regex(regex):
            return {
                "valid": False,
                "error": "Invalid regex pattern",
                "matches": []
            }

        try:
            pattern = re.compile(regex, re.IGNORECASE)
            matches = []

            for test_str in test_strings:
                is_match = bool(pattern.search(test_str))
                matches.append(
                    {
                        "string": test_str,
                        "matched": is_match
                    }
                )

            return {
                "valid": True,
                "error": None,
                "matches": matches
            }
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "matches": []
            }

    def match_column(self, column_name: str) -> Optional[Dict[str, Any]]:
        """
        Find matching heuristic for a column name
        
        Args:
            column_name: Column name to match
        
        Returns:
            Best matching heuristic or None
        """
        enabled_heuristics = self.get_all(enabled_only=True)

        for heuristic in enabled_heuristics:
            try:
                pattern = re.compile(heuristic["regex"], re.IGNORECASE)
                if pattern.search(column_name):
                    return heuristic
            except re.error:
                # Skip invalid regex patterns
                continue

        return None

    @staticmethod
    def validate_regex(regex: str) -> bool:
        """
        Validate regex pattern
        
        Args:
            regex: Regular expression pattern
        
        Returns:
            True if valid, False otherwise
        """
        if not regex:
            return False

        try:
            re.compile(regex)
            return True
        except re.error:
            return False

    def _generate_id(self, existing_ids: List[str]) -> str:
        """
        Generate unique heuristic ID
        
        Args:
            existing_ids: List of existing IDs
        
        Returns:
            New unique ID
        """
        # Extract numeric part from existing IDs (e.g., h001 -> 1)
        numbers = []
        for id_str in existing_ids:
            if id_str.startswith("h"):
                try:
                    numbers.append(int(id_str[1:]))
                except ValueError:
                    continue

        # Find next available number
        if numbers:
            next_num = max(numbers) + 1
        else:
            next_num = 1

        return f"h{next_num:03d}"

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about heuristic patterns

        Returns:
            Dictionary with pattern statistics
        """
        return self.heuristics.get("statistics", {
            "total_patterns": len(self.heuristics.get("heuristics", [])),
            "manual_patterns": 0,
            "auto_learned_patterns": 0,
            "last_updated": None,
            "last_auto_learn": None
        })

    def get_by_source(self, source: str, enabled_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get heuristics filtered by source

        Args:
            source: Source to filter by ("manual" or "auto_learned")
            enabled_only: Only return enabled heuristics

        Returns:
            List of heuristic dictionaries
        """
        heuristics = self.heuristics.get("heuristics", [])
        filtered = [h for h in heuristics if h.get("source") == source]

        if enabled_only:
            filtered = [h for h in filtered if h.get("enabled", True)]

        # Sort by priority (descending)
        return sorted(filtered, key=lambda h: h.get("priority", 0), reverse=True)
