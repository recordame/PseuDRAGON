"""
Heuristic Pattern Learner for PseuDRAGON
PseuDRAGON 휴리스틱 패턴 학습기

Automatically extracts regex patterns from expert feedback to enable
fast PII classification without LLM calls. Integrates with the unified
heuristics.json file used by HeuristicManager.

전문가 피드백에서 정규식 패턴을 자동으로 추출하여
LLM 호출 없이 빠른 PII 분류를 가능하게 합니다.
HeuristicManager가 사용하는 통합 heuristics.json 파일과 통합됩니다.

Processing hierarchy:
처리 우선순위:
1. Exact Match (정확한 매칭)
2. Heuristics (수동 + 자동 학습된 휴리스틱) - This module
3. RAG Similarity (RAG 유사도)
4. LLM Verification (LLM 검증)
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
    """
    Learns regex patterns from expert feedback and saves to unified heuristics.json
    전문가 피드백으로부터 정규식 패턴을 학습하고 통합 heuristics.json에 저장

    Analyzes column_exact and pii_classification data to extract:
    - Suffix patterns (e.g., _ACNO$ -> PII, _DT$ -> Non-PII)
    - Prefix patterns (e.g., ^USR_ -> PII)
    - Comment keywords (e.g., "account number" -> PII)

    Auto-learned patterns are marked with source="auto_learned" and can be
    edited or deleted by experts through the web interface.

    column_exact 및 pii_classification 데이터를 분석하여 추출:
    - 접미사 패턴 (예: _ACNO$ -> PII, _DT$ -> Non-PII)
    - 접두사 패턴 (예: ^USR_ -> PII)
    - 코멘트 키워드 (예: "account number" -> PII)

    자동 학습된 패턴은 source="auto_learned"로 표시되며
    웹 인터페이스를 통해 전문가가 편집하거나 삭제할 수 있습니다.
    """

    # Minimum occurrences for a pattern to be considered reliable
    # 패턴이 신뢰할 수 있는 것으로 간주되기 위한 최소 발생 횟수
    MIN_PATTERN_OCCURRENCES = 2

    # Minimum confidence threshold for a pattern to be saved
    # 패턴이 저장되기 위한 최소 신뢰도 임계값
    MIN_CONFIDENCE = 0.8

    def __init__(
        self,
        feedback_file: str = "resources/expert_feedback.json",
        heuristics_file: str = "resources/heuristics.json"
    ):
        """
        Initialize the pattern learner
        패턴 학습기 초기화

        Args:
            feedback_file: Path to expert feedback JSON file
            heuristics_file: Path to unified heuristics JSON file (same as HeuristicManager)
        """
        self.feedback_file = feedback_file
        self.heuristics_file = heuristics_file

    def _load_heuristics(self) -> Dict[str, Any]:
        """
        Load existing heuristics from unified file
        통합 파일에서 기존 휴리스틱 로드

        Returns:
            Dictionary of heuristics
        """
        if not os.path.exists(self.heuristics_file):
            logger.info(f"No existing heuristics file found at {self.heuristics_file}")
            return self._create_empty_heuristics()

        try:
            with open(self.heuristics_file, 'r', encoding='utf-8') as f:
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
        """
        Migrate v1.0 heuristics format to v2.0
        v1.0 휴리스틱 형식을 v2.0으로 마이그레이션

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
        """
        Create empty heuristics structure (v2.0 format)
        빈 휴리스틱 구조 생성 (v2.0 형식)

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

    def _save_heuristics(self, heuristics: Dict[str, Any]):
        """
        Save heuristics to unified file
        통합 파일에 휴리스틱 저장
        """
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
            auto_count = sum(1 for h in heuristics["heuristics"] if h.get("source") == "auto_learned")
            heuristics["statistics"]["total_patterns"] = len(heuristics["heuristics"])
            heuristics["statistics"]["manual_patterns"] = manual_count
            heuristics["statistics"]["auto_learned_patterns"] = auto_count

            with open(self.heuristics_file, 'w', encoding='utf-8') as f:
                json.dump(heuristics, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(heuristics['heuristics'])} heuristic patterns to {self.heuristics_file}")
            logger.info(f"  - Manual: {manual_count}, Auto-learned: {auto_count}")
        except Exception as e:
            logger.error(f"Failed to save heuristics: {e}")

    def _generate_id(self, existing_ids: List[str], prefix: str = "auto") -> str:
        """
        Generate unique heuristic ID
        고유한 휴리스틱 ID 생성

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
                    numbers.append(int(id_str[len(prefix):]))
                except ValueError:
                    continue

        # Find next available number
        next_num = max(numbers) + 1 if numbers else 1
        return f"{prefix}{next_num:03d}"

    def learn_patterns_from_feedback(self) -> Dict[str, Any]:
        """
        Main method to analyze expert feedback and extract patterns
        전문가 피드백을 분석하고 패턴을 추출하는 메인 메서드

        Returns:
            Dictionary of learned patterns (unified heuristics)
        """
        logger.info("Starting pattern learning from expert feedback...")

        # Load expert feedback
        if not os.path.exists(self.feedback_file):
            logger.warning(f"Expert feedback file not found: {self.feedback_file}")
            return self._load_heuristics()

        try:
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                feedback = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load expert feedback: {e}")
            return self._load_heuristics()

        # Load existing heuristics (preserve manual patterns)
        heuristics = self._load_heuristics()

        # Remove old auto-learned patterns (will be replaced with new ones)
        heuristics["heuristics"] = [
            h for h in heuristics["heuristics"]
            if h.get("source") != "auto_learned"
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
                pii_columns.append({
                    "column_name": column_name,
                    "column_comment": column_comment,
                    "total_selections": total_selections
                })
            elif pii_type == "Non-PII":
                non_pii_columns.append({
                    "column_name": column_name,
                    "column_comment": column_comment,
                    "total_selections": total_selections
                })

        # Process pii_classification entries (user corrections)
        for key, entry in feedback.get("pii_classification", {}).items():
            column_name = entry.get("column_name", "")
            classification = entry.get("classification", "")
            column_comment = entry.get("column_comment", "")
            change_count = entry.get("change_count", 0)

            # User corrections have high weight
            weight = change_count * 5  # Give 5x weight to user corrections

            if classification == "PII":
                pii_columns.append({
                    "column_name": column_name,
                    "column_comment": column_comment,
                    "total_selections": weight
                })
            elif classification == "Non-PII":
                non_pii_columns.append({
                    "column_name": column_name,
                    "column_comment": column_comment,
                    "total_selections": weight
                })

        logger.info(f"Found {len(pii_columns)} PII columns and {len(non_pii_columns)} Non-PII columns")

        # Extract patterns
        pii_patterns = self._extract_patterns(pii_columns, "PII")
        non_pii_patterns = self._extract_patterns(non_pii_columns, "Non-PII")

        # Resolve conflicts (same pattern appearing in both PII and Non-PII)
        pii_patterns, non_pii_patterns = self._resolve_conflicts(pii_patterns, non_pii_patterns)

        # Convert patterns to heuristic entries
        existing_ids = [h.get("id", "") for h in heuristics["heuristics"]]
        now = datetime.now().isoformat()

        # Build a map of existing regex patterns to their heuristic entries for sample_columns update
        # 기존 regex 패턴과 해당 heuristic entry 매핑 (sample_columns 업데이트용)
        existing_pattern_map = {}
        for h in heuristics["heuristics"]:
            regex = h.get("regex", "")
            if regex:
                existing_pattern_map[regex] = h

        # Helper to add columns to existing pattern's sample_columns if matched
        # Returns list of unmatched columns that need new patterns
        def add_columns_to_existing_pattern(sample_columns: List[str], pii_type: str) -> List[str]:
            """
            Check if columns match existing patterns and add to sample_columns if not already present.
            Returns list of columns that didn't match any existing pattern.
            기존 패턴과 매칭되는 컬럼을 sample_columns에 추가하고, 매칭되지 않은 컬럼 목록 반환
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
                            # 중복 방지: sample_columns에 없는 경우에만 추가
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
            """
            Create a new heuristic pattern for exact column name match.
            Will be refined later through refinement process.
            컬럼명 전체를 패턴으로 생성 (이후 refinement 과정에서 정제됨)
            """
            col_upper = column_name.upper()
            # Create pattern that matches the exact column name as suffix
            # e.g., "FINTECH_USE_NUM" -> "_FINTECH_USE_NUM$" or "^FINTECH_USE_NUM$"
            if "_" in col_upper:
                # Use suffix pattern format for columns with underscore
                pattern = f"_{col_upper.split('_', 1)[1]}$" if col_upper.count('_') > 0 else f"^{col_upper}$"
                # Actually, use the full column name as suffix for better refinement
                pattern = f"^{col_upper}$"
            else:
                pattern = f"^{col_upper}$"

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
                "updated_at": now
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
            ("keyword", pii_patterns.get("keyword", {}))
        ]:
            for pattern, data in patterns.items():
                # Skip if pattern already exists
                # 이미 동일한 regex 패턴이 있으면 스킵 (단, sample_columns는 업데이트)
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
                # 기존 패턴과 매칭되지 않는 컬럼 확인
                unmatched = add_columns_to_existing_pattern(data["samples"], "PII")

                # If all columns matched existing patterns, skip this pattern
                if not unmatched:
                    logger.debug(f"Skipping pattern {pattern} - samples already covered by existing patterns")
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
                    "updated_at": now
                }
                heuristics["heuristics"].append(new_entry)
                existing_pattern_map[pattern] = new_entry

        # Process all PII columns that didn't match any pattern - create exact patterns
        # 어떤 패턴에도 매칭되지 않은 PII 컬럼에 대해 exact 패턴 생성
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
            ("keyword", non_pii_patterns.get("keyword", {}))
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
                    logger.debug(f"Skipping pattern {pattern} - samples already covered by existing patterns")
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
                    "updated_at": now
                }
                heuristics["heuristics"].append(new_entry)
                existing_pattern_map[pattern] = new_entry

        # Process all Non-PII columns that didn't match any pattern - create exact patterns
        # 어떤 패턴에도 매칭되지 않은 Non-PII 컬럼에 대해 exact 패턴 생성
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
        # sample_columns에서 더 세밀한 패턴을 추출하는 refinement 실행
        heuristics = self.refine_heuristics()

        return heuristics

    def _extract_patterns(
        self,
        columns: List[Dict[str, Any]],
        classification: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract suffix, prefix, and keyword patterns from column list
        컬럼 목록에서 접미사, 접두사, 키워드 패턴 추출

        Args:
            columns: List of column entries
            classification: "PII" or "Non-PII"

        Returns:
            Dictionary of extracted patterns by type
        """
        patterns = {
            "suffix": {},
            "prefix": {},
            "keyword": {}
        }

        # Collect suffixes and prefixes with weights
        suffix_counts = defaultdict(lambda: {"count": 0, "samples": [], "weight": 0})
        prefix_counts = defaultdict(lambda: {"count": 0, "samples": [], "weight": 0})
        keyword_counts = defaultdict(lambda: {"count": 0, "samples": [], "weight": 0})

        for col in columns:
            column_name = col["column_name"].upper()
            column_comment = col.get("column_comment", "")
            weight = col.get("total_selections", 1)

            # Extract suffixes using longest match from the end
            # PII 특징은 주로 suffix에 나타나므로 뒤에서부터 longest match로 패턴 추출
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

            # NOTE: keyword 패턴은 column_comment가 아닌 column_name에서만 추출
            # column_comment에서 추출된 패턴은 실제 컬럼명과 매칭되지 않으므로 제거함
            # Keyword patterns are NOT extracted from column_comment anymore
            # because they don't match actual column names

        # Filter and create final patterns
        total_columns = len(columns) if columns else 1

        # Process suffixes - longest match 우선
        # 먼저 모든 후보 suffix를 수집
        suffix_candidates = {}
        for suffix, data in suffix_counts.items():
            if data["count"] >= self.MIN_PATTERN_OCCURRENCES:
                confidence = min(1.0, data["count"] / total_columns + 0.1 * len(data["samples"]))
                if confidence >= self.MIN_CONFIDENCE:
                    suffix_candidates[suffix] = {
                        "confidence": round(confidence, 3),
                        "count": data["count"],
                        "samples": data["samples"][:5]
                    }

        # Longest match 우선: 동일한 sample 집합을 가진 짧은 패턴 제거
        # 예: "_NUM$"과 "_ACCOUNT_NUM$"이 같은 컬럼들에서 나왔다면 "_ACCOUNT_NUM$"만 유지
        suffixes_to_remove = set()
        suffix_list = sorted(suffix_candidates.keys(), key=len, reverse=True)  # 긴 것부터

        for i, longer_suffix in enumerate(suffix_list):
            longer_samples = set(suffix_candidates[longer_suffix]["samples"])
            for shorter_suffix in suffix_list[i+1:]:
                if shorter_suffix in suffixes_to_remove:
                    continue
                shorter_samples = set(suffix_candidates[shorter_suffix]["samples"])
                # 짧은 패턴의 샘플이 긴 패턴의 샘플에 모두 포함되면 짧은 패턴 제거
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
                        "samples": data["samples"][:5]
                    }

        # Process keywords
        for keyword, data in keyword_counts.items():
            if data["count"] >= self.MIN_PATTERN_OCCURRENCES:
                confidence = min(1.0, data["count"] / total_columns + 0.1 * len(data["samples"]))
                if confidence >= self.MIN_CONFIDENCE:
                    patterns["keyword"][keyword] = {
                        "confidence": round(confidence, 3),
                        "count": data["count"],
                        "samples": data["samples"][:5]
                    }

        logger.info(f"Extracted patterns for {classification}: "
                   f"{len(patterns['suffix'])} suffixes, "
                   f"{len(patterns['prefix'])} prefixes, "
                   f"{len(patterns['keyword'])} keywords")

        return patterns

    def _resolve_conflicts(
        self,
        pii_patterns: Dict[str, Dict[str, Any]],
        non_pii_patterns: Dict[str, Dict[str, Any]]
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        Resolve conflicts when same pattern appears in both PII and Non-PII
        동일한 패턴이 PII와 Non-PII 모두에 나타날 때 충돌 해결

        Strategy: Keep pattern only in the category with higher confidence/count
        전략: 더 높은 신뢰도/빈도를 가진 카테고리에서만 패턴 유지

        Args:
            pii_patterns: PII patterns dictionary
            non_pii_patterns: Non-PII patterns dictionary

        Returns:
            Tuple of resolved (pii_patterns, non_pii_patterns)
        """
        for pattern_type in ["suffix", "prefix", "keyword"]:
            pii_set = set(pii_patterns.get(pattern_type, {}).keys())
            non_pii_set = set(non_pii_patterns.get(pattern_type, {}).keys())
            conflicts = pii_set & non_pii_set

            for pattern in conflicts:
                pii_conf = pii_patterns[pattern_type][pattern]["confidence"]
                non_pii_conf = non_pii_patterns[pattern_type][pattern]["confidence"]

                if pii_conf >= non_pii_conf:
                    del non_pii_patterns[pattern_type][pattern]
                    logger.info(f"Conflict resolved: {pattern_type} '{pattern}' kept as PII (conf: {pii_conf} vs {non_pii_conf})")
                else:
                    del pii_patterns[pattern_type][pattern]
                    logger.info(f"Conflict resolved: {pattern_type} '{pattern}' kept as Non-PII (conf: {non_pii_conf} vs {pii_conf})")

        return pii_patterns, non_pii_patterns

    def _deduplicate_suffix_patterns_by_samples(
        self,
        heuristics: Dict[str, Any],
        now: str
    ) -> Dict[str, Any]:
        """
        Remove redundant suffix patterns that have identical or subset sample_columns.
        동일하거나 부분집합인 sample_columns를 가진 중복 suffix 패턴 제거

        When two suffix patterns have the same sample_columns, keep only the longer (more specific) pattern.
        두 suffix 패턴이 동일한 sample_columns를 가지면, 더 긴(더 구체적인) 패턴만 유지

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
            if (h.get("pattern_type") == "suffix" and
                h.get("source") == "auto_learned" and
                h.get("enabled", True)):
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

                for shorter_pattern in patterns[i + 1:]:
                    shorter_regex = shorter_pattern.get("regex", "")
                    shorter_samples = set(c.upper() for c in shorter_pattern.get("sample_columns", []))
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
                h for h in heuristics["heuristics"]
                if h.get("id") not in patterns_to_remove
            ]
            logger.info(f"Removed {len(patterns_to_remove)} redundant suffix patterns via deduplication")

        return heuristics

    def refine_heuristics(self) -> Dict[str, Any]:
        """
        Refine existing heuristics by extracting more specific patterns from sample_columns
        기존 휴리스틱의 sample_columns를 분석하여 더 세밀한 패턴으로 분리

        For example, if "_NUM$" has samples ["CNTR_ACCOUNT_NUM", "FINTECH_USE_NUM", "RECV_CLIENT_ACCOUNT_NUM"],
        this method will create more specific patterns like "_ACCOUNT_NUM$" and "_USE_NUM$".

        예: "_NUM$"에 ["CNTR_ACCOUNT_NUM", "FINTECH_USE_NUM", "RECV_CLIENT_ACCOUNT_NUM"]가 있으면
        "_ACCOUNT_NUM$", "_USE_NUM$" 같은 더 세밀한 패턴을 생성

        Returns:
            Updated heuristics dictionary
        """
        logger.info("Starting heuristics refinement...")
        heuristics = self._load_heuristics()
        now = datetime.now().isoformat()

        # Get existing IDs for new pattern generation
        existing_ids = [h.get("id", "") for h in heuristics["heuristics"]]

        # Track new patterns to add
        new_patterns_to_add = []

        # First, collect all exact patterns and try to merge them into suffix patterns
        # exact 패턴들을 수집하여 공통 suffix 패턴으로 병합 시도
        exact_patterns_by_pii_type = {"PII": [], "Non-PII": []}
        # Map: column -> exact pattern entry (for removing merged columns)
        column_to_exact_pattern = {}

        for h in heuristics["heuristics"]:
            if h.get("pattern_type") == "exact" and h.get("enabled", True):
                pii_type = h.get("pii_type", "PII")
                sample_cols = h.get("sample_columns", [])
                if sample_cols:
                    exact_patterns_by_pii_type[pii_type].extend(sample_cols)
                    for col in sample_cols:
                        column_to_exact_pattern[col.upper()] = h

        # Track columns that were merged into new suffix patterns
        # 새로운 suffix 패턴으로 병합된 컬럼 추적
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
                    # Check if this suffix already exists
                    existing_pattern = next(
                        (h2 for h2 in heuristics["heuristics"]
                         if h2.get("regex") == suffix and h2.get("pii_type") == pii_type),
                        None
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
                        # exact 패턴 병합 시 confidence는 매칭된 컬럼 수에 기반
                        # MIN_PATTERN_OCCURRENCES 이상이면 유효한 패턴으로 간주
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
                                "updated_at": now
                            }
                            new_patterns_to_add.append(new_entry)
                            # Track merged columns
                            for col in matched_cols:
                                merged_columns.add(col.upper())
                            logger.info(f"Merged exact patterns into suffix: {suffix} "
                                       f"({len(matched_cols)} columns: {', '.join(matched_cols[:3])})")

        # Remove merged columns from their original exact patterns
        # 병합된 컬럼을 원본 exact 패턴에서 제거
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
        # 남은 샘플이 없는 exact 패턴 제거
        if exact_patterns_to_remove:
            heuristics["heuristics"] = [
                h for h in heuristics["heuristics"]
                if h.get("id") not in exact_patterns_to_remove
            ]
            logger.info(f"Removed {len(exact_patterns_to_remove)} empty exact patterns after merging")

        # Process each suffix heuristic for refinement
        for h in heuristics["heuristics"]:
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
            # 샘플 컬럼들에서 가능한 모든 더 긴 suffix 추출
            suffix_groups = defaultdict(list)

            for col in sample_columns:
                col_upper = col.upper()
                if "_" not in col_upper:
                    continue

                parts = col_upper.split("_")
                # Generate suffixes longer than current pattern
                # 현재 패턴보다 긴 suffix만 생성
                for i in range(1, len(parts)):
                    suffix = "_" + "_".join(parts[i:]) + "$"
                    # Only consider suffixes longer than current
                    if len(suffix) > len(current_regex):
                        suffix_groups[suffix].append(col)

            # Find suffixes that can form independent patterns
            # 독립적인 패턴을 형성할 수 있는 suffix 찾기
            refined_patterns = {}
            for suffix, columns in suffix_groups.items():
                if len(columns) >= self.MIN_PATTERN_OCCURRENCES:
                    # Check if this suffix already exists
                    existing_pattern = next(
                        (h2 for h2 in heuristics["heuristics"]
                         if h2.get("regex") == suffix and h2.get("pii_type") == pii_type),
                        None
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
            # Longest match 원칙을 사용하여 최적의 refined 패턴 선택
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
                            "updated_at": now
                        }
                        new_patterns_to_add.append(new_entry)

                        logger.info(f"Refined pattern: {current_regex} -> {suffix} "
                                   f"({len(columns)} columns: {', '.join(columns[:3])})")

                # Update parent pattern: remove columns that are now in refined patterns
                # 부모 패턴에서 refined 패턴으로 이동한 컬럼 제거
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
        # 동일한 sample_columns를 가진 suffix 패턴들에 대해 longest match 중복 제거
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
        self,
        column_name: str,
        column_comment: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Classify a column using learned heuristic patterns
        학습된 휴리스틱 패턴을 사용하여 컬럼 분류

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
        enabled_heuristics = [
            h for h in heuristics.get("heuristics", [])
            if h.get("enabled", True)
        ]
        enabled_heuristics.sort(key=lambda h: h.get("priority", 0), reverse=True)

        column_upper = column_name.upper()

        for h in enabled_heuristics:
            try:
                pattern = re.compile(h["regex"], re.IGNORECASE)

                # Check column name only (NOT column_comment)
                # 컬럼명만 확인 (column_comment는 실제 매칭에 사용하지 않음)
                if pattern.search(column_upper):
                    return {
                        "classification": h.get("pii_type", "PII"),
                        "match_type": h.get("pattern_type", "unknown"),
                        "pattern": h.get("regex", ""),
                        "confidence": h.get("confidence", 0.8),
                        "samples": h.get("sample_columns", []),
                        "source": h.get("source", "unknown"),
                        "heuristic_id": h.get("id", "")
                    }
            except re.error:
                # Skip invalid regex patterns
                continue

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about learned patterns
        학습된 패턴에 대한 통계 가져오기

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
                "exact": len([h for h in pii_auto if h.get("pattern_type") == "exact"])
            },
            "non_pii_patterns": {
                "suffixes": len([h for h in non_pii_auto if h.get("pattern_type") == "suffix"]),
                "prefixes": len([h for h in non_pii_auto if h.get("pattern_type") == "prefix"]),
                "keywords": len([h for h in non_pii_auto if h.get("pattern_type") == "keyword"]),
                "exact": len([h for h in non_pii_auto if h.get("pattern_type") == "exact"])
            },
            "last_updated": heuristics.get("statistics", {}).get("last_updated"),
            "last_auto_learn": heuristics.get("statistics", {}).get("last_auto_learn")
        }
