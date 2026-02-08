"""
Heuristic Manager
휴리스틱 관리자

Manages heuristic rules for PII detection.
PII 탐지를 위한 휴리스틱 규칙을 관리합니다.

This module provides:
이 모듈은 다음을 제공합니다:
- CRUD operations for heuristic patterns / 휴리스틱 패턴의 CRUD 작업
- Regex validation / 정규식 검증
- Priority-based pattern matching / 우선순위 기반 패턴 매칭
- Integration with Stage 1 PII detection / Stage 1 PII 탐지와의 통합
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class HeuristicManager:
    """
    Manager for heuristic rules
    휴리스틱 규칙 관리자
    
    Handles loading, saving, and querying heuristic patterns.
    휴리스틱 패턴의 로딩, 저장 및 쿼리를 처리합니다.
    """

    def __init__(self, heuristics_file: Optional[str] = None):
        """
        Initialize heuristic manager
        휴리스틱 관리자 초기화
        
        Args:
            heuristics_file: Path to heuristics.json file
                           heuristics.json 파일 경로
        """
        if heuristics_file is None:
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

    def load(self) -> None:
        """
        Load heuristics from file
        파일에서 휴리스틱 로드
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
        v1.0 휴리스틱 형식을 v2.0으로 마이그레이션

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
        파일에 휴리스틱 저장
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
        모든 휴리스틱 가져오기
        
        Args:
            enabled_only: Only return enabled heuristics
                         활성화된 휴리스틱만 반환
        
        Returns:
            List of heuristic dictionaries
            휴리스틱 딕셔너리 목록
        """
        heuristics = self.heuristics.get("heuristics", [])

        if enabled_only:
            heuristics = [h for h in heuristics if h.get("enabled", True)]

        # Sort by priority (descending)
        return sorted(heuristics, key=lambda h: h.get("priority", 0), reverse=True)

    def get_by_id(self, heuristic_id: str) -> Optional[Dict[str, Any]]:
        """
        Get heuristic by ID
        ID로 휴리스틱 가져오기
        
        Args:
            heuristic_id: Heuristic ID
                         휴리스틱 ID
        
        Returns:
            Heuristic dictionary or None
            휴리스틱 딕셔너리 또는 None
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
        새 휴리스틱 추가

        Args:
            name: Heuristic name
                 휴리스틱 이름
            regex: Regular expression pattern
                  정규식 패턴
            pii_type: PII type (PII or Non-PII)
                     PII 유형
            rationale: Reason for this heuristic
                      휴리스틱의 이유
            priority: Priority (higher = more important)
                     우선순위 (높을수록 중요)
            enabled: Whether heuristic is enabled
                    휴리스틱 활성화 여부
            source: Source of heuristic ("manual" or "auto_learned")
                   휴리스틱 출처 ("manual" 또는 "auto_learned")
            pattern_type: Type of pattern ("suffix", "prefix", "keyword", "custom")
                         패턴 유형 ("suffix", "prefix", "keyword", "custom")

        Returns:
            Created heuristic dictionary
            생성된 휴리스틱 딕셔너리

        Raises:
            ValueError: If regex is invalid
                       정규식이 유효하지 않은 경우
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
        기존 휴리스틱 업데이트
        
        Args:
            heuristic_id: Heuristic ID to update
                         업데이트할 휴리스틱 ID
            **kwargs: Fields to update
                     업데이트할 필드
        
        Returns:
            Updated heuristic or None if not found
            업데이트된 휴리스틱 또는 찾을 수 없으면 None
        
        Raises:
            ValueError: If regex is invalid
                       정규식이 유효하지 않은 경우
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
        휴리스틱 삭제
        
        Args:
            heuristic_id: Heuristic ID to delete
                         삭제할 휴리스틱 ID
        
        Returns:
            True if deleted, False if not found
            삭제되면 True, 찾을 수 없으면 False
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
        샘플 문자열에 대해 정규식 패턴 테스트
        
        Args:
            regex: Regular expression pattern to test
                  테스트할 정규식 패턴
            test_strings: List of strings to test against
                         테스트할 문자열 목록
        
        Returns:
            Dictionary with test results
            테스트 결과가 포함된 딕셔너리
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
        컬럼 이름에 대한 일치하는 휴리스틱 찾기
        
        Args:
            column_name: Column name to match
                        일치시킬 컬럼 이름
        
        Returns:
            Best matching heuristic or None
            최적 일치 휴리스틱 또는 None
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
        정규식 패턴 검증
        
        Args:
            regex: Regular expression pattern
                  정규식 패턴
        
        Returns:
            True if valid, False otherwise
            유효하면 True, 그렇지 않으면 False
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
        고유한 휴리스틱 ID 생성
        
        Args:
            existing_ids: List of existing IDs
                         기존 ID 목록
        
        Returns:
            New unique ID
            새로운 고유 ID
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
        휴리스틱 패턴에 대한 통계 가져오기

        Returns:
            Dictionary with pattern statistics
            패턴 통계가 포함된 딕셔너리
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
        소스별로 필터링된 휴리스틱 가져오기

        Args:
            source: Source to filter by ("manual" or "auto_learned")
                   필터링할 소스 ("manual" 또는 "auto_learned")
            enabled_only: Only return enabled heuristics
                         활성화된 휴리스틱만 반환

        Returns:
            List of heuristic dictionaries
            휴리스틱 딕셔너리 목록
        """
        heuristics = self.heuristics.get("heuristics", [])
        filtered = [h for h in heuristics if h.get("source") == source]

        if enabled_only:
            filtered = [h for h in filtered if h.get("enabled", True)]

        # Sort by priority (descending)
        return sorted(filtered, key=lambda h: h.get("priority", 0), reverse=True)
