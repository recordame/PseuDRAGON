"""
Policy Domain-Specific Language (DSL) for PseuDRAGON Framework
PseuDRAGON 프레임워크를 위한 정책 도메인 특화 언어 (DSL)

Implements the structured policy language described in the paper:
논문에 설명된 구조화된 정책 언어를 구현합니다:

Π = {(t, c, a, θ)}
where / 여기서:
  t = table name / 테이블 이름
  c = column name / 컬럼 이름
  a = action (KEEP, DELETE, HASH, GENERALIZE, etc.) / 액션
  θ = action parameters / 액션 매개변수

This module defines the core data structures for representing
pseudonymization policies at the column and table level.
이 모듈은 컬럼 및 테이블 수준에서 가명처리 정책을 표현하기 위한
핵심 데이터 구조를 정의합니다.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ActionType(str, Enum):
    """
    Enumeration of pseudonymization actions
    가명화 액션 열거형

    These correspond to the 'a' in the policy tuple (t, c, a, θ)
    정책 튜플 (t, c, a, θ)의 'a'에 해당합니다
    """
    KEEP = "KEEP"
    DELETE = "DELETE"
    HASH = "HASH"
    MASK = "MASK"
    TOKENIZE = "TOKENIZE"
    GENERALIZE = "GENERALIZE"
    ROUND = "ROUND"
    ENCRYPT = "ENCRYPT"

    @classmethod
    def from_string(cls, action_str: str) -> 'ActionType':
        """
        Create ActionType from string
        문자열에서 ActionType 생성
        
        Args:
            action_str: Action type as string
                       문자열로 된 액션 타입
        
        Returns:
            ActionType enum value
            ActionType 열거형 값
        
        Raises:
            ValueError: If action string is invalid
                       액션 문자열이 유효하지 않을 때
        """
        try:
            return cls(action_str.upper())
        except ValueError:
            raise ValueError(f"Invalid action type: {action_str}")


@dataclass
class PolicyAction:
    """
    Single policy action for a column
    컬럼에 대한 단일 정책 액션

    Represents a specific pseudonymization action with its parameters,
    rationale, and legal evidence.
    매개변수, 근거 및 법적 증거와 함께 특정 가명처리 액션을 나타냅니다.

    Attributes:
        action: Type of action to perform
               수행할 액션 유형
        parameters: Action-specific parameters (θ in the paper)
                   액션별 매개변수 (논문의 θ)
        rationale: Human-readable explanation
                  사람이 읽을 수 있는 설명
        legal_evidence: Citation from legal knowledge base
                       법적 지식 베이스의 인용
        code_snippet: Python code implementation for this action
                     이 액션에 대한 Python 코드 구현
    """
    action: ActionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    legal_evidence: str = ""
    code_snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization
        직렬화를 위해 딕셔너리로 변환

        Returns:
            Dictionary representation
            딕셔너리 표현
        """
        return {'action': self.action.value, 'parameters': self.parameters, 'rationale': self.rationale, 'legal_evidence': self.legal_evidence, 'code_snippet': self.code_snippet}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PolicyAction':
        """
        Create from dictionary
        딕셔너리에서 생성

        Args:
            data: Dictionary containing action data
                 액션 데이터를 포함하는 딕셔너리

        Returns:
            PolicyAction instance
            PolicyAction 인스턴스

        Raises:
            ValueError: If action type is invalid
                       액션 타입이 유효하지 않을 때
        """
        return cls(
            action=ActionType(data['action']),
            parameters=data.get('parameters', {}),
            rationale=data.get('rationale', ''),
            legal_evidence=data.get('legal_evidence', ''),
            code_snippet=data.get('code_snippet', '')
        )

    def __repr__(self) -> str:
        """
        String representation
        문자열 표현
        """
        params_str = f", params={self.parameters}" if self.parameters else ""
        return f"PolicyAction({self.action.value}{params_str})"


@dataclass
class ColumnPolicy:
    """
    Policy for a single column
    단일 컬럼에 대한 정책
    
    Represents the complete policy for a column, including its PII type,
    chosen action, and alternative actions for human-in-the-loop review.
    PII 유형, 선택된 액션 및 사람 개입 검토를 위한 대안 액션을 포함하여
    컬럼에 대한 완전한 정책을 나타냅니다.
    
    Attributes:
        column_name: Name of the column
                    컬럼 이름
        pii_type: Type of PII (PII or Non-PII)
                 PII 유형 (식별자, 비PII)
        is_pii: Whether this column contains PII
               이 컬럼이 PII를 포함하는지 여부
        action: The chosen action for this column
               이 컬럼에 대해 선택된 액션
        candidate_actions: Alternative actions (for HITL)
                          대안 액션 (사람 개입용)
    """
    column_name: str
    pii_type: str
    is_pii: bool
    action: PolicyAction
    candidate_actions: List[PolicyAction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary
        딕셔너리로 변환
        
        Returns:
            Dictionary representation
            딕셔너리 표현
        """
        return {'column_name': self.column_name, 'pii_type': self.pii_type, 'is_pii': self.is_pii, 'action': self.action.to_dict(), 'candidate_actions': [a.to_dict() for a in self.candidate_actions]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ColumnPolicy':
        """
        Create from dictionary
        딕셔너리에서 생성
        
        Args:
            data: Dictionary containing column policy data
                 컬럼 정책 데이터를 포함하는 딕셔너리
        
        Returns:
            ColumnPolicy instance
            ColumnPolicy 인스턴스
        """
        return cls(
            column_name=data['column_name'],
            pii_type=data['pii_type'],
            is_pii=data['is_pii'],
            action=PolicyAction.from_dict(data['action']),
            candidate_actions=[PolicyAction.from_dict(a) for a in data.get('candidate_actions', [])]
        )

    def add_candidate_action(self, action: PolicyAction) -> None:
        """
        Add a candidate action
        후보 액션 추가
        
        Args:
            action: PolicyAction to add as candidate
                   후보로 추가할 PolicyAction
        """
        self.candidate_actions.append(action)

    def __repr__(self) -> str:
        """
        String representation
        문자열 표현
        """
        return f"ColumnPolicy({self.column_name}, {self.pii_type}, {self.action.action.value})"


@dataclass
class Policy:
    """
    Complete policy for a table (Π in the paper)
    테이블에 대한 완전한 정책 (논문의 Π)

    Represents: Π = {(t, c, a, θ)}

    This is the top-level policy object that contains all column policies
    for a specific table, along with metadata and preferred method.
    이것은 특정 테이블에 대한 모든 컬럼 정책을 메타데이터 및 선호 기법과 함께
    포함하는 최상위 정책 객체입니다.

    Attributes:
        table_name: Name of the table (t in the paper)
                   테이블 이름 (논문의 t)
        columns: Dictionary mapping column names to their policies
                컬럼 이름을 정책에 매핑하는 딕셔너리
        preferred_method: User's preferred pseudonymization method
                           사용자가 선호하는 가명화 기법
        metadata: Additional metadata
                 추가 메타데이터
    """
    table_name: str
    columns: Dict[str, ColumnPolicy] = field(default_factory=dict)
    preferred_method: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_column_policy(self, col_policy: ColumnPolicy) -> None:
        """
        Add a column policy
        컬럼 정책 추가

        Args:
            col_policy: ColumnPolicy to add
                       추가할 ColumnPolicy
        """
        self.columns[col_policy.column_name] = col_policy

    def get_column_policy(self, column_name: str) -> Optional[ColumnPolicy]:
        """
        Get policy for a specific column
        특정 컬럼에 대한 정책 가져오기

        Args:
            column_name: Name of the column
                        컬럼 이름

        Returns:
            ColumnPolicy if found, None otherwise
            찾으면 ColumnPolicy, 없으면 None
        """
        return self.columns.get(column_name)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert policy to dictionary format
        정책을 딕셔너리 형식으로 변환

        Returns:
            Dictionary representation of the policy
            정책의 딕셔너리 표현
        """
        return {
            "table_name": self.table_name,
            "preferred_method": self.preferred_method,
            "columns": {col_name: col_policy.to_dict() for col_name, col_policy in self.columns.items()},
            "metadata": self.metadata
        }

    def get_pii_columns(self) -> List[str]:
        """
        Get list of PII column names
        PII 컬럼 이름 목록 가져오기
        
        Returns:
            List of column names that contain PII
            PII를 포함하는 컬럼 이름 목록
        """
        return [col for col, policy in self.columns.items() if policy.is_pii]

    def get_non_pii_columns(self) -> List[str]:
        """
        Get list of non-PII column names
        비PII 컬럼 이름 목록 가져오기
        
        Returns:
            List of column names that don't contain PII
            PII를 포함하지 않는 컬럼 이름 목록
        """
        return [col for col, policy in self.columns.items() if not policy.is_pii]

    def get_columns_by_action(self, action_type: ActionType) -> List[str]:
        """
        Get columns with a specific action type
        특정 액션 타입을 가진 컬럼 가져오기
        
        Args:
            action_type: ActionType to filter by
                        필터링할 ActionType
        
        Returns:
            List of column names with the specified action
            지정된 액션을 가진 컬럼 이름 목록
        """
        return [col for col, policy in self.columns.items() if policy.action.action == action_type]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization
        직렬화를 위해 딕셔너리로 변환
        
        Returns:
            Dictionary representation
            딕셔너리 표현
        """
        return {'table_name': self.table_name, 'columns': {name: pol.to_dict() for name, pol in self.columns.items()}, 'preferred_method': self.preferred_method, 'metadata': self.metadata}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Policy':
        """
        Create from dictionary
        딕셔너리에서 생성
        
        Args:
            data: Dictionary containing policy data
                 정책 데이터를 포함하는 딕셔너리
        
        Returns:
            Policy instance
            Policy 인스턴스
        """
        policy = cls(table_name=data['table_name'], preferred_method=data.get('preferred_method', ''), metadata=data.get('metadata', {}))
        for col_name, col_data in data.get('columns', {}).items():
            policy.columns[col_name] = ColumnPolicy.from_dict(col_data)
        return policy

    def __repr__(self) -> str:
        """
        String representation
        문자열 표현
        """
        pii_count = len(self.get_pii_columns())
        return f"Policy(table={self.table_name}, columns={len(self.columns)}, pii={pii_count})"
