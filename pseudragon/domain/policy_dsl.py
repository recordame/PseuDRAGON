"""
Policy Domain-Specific Language (DSL) for PseuDRAGON Framework

Implements the structured policy language described in the paper:

Π = {(t, c, a, θ)}
where
  t = table name
  c = column name
  a = action (KEEP, DELETE, HASH, GENERALIZE, etc.)
  θ = action parameters

This module defines the core data structures for representing
pseudonymization policies at the column and table level.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ActionType(str, Enum):
    """
    Enumeration of pseudonymization actions

    These correspond to the 'a' in the policy tuple (t, c, a, θ)
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
        
        Args:
            action_str: Action type as string
        
        Returns:
            ActionType enum value
        
        Raises:
            ValueError: If action string is invalid
        """
        try:
            return cls(action_str.upper())
        except ValueError:
            raise ValueError(f"Invalid action type: {action_str}")


@dataclass
class PolicyAction:
    """
    Single policy action for a column

    Represents a specific pseudonymization action with its parameters,
    rationale, and legal evidence.

    Attributes:
        action: Type of action to perform
        parameters: Action-specific parameters (θ in the paper)
        rationale: Human-readable explanation
        legal_evidence: Citation from legal knowledge base
        code_snippet: Python code implementation for this action
    """
    action: ActionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    legal_evidence: str = ""
    code_snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization

        Returns:
            Dictionary representation
        """
        return {'action': self.action.value, 'parameters': self.parameters, 'rationale': self.rationale, 'legal_evidence': self.legal_evidence, 'code_snippet': self.code_snippet}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PolicyAction':
        """
        Create from dictionary

        Args:
            data: Dictionary containing action data

        Returns:
            PolicyAction instance

        Raises:
            ValueError: If action type is invalid
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
        """
        params_str = f", params={self.parameters}" if self.parameters else ""
        return f"PolicyAction({self.action.value}{params_str})"


@dataclass
class ColumnPolicy:
    """
    Policy for a single column
    
    Represents the complete policy for a column, including its PII type,
    chosen action, and alternative actions for human-in-the-loop review.
    
    Attributes:
        column_name: Name of the column
        pii_type: Type of PII (PII or Non-PII)
        is_pii: Whether this column contains PII
        action: The chosen action for this column
        candidate_actions: Alternative actions (for HITL)
    """
    column_name: str
    pii_type: str
    is_pii: bool
    action: PolicyAction
    candidate_actions: List[PolicyAction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary
        
        Returns:
            Dictionary representation
        """
        return {'column_name': self.column_name, 'pii_type': self.pii_type, 'is_pii': self.is_pii, 'action': self.action.to_dict(), 'candidate_actions': [a.to_dict() for a in self.candidate_actions]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ColumnPolicy':
        """
        Create from dictionary
        
        Args:
            data: Dictionary containing column policy data
        
        Returns:
            ColumnPolicy instance
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
        
        Args:
            action: PolicyAction to add as candidate
        """
        self.candidate_actions.append(action)

    def __repr__(self) -> str:
        """
        String representation
        """
        return f"ColumnPolicy({self.column_name}, {self.pii_type}, {self.action.action.value})"


@dataclass
class Policy:
    """
    Complete policy for a table (Π in the paper)

    Represents: Π = {(t, c, a, θ)}

    This is the top-level policy object that contains all column policies
    for a specific table, along with metadata and preferred method.

    Attributes:
        table_name: Name of the table (t in the paper)
        columns: Dictionary mapping column names to their policies
        preferred_method: User's preferred pseudonymization method
        metadata: Additional metadata
    """
    table_name: str
    columns: Dict[str, ColumnPolicy] = field(default_factory=dict)
    preferred_method: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_column_policy(self, col_policy: ColumnPolicy) -> None:
        """
        Add a column policy

        Args:
            col_policy: ColumnPolicy to add
        """
        self.columns[col_policy.column_name] = col_policy

    def get_column_policy(self, column_name: str) -> Optional[ColumnPolicy]:
        """
        Get policy for a specific column

        Args:
            column_name: Name of the column

        Returns:
            ColumnPolicy if found, None otherwise
        """
        return self.columns.get(column_name)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert policy to dictionary format

        Returns:
            Dictionary representation of the policy
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
        
        Returns:
            List of column names that contain PII
        """
        return [col for col, policy in self.columns.items() if policy.is_pii]

    def get_non_pii_columns(self) -> List[str]:
        """
        Get list of non-PII column names
        
        Returns:
            List of column names that don't contain PII
        """
        return [col for col, policy in self.columns.items() if not policy.is_pii]

    def get_columns_by_action(self, action_type: ActionType) -> List[str]:
        """
        Get columns with a specific action type
        
        Args:
            action_type: ActionType to filter by
        
        Returns:
            List of column names with the specified action
        """
        return [col for col, policy in self.columns.items() if policy.action.action == action_type]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization
        
        Returns:
            Dictionary representation
        """
        return {'table_name': self.table_name, 'columns': {name: pol.to_dict() for name, pol in self.columns.items()}, 'preferred_method': self.preferred_method, 'metadata': self.metadata}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Policy':
        """
        Create from dictionary
        
        Args:
            data: Dictionary containing policy data
        
        Returns:
            Policy instance
        """
        policy = cls(table_name=data['table_name'], preferred_method=data.get('preferred_method', ''), metadata=data.get('metadata', {}))
        for col_name, col_data in data.get('columns', {}).items():
            policy.columns[col_name] = ColumnPolicy.from_dict(col_data)
        return policy

    def __repr__(self) -> str:
        """
        String representation
        """
        pii_count = len(self.get_pii_columns())
        return f"Policy(table={self.table_name}, columns={len(self.columns)}, pii={pii_count})"
