"""
Policy Validators
정책 검증기

Implements consistency checks for policy validation.
정책 검증을 위한 일관성 검사를 구현합니다.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pseudragon.domain.policy_dsl import ActionType, Policy


@dataclass
class ValidationViolation:
    """
    Represents a policy validation violation
    정책 검증 위반을 나타냄
    
    Attributes:
        severity: Violation severity level ('error', 'warning', 'info')
                 위반 심각도 수준
        message: Description of the violation
                위반에 대한 설명
        column: Column name associated with the violation (if applicable)
               위반과 관련된 컬럼 이름 (해당하는 경우)
        suggestion: Suggested fix for the violation
                   위반에 대한 수정 제안
    """
    severity: str
    message: str
    column: Optional[str] = None
    suggestion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary
        딕셔너리로 변환
        
        Returns:
            Dictionary representation
            딕셔너리 표현
        """
        return {'severity': self.severity, 'message': self.message, 'column': self.column, 'suggestion': self.suggestion}

    def __repr__(self) -> str:
        """
        String representation
        문자열 표현
        """
        col_str = f" [{self.column}]" if self.column else ""
        return f"{self.severity.upper()}{col_str}: {self.message}"

    def is_error(self) -> bool:
        """
        Check if violation is an error
        위반이 오류인지 확인
        """
        return self.severity == 'error'

    def is_warning(self) -> bool:
        """
        Check if violation is a warning
        위반이 경고인지 확인
        """
        return self.severity == 'warning'


class LegalComplianceChecker:
    """
    Legal Compliance Checker
    법적 준수 검사기
    
    Verifies that all PII columns are processed according to PIPA/GDPR.
    모든 PII 컬럼이 PIPA/GDPR에 따라 처리되는지 확인합니다.
    """

    ALLOWED_PII_ACTIONS = {ActionType.DELETE, ActionType.HASH, ActionType.MASK, ActionType.TOKENIZE, ActionType.GENERALIZE, ActionType.ENCRYPT}

    @staticmethod
    def check(policy: Policy) -> list[ValidationViolation]:
        """
        Check legal compliance
        법적 준수 검사

        Args:
            policy: Policy to validate
                   검증할 정책

        Returns:
            List of violations
            위반 목록
        """
        violations = []

        for col_name, col_policy in policy.columns.items():
            if col_policy.is_pii and col_policy.action.action == ActionType.KEEP:
                violations.append(
                    ValidationViolation(
                        severity='error',
                        message=f"PII column '{col_name}' cannot be kept without processing",
                        column=col_name,
                        suggestion="Apply HASH, GENERALIZE, MASK, TOKENIZE, ENCRYPT, or DELETE action"
                    )
                )

        return violations


class ConsistencyChecker:
    """
    Consistency Checker
    일관성 검사기
    
    Ensures no logical contradictions in the policy.
    정책에 논리적 모순이 없는지 확인합니다.
    
    Example: A column is both deleted and used in aggregation.
    예: 컬럼이 삭제되면서 동시에 집계에 사용됨.
    """

    @staticmethod
    def check(policy: Policy) -> List[ValidationViolation]:
        """
        Check consistency
        일관성 검사
        
        Args:
            policy: Policy to validate
                   검증할 정책
        
        Returns:
            List of violations
            위반 목록
        """
        violations = []

        deleted_columns = ConsistencyChecker._get_deleted_columns(policy)
        violations.extend(ConsistencyChecker._check_group_by_consistency(policy, deleted_columns))

        return violations

    @staticmethod
    def _get_deleted_columns(policy: Policy) -> set:
        """
        Get set of deleted columns
        삭제된 컬럼 집합 가져오기
        """
        return {col for col, pol in policy.columns.items() if pol.action.action == ActionType.DELETE}

    @staticmethod
    def _check_group_by_consistency(policy: Policy, deleted_columns: set) -> List[ValidationViolation]:
        """
        Check that group_by columns are not deleted
        group_by 컬럼이 삭제되지 않았는지 확인
        """
        violations = []

        for col_name, col_policy in policy.columns.items():
            params = col_policy.action.parameters

            if 'group_by' in params:
                group_cols = params['group_by']
                if isinstance(group_cols, list):
                    for gcol in group_cols:
                        if gcol in deleted_columns:
                            violations.append(
                                ValidationViolation(
                                    severity='error',
                                    message=f"Column '{col_name}' groups by deleted column '{gcol}'",
                                    column=col_name,
                                    suggestion=f"Do not delete '{gcol}' or remove it from group_by"
                                )
                            )

        return violations


class UtilityPreservationChecker:
    """
    Utility Preservation Checker
    유용성 보존 검사기
    
    Verifies that preferred methods can still be achieved after pseudonymization.
    선호 방법이 가명화 후에도 달성 가능한지 확인합니다.

    Example: Preferred method mentions "user-level analysis" but user_id is deleted.
    예시: 선호 방법이 "사용자 수준 분석"을 언급하지만 user_id가 삭제됨.
    """

    OBJECTIVE_KEYWORDS = {
        'user': ['user_id', 'user', 'customer_id', 'customer'],
        'age': ['age', 'birth_date', 'birthday', 'dob'],
        'location': ['address', 'city', 'region', 'location'],
        'time': ['date', 'time', 'timestamp', 'created_at']
    }

    @staticmethod
    def check(policy: Policy, preferred_method: str) -> List[ValidationViolation]:
        """
        Check if preferred method can still be achieved.
        선호 방법이 달성 가능한지 확인합니다.

        Args:
            policy: Policy to check
            preferred_method: User's preferred pseudonymization method

        Returns:
            List of violations
        """
        violations = []

        if not preferred_method:
            return violations

        preferred_method_lower = preferred_method.lower()
        deleted_columns = {col for col, pol in policy.columns.items() if pol.action.action == ActionType.DELETE}

        for keyword, related_cols in UtilityPreservationChecker.OBJECTIVE_KEYWORDS.items():
            if keyword in preferred_method_lower:
                deleted_related = [col for col in deleted_columns if any(rc in col.lower() for rc in related_cols)]
                if deleted_related:
                    violations.append(
                        ValidationViolation(
                            severity='warning',
                            message=f"Preferred method mentions '{keyword}' but related columns are deleted: {deleted_related}",
                            suggestion=f"Consider using HASH or MASK instead of DELETE"
                        )
                    )

        return violations


class PrivacyMetricsChecker:
    """
    Privacy Metrics Checker
    프라이버시 메트릭 검사기
    
    Estimates k-anonymity and re-identification risk.
    k-익명성과 재식별 위험을 추정합니다.
    
    Note: This is a simplified heuristic. Full implementation would
    require actual data analysis.
    참고: 이것은 단순화된 휴리스틱입니다. 완전한 구현은
    실제 데이터 분석이 필요합니다.
    """

    DIRECT_IDENTIFIER_TYPES = {'PII'}
    SAFE_ACTIONS = {ActionType.DELETE, ActionType.HASH}

    @staticmethod
    def check(policy: Policy) -> List[ValidationViolation]:
        """
        Check privacy metrics
        프라이버시 메트릭 검사

        Args:
            policy: Policy to validate
                   검증할 정책

        Returns:
            List of violations
            위반 목록
        """
        violations = []

        violations.extend(PrivacyMetricsChecker._check_direct_identifiers(policy))

        return violations

    @staticmethod
    def _check_direct_identifiers(policy: Policy) -> List[ValidationViolation]:
        """
        Check direct identifier handling
        직접 식별자 처리 검사
        """
        violations = []

        direct_identifiers = [col for col, pol in policy.columns.items() if pol.pii_type in PrivacyMetricsChecker.DIRECT_IDENTIFIER_TYPES and pol.action.action == ActionType.KEEP]

        if direct_identifiers:
            violations.append(
                ValidationViolation(
                    severity='error',
                    message=f"Direct identifiers must not be kept: {direct_identifiers}",
                    column=direct_identifiers[0] if direct_identifiers else None,
                    suggestion="Apply HASH, TOKENIZE, or DELETE"
                )
            )

        return violations


class CompletenessChecker:
    """
    Completeness Checker
    완전성 검사기
    
    Verifies that all necessary columns are addressed.
    모든 필요한 컬럼이 처리되었는지 확인합니다.
    """

    @staticmethod
    def check(policy: Policy) -> List[ValidationViolation]:
        """
        Check completeness
        완전성 검사
        
        Args:
            policy: Policy to validate
                   검증할 정책
        
        Returns:
            List of violations
            위반 목록
        """
        violations = []

        unprocessed_pii = [col for col, pol in policy.columns.items() if pol.is_pii and not pol.action.action]

        if unprocessed_pii:
            violations.append(ValidationViolation(severity='error', message=f"PII columns without action: {unprocessed_pii}", suggestion="Assign an action to all PII columns"))

        return violations


class PolicyValidator:
    """
    Policy Validator
    정책 검증기
    
    Main validator class that orchestrates all validation checks.
    모든 검증 검사를 조율하는 메인 검증기 클래스.
    
    Validates policies according to the paper's 5 checks (Line 558-564).
    논문의 5가지 검사에 따라 정책을 검증합니다 (558-564줄).
    """

    def __init__(self, legal_kb=None):
        """
        Initialize PolicyValidator
        PolicyValidator 초기화
        
        Args:
            legal_kb: Legal knowledge base (optional)
                     법적 지식 베이스 (선택사항)
        """
        self.legal_kb = legal_kb

    def validate(self, policy: Policy, preferred_method: str = "") -> list[ValidationViolation]:
        """
        Validate policy for consistency and utility preservation.
        정책의 일관성 및 유용성 보존을 검증합니다.

        Args:
            policy: Policy to validate
            preferred_method: User's preferred pseudonymization method
                             사용자의 선호하는 가명화 방법

        Returns:
            List of violations (empty if policy is valid)
            위반 목록 (정책이 유효하면 비어있음)
        """
        violations = []

        violations.extend(self.check_legal_compliance(policy))
        violations.extend(self.check_consistency(policy))
        violations.extend(self.check_utility_preservation(policy, preferred_method))
        violations.extend(self.check_privacy_metrics(policy))
        violations.extend(self.check_completeness(policy))

        return violations

    def check_legal_compliance(self, policy: Policy) -> list[ValidationViolation]:
        """
        Check 1: Verify all PII columns are processed according to PIPA/GDPR
        검사 1: 모든 PII 컬럼이 PIPA/GDPR에 따라 처리되는지 확인

        Args:
            policy: Policy to validate
                   검증할 정책

        Returns:
            List of violations
            위반 목록
        """
        return LegalComplianceChecker.check(policy)

    def check_consistency(self, policy: Policy) -> list[ValidationViolation]:
        """
        Check 2: Ensure no logical contradictions
        검사 2: 논리적 모순이 없는지 확인

        Args:
            policy: Policy to validate
                   검증할 정책

        Returns:
            List of violations
            위반 목록
        """
        return ConsistencyChecker.check(policy)

    def check_utility_preservation(self, policy: Policy, preferred_method: str) -> list[ValidationViolation]:
        """
        Check 3: Verify preferred method can still be achieved
        검증 3: 선호 방법이 달성 가능한지 확인

        Args:
            policy: Policy to check
            preferred_method: User's preferred pseudonymization method

        Returns:
            List of violations
        """
        return UtilityPreservationChecker.check(policy, preferred_method)
    def check_privacy_metrics(self, policy: Policy) -> List[ValidationViolation]:
        """
        Check 4: Estimate k-anonymity and re-identification risk
        검사 4: k-익명성과 재식별 위험 추정
        
        Args:
            policy: Policy to validate
                   검증할 정책
        
        Returns:
            List of violations
            위반 목록
        """
        return PrivacyMetricsChecker.check(policy)

    def check_completeness(self, policy: Policy) -> list[ValidationViolation]:
        """
        Check 5: Verify all necessary columns are addressed
        검사 5: 모든 필요한 컬럼이 처리되었는지 확인
        
        Args:
            policy: Policy to validate
                   검증할 정책
        
        Returns:
            List of violations
            위반 목록
        """
        return CompletenessChecker.check(policy)

    def suggest_fixes(self, policy: Policy, violations: list[ValidationViolation]) -> Policy:
        """
        Automatically suggest fixes for violations (Line 545)
        위반에 대한 수정 자동 제안 (545줄)
        
        Args:
            policy: Policy with violations
                   위반이 있는 정책
            violations: List of violations to fix
                       수정할 위반 목록
        
        Returns:
            Modified policy with suggested fixes
            제안된 수정이 적용된 수정된 정책
        """
        for violation in violations:
            if violation.is_error() and violation.column:
                col_policy = policy.get_column_policy(violation.column)
                if col_policy:
                    if "cannot be kept" in violation.message:
                        col_policy.action.action = ActionType.HASH
                    elif "Direct identifiers" in violation.message:
                        col_policy.action.action = ActionType.HASH

        return policy

    def has_errors(self, violations: list[ValidationViolation]) -> bool:
        """
        Check if there are any error-level violations
        오류 수준 위반이 있는지 확인
        
        Args:
            violations: List of violations
                       위반 목록
        
        Returns:
            True if there are errors, False otherwise
            오류가 있으면 True, 그렇지 않으면 False
        """
        return any(v.is_error() for v in violations)

    def get_summary(self, violations: list[ValidationViolation]) -> dict[str, int]:
        """
        Get summary of violations by severity
        심각도별 위반 요약 가져오기
        
        Args:
            violations: List of violations
                       위반 목록
        
        Returns:
            Dictionary with counts by severity
            심각도별 개수를 포함하는 딕셔너리
        """
        summary = {'error': 0, 'warning': 0, 'info': 0}
        for v in violations:
            if v.severity in summary:
                summary[v.severity] += 1
        return summary
