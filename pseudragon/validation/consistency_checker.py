"""Policy Validators

Implements consistency checks for policy validation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pseudragon.domain.policy_dsl import ActionType, Policy


@dataclass
class ValidationViolation:
    """Represents a policy validation violation

    Attributes:
        severity: Violation severity level ('error', 'warning', 'info')

        message: Description of the violation

        column: Column name associated with the violation (if applicable)

        suggestion: Suggested fix for the violation
    """
    severity: str
    message: str
    column: Optional[str] = None
    suggestion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary

        Returns:
            Dictionary representation
        """
        return {'severity': self.severity, 'message': self.message, 'column': self.column, 'suggestion': self.suggestion}

    def __repr__(self) -> str:
        """String representation"""
        col_str = f" [{self.column}]" if self.column else ""
        return f"{self.severity.upper()}{col_str}: {self.message}"

    def is_error(self) -> bool:
        """Check if violation is an error"""
        return self.severity == 'error'

    def is_warning(self) -> bool:
        """Check if violation is a warning"""
        return self.severity == 'warning'


class LegalComplianceChecker:
    """Legal Compliance Checker

    Verifies that all PII columns are processed according to PIPA/GDPR.
    """

    ALLOWED_PII_ACTIONS = {ActionType.DELETE, ActionType.HASH, ActionType.MASK, ActionType.TOKENIZE, ActionType.GENERALIZE, ActionType.ENCRYPT}

    @staticmethod
    def check(policy: Policy) -> list[ValidationViolation]:
        """Check legal compliance

        Args:
            policy: Policy to validate

        Returns:
            List of violations
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
    """Consistency Checker

    Ensures no logical contradictions in the policy.

    Example: A column is both deleted and used in aggregation.
    """

    @staticmethod
    def check(policy: Policy) -> List[ValidationViolation]:
        """Check consistency

        Args:
            policy: Policy to validate

        Returns:
            List of violations
        """
        violations = []

        deleted_columns = ConsistencyChecker._get_deleted_columns(policy)
        violations.extend(ConsistencyChecker._check_group_by_consistency(policy, deleted_columns))

        return violations

    @staticmethod
    def _get_deleted_columns(policy: Policy) -> set:
        """Get set of deleted columns"""
        return {col for col, pol in policy.columns.items() if pol.action.action == ActionType.DELETE}

    @staticmethod
    def _check_group_by_consistency(policy: Policy, deleted_columns: set) -> List[ValidationViolation]:
        """Check that group_by columns are not deleted"""
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
    """Utility Preservation Checker

    Verifies that preferred methods can still be achieved after pseudonymization.

    Example: Preferred method mentions "user-level analysis" but user_id is deleted.
    """

    OBJECTIVE_KEYWORDS = {
        'user': ['user_id', 'user', 'customer_id', 'customer'],
        'age': ['age', 'birth_date', 'birthday', 'dob'],
        'location': ['address', 'city', 'region', 'location'],
        'time': ['date', 'time', 'timestamp', 'created_at']
    }

    @staticmethod
    def check(policy: Policy, preferred_method: str) -> List[ValidationViolation]:
        """Check if preferred method can still be achieved.

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
    """Privacy Metrics Checker

    Estimates k-anonymity and re-identification risk.

    Note: This is a simplified heuristic. Full implementation would
    require actual data analysis.
    """

    DIRECT_IDENTIFIER_TYPES = {'PII'}
    SAFE_ACTIONS = {ActionType.DELETE, ActionType.HASH}

    @staticmethod
    def check(policy: Policy) -> List[ValidationViolation]:
        """Check privacy metrics

        Args:
            policy: Policy to validate

        Returns:
            List of violations
        """
        violations = []

        violations.extend(PrivacyMetricsChecker._check_direct_identifiers(policy))

        return violations

    @staticmethod
    def _check_direct_identifiers(policy: Policy) -> List[ValidationViolation]:
        """Check direct identifier handling"""
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
    """Completeness Checker

    Verifies that all necessary columns are addressed.
    """

    @staticmethod
    def check(policy: Policy) -> List[ValidationViolation]:
        """Check completeness

        Args:
            policy: Policy to validate

        Returns:
            List of violations
        """
        violations = []

        unprocessed_pii = [col for col, pol in policy.columns.items() if pol.is_pii and not pol.action.action]

        if unprocessed_pii:
            violations.append(ValidationViolation(severity='error', message=f"PII columns without action: {unprocessed_pii}", suggestion="Assign an action to all PII columns"))

        return violations


class PolicyValidator:
    """Policy Validator

    Main validator class that orchestrates all validation checks.

    Validates policies according to the paper's 5 checks (Line 558-564).
    """

    def __init__(self, legal_kb=None):
        """Initialize PolicyValidator

        Args:
            legal_kb: Legal knowledge base (optional)
        """
        self.legal_kb = legal_kb

    def validate(self, policy: Policy, preferred_method: str = "") -> list[ValidationViolation]:
        """Validate policy for consistency and utility preservation.

        Args:
            policy: Policy to validate
            preferred_method: User's preferred pseudonymization method

        Returns:
            List of violations (empty if policy is valid)
        """
        violations = []

        violations.extend(self.check_legal_compliance(policy))
        violations.extend(self.check_consistency(policy))
        violations.extend(self.check_utility_preservation(policy, preferred_method))
        violations.extend(self.check_privacy_metrics(policy))
        violations.extend(self.check_completeness(policy))

        return violations

    def check_legal_compliance(self, policy: Policy) -> list[ValidationViolation]:
        """Check 1: Verify all PII columns are processed according to PIPA/GDPR

        Args:
            policy: Policy to validate

        Returns:
            List of violations
        """
        return LegalComplianceChecker.check(policy)

    def check_consistency(self, policy: Policy) -> list[ValidationViolation]:
        """Check 2: Ensure no logical contradictions

        Args:
            policy: Policy to validate

        Returns:
            List of violations
        """
        return ConsistencyChecker.check(policy)

    def check_utility_preservation(self, policy: Policy, preferred_method: str) -> list[ValidationViolation]:
        """Check 3: Verify preferred method can still be achieved

        Args:
            policy: Policy to check
            preferred_method: User's preferred pseudonymization method

        Returns:
            List of violations
        """
        return UtilityPreservationChecker.check(policy, preferred_method)
    def check_privacy_metrics(self, policy: Policy) -> List[ValidationViolation]:
        """Check 4: Estimate k-anonymity and re-identification risk

        Args:
            policy: Policy to validate

        Returns:
            List of violations
        """
        return PrivacyMetricsChecker.check(policy)

    def check_completeness(self, policy: Policy) -> list[ValidationViolation]:
        """Check 5: Verify all necessary columns are addressed

        Args:
            policy: Policy to validate

        Returns:
            List of violations
        """
        return CompletenessChecker.check(policy)

    def suggest_fixes(self, policy: Policy, violations: list[ValidationViolation]) -> Policy:
        """Automatically suggest fixes for violations (Line 545)

        Args:
            policy: Policy with violations

            violations: List of violations to fix

        Returns:
            Modified policy with suggested fixes
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
        """Check if there are any error-level violations

        Args:
            violations: List of violations

        Returns:
            True if there are errors, False otherwise
        """
        return any(v.is_error() for v in violations)

    def get_summary(self, violations: list[ValidationViolation]) -> dict[str, int]:
        """Get summary of violations by severity

        Args:
            violations: List of violations

        Returns:
            Dictionary with counts by severity
        """
        summary = {'error': 0, 'warning': 0, 'info': 0}
        for v in violations:
            if v.severity in summary:
                summary[v.severity] += 1
        return summary
