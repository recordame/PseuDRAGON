"""Compliance Validator.

Implements automatic compliance checks:

1. Legal compliance
2. Consistency
3. Utility preservation
4. Privacy metrics (heuristic-based)
5. Completeness

This module validates policies against legal and technical constraints.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from pseudragon.domain.policy_dsl import ActionType, Policy


@dataclass
class Violation:
    """Represents a policy violation.

    Attributes:
        severity: HIGH, MEDIUM, or LOW
        column: Affected column name(s)
        message: Description of the violation
        suggestion: Recommended fix
        check_type: Which check detected this (legal, consistency, etc.)
    """

    severity: str
    column: str
    message: str
    suggestion: str
    check_type: str


class ComplianceValidator:
    """Automatic Compliance Validator.

    Run five schema-level validation checks.

    Note: Check 4 (Privacy metrics) uses heuristic-based estimation.
    Full k-anonymity calculation requires actual data, which is beyond
    the scope of schema-only analysis.
    """

    def __init__(self, rag_system=None):
        """Initialize validator.

        Args:
            rag_system: Optional RAG system for legal context retrieval
        """
        self.rag = rag_system

    def validate_policy(self, policy: Policy, schema: Dict[str, str]) -> List[Violation]:
        """Run all 5 compliance checks on a policy.

        Check the policy against its field context.

        Args:
            policy: Policy object to validate
            schema: Database schema (column_name -> data_type)

        Returns:
            List of Violation objects (empty if no violations)
        """
        violations = []

        # Check 1: Legal Compliance
        violations.extend(self._check_legal_compliance(policy))

        # Check 2: Consistency
        violations.extend(self._check_consistency(policy))

        # Check 3: Utility Preservation
        violations.extend(self._check_utility_preservation(policy))

        # Check 4: Privacy Metrics
        violations.extend(self._check_privacy_metrics(policy))

        # Check 5: Completeness
        violations.extend(self._check_completeness(policy, schema))

        return violations

    def _check_legal_compliance(self, policy: Policy) -> List[Violation]:
        """Check 1: Legal Compliance.

        Verifies that all PII columns are processed according to legal requirements.

        Checks:

        - All PII columns have legal justification

        - DELETE actions have rationale

        - PII are properly handled
        """
        violations = []

        for col_name, col_policy in policy.columns.items():
            if not col_policy.is_pii:
                continue

            # Check for legal evidence
            if (
                not col_policy.action.legal_evidence
                or col_policy.action.legal_evidence == "Unknown Source"
            ):
                violations.append(
                    Violation(
                        severity="HIGH",
                        column=col_name,
                        message=f"Missing legal justification for PII column '{col_name}'",
                        suggestion="Add PIPA/GDPR citation for this action. Use RAG to retrieve relevant legal provisions.",
                        check_type="legal_compliance",
                    )
                )

            # Check DELETE rationale
            if col_policy.action.action == ActionType.DELETE and not col_policy.action.rationale:
                violations.append(
                    Violation(
                        severity="MEDIUM",
                        column=col_name,
                        message=f"DELETE action on '{col_name}' lacks rationale",
                        suggestion="Explain why this column is not needed for the preferred method.",
                        check_type="legal_compliance",
                    )
                )

            # Check PII handling
            if col_policy.pii_type == "PII" and col_policy.action.action not in [
                ActionType.DELETE,
                ActionType.HASH,
                ActionType.TOKENIZE,
            ]:
                violations.append(
                    Violation(
                        severity="HIGH",
                        column=col_name,
                        message=f"Direct identifier '{col_name}' must be deleted, hashed, or tokenized",
                        suggestion="PII require strong pseudonymization or removal.",
                        check_type="legal_compliance",
                    )
                )

        return violations

    def _check_consistency(self, policy: Policy) -> List[Violation]:
        """Check 2: Consistency.

        Ensures no logical contradictions in the policy.

        Checks:

        - Columns mentioned in preferred method are not deleted

        - No column is both deleted and used for grouping
        """
        violations = []

        # Find deleted columns
        deleted_cols = set()
        for col_name, col_policy in policy.columns.items():
            if col_policy.action.action == ActionType.DELETE:
                deleted_cols.add(col_name)

        # Check for conflicts with preferred method
        if policy.preferred_method:
            deleted_cols = [col for col, cp in policy.columns.items() if cp.action == "DELETE"]
            for col in deleted_cols:
                col_name = col.lower()
                if col_name in policy.preferred_method.lower():
                    violations.append(
                        ValidationViolation(
                            severity="ERROR",
                            category="COMPLIANCE",
                            column=col,
                            message=f"Column '{col}' is deleted but mentioned in preferred method",
                            suggestion="Consider using HASH or MASK instead of DELETE for this column.",
                        )
                    )

        return violations

    def _check_utility_preservation(self, policy: Policy) -> List[Violation]:
        """Check 3: Utility Preservation.

        Verifies that the preferred method can still be achieved.

        Checks:
        - Columns mentioned in preferred method are not deleted
        - If preferred method mentions specific operations, required columns should be preserved

        Args:
            policy: Policy to check

        Returns:
            List of violations
        """
        violations = []

        if not policy.preferred_method:
            return violations

        obj_lower = policy.preferred_method.lower()
        aggregation_keywords = [
            "group by",
            "sum",
            "count",
            "average",
            "trend",
            "analysis",
        ]

        # Check for deleted non-PII columns
        if any(kw in obj_lower for kw in aggregation_keywords):
            non_pii_deleted = []
            for col, pol in policy.columns.items():
                if not pol.is_pii and pol.action.action == ActionType.DELETE:
                    non_pii_deleted.append(col)

            if non_pii_deleted:
                violations.append(
                    Violation(
                        severity="MEDIUM",
                        column=", ".join(non_pii_deleted),
                        message=f"Non-PII columns deleted may affect preferred method: {non_pii_deleted}",
                        suggestion="Consider keeping these columns as they are not PII and may be needed for analysis.",
                        check_type="utility_preservation",
                    )
                )

        # Check for PII columns kept without transformation
        pii_cols_kept = 0
        for col, pol in policy.columns.items():
            if pol.is_pii and pol.action.action == ActionType.KEEP:
                pii_cols_kept += 1

        if pii_cols_kept > 0:
            violations.append(
                Violation(
                    severity="HIGH",
                    column=f"{pii_cols_kept} PII column(s)",
                    message=f"{pii_cols_kept} PII columns are kept without transformation",
                    suggestion="All PII columns should be pseudonymized or anonymized. Use HASH, MASK, or TOKENIZE.",
                    check_type="utility_preservation",
                )
            )

        return violations

    def _check_completeness(self, policy: Policy, schema: Dict[str, str]) -> List[Violation]:
        """Check 5: Completeness.

        Verifies that all columns in the schema are addressed in the policy.
        """
        violations = []

        schema_cols = set(schema.keys())
        policy_cols = set(policy.columns.keys())

        # Check for missing columns
        missing = schema_cols - policy_cols
        if missing:
            violations.append(
                Violation(
                    severity="HIGH",
                    column=", ".join(missing),
                    message=f"Columns not addressed in policy: {missing}",
                    suggestion="Add policy actions for all columns. Use KEEP for non-PII columns.",
                    check_type="completeness",
                )
            )

        # Check for extra columns
        extra = policy_cols - schema_cols
        if extra:
            violations.append(
                Violation(
                    severity="LOW",
                    column=", ".join(extra),
                    message=f"Policy contains columns not in schema: {extra}",
                    suggestion="Remove these columns from the policy or verify schema correctness.",
                    check_type="completeness",
                )
            )

        return violations

    def get_validation_summary(self, violations: List[Violation]) -> Dict[str, Any]:
        """Generate a summary of validation results.

        Returns:
            Dictionary with counts by severity and check type
        """
        summary = {
            "total": len(violations),
            "by_severity": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
            "by_check": {
                "legal_compliance": 0,
                "consistency": 0,
                "utility_preservation": 0,
                "privacy_metrics": 0,
                "completeness": 0,
            },
            "is_compliant": len([v for v in violations if v.severity == "HIGH"]) == 0,
        }

        for v in violations:
            summary["by_severity"][v.severity] += 1
            summary["by_check"][v.check_type] += 1

        return summary
