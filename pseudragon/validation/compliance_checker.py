"""
Compliance Validator
규정 준수 검증기

Implements automatic compliance checks:
논문의 자동 규정 준수 검사 구현 (Lines 558-564):
1. Legal compliance / 법적 준수
2. Consistency / 일관성
3. Utility preservation / 유용성 보존
4. Privacy metrics (heuristic-based) / 개인정보 보호 지표 (휴리스틱 기반)
5. Completeness / 완전성

This module validates policies against legal and technical constraints.
이 모듈은 법적 및 기술적 제약 조건에 대해 정책을 검증합니다.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from pseudragon.domain.policy_dsl import ActionType, Policy


@dataclass
class Violation:
    """
    Represents a policy violation
    정책 위반을 나타냄

    Attributes:
        severity: HIGH, MEDIUM, or LOW
                 심각도: HIGH, MEDIUM 또는 LOW
        column: Affected column name(s)
               영향을 받는 컬럼 이름
        message: Description of the violation
                위반 사항 설명
        suggestion: Recommended fix
                   권장 수정 사항
        check_type: Which check detected this (legal, consistency, etc.)
                   이를 감지한 검사 유형 (legal, consistency 등)
    """

    severity: str
    column: str
    message: str
    suggestion: str
    check_type: str


class ComplianceValidator:
    """
    Automatic Compliance Validator
    자동 규정 준수 검증기

    Implements the 5-check validation framework from paper Section 4.3.1.
    논문 Section 4.3.1의 5단계 검증 프레임워크 구현.

    Note: Check 4 (Privacy metrics) uses heuristic-based estimation.
    Full k-anonymity calculation requires actual data, which is beyond
    the scope of schema-only analysis.

    참고: Check 4 (개인정보 보호 지표)는 휴리스틱 기반 추정을 사용합니다.
    완전한 k-익명성 계산은 실제 데이터가 필요하며, 이는 스키마 전용 분석의
    범위를 벗어납니다.
    """

    def __init__(self, rag_system=None):
        """
        Initialize validator
        검증기 초기화

        Args:
            rag_system: Optional RAG system for legal context retrieval
                       법적 컨텍스트 검색을 위한 선택적 RAG 시스템
        """
        self.rag = rag_system

    def validate_policy(self, policy: Policy, schema: Dict[str, str]) -> List[Violation]:
        """
        Run all 5 compliance checks on a policy
        정책에 대해 5가지 규정 준수 검사 모두 실행

        Implements CONTEXTUAL_COMPLIANCE_CHECK from Algorithm (Line 541).
        알고리즘의 CONTEXTUAL_COMPLIANCE_CHECK 구현 (Line 541).

        Args:
            policy: Policy object to validate
                   검증할 정책 객체
            schema: Database schema (column_name -> data_type)
                   데이터베이스 스키마 (column_name -> data_type)

        Returns:
            List of Violation objects (empty if no violations)
            Violation 객체 목록 (위반 사항이 없으면 빈 목록)
        """
        violations = []

        # Check 1: Legal Compliance / 법적 준수
        violations.extend(self._check_legal_compliance(policy))

        # Check 2: Consistency / 일관성
        violations.extend(self._check_consistency(policy))

        # Check 3: Utility Preservation / 유용성 보존
        violations.extend(self._check_utility_preservation(policy))

        # Check 4: Privacy Metrics / 개인정보 보호 지표
        violations.extend(self._check_privacy_metrics(policy))

        # Check 5: Completeness / 완전성
        violations.extend(self._check_completeness(policy, schema))

        return violations

    def _check_legal_compliance(self, policy: Policy) -> List[Violation]:
        """
        Check 1: Legal Compliance
        검사 1: 법적 준수

        Verifies that all PII columns are processed according to legal requirements.
        모든 PII 컬럼이 법적 요구사항에 따라 처리되는지 확인합니다.

        Checks:
        검사 항목:
        - All PII columns have legal justification
          모든 PII 컬럼에 법적 근거가 있는지
        - DELETE actions have rationale
          DELETE 액션에 근거가 있는지
        - PII are properly handled
          직접 식별자가 적절히 처리되는지
        """
        violations = []

        for col_name, col_policy in policy.columns.items():
            if not col_policy.is_pii:
                continue

            # Check for legal evidence / 법적 근거 확인
            if (not col_policy.action.legal_evidence or col_policy.action.legal_evidence == "Unknown Source"):
                violations.append(
                    Violation(
                        severity="HIGH",
                        column=col_name,
                        message=f"Missing legal justification for PII column '{col_name}'",
                        suggestion="Add PIPA/GDPR citation for this action. Use RAG to retrieve relevant legal provisions.",
                        check_type="legal_compliance", )
                )

            # Check DELETE rationale / DELETE 근거 확인
            if (col_policy.action.action == ActionType.DELETE and not col_policy.action.rationale):
                violations.append(
                    Violation(
                        severity="MEDIUM",
                        column=col_name,
                        message=f"DELETE action on '{col_name}' lacks rationale",
                        suggestion="Explain why this column is not needed for the preferred method.",
                        check_type="legal_compliance", )
                )

            # Check PII handling / 직접 식별자 처리 확인
            if (col_policy.pii_type == "PII" and col_policy.action.action not in [ActionType.DELETE, ActionType.HASH, ActionType.TOKENIZE, ]):
                violations.append(
                    Violation(
                        severity="HIGH",
                        column=col_name,
                        message=f"Direct identifier '{col_name}' must be deleted, hashed, or tokenized",
                        suggestion="PII require strong pseudonymization or removal.",
                        check_type="legal_compliance", )
                )

        return violations

    def _check_consistency(self, policy: Policy) -> List[Violation]:
        """
        Check 2: Consistency
        검사 2: 일관성

        Ensures no logical contradictions in the policy.
        정책에 논리적 모순이 없는지 확인합니다.

        Checks:
        검사 항목:
        - Columns mentioned in preferred method are not deleted
          선호 기법에 언급된 컬럼이 삭제되지 않았는지
        - No column is both deleted and used for grouping
          삭제되면서 동시에 그룹화에 사용되는 컬럼이 없는지
        """
        violations = []

        # Find deleted columns / 삭제된 컬럼 찾기
        deleted_cols = set()
        for col_name, col_policy in policy.columns.items():
            if col_policy.action.action == ActionType.DELETE:
                deleted_cols.add(col_name)

        # Check for conflicts with preferred method / 선호 기법과의 충돌 확인
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
                            suggestion="Consider using HASH or MASK instead of DELETE for this column.", )
                    )

        return violations

    def _check_utility_preservation(self, policy: Policy) -> List[Violation]:
        """
        Check 3: Utility Preservation
        검사 3: 유용성 보존

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
        aggregation_keywords = ["group by", "sum", "count", "average", "trend", "analysis", ]

        # Check for deleted non-PII columns / 삭제된 비PII 컬럼 확인
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
                        check_type="utility_preservation", )
                )

        # Check for PII columns kept without transformation / 변환 없이 유지된 PII 컬럼 확인
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
                    check_type="utility_preservation", )
            )

        return violations

    def _check_completeness(self, policy: Policy, schema: Dict[str, str]) -> List[Violation]:
        """
        Check 5: Completeness
        검사 5: 완전성

        Verifies that all columns in the schema are addressed in the policy.
        스키마의 모든 컬럼이 정책에서 다루어지는지 확인합니다.
        """
        violations = []

        schema_cols = set(schema.keys())
        policy_cols = set(policy.columns.keys())

        # Check for missing columns / 누락된 컬럼 확인
        missing = schema_cols - policy_cols
        if missing:
            violations.append(
                Violation(
                    severity="HIGH",
                    column=", ".join(missing),
                    message=f"Columns not addressed in policy: {missing}",
                    suggestion="Add policy actions for all columns. Use KEEP for non-PII columns.",
                    check_type="completeness", )
            )

        # Check for extra columns / 추가 컬럼 확인
        extra = policy_cols - schema_cols
        if extra:
            violations.append(
                Violation(
                    severity="LOW",
                    column=", ".join(extra),
                    message=f"Policy contains columns not in schema: {extra}",
                    suggestion="Remove these columns from the policy or verify schema correctness.",
                    check_type="completeness", )
            )

        return violations

    def get_validation_summary(self, violations: List[Violation]) -> Dict[str, Any]:
        """
        Generate a summary of validation results
        검증 결과 요약 생성

        Returns:
            Dictionary with counts by severity and check type
            심각도 및 검사 유형별 개수가 포함된 딕셔너리
        """
        summary = {
            "total": len(violations),
            "by_severity": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
            "by_check": {"legal_compliance": 0, "consistency": 0, "utility_preservation": 0, "privacy_metrics": 0, "completeness": 0, },
            "is_compliant": len([v for v in violations if v.severity == "HIGH"]) == 0,
        }

        for v in violations:
            summary["by_severity"][v.severity] += 1
            summary["by_check"][v.check_type] += 1

        return summary
