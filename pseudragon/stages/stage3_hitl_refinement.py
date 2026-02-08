"""
Stage 3: Human-in-the-Loop (HITL) Refinement
3단계: 사람 개입 (HITL) 개선

Implements the interactive policy refinement:
논문의 알고리즘 1에서 대화형 정책 개선을 구현합니다:
- Automatic validation (5 checks) / 자동 검증 (5가지 체크)
- User review and modification / 사용자 검토 및 수정
- Real-time feedback / 실시간 피드백
- Audit trail / 감사 추적
"""

# Standard library imports
# 표준 라이브러리 import
import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

# Project-specific imports
# 프로젝트 관련 import
from pseudragon.domain.policy_dsl import Policy
from pseudragon.validation.consistency_checker import PolicyValidator, ValidationViolation


class Stage3HITLRefinement:
    """
    Stage 3: HITL Refinement
    3단계: HITL 개선
    
    Allows users to review and modify policies with automatic validation.
    자동 검증으로 정책을 검토하고 수정할 수 있습니다.
    
    Implements STAGE3_HITL from the paper.
    논문의 STAGE3_HITL을 구현합니다.
    """

    def __init__(self, validator: PolicyValidator):
        """
        Initialize Stage 3 HITL Refinement
        3단계 HITL 개선 초기화
        
        Args:
            validator: Policy validator for checking policy consistency
                      정책 일관성을 확인하기 위한 정책 검증기
        """
        self.validator = validator

    def refine_policy(
        self,
        policy: Policy,
        preferred_method: str,
        expert_feedback: Dict[str, Any],
        log_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[Policy, List[ValidationViolation]]:
        """
        Refine policy based on expert feedback and validate
        전문가 피드백을 기반으로 정책 개선 및 검증

        Args:
            policy: Initial policy from Stage 2
                   2단계의 초기 정책
            preferred_method: User's preferred pseudonymization method
                               사용자가 선호하는 가명화 방법
            expert_feedback: Expert modifications
                            전문가 수정사항
            log_callback: Optional callback for logging
                         로깅을 위한 선택적 콜백

        Returns:
            Tuple of (refined policy, validation violations)
            (개선된 정책, 검증 위반사항) 튜플
        """
        self._log("[Stage 3] HITL Refinement", log_callback)

        # Create deep copy to avoid modifying original policy
        # 원본 정책 수정을 방지하기 위해 깊은 복사 생성
        policy_current = copy.deepcopy(policy)

        # Note: In the web interface, expert feedback is applied directly via the UI
        # before this method is called. This method focuses on validation.
        # 참고: 웹 인터페이스에서는 이 메서드가 호출되기 전에 전문가 피드백이
        # UI를 통해 직접 적용됩니다. 이 메서드는 검증에 초점을 맞춥니다.

        # Validate the refined policy
        # 개선된 정책 검증
        violations = self.validator.validate(policy_current, preferred_method)

        return policy_current, violations

    def validate_user_changes(
        self,
        policy: Policy,
        preferred_method: str
    ) -> List[ValidationViolation]:
        """
        Validate user-modified policy in real-time
        실시간으로 사용자가 수정한 정책을 검증합니다

        This is called by the web UI when user makes changes.
        웹 UI에서 사용자가 변경을 할 때 호출됩니다.
        
        Args:
            policy: Policy to validate
                   검증할 정책
            preferred_method: Preferred anonymization method
                               선호하는 익명화 방법

        Returns:
            List of validation violations
            검증 위반사항 목록
        """
        return self.validator.validate(policy, preferred_method)

    def _log(self, message: str, callback: Optional[Callable] = None) -> None:
        """
        Helper for logging messages
        메시지 로깅을 위한 헬퍼 함수
        
        Args:
            message: Message to log
                    로그할 메시지
            callback: Optional callback function
                     선택적 콜백 함수
        """
        if callback:
            callback(message)
        else:
            print(message)
