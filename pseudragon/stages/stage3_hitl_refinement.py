"""Stage 3: Human-in-the-Loop (HITL) Refinement.

Implements the interactive policy refinement:

- Automatic validation (5 checks)
- User review and modification
- Real-time feedback
- Audit trail
"""


import copy
from typing import Any, Callable, Dict, List, Optional, Tuple


from pseudragon.domain.policy_dsl import Policy
from pseudragon.validation.consistency_checker import PolicyValidator, ValidationViolation


class Stage3HITLRefinement:
    """Stage 3: HITL Refinement.

    Allows users to review and modify policies with automatic validation.
    """

    def __init__(self, validator: PolicyValidator):
        """Initialize Stage 3 HITL Refinement.

        Args:
            validator: Policy validator for checking policy consistency
        """
        self.validator = validator

    def refine_policy(
        self,
        policy: Policy,
        preferred_method: str,
        expert_feedback: Dict[str, Any],
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[Policy, List[ValidationViolation]]:
        """Refine policy based on expert feedback and validate.

        Args:
            policy: Initial policy from Stage 2
            preferred_method: User's preferred pseudonymization method
            expert_feedback: Expert modifications
            log_callback: Optional callback for logging

        Returns:
            Tuple of (refined policy, validation violations)
        """
        self._log("[Stage 3] HITL Refinement", log_callback)

        # Create deep copy to avoid modifying original policy

        policy_current = copy.deepcopy(policy)

        # Note: In the web interface, expert feedback is applied directly via the UI
        # before this method is called. This method focuses on validation.

        # Validate the refined policy

        violations = self.validate_user_changes(policy_current, preferred_method)

        return policy_current, violations

    def validate_user_changes(
        self, policy: Policy, preferred_method: str
    ) -> List[ValidationViolation]:
        """Validate user-modified policy in real-time.

        This is called by the web UI when user makes changes.

        Args:
            policy: Policy to validate
            preferred_method: Preferred anonymization method

        Returns:
            List of validation violations
        """
        from pseudragon.validation.method_contract import (
            normalize_method_metadata,
            MethodContractError,
        )

        violations = self.validator.validate(policy, preferred_method)
        for column, col_policy in policy.columns.items():
            try:
                normalize_method_metadata(col_policy.action.to_dict(), column)
            except MethodContractError as exc:
                violations.append(
                    ValidationViolation(
                        severity="error",
                        message=str(exc),
                        column=column,
                        suggestion="Select a verified source-derived candidate or supply and review an explicit method contract.",
                    )
                )
        return violations

    def _log(self, message: str, callback: Optional[Callable] = None) -> None:
        """Helper for logging messages.

        Args:
            message: Message to log
            callback: Optional callback function
        """
        if callback:
            callback(message)
        else:
            print(message)
