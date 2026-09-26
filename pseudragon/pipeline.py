"""PseuDRAGON Pipeline Orchestrator.

Coordinate field identification, policy synthesis, expert review and code generation.
"""


from typing import Any, Callable, Dict, Optional


from pseudragon.domain.policy_dsl import Policy
from pseudragon.validation import PolicyValidator, ValidationViolation


class PipelineError(Exception):
    """Custom exception for pipeline operations."""


class PipelineStage:
    """Pipeline Stage Enumeration."""

    STAGE1_PII_DETECTION = "Stage 1: PII Detection"
    STAGE2_POLICY_SYNTHESIS = "Stage 2: Policy Synthesis"
    STAGE3_HITL_REFINEMENT = "Stage 3: HITL Refinement"
    STAGE4_CODE_GENERATION = "Stage 4: Code Generation"


class PseuDRAGONPipeline:
    """Main Pipeline Orchestrator for PseuDRAGON Framework.

    The pipeline consists of 4 stages:

    1. PII Detection: Identify PII columns using hybrid approach

    2. Policy Synthesis: Generate pseudonymization policies

    3. HITL Refinement: Human-in-the-loop policy refinement

    4. Code Generation: Generate executable Python code
    """

    def __init__(
        self,
        rag_system,
        default_llm_client,
        llm_client_stage_1=None,
        llm_client_stage_2=None,
        expert_preference_manager=None,
    ):
        """Initialize PseuDRAGON Pipeline.

        Args:
            rag_system: RAG system for legal knowledge retrieval
            llm_client: Default OpenAI client for LLM operations
            llm_client_stage_1: Optional OpenAI client for Stage 1 (PII Detection)
            llm_client_stage_2: Optional OpenAI client for Stage 2 (Policy Synthesis)
            expert_preference_manager: Optional expert preference manager for learned PII classifications
        """
        self.rag = rag_system
        self.default_llm_client = default_llm_client
        self.llm_client_stage_1 = llm_client_stage_1 if llm_client_stage_1 else default_llm_client
        self.llm_client_stage_2 = llm_client_stage_2 if llm_client_stage_2 else default_llm_client
        self.validator = PolicyValidator()
        self.expert_preference_manager = expert_preference_manager

        self._initialize_stages()
        self._load_feedback_knowledge()

    def _initialize_stages(self) -> None:
        """Initialize all pipeline stages."""

        from pseudragon.stages.stage1_pii_detection import Stage1PIIDetection
        from pseudragon.stages.stage2_policy_synthesis import Stage2PolicySynthesis
        from pseudragon.stages.stage3_hitl_refinement import Stage3HITLRefinement
        from pseudragon.stages.stage4_code_generation import Stage4CodeGeneration

        self.stage1 = Stage1PIIDetection(
            self.rag,
            self.llm_client_stage_1,
            expert_preference_manager=self.expert_preference_manager,
        )
        self.stage2 = Stage2PolicySynthesis(self.rag, self.llm_client_stage_2)
        self.stage3 = Stage3HITLRefinement(self.validator)
        self.stage4 = Stage4CodeGeneration(self.llm_client_stage_2)

    def _load_feedback_knowledge(self) -> None:
        """Load feedback knowledge base from past HITL sessions.

        This method loads audit logs from the output directory and integrates
        expert feedback into the RAG knowledge base for improved policy synthesis.
        """
        import os

        output_dir = "output"
        if os.path.exists(output_dir):
            try:
                self.rag.load_feedback_knowledge(output_dir)
            except Exception as e:
                print(f"[WARNING] Failed to load feedback knowledge: {e}")

    def run(
        self,
        schema: Dict[str, Any],
        table_name: str,
        preferred_method: str,
        db_config: Optional[str] = None,
        hitl_callback: Optional[Callable] = None,
        log_callback: Optional[Callable] = None,
        output_dir: Optional[str] = None,
    ) -> str:
        """Execute the full PseuDRAGON pipeline.

        Args:
            schema: Database schema metadata (column names, types, and comments)
            table_name: Name of the table
            preferred_method: User's preferred pseudonymization method
            db_config: Database connection path or config
            hitl_callback: Callback for human-in-the-loop interaction
            log_callback: Callback for logging

        Returns:
            Generated Python code as string

        Raises:
            PipelineError: If any stage fails
        """
        try:
            # self._log_stage_start("PseuDRAGON Pipeline", log_callback)

            pii_metadata = self._run_stage1(schema, table_name, log_callback)
            policy_initial = self._run_stage2(
                pii_metadata, table_name, preferred_method, log_callback
            )

            # Generate integrated report after Stage 1 and 2
            self._generate_integrated_report(
                table_name, pii_metadata, policy_initial, log_callback, output_dir
            )

            policy_final = self._run_stage3(
                policy_initial, preferred_method, hitl_callback, log_callback
            )
            code = self._run_stage4(schema, policy_final, pii_metadata, db_config, log_callback)

            # self._log_stage_complete("PseuDRAGON Pipeline", log_callback)

            return code

        except Exception as e:
            error_msg = f"Pipeline execution failed: {str(e)}"
            self._log(error_msg, log_callback)
            raise PipelineError(error_msg) from e

    def _run_stage1(
        self, schema: Dict[str, Any], table_name: str, log_callback: Optional[Callable]
    ) -> Dict[str, Any]:
        """Run Stage 1: PII Detection.

        Args:
            schema: Database schema
            table_name: Table name
            log_callback: Logging callback

        Returns:
            PII metadata
        """
        # self._log_stage_start(PipelineStage.STAGE1_PII_DETECTION, log_callback)
        pii_metadata = self.stage1.identify_pii(schema, table_name, log_callback)
        # self._log_stage_complete(PipelineStage.STAGE1_PII_DETECTION, log_callback)
        return pii_metadata

    def _run_stage2(
        self,
        pii_metadata: Dict[str, Any],
        table_name: str,
        preferred_method: str,
        purpose_goal: str = "",
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> Policy:
        """Run Stage 2: Policy Synthesis.

        Args:
            pii_metadata: PII metadata from Stage 1
            table_name: Table name
            preferred_method: User's preferred pseudonymization method
            purpose_goal: User's intended purpose for data pseudonymization
            log_callback: Logging callback

        Returns:
            Initial policy
        """
        # self._log_stage_start(PipelineStage.STAGE2_POLICY_SYNTHESIS, log_callback)
        policy = self.stage2.synthesize_policy(
            pii_metadata, table_name, preferred_method, purpose_goal, log_callback
        )
        # self._log_stage_complete(PipelineStage.STAGE2_POLICY_SYNTHESIS, log_callback)
        return policy

    def _run_stage3(
        self,
        policy: Policy,
        preferred_method: str,
        hitl_callback: Optional[Callable[[Policy], Policy]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> tuple[Policy, list[ValidationViolation]]:
        """Run Stage 3: HITL Refinement.

        Args:
            policy: Initial policy from Stage 2
            preferred_method: User's preferred pseudonymization method
            hitl_callback: Callback for HITL interaction
            log_callback: Logging callback

        Returns:
            Tuple of (refined policy, violations)
        """
        # self._log_stage_start(PipelineStage.STAGE3_HITL_REFINEMENT, log_callback)
        policy_refined, violations = self.stage3.refine_policy(
            policy, preferred_method, hitl_callback, log_callback
        )
        # self._log_stage_complete(PipelineStage.STAGE3_HITL_REFINEMENT, log_callback)

        return policy_refined, violations

    def _run_stage4(
        self,
        schema: Dict[str, Any],
        policy: Policy,
        pii_metadata: Dict[str, Any],
        db_config: Optional[str],
        log_callback: Optional[Callable],
    ) -> str:
        """Run Stage 4: Code Generation.

        Args:
            schema: Database schema
            policy: Final policy from Stage 3
            pii_metadata: PII metadata
            db_config: Database connection path or config
            log_callback: Logging callback

        Returns:
            Generated Python code
        """
        # self._log_stage_start(PipelineStage.STAGE4_CODE_GENERATION, log_callback)
        code = self.stage4.generate_code(schema, policy, pii_metadata, db_config, log_callback)
        # self._log_stage_complete(PipelineStage.STAGE4_CODE_GENERATION, log_callback)
        return code

    def _generate_integrated_report(
        self,
        table_name: str,
        pii_metadata: Dict[str, Any],
        policy: Any,
        log_callback: Optional[Callable],
        output_dir: Optional[str] = None,
    ) -> None:
        """Generate integrated report combining Stage 1 and Stage 2 results.

        Args:
            table_name: Table name
            pii_metadata: PII metadata from Stage 1
            policy: Policy from Stage 2
            log_callback: Logging callback
            output_dir: Optional output directory
        """
        from pseudragon.reporting.compliance_reporter import ReportGenerator

        self._log("[Report] Generating integrated report...", log_callback)

        try:
            report_gen = ReportGenerator()
            report_path = report_gen.generate_integrated_report(
                table_name, pii_metadata, policy, output_dir=output_dir
            )
            # Store the report path so web interface can access it

            self._last_integrated_report = report_path
            self._log(f"[OK] Integrated report saved: {report_path}", log_callback)
        except Exception as e:
            self._log(
                f"[WARN] Failed to generate integrated report: {str(e)}",
                log_callback,
            )

    def _log(self, message: str, callback: Optional[Callable] = None) -> None:
        """Log a message.

        Args:
            message: Message to log
            callback: Optional logging callback
        """
        if callback:
            callback(message)
        else:
            print(message)

    # def _log_stage_start(self, stage_name: str, callback: Optional[Callable] = None) -> None:
    #     """
    #     Log stage start

    # .
    #     Args:
    #         stage_name: Name of the stage

    #         callback: Optional logging callback

    #     """
    #     self._log(f"▶ Starting {stage_name}", callback)
    # .
    # def _log_stage_complete(self, stage_name: str, callback: Optional[Callable] = None) -> None:
    #     """
    #     Log stage completion

    # .
    #     Args:
    #         stage_name: Name of the stage

    #         callback: Optional logging callback

    #     """
    #     self._log(f"✓ Completed {stage_name}", callback)

    # Public API methods for individual stage execution

    def identify_pii(
        self, schema: Dict[str, Any], table_name: str, log_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Public API: Execute Stage 1 - PII Identification.

        Args:
            schema: Database schema metadata (column names, types, and comments)
            table_name: Name of the table
            log_callback: Optional callback for logging

        Returns:
            PII metadata dictionary
        """
        return self._run_stage1(schema, table_name, log_callback)

    def synthesize_policy(
        self,
        pii_metadata: Dict[str, Any],
        table_name: str,
        preferred_method: str = "",
        purpose_goal: str = "",
        log_callback: Optional[Callable] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Public API: Execute Stage 2 - Policy Synthesis.

        Args:
            pii_metadata: PII metadata from Stage 1
            table_name: Name of the table
            preferred_method: User's preferred pseudonymization method
            purpose_goal: User's intended purpose for data pseudonymization
            log_callback: Optional callback for logging
            output_dir: output directory for each session

        Returns:
            Policy dictionary (per-column policies)
        """
        policy = self._run_stage2(
            pii_metadata, table_name, preferred_method, purpose_goal, log_callback
        )

        # Generate integrated report after Stage 2

        self._generate_integrated_report(table_name, pii_metadata, policy, log_callback, output_dir)

        # Convert Policy object to dictionary format expected by app.py

        policy_dict = {}
        for col_name, col_policy in policy.columns.items():
            # Convert PolicyAction objects to method dictionaries
            methods = []
            for action in col_policy.candidate_actions:
                methods.append(
                    {
                        "method": action.action.value,
                        "applicability": "Pending review"
                        if action.parameters.get("requirement_origin") == "source_table_parser_v1"
                        else "High",
                        "description": action.rationale,  # Unified language description
                        "code_snippet": action.code_snippet,  # Include code snippet
                        "example_implementation": action.code_snippet,  # Also use as example
                        "parameters": action.parameters,
                        "contract_verification": action.contract_verification,
                        "contract_code_sha256": action.contract_code_sha256,
                        "legal_evidence": action.legal_evidence,  # Add legal evidence for frontend display
                    }
                )

            # DEBUG: Log Non-PII columns with their methods
            if not col_policy.is_pii:
                print(f"[DEBUG pipeline.py] Converting Non-PII column '{col_name}' to dict:")
                print(f"  - is_pii: {col_policy.is_pii}")
                print(f"  - pii_type: {col_policy.pii_type}")
                print(f"  - candidate_actions count: {len(col_policy.candidate_actions)}")
                print(f"  - methods count: {len(methods)}")
                print(f"  - methods: {[t['method'] for t in methods]}")

            policy_dict[col_name] = {
                "is_pii": col_policy.is_pii,
                "pii_type": col_policy.pii_type,
                "recommended_methods": methods,
                "evidence_source": col_policy.action.legal_evidence
                if col_policy.action
                else "Unknown",
                "retrieved_evidence": col_policy.retrieved_evidence,
                "retrieval_trace": col_policy.retrieval_trace,
            }

        return policy_dict

    def generate_python_code(
        self,
        schema: Dict[str, Any],
        policy_data: Dict[str, Any],
        pii_metadata: Dict[str, Any],
        table_name: str = "",
        db_config: Any = None,
        log_callback: Optional[Callable] = None,
    ) -> str:
        """Public API: Execute Stage 4 - Code Generation.

        Args:
            schema: Database schema
            policy_data: Policy data (from Stage 2/3)
            pii_metadata: PII metadata (from Stage 1)
            table_name: Name of the table
            db_config: Database configuration
            log_callback: Optional callback for logging

        Returns:
            Generated Python code as string
        """
        # Convert policy_data dict back to Policy object

        from pseudragon.domain.policy_dsl import (
            Policy,
            ColumnPolicy,
            PolicyAction,
            ActionType,
        )

        policy = Policy(table_name=table_name)

        for col_name, col_data in policy_data.items():
            # Get the first method as the primary action
            methods_list = col_data.get("recommended_methods", [])

            if not methods_list:
                # No methods - default to KEEP
                primary_action = PolicyAction(
                    action=ActionType.KEEP,
                    rationale="No specific method provided",
                    code_snippet="# No transformation needed",
                )
                candidate_actions = []
            else:
                # Convert first method to primary action
                from pseudragon.validation.method_contract import normalize_method_metadata

                first_method = normalize_method_metadata(methods_list[0], col_name)
                primary_action = PolicyAction(
                    action=self._map_method_to_action_type(first_method.get("method", "KEEP")),
                    parameters=first_method.get("parameters", {}),
                    rationale=first_method.get("description", first_method.get("rationale", "")),
                    legal_evidence=first_method.get(
                        "legal_evidence",
                        first_method.get("legal_source", col_data.get("legal_source", "")),
                    ),
                    code_snippet=first_method.get("code_snippet", ""),
                    contract_verification=first_method.get("contract_verification"),
                    contract_code_sha256=first_method.get("contract_code_sha256"),
                )

                # Convert remaining methods to candidate actions
                candidate_actions = []
                for method in methods_list:
                    candidate_actions.append(
                        PolicyAction(
                            action=self._map_method_to_action_type(method.get("method", "KEEP")),
                            parameters=method.get("parameters", {}),
                            rationale=method.get("description", method.get("rationale", "")),
                            legal_evidence=method.get(
                                "legal_evidence",
                                method.get("legal_source", col_data.get("legal_source", "")),
                            ),
                            code_snippet=method.get("code_snippet", ""),
                            contract_verification=method.get("contract_verification"),
                            contract_code_sha256=method.get("contract_code_sha256"),
                        )
                    )

            col_policy = ColumnPolicy(
                column_name=col_name,
                pii_type=col_data.get("pii_type", "Unknown"),
                is_pii=col_data.get("is_pii", False),
                action=primary_action,
                candidate_actions=candidate_actions,
                retrieved_evidence=col_data.get("retrieved_evidence", []),
                retrieval_trace=col_data.get("retrieval_trace"),
            )

            policy.add_column_policy(col_policy)

        return self._run_stage4(schema, policy, pii_metadata, db_config, log_callback)

    def _map_method_to_action_type(self, method: str) -> "ActionType":
        """Map method string to ActionType enum.

        Args:
            method: Method name as string

        Returns:
            ActionType enum value
        """
        from pseudragon.domain.policy_dsl import ActionType

        method_upper = method.upper()

        mapping = {
            "HASH": ActionType.HASH,
            "HASHING": ActionType.HASH,
            "MASK": ActionType.MASK,
            "MASKING": ActionType.MASK,
            "TOKENIZE": ActionType.TOKENIZE,
            "TOKENIZATION": ActionType.TOKENIZE,
            "GENERALIZE": ActionType.GENERALIZE,
            "GENERALIZATION": ActionType.GENERALIZE,
            "ROUND": ActionType.ROUND,
            "ROUNDING": ActionType.ROUND,
            "ENCRYPT": ActionType.ENCRYPT,
            "ENCRYPTION": ActionType.ENCRYPT,
            "DELETE": ActionType.DELETE,
            "DELETION": ActionType.DELETE,
            "KEEP": ActionType.KEEP,
            "KEEP ORIGINAL": ActionType.KEEP,
        }

        return mapping.get(method_upper, ActionType.KEEP)
