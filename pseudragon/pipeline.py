"""
PseuDRAGON Pipeline Orchestrator
PseuDRAGON 파이프라인 오케스트레이터

Main pipeline that coordinates all 4 stages as described in the paper.
논문에 설명된 4단계를 조율하는 메인 파이프라인입니다.

This module implements Algorithm 1 from the paper,
orchestrating the complete pseudonymization workflow.
이 모듈은 논문의 알고리즘을 구현하며,
완전한 가명처리 워크플로우를 조율합니다.
"""

# Standard library imports
# 표준 라이브러리 import
from typing import Any, Callable, Dict, Optional

# Project-specific imports
# 프로젝트 관련 import
from pseudragon.domain.policy_dsl import Policy
from pseudragon.validation import PolicyValidator, ValidationViolation


class PipelineError(Exception):
    """
    Custom exception for pipeline operations
    파이프라인 작업을 위한 커스텀 예외
    """


class PipelineStage:
    """
    Pipeline Stage Enumeration
    파이프라인 스테이지 열거
    """

    STAGE1_PII_DETECTION = "Stage 1: PII Detection"
    STAGE2_POLICY_SYNTHESIS = "Stage 2: Policy Synthesis"
    STAGE3_HITL_REFINEMENT = "Stage 3: HITL Refinement"
    STAGE4_CODE_GENERATION = "Stage 4: Code Generation"


class PseuDRAGONPipeline:
    """
    Main Pipeline Orchestrator for PseuDRAGON Framework
    PseuDRAGON 프레임워크를 위한 메인 파이프라인 오케스트레이터

    Implements Algorithm 1 from the paper.
    논문의 알고리즘 1(202-225줄)을 구현합니다.

    The pipeline consists of 4 stages:
    파이프라인은 4단계로 구성됩니다:

    1. PII Detection: Identify PII columns using hybrid approach
       PII 탐지: 하이브리드 접근법을 사용하여 PII 컬럼 식별

    2. Policy Synthesis: Generate pseudonymization policies
       정책 합성: 가명처리 정책 생성

    3. HITL Refinement: Human-in-the-loop policy refinement
       HITL 개선: 사람 개입 정책 개선

    4. Code Generation: Generate executable Python code
       코드 생성: 실행 가능한 Python 코드 생성
    """

    def __init__(self, rag_system, default_llm_client, llm_client_stage_1=None, llm_client_stage_2=None, expert_preference_manager=None):
        """
        Initialize PseuDRAGON Pipeline
        PseuDRAGON 파이프라인 초기화

        Args:
            rag_system: RAG system for legal knowledge retrieval
                       법적 지식 검색을 위한 RAG 시스템
            llm_client: Default OpenAI client for LLM operations
                       LLM 작업을 위한 기본 OpenAI 클라이언트
            llm_client_stage_1: Optional OpenAI client for Stage 1 (PII Detection)
                          Stage 1 (PII 탐지)을 위한  클라이언트
            llm_client_stage_2: Optional OpenAI client for Stage 2 (Policy Synthesis)
                          Stage 2 (정책 합성)을 위한 클라이언트
            expert_preference_manager: Optional expert preference manager for learned PII classifications
                                      학습된 PII 분류를 위한 전문가 선호도 관리자
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
        """
        Initialize all pipeline stages
        모든 파이프라인 스테이지 초기화
        """

        from pseudragon.stages.stage1_pii_detection import Stage1PIIDetection
        from pseudragon.stages.stage2_policy_synthesis import Stage2PolicySynthesis
        from pseudragon.stages.stage3_hitl_refinement import Stage3HITLRefinement
        from pseudragon.stages.stage4_code_generation import Stage4CodeGeneration

        self.stage1 = Stage1PIIDetection(
            self.rag,
            self.llm_client_stage_1,
            expert_preference_manager=self.expert_preference_manager
        )
        self.stage2 = Stage2PolicySynthesis(self.rag, self.llm_client_stage_2)
        self.stage3 = Stage3HITLRefinement(self.validator)
        self.stage4 = Stage4CodeGeneration(self.llm_client_stage_2)

    def _load_feedback_knowledge(self) -> None:
        """
        Load feedback knowledge base from past HITL sessions
        과거 HITL 세션에서 피드백 지식베이스 로드
        
        This method loads audit logs from the output directory and integrates
        expert feedback into the RAG knowledge base for improved policy synthesis.
        이 메서드는 output 디렉토리에서 감사 로그를 로드하고
        개선된 정책 합성을 위해 전문가 피드백을 RAG 지식베이스에 통합합니다.
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
            output_dir: Optional[str] = None
    ) -> str:
        """
        Execute the full PseuDRAGON pipeline
        전체 PseuDRAGON 파이프라인 실행

        Args:
            schema: Database schema (column names and sample values)
                   데이터베이스 스키마 (컬럼 이름 및 샘플 값)
            table_name: Name of the table
                       테이블 이름
            preferred_method: User's preferred pseudonymization method
                                사용자의 선호하는 익명화 기법
            db_config: Database connection path or config
                      데이터베이스 연결 경로 또는 설정
            hitl_callback: Callback for human-in-the-loop interaction
                          사람 개입 상호작용을 위한 콜백
            log_callback: Callback for logging
                         로깅을 위한 콜백

        Returns:
            Generated Python code as string
            문자열로 된 생성된 Python 코드

        Raises:
            PipelineError: If any stage fails
                          스테이지 실패 시
        """
        try:
            # self._log_stage_start("PseuDRAGON Pipeline", log_callback)

            pii_metadata = self._run_stage1(schema, table_name, log_callback)
            policy_initial = self._run_stage2(pii_metadata, table_name, preferred_method, log_callback)

            # Generate integrated report after Stage 1 and 2
            self._generate_integrated_report(table_name, pii_metadata, policy_initial, log_callback, output_dir)

            policy_final = self._run_stage3(policy_initial, preferred_method, hitl_callback, log_callback)
            code = self._run_stage4(schema, policy_final, pii_metadata, db_config, log_callback)

            # self._log_stage_complete("PseuDRAGON Pipeline", log_callback)

            return code

        except Exception as e:
            error_msg = f"Pipeline execution failed: {str(e)}"
            self._log(error_msg, log_callback)
            raise PipelineError(error_msg) from e

    def _run_stage1(self, schema: Dict[str, Any], table_name: str, log_callback: Optional[Callable]) -> Dict[str, Any]:
        """
        Run Stage 1: PII Detection
        스테이지 1 실행: PII 탐지

        Args:
            schema: Database schema
                   데이터베이스 스키마
            table_name: Table name
                       테이블 이름
            log_callback: Logging callback
                         로깅 콜백

        Returns:
            PII metadata
            PII 메타데이터
        """
        # self._log_stage_start(PipelineStage.STAGE1_PII_DETECTION, log_callback)
        pii_metadata = self.stage1.identify_pii(schema, table_name, log_callback)
        # self._log_stage_complete(PipelineStage.STAGE1_PII_DETECTION, log_callback)
        return pii_metadata

    def _run_stage2(self, pii_metadata: Dict[str, Any], table_name: str, preferred_method: str, purpose_goal: str = "", log_callback: Optional[Callable[[str], None]] = None) -> Policy:
        """
        Run Stage 2: Policy Synthesis
        스테이지 2 실행: 정책 합성

        Args:
            pii_metadata: PII metadata from Stage 1
                         스테이지 1의 PII 메타데이터
            table_name: Table name
                       테이블 이름
            preferred_method: User's preferred pseudonymization method
                               사용자가 선호하는 가명화 기법
            purpose_goal: User's intended purpose for data pseudonymization
                         사용자의 데이터 가명처리 의도(목적)
            log_callback: Logging callback
                         로깅 콜백

        Returns:
            Initial policy
            초기 정책
        """
        # self._log_stage_start(PipelineStage.STAGE2_POLICY_SYNTHESIS, log_callback)
        policy = self.stage2.synthesize_policy(pii_metadata, table_name, preferred_method, purpose_goal, log_callback)
        # self._log_stage_complete(PipelineStage.STAGE2_POLICY_SYNTHESIS, log_callback)
        return policy

    def _run_stage3(self, policy: Policy, preferred_method: str, hitl_callback: Optional[Callable[[Policy], Policy]] = None, log_callback: Optional[Callable[[str], None]] = None) -> tuple[
        Policy, list[ValidationViolation]]:
        """
        Run Stage 3: HITL Refinement
        스테이지 3 실행: HITL 개선

        Args:
            policy: Initial policy from Stage 2
                   스테이지 2의 초기 정책
            preferred_method: User's preferred pseudonymization method
                               사용자가 선호하는 가명화 기법
            hitl_callback: Callback for HITL interaction
                          HITL 상호작용 콜백
            log_callback: Logging callback
                         로깅 콜백

        Returns:
            Tuple of (refined policy, violations)
            (개선된 정책, 위반 사항) 튜플
        """
        # self._log_stage_start(PipelineStage.STAGE3_HITL_REFINEMENT, log_callback)
        policy_refined, violations = self.stage3.refine_policy(policy, preferred_method, hitl_callback, log_callback)
        # self._log_stage_complete(PipelineStage.STAGE3_HITL_REFINEMENT, log_callback)

        return policy_refined, violations

    def _run_stage4(self, schema: Dict[str, Any], policy: Policy, pii_metadata: Dict[str, Any], db_config: Optional[str], log_callback: Optional[Callable]) -> str:
        """
        Run Stage 4: Code Generation
        스테이지 4 실행: 코드 생성

        Args:
            schema: Database schema
                   데이터베이스 스키마
            policy: Final policy from Stage 3
                   스테이지 3의 최종 정책
            pii_metadata: PII metadata
                         PII 메타데이터
            db_config: Database connection path or config
                      데이터베이스 연결 경로 또는 설정
            log_callback: Logging callback
                         로깅 콜백

        Returns:
            Generated Python code
            생성된 Python 코드
        """
        # self._log_stage_start(PipelineStage.STAGE4_CODE_GENERATION, log_callback)
        code = self.stage4.generate_code(schema, policy, pii_metadata, db_config, log_callback)
        # self._log_stage_complete(PipelineStage.STAGE4_CODE_GENERATION, log_callback)
        return code

    def _generate_integrated_report(self, table_name: str, pii_metadata: Dict[str, Any], policy: Any, log_callback: Optional[Callable], output_dir: Optional[str] = None) -> None:
        """
        Generate integrated report combining Stage 1 and Stage 2 results
        스테이지 1과 2 결과를 결합한 통합 보고서 생성

        Args:
            table_name: Table name
                       테이블 이름
            pii_metadata: PII metadata from Stage 1
                         스테이지 1의 PII 메타데이터
            policy: Policy from Stage 2
                   스테이지 2의 정책
            log_callback: Logging callback
                         로깅 콜백
            output_dir: Optional output directory
                       선택적 출력 디렉토리
        """
        from pseudragon.reporting.compliance_reporter import ReportGenerator

        self._log("[Report] Generating integrated report...", log_callback)

        try:
            report_gen = ReportGenerator()
            report_path = report_gen.generate_integrated_report(table_name, pii_metadata, policy, output_dir=output_dir)
            # Store the report path so web interface can access it
            # 웹 인터페이스가 접근할 수 있도록 보고서 경로 저장
            self._last_integrated_report = report_path
            self._log(f"[OK] Integrated report saved: {report_path}", log_callback)
        except Exception as e:
            self._log(f"[WARN] Failed to generate integrated report: {str(e)}", log_callback, )

    def _log(self, message: str, callback: Optional[Callable] = None) -> None:
        """
        Log a message
        메시지 로깅

        Args:
            message: Message to log
                    로깅할 메시지
            callback: Optional logging callback
                     선택적 로깅 콜백
        """
        if callback:
            callback(message)
        else:
            print(message)

    # def _log_stage_start(self, stage_name: str, callback: Optional[Callable] = None) -> None:
    #     """
    #     Log stage start
    #     스테이지 시작 로깅
    #
    #     Args:
    #         stage_name: Name of the stage
    #                    스테이지 이름
    #         callback: Optional logging callback
    #                  선택적 로깅 콜백
    #     """
    #     self._log(f"▶ Starting {stage_name}", callback)
    #
    # def _log_stage_complete(self, stage_name: str, callback: Optional[Callable] = None) -> None:
    #     """
    #     Log stage completion
    #     스테이지 완료 로깅
    #
    #     Args:
    #         stage_name: Name of the stage
    #                    스테이지 이름
    #         callback: Optional logging callback
    #                  선택적 로깅 콜백
    #     """
    #     self._log(f"✓ Completed {stage_name}", callback)

    # Public API methods for individual stage execution
    # 개별 스테이지 실행을 위한 Public API 메서드

    def identify_pii(self, schema: Dict[str, Any], table_name: str, log_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Public API: Execute Stage 1 - PII Identification
        Public API: 스테이지 1 실행 - PII 식별

        Args:
            schema: Database schema (column names and sample values)
                   데이터베이스 스키마 (컬럼 이름 및 샘플 값)
            table_name: Name of the table
                       테이블 이름
            log_callback: Optional callback for logging
                         로깅을 위한 선택적 콜백

        Returns:
            PII metadata dictionary
            PII 메타데이터 딕셔너리
        """
        return self._run_stage1(schema, table_name, log_callback)

    def synthesize_policy(
            self,
            pii_metadata: Dict[str, Any],
            table_name: str,
            preferred_method: str = "",
            purpose_goal: str = "",
            log_callback: Optional[Callable] = None,
            output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Public API: Execute Stage 2 - Policy Synthesis
        Public API: 스테이지 2 실행 - 정책 합성

        Args:
            pii_metadata: PII metadata from Stage 1
                         스테이지 1의 PII 메타데이터
            table_name: Name of the table
                       테이블 이름
            preferred_method: User's preferred pseudonymization method
                    사용자가 선호하는 가명화 기법
            purpose_goal: User's intended purpose for data pseudonymization
                         사용자의 데이터 가명처리 의도(목적)
            log_callback: Optional callback for logging
                         로깅을 위한 선택적 콜백
            output_dir: output directory for each session
                        산출물이 저장될 세션별 폴더 경로

        Returns:
            Policy dictionary (per-column policies)
            정책 딕셔너리 (컬럼별 정책)
        """
        policy = self._run_stage2(pii_metadata, table_name, preferred_method, purpose_goal, log_callback)

        # Generate integrated report after Stage 2
        # 스테이지 2 이후 통합 보고서 생성
        self._generate_integrated_report(table_name, pii_metadata, policy, log_callback, output_dir)

        # Convert Policy object to dictionary format expected by app.py
        # Policy 객체를 app.py가 기대하는 딕셔너리 형식으로 변환
        policy_dict = {}
        for col_name, col_policy in policy.columns.items():
            # Convert PolicyAction objects to method dictionaries
            methods = []
            for action in col_policy.candidate_actions:
                methods.append(
                    {
                        "method": action.action.value, "applicability": "High",  # Default value
                        "description": action.rationale,  # Unified language description
                        "code_snippet": action.code_snippet,  # CRITICAL: Include code snippet
                        "example_implementation": action.code_snippet,  # Also use as example
                        "parameters": action.parameters,
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
                "evidence_source": col_policy.action.legal_evidence if col_policy.action else "Unknown",
            }

        return policy_dict

    def generate_python_code(
            self,
            schema: Dict[str, Any],
            policy_data: Dict[str, Any],
            pii_metadata: Dict[str, Any],
            table_name: str = "",
            db_config: Any = None,
            log_callback: Optional[Callable] = None
    ) -> str:
        """
        Public API: Execute Stage 4 - Code Generation
        Public API: 스테이지 4 실행 - 코드 생성

        Args:
            schema: Database schema
                   데이터베이스 스키마
            policy_data: Policy data (from Stage 2/3)
                        정책 데이터 (스테이지 2/3에서)
            pii_metadata: PII metadata (from Stage 1)
                         PII 메타데이터 (스테이지 1에서)
            table_name: Name of the table
                       테이블 이름
            db_config: Database configuration
                      데이터베이스 설정
            log_callback: Optional callback for logging
                         로깅을 위한 선택적 콜백

        Returns:
            Generated Python code as string
            문자열로 된 생성된 Python 코드
        """
        # Convert policy_data dict back to Policy object
        # policy_data 딕셔너리를 Policy 객체로 다시 변환
        from pseudragon.domain.policy_dsl import (Policy, ColumnPolicy, PolicyAction, ActionType, )

        policy = Policy(table_name=table_name)

        for col_name, col_data in policy_data.items():
            # Get the first method as the primary action
            methods_list = col_data.get("recommended_methods", [])

            if not methods_list:
                # No methods - default to KEEP
                primary_action = PolicyAction(action=ActionType.KEEP, rationale="No specific method provided", code_snippet="# No transformation needed", )
                candidate_actions = []
            else:
                # Convert first method to primary action
                first_method = methods_list[0]
                primary_action = PolicyAction(
                    action=self._map_method_to_action_type(first_method.get("method", "KEEP")),
                    parameters=first_method.get("parameters", {}),
                    rationale=first_method.get("description", first_method.get("rationale", "")),
                    legal_evidence=col_data.get("legal_source", ""),
                    code_snippet=first_method.get("code_snippet", ""), )

                # Convert remaining methods to candidate actions
                candidate_actions = []
                for method in methods_list:
                    candidate_actions.append(
                        PolicyAction(
                            action=self._map_method_to_action_type(method.get("method", "KEEP")),
                            parameters=method.get("parameters", {}),
                            rationale=method.get("description", method.get("rationale", "")),
                            legal_evidence=col_data.get("legal_source", ""),
                            code_snippet=method.get("code_snippet", ""), )
                    )

            col_policy = ColumnPolicy(
                column_name=col_name,
                pii_type=col_data.get("pii_type", "Unknown"),
                is_pii=col_data.get("is_pii", False),
                action=primary_action,
                candidate_actions=candidate_actions, )

            policy.add_column_policy(col_policy)

        return self._run_stage4(schema, policy, pii_metadata, db_config, log_callback)

    def _map_method_to_action_type(self, method: str) -> "ActionType":
        """
        Map method string to ActionType enum
        메서드 문자열을 ActionType enum으로 매핑

        Args:
            method: Method name as string
                   문자열로 된 메서드 이름

        Returns:
            ActionType enum value
            ActionType 열거형 값
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
