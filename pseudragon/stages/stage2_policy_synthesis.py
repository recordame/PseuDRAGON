"""
Optimized Stage 2: Policy Synthesis with Universal Caching
범용 캐싱을 사용한 최적화된 Stage 2: 정책 합성

Supports both OpenAI (prompt caching) and Ollama (KV cache).
OpenAI (프롬프트 캐싱)와 Ollama (KV 캐시)를 모두 지원합니다.
"""

# Standard library imports
# 표준 라이브러리 import
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

# Project-specific imports
# 프로젝트 관련 import
from config.config import PolicyConfig, Settings
from pseudragon.domain.policy_dsl import ActionType, ColumnPolicy, Policy, PolicyAction
from pseudragon.llm.json_parser import safe_parse_json
from pseudragon.llm.session_manager import UniversalSessionManager


def load_prompt(stage_name: str, prompt_type: str = "system") -> str:
    """
    Load prompt template from resources/prompts/ based on PROMPT_MODE
    PROMPT_MODE에 따라 resources/prompts/에서 프롬프트 템플릿 로드
    
    Args:
        stage_name: Name of the stage (e.g., "stage2_policy_synthesis")
                   스테이지 이름
        prompt_type: Type of prompt ("system" or "user")
                    프롬프트 유형
    
    Returns:
        Prompt text content
        프롬프트 텍스트 내용
    """
    prompt_dir = os.path.join(os.path.dirname(__file__), "..", "..")

    # Extract stage number (stage1 or stage2) from stage_name
    # stage_name에서 스테이지 번호(stage1 또는 stage2) 추출
    if "stage1" in stage_name:
        stage = "stage1"
    elif "stage2" in stage_name:
        stage = "stage2"
    else:
        # Fallback to old behavior for other stages
        # 다른 스테이지는 이전 방식으로 처리
        filename = f"{stage_name}_{prompt_type}.txt"
        filepath = os.path.join(prompt_dir, "resources", "prompts", filename)
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    # Use Settings.get_prompt_path() for stage1 and stage2
    # stage1과 stage2는 Settings.get_prompt_path() 사용
    prompt_path = Settings.get_prompt_path(stage, prompt_type)
    filepath = os.path.join(prompt_dir, prompt_path)

    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


class Stage2PolicySynthesis:
    """
    Optimized Stage 2: Policy Synthesis with Universal Session Management
    범용 세션 관리를 사용한 최적화된 Stage 2: 정책 합성
    
    Key features:
    - Auto-detects LLM provider (OpenAI, Ollama, etc.)
    - Applies provider-specific caching optimizations
    - System prompt sent only once per table
    - Estimated 60-75% reduction in system prompt tokens
    
    Provider-specific optimizations:
    - OpenAI: Automatic prompt caching
    - Ollama: KV cache with keep_alive
    - Others: Manual conversation history
    """

    def __init__(self, rag_system, llm_client, provider: str = "auto", base_url: Optional[str] = None):
        """
        Initialize optimized Stage 2 with provider detection.
        
        Args:
            rag_system: RAG system for context retrieval
            llm_client: LLM client
            provider: LLM provider ("auto", "openai", "ollama", "other")
            base_url: Base URL for API (used for auto-detection)
        """
        self.rag = rag_system
        self.client = llm_client

        # Get base_url from Settings if not provided
        if base_url is None:
            base_url = getattr(Settings, 'STAGE_2_OPENAI_BASE_URL', None)

        # Initialize universal session manager with provider detection
        self.session_manager = UniversalSessionManager(llm_client, provider=provider, base_url=base_url)

        # Load system prompt once
        self.system_prompt = load_prompt("stage2_policy_synthesis", "system")

        # Initialize audit logger (optional, for COT logging)
        self.audit_logger = None

    def synthesize_policy(self, pii_analysis: Dict[str, Any], table_name: str, preferred_method: str = "", purpose_goal: str = "", log_callback: Optional[Callable] = None) -> Policy:
        """
        Synthesize policy from PII analysis and preferred method using session.
        세션을 사용하여 PII 분석과 선호 기법으로부터 정책을 합성합니다.

        Implements f_policy() from Algorithm 1 with session optimization.
        세션 최적화를 포함한 알고리즘 1의 f_policy()를 구현합니다.

        Now supports batch parallel processing for LLM calls.
        이제 LLM 호출을 위한 배치 병렬 처리를 지원합니다.

        Args:
            pii_analysis: PII analysis result from Stage 1
            table_name: Name of the table being processed
            preferred_method: User's preferred pseudonymization method
            purpose_goal: User's intended purpose for data pseudonymization (e.g., statistical analysis, ML, research)
            log_callback: Optional callback for logging

        Returns:
            Policy object with candidate actions for each PII column
        """
        self._log(f"[Stage 2] Policy Synthesis for '{table_name}'", log_callback)
        if purpose_goal:
            self._log(f"  > Pseudonymization Purpose: {purpose_goal}", log_callback)
        if preferred_method:
            self._log(f"  > Preferred method: {preferred_method}", log_callback)

        # Create or get session for this table
        # 이 테이블에 대한 세션 생성 또는 가져오기
        # CRITICAL: max_history=0 prevents token overflow
        # 중요: max_history=0으로 토큰 오버플로우 방지
        # Each column's policy synthesis is independent
        # 각 컬럼의 정책 합성은 독립적
        session_id = f"stage2_{table_name}"
        session = self.session_manager.get_or_create_session(
            session_id=session_id,
            system_prompt=self.system_prompt,
            ttl_minutes=30,
            max_history=0  # No history - each column is independent
            # 히스토리 없음 - 각 컬럼이 독립적
        )

        self._log(f"Session ID: {session_id}", log_callback)

        policy = Policy(table_name=table_name, preferred_method=preferred_method)

        # Store purpose_goal for use in method generation
        self._purpose_goal = purpose_goal

        # Categorize columns for parallel processing
        # 병렬 처리를 위한 컬럼 분류
        pii_columns: List[Tuple[str, Dict[str, Any]]] = []
        non_pii_columns: List[Tuple[str, Dict[str, Any]]] = []

        for column, analysis in pii_analysis.items():
            # CRITICAL: Data consistency check
            # Check pii_type first to determine if column is truly PII
            # This handles cases where is_pii and pii_type are inconsistent
            pii_type = analysis["pii_type"]
            is_truly_pii = analysis["is_pii"] and pii_type not in ["Non-PII", "Unknown", ""]

            # If pii_type indicates PII but is_pii is False, treat as PII (conservative approach)
            if pii_type in ["PII"]:
                is_truly_pii = True
                self._log(f"  [WARN]️  Data consistency: {column} has pii_type='{pii_type}', treating as PII", log_callback)

            if is_truly_pii:
                pii_columns.append((column, analysis))
            else:
                non_pii_columns.append((column, analysis))

        # Process PII and Non-PII columns in parallel batches
        # PII 및 Non-PII 컬럼을 배치로 병렬 처리
        batch_size = Settings.LLM_BATCH_SIZE
        max_workers = Settings.MAX_CONCURRENT_LLM_REQUESTS

        # Process Non-PII columns in parallel
        # Non-PII 컬럼 병렬 처리
        if non_pii_columns:
            self._log(f"\n  [Parallel] Parallel Non-PII processing: {len(non_pii_columns)} columns (batch_size={batch_size})", log_callback)

            for batch_start in range(0, len(non_pii_columns), batch_size):
                batch_end = min(batch_start + batch_size, len(non_pii_columns))
                batch = non_pii_columns[batch_start:batch_end]
                batch_num = (batch_start // batch_size) + 1
                total_batches = (len(non_pii_columns) + batch_size - 1) // batch_size

                self._log(f"  [Batch] Non-PII batch {batch_num}/{total_batches} ({len(batch)} columns)", log_callback)

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_column = {}
                    for column, analysis in batch:
                        future = executor.submit(
                            self._process_non_pii_column,
                            session=session,
                            column=column,
                            analysis=analysis,
                            log_callback=log_callback
                        )
                        future_to_column[future] = (column, analysis)

                    for future in as_completed(future_to_column):
                        column, analysis = future_to_column[future]
                        try:
                            col_policy = future.result()
                            policy.add_column_policy(col_policy)
                        except Exception as e:
                            self._log(f"   [ERROR] Error processing Non-PII {column}: {str(e)}", log_callback)
                            # Fallback policy
                            col_policy = ColumnPolicy(
                                column_name=column,
                                pii_type=analysis["pii_type"],
                                is_pii=False,
                                action=PolicyAction(action=ActionType.KEEP, rationale="Default: Keep non-PII column"),
                                candidate_actions=[
                                    PolicyAction(action=ActionType.KEEP, rationale="Keep this non-PII column for analysis"),
                                    PolicyAction(action=ActionType.DELETE, rationale="Remove this column if not needed"),
                                ]
                            )
                            policy.add_column_policy(col_policy)

        # Process PII columns in parallel
        # PII 컬럼 병렬 처리
        if pii_columns:
            self._log(f"\n  [Parallel] Parallel PII processing: {len(pii_columns)} columns (batch_size={batch_size})", log_callback)

            for batch_start in range(0, len(pii_columns), batch_size):
                batch_end = min(batch_start + batch_size, len(pii_columns))
                batch = pii_columns[batch_start:batch_end]
                batch_num = (batch_start // batch_size) + 1
                total_batches = (len(pii_columns) + batch_size - 1) // batch_size

                self._log(f"  [Batch] PII batch {batch_num}/{total_batches} ({len(batch)} columns)", log_callback)

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_column = {}
                    for column, analysis in batch:
                        future = executor.submit(
                            self._process_pii_column,
                            session=session,
                            column=column,
                            analysis=analysis,
                            preferred_method=preferred_method,
                            purpose_goal=purpose_goal,
                            log_callback=log_callback
                        )
                        future_to_column[future] = (column, analysis)

                    for future in as_completed(future_to_column):
                        column, analysis = future_to_column[future]
                        try:
                            col_policy = future.result()
                            policy.add_column_policy(col_policy)
                        except Exception as e:
                            self._log(f"   [ERROR] Error processing PII {column}: {str(e)}", log_callback)
                            # Fallback policy
                            pii_type = analysis["pii_type"]
                            col_policy = ColumnPolicy(
                                column_name=column,
                                pii_type=pii_type,
                                is_pii=True,
                                action=PolicyAction(
                                    action=ActionType.from_string(PolicyConfig.get_default_action(is_pii=True)),
                                    rationale=f"Default action for PII: {PolicyConfig.DEFAULT_PII_ACTION}"
                                ),
                                candidate_actions=self._generate_fallback_methods(column, pii_type, "Fallback")
                            )
                            policy.add_column_policy(col_policy)

        # Get session stats for monitoring
        stats = session.get_stats()
        self._log(f"\nTokens saved in this session: ~{stats['estimated_tokens_saved']}", log_callback)

        return policy

    def _process_non_pii_column(self, session, column: str, analysis: Dict[str, Any], log_callback: Optional[Callable] = None) -> ColumnPolicy:
        """
        Process a single Non-PII column (used for parallel processing).
        단일 Non-PII 컬럼 처리 (병렬 처리에 사용).

        Args:
            session: LLM session
            column: Column name
            analysis: PII analysis result for this column
            log_callback: Optional logging callback

        Returns:
            ColumnPolicy for the Non-PII column
        """
        data_type = analysis.get("data_type", "")
        column_comment = analysis.get("column_comment", "")

        self._log(f"  -> Generating Non-PII methods for: {column}", log_callback)

        methods = self._generate_non_pii_methods_with_session(
            session=session,
            column=column,
            column_comment=column_comment,
            data_type=data_type,
            log_callback=log_callback,
        )

        default_action = PolicyConfig.get_default_action(is_pii=False)

        col_policy = ColumnPolicy(
            column_name=column,
            pii_type=analysis["pii_type"],
            is_pii=False,
            action=(methods
            [0] if methods
             else PolicyAction(
                action=ActionType.from_string(default_action),
                rationale=f"Non-PII column - default action: {default_action}",
            )),
            candidate_actions=methods
             if methods
             else [
                PolicyAction(action=ActionType.KEEP, rationale="Keep this non-PII column for analysis"),
                PolicyAction(action=ActionType.DELETE, rationale="Remove this column if not needed"),
            ]
        )

        self._log(f"   [OK] Generated {len(methods)} Non-PII methods for {column}", log_callback)
        return col_policy

    def _process_pii_column(self, session, column: str, analysis: Dict[str, Any], preferred_method: str, purpose_goal: str, log_callback: Optional[Callable] = None) -> ColumnPolicy:
        """
        Process a single PII column (used for parallel processing).
        단일 PII 컬럼 처리 (병렬 처리에 사용).

        Args:
            session: LLM session
            column: Column name
            analysis: PII analysis result for this column
            preferred_method: User's preferred method
            purpose_goal: User's purpose goal
            log_callback: Optional logging callback

        Returns:
            ColumnPolicy for the PII column
        """
        pii_type = analysis["pii_type"]
        data_type = analysis.get("data_type", "")
        column_comment = analysis.get("column_comment", "")

        self._log(f"  -> Generating policy for: {column}", log_callback)

        query = f"Pseudonymization and anonymization methods for {pii_type} ({column}) considering preferred method '{preferred_method}'"
        context, source_docs = self.rag.retrieve(query)

        methods = self._generate_pii_methods_with_session(
            session=session,
            column=column,
            pii_type=pii_type,
            preferred_method=preferred_method,
            purpose_goal=purpose_goal,
            context=context,
            data_type=data_type,
            column_comment=column_comment,
            source_docs=source_docs,
            log_callback=log_callback
        )

        col_policy = ColumnPolicy(
            column_name=column,
            pii_type=pii_type,
            is_pii=True,
            action=(methods
            [0] if methods
             else PolicyAction(
                action=ActionType.from_string(PolicyConfig.get_default_action(is_pii=True)),
                rationale=f"Default action for PII: {PolicyConfig.DEFAULT_PII_ACTION}"
            )),
            candidate_actions=methods

        )

        self._log(f"   [OK] Generated {len(methods)} methods for {column}", log_callback)
        return col_policy

    def _generate_non_pii_methods_with_session(
            self,
            session,
            column: str,
            column_comment: str = "",
            data_type: str = "",
            log_callback: Optional[Callable] = None
    ) -> List[PolicyAction]:
        """
        Generate recommended methods
         for Non-PII columns using LLM.
        LLM을 사용하여 Non-PII 컬럼에 대한 추천 기법 생성.

        Args:
            session: Active LLM session
            column: Column name
            column_comment: Column description
            data_type: Data type of the column
            log_callback: Optional callback for logging

        Returns:
            List of PolicyAction objects for Non-PII columns
        """
        # Load Non-PII specific prompts
        try:
            non_pii_system_prompt = load_prompt("non_pii_methods", "system")
            non_pii_user_prompt_template = load_prompt("non_pii_methods", "user")
        except FileNotFoundError:
            self._log(f"  WARNING: Non-PII prompts not found, using default KEEP/DELETE", log_callback)
            return [
                PolicyAction(action=ActionType.KEEP, rationale="Keep this non-PII column for analysis"),
                PolicyAction(action=ActionType.DELETE, rationale="Remove this column if not needed"),
            ]

        # Create separate session for Non-PII (uses different system prompt)
        non_pii_session_id = f"stage2_non_pii_{session.session_id}"
        non_pii_session = self.session_manager.get_or_create_session(
            session_id=non_pii_session_id,
            system_prompt=non_pii_system_prompt,
            ttl_minutes=30,
            max_history=0
        )

        # Format user prompt
        user_prompt = non_pii_user_prompt_template.format(
            column=column,
            column_desc=column_comment if column_comment else "No description",
            data_type=data_type if data_type else "Unknown"
        )

        # Prepare streaming settings
        stream_enabled = Settings.ENABLE_STREAMING and non_pii_session.supports_streaming()
        stream_callback = None

        if stream_enabled and Settings.STREAMING_LOG_CHUNKS:
            accumulated_chunks = []

            def log_stream_chunk(chunk: str):
                accumulated_chunks.append(chunk)

            stream_callback = log_stream_chunk

        # Call LLM
        try:
            response = non_pii_session.chat(
                user_prompt=user_prompt,
                model=Settings.LLM_STAGE_2,
                temperature=0.1,
                response_format={"type": "json_object"},
                stream=stream_enabled,
                stream_callback=stream_callback
            )

            content = response.choices[0].message.content
            result = safe_parse_json(
                content, default_response={
                    "recommended_methods": [
                        {"method": "KEEP", "description": "Keep this non-PII column for analysis", "code_snippet": f"# Keep df['{column}'] as-is"},
                        {"method": "DELETE", "description": "Remove this column if not needed", "code_snippet": f"# df.drop(columns=['{column}'], inplace=True)"}
                    ]
                }
            )

            # Parse methods

            methods = []
            for method in result.get("recommended_methods", []):
                action_type = self._map_method_to_action(method["method"])
                rationale = method.get("description", "")
                code_snippet = method.get("code_snippet", "")

                methods.append(
                    PolicyAction(
                        action=action_type,
                        parameters=method.get("parameters", {}),
                        code_snippet=code_snippet,
                        rationale=rationale,
                        legal_evidence=method.get("legal_source", "Data Minimization Principle")
                    )
                )

            # Ensure KEEP and DELETE are always included
            has_keep = any(t.action == ActionType.KEEP for t in methods
            )
            has_delete = any(t.action == ActionType.DELETE for t in methods
            )

            if not has_keep:
                methods.append(
                    PolicyAction(
                        action=ActionType.KEEP,
                        rationale="Keep this non-PII column for analysis",
                        code_snippet=f"# Keep df['{column}'] as-is"
                    )
                )

            if not has_delete:
                methods.append(
                    PolicyAction(
                        action=ActionType.DELETE,
                        rationale="Remove this column if not needed",
                        code_snippet=f"# df.drop(columns=['{column}'], inplace=True)"
                    )
                )

            return methods


        except Exception as e:
            self._log(f"  ERROR generating Non-PII methods for {column}: {e}", log_callback)
            return [
                PolicyAction(action=ActionType.KEEP, rationale="Keep this non-PII column for analysis", code_snippet=f"# Keep df['{column}'] as-is"),
                PolicyAction(action=ActionType.DELETE, rationale="Remove this column if not needed", code_snippet=f"# df.drop(columns=['{column}'], inplace=True)"),
            ]

    def _generate_pii_methods_with_session(
            self,
            session,
            column: str,
            pii_type: str,
            preferred_method: str,
            purpose_goal: str,
            context: str,
            data_type: str = "",
            column_comment: str = "",
            source_docs: list[str] = None,
            log_callback: Optional[Callable] = None
    ) -> List[PolicyAction]:
        """
        Generate recommended methods
         using session (system prompt already set).
        세션을 사용한 추천 기법 생성 (시스템 프롬프트 이미 설정됨).

        Args:
            session: Active LLM session
            column: Column name
            pii_type: PII type from Stage 1
            preferred_method: User's preferred pseudonymization method
            purpose_goal: User's intended purpose for data pseudonymization
            context: Retrieved legal context
            column_comment: Optional column description
            source_docs: List of source document names
            log_callback: Optional callback for logging

        Returns:
            List of PolicyAction objects
        """
        user_prompt_template = load_prompt("stage2_policy_synthesis", "user")

        user_prompt = user_prompt_template.format(
            context=context,
            column=column,
            pii_type=pii_type,
            data_type=data_type,
            column_desc=column_comment,
            preferred_method=(preferred_method if preferred_method else "General Pseudonymization"),
            purpose_goal=(purpose_goal if purpose_goal else "General data analysis"),
        )

        # Prepare streaming settings
        stream_enabled = Settings.ENABLE_STREAMING and session.supports_streaming()
        stream_callback = None

        if stream_enabled and Settings.STREAMING_LOG_CHUNKS:
            # Accumulate chunks for logging (but don't log individual chunks)
            accumulated_chunks = []

            def log_stream_chunk(chunk: str):
                """Real-time logging callback for streaming chunks."""
                accumulated_chunks.append(chunk)
                # Note: Individual chunks are not logged to avoid excessive output

            stream_callback = log_stream_chunk

        max_retries = 3
        for attempt in range(max_retries):
            # Note: Streaming header removed to reduce log verbosity

            # Use session.chat() with streaming support
            response = session.chat(
                user_prompt=user_prompt,
                model=Settings.LLM_STAGE_2,
                temperature=0.1,
                response_format={"type": "json_object"},
                stream=stream_enabled,
                stream_callback=stream_callback
            )

            # Note: Streaming completion log removed to reduce verbosity

            # Safely parse JSON response (handles Ollama and other providers)
            content = response.choices[0].message.content
            result = safe_parse_json(content, default_response={"recommended_methods": [], "evidence_source": "Failed to parse LLM response"})

            # Log Chain-of-Thought reasoning if available
            if "chain_of_thought" in result:
                self._log_cot_reasoning(column, pii_type, result["chain_of_thought"], log_callback)

                # Also log to audit logger if available
                if self.audit_logger:
                    methods_count = len(result.get("recommended_methods", []))
                    self.audit_logger.log_stage2_cot_reasoning(
                        table="",  # Table name not available in this context
                        column=column,
                        pii_type=pii_type,
                        chain_of_thought=result["chain_of_thought"],
                        methods_count=methods_count
                    )

            methods_data = result.get("recommended_methods", [])
            all_have_code = all(method.get("code_snippet", "").strip() != "" for method in methods_data if method.get("method", "").upper() not in ["KEEP", "DELETE"])

            if all_have_code or attempt == max_retries - 1:
                if not all_have_code:
                    self._log(f"  WARNING: After {max_retries} attempts, LLM still didn't provide all code_snippets for {column}", None, )
                break
            else:
                self._log(f"  Retry {attempt + 1}/{max_retries}: LLM didn't provide code_snippet, retrying...", None, )

        evidence = result.get("evidence_source", "Unknown Source")

        if source_docs:
            # Filter out feedback sources for the evidence field
            legal_sources = [s for s in source_docs if not s.startswith("feedback:")]
            if legal_sources:
                evidence = ", ".join(legal_sources)

        methods = []
        for method in result.get("recommended_methods", []):
            action_type = self._map_method_to_action(method["method"])

            # Use description field as rationale (unified language)
            rationale = method.get("description", "")

            # Get code snippet from LLM or use standard snippet
            code_snippet = method.get("code_snippet", "")

            # Extract legal evidence from method or fall back to general evidence
            method_legal_evidence = method.get("legal_source", evidence)

            methods.append(PolicyAction(action=action_type, parameters=method.get("parameters", {}), code_snippet=code_snippet, rationale=rationale, legal_evidence=method_legal_evidence))

        # Separate KEEP/DELETE from other methods

        keep_delete_methods = []
        other_methods = []

        for method in methods:
            if method.action in [ActionType.KEEP, ActionType.DELETE]:
                keep_delete_methods.append(method)
            else:
                other_methods.append(method)

        # Validate: Ensure at least 2 anonymization methods

        # If LLM failed to generate sufficient methods
        # , use fallback
        if len(other_methods) < 2:
            self._log(f"  WARNING: Only {len(other_methods)} anonymization method(s) generated for {column}. Adding fallback methods...", None)
            fallback_methods = self._generate_fallback_methods(column, pii_type, evidence)

            # Add fallback methods
            #  that aren't already present
            existing_actions = {t.action for t in other_methods}
            for fallback_method in fallback_methods:
                if fallback_method.action not in existing_actions:
                    other_methods.append(fallback_method)
                    if len(other_methods) >= 2:
                        break

        # Always add KEEP option if not already present
        has_keep = any(t.action == ActionType.KEEP for t in keep_delete_methods)
        if not has_keep:
            keep_delete_methods.append(
                PolicyAction(action=ActionType.KEEP, rationale="Keep the column as-is without any transformation for analysis purposes.", legal_evidence=evidence, code_snippet="")
            )

        # Always add DELETE option if not already present
        has_delete = any(t.action == ActionType.DELETE for t in keep_delete_methods)
        if not has_delete:
            # DELETE doesn't need code snippet
            keep_delete_methods.append(PolicyAction(action=ActionType.DELETE, rationale="Remove the column if not needed for analysis.", legal_evidence=evidence, code_snippet=""))

        # Return other methods
        #  first, then KEEP/DELETE at the end
        return other_methods + keep_delete_methods

    def _generate_fallback_methods(self, column: str, pii_type: str, legal_evidence: str) -> List[PolicyAction]:
        """
        Generate fallback anonymization methods
         when LLM fails to produce sufficient methods

        LLM이 충분한 기법을 생성하지 못할 때 대체 익명화 기법 생성

        Args:
            column: Column name
            pii_type: PII type (PII)
            legal_evidence: Legal evidence source

        Returns:
            List of fallback PolicyAction objects
        """
        fallback_methods = []

        # HASH: Universal method for all PII types
        fallback_methods.append(
            PolicyAction(
                action=ActionType.HASH,
                rationale=f"SHA-256 hashing ensures irreversible pseudonymization for {pii_type}",
                legal_evidence=legal_evidence,
                code_snippet=f"import hashlib\nhashed_{column} = hashlib.sha256(str(record['{column}']).encode()).hexdigest()",
                parameters={"algorithm": "SHA-256"}
            )
        )

        # MASK: Suitable for PII
        if pii_type == "PII":
            fallback_methods.append(
                PolicyAction(
                    action=ActionType.MASK,
                    rationale=f"Partial masking retains data utility while protecting {pii_type}",
                    legal_evidence=legal_evidence,
                    code_snippet=f"masked_{column} = str(record['{column}'])[:-4] + '****' if len(str(record['{column}'])) > 4 else '****'",
                    parameters={"mask_char": "*", "keep_last": 4}
                )
            )
        else:
            # For Sensitive Info, use more aggressive masking
            fallback_methods.append(
                PolicyAction(
                    action=ActionType.MASK,
                    rationale=f"Full masking ensures complete protection of {pii_type}",
                    legal_evidence=legal_evidence,
                    code_snippet=f"masked_{column} = '*' * len(str(record['{column}']))",
                    parameters={"mask_char": "*", "full_mask": True}
                )
            )

        # TOKENIZATION: Suitable for PII (reversible)
        if pii_type == "PII":
            # Note: This code snippet assumes token_map dictionary is initialized
            # in the execution context (e.g., token_map = {} before processing)
            fallback_methods.append(
                PolicyAction(
                    action=ActionType.TOKENIZE,
                    rationale=f"Tokenization allows reversible pseudonymization for {pii_type}",
                    legal_evidence=legal_evidence,
                    code_snippet=f"import uuid\nif '{column}' not in token_map:\n    token_map['{column}'] = {{}}\nif record['{column}'] not in token_map['{column}']:\n    token_map['{column}']["
                                 f"record['{column}']] = str(uuid.uuid4())\ntokenized_{column} = token_map['{column}'][record['{column}']]",
                    parameters={"reversible": True}
                )
            )

        return fallback_methods

    def _log_cot_reasoning(self, column: str, pii_type: str, cot: Dict[str, Any], log_callback: Optional[Callable] = None):
        """
        Log chain-of-thought reasoning for policy synthesis.
        정책 합성의 chain-of-thought 추론 로깅.

        Args:
            column: Column name
            pii_type: PII type
            cot: Chain-of-thought dictionary from LLM response
            log_callback: Optional callback for logging
        """
        self._log(f"\n   [CoT] [Chain-of-Thought] {column} ({pii_type}):", log_callback)

        # PII Analysis
        if "pii_analysis" in cot:
            self._log(f"      [Stats] PII Analysis: {cot['pii_analysis']}", log_callback)

        # method Evaluation
        if "method_evaluation" in cot and cot["method_evaluation"]:
            self._log(f"      [Method] method Evaluation:", log_callback)
            for i, eval_item in enumerate(cot["method_evaluation"], 1):
                self._log(f"         {i}. {eval_item}", log_callback)

        # Legal Compliance
        if "legal_compliance" in cot and cot["legal_compliance"]:
            self._log(f"      Legal Compliance:", log_callback)
            for req in cot["legal_compliance"]:
                self._log(f"         - {req}", log_callback)

        # Implementation Rationale
        if "implementation_rationale" in cot:
            self._log(f"      [Tip] Implementation: {cot['implementation_rationale']}", log_callback)

        self._log("", log_callback)  # Empty line for readability

    def set_audit_logger(self, audit_logger):
        """
        Set audit logger for COT logging.
        COT 로깅을 위한 감사 로거 설정.

        Args:
            audit_logger: AuditLogger instance
        """
        self.audit_logger = audit_logger

    def _map_method_to_action(self, method: str) -> ActionType:
        """
        Map LLM-generated method name to ActionType enum
        LLM이 생성한 메서드 이름을 ActionType enum으로 매핑합니다
        """
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
            "KEEP": ActionType.KEEP
        }

        return mapping.get(method_upper, ActionType.HASH)

    def _log(self, message: str, callback: Optional[Callable] = None):
        """
        Helper for logging
        로깅을 위한 헬퍼 함수
        """
        if callback:
            callback(message)
        else:
            print(message)

    def cleanup_sessions(self, log_callback: Optional[Callable] = None):
        """Clean up all sessions (call at end of pipeline)."""
        stats = self.session_manager.get_all_stats()
        self._log(f"[Stage 2] Session Manager Stats: {json.dumps(stats, indent=2)}", log_callback)

        # Close all sessions
        for session_id in list(self.session_manager.sessions.keys()):
            self.session_manager.close_session(session_id)
