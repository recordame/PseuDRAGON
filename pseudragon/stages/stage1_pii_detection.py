"""
Optimized Stage 1: PII Detection with Universal Caching
범용 캐싱을 사용한 최적화된 Stage 1: PII 탐지

Supports both OpenAI (prompt caching) and Ollama (KV cache).
OpenAI (프롬프트 캐싱)와 Ollama (KV 캐시)를 모두 지원합니다.
"""

# Standard library imports
# 표준 라이브러리 import
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

# Project-specific imports
# 프로젝트 관련 import
from config.config import HeuristicConfig, Settings
from pseudragon.llm.json_parser import safe_parse_json
from pseudragon.llm.session_manager import UniversalSessionManager


def load_prompt(stage_name: str, prompt_type: str = "system") -> str:
    """
    Load prompt template from resources/prompts/ based on PROMPT_MODE
    PROMPT_MODE에 따라 resources/prompts/에서 프롬프트 템플릿 로드
    
    Args:
        stage_name: Name of the stage (e.g., "stage1_pii_identification")
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


class Stage1PIIDetection:
    """
    Optimized Stage 1: PII Identification with Universal Session Management
    범용 세션 관리를 사용한 최적화된 Stage 1: PII 식별
    
    Key features:
    - Auto-detects LLM provider (OpenAI, Ollama, etc.)
    - Applies provider-specific caching optimizations
    - System prompt sent only once per table
    - Estimated 70-90% reduction in system prompt tokens
    
    Provider-specific optimizations:
    - OpenAI: Automatic prompt caching
    - Ollama: KV cache with keep_alive
    - Others: Manual conversation history
    """

    def __init__(self, rag_system, llm_client, provider: str = "auto", base_url: Optional[str] = None, expert_preference_manager=None):
        """
        Initialize optimized Stage 1 with provider detection.

        Args:
            rag_system: RAG system for context retrieval
            llm_client: LLM client
            provider: LLM provider ("auto", "openai", "ollama", "other")
            base_url: Base URL for API (used for auto-detection)
            expert_preference_manager: Optional user preference manager for learned PII classifications
        """
        self.rag = rag_system
        self.client = llm_client
        self.expert_preference_manager = expert_preference_manager

        # Get base_url from Settings if not provided
        if base_url is None:
            base_url = getattr(Settings, 'STAGE_1_OPENAI_BASE_URL', None)

        # Initialize universal session manager with provider detection
        self.session_manager = UniversalSessionManager(llm_client, provider=provider, base_url=base_url)

        # Load system prompt once
        self.system_prompt = load_prompt("stage1_pii_identification", "system")

        # Load heuristic patterns from HeuristicManager
        from pseudragon.heuristics.heuristic_manager import HeuristicManager
        self.heuristic_manager = HeuristicManager()

        # Load environment-based patterns as fallback
        self.heuristic_patterns = HeuristicConfig.get_heuristic_patterns()
        self.pii_type_keywords = HeuristicConfig.get_pii_type_keywords()

        # Load Non-PII patterns
        self.non_pii_patterns = HeuristicConfig.get_non_pii_patterns()

        # Initialize audit logger (optional, for COT logging)
        self.audit_logger = None

    def identify_pii(self, schema: Dict[str, Any], table_name: str, log_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Identify PII columns using session-based approach with provider detection.
        제공자 감지를 통한 세션 기반 접근법으로 PII 컬럼을 식별합니다.

        Now supports batch parallel processing for LLM verification.
        이제 LLM 검증을 위한 배치 병렬 처리를 지원합니다.

        Args:
            schema: Enhanced schema with column metadata
            table_name: Name of the table
            log_callback: Optional logging callback

        Returns:
            Dictionary mapping column names to PII analysis results
        """
        results = {}

        # Create or get session for this table
        # 이 테이블에 대한 세션 생성 또는 가져오기
        # CRITICAL: max_history=0 prevents token overflow
        # 중요: max_history=0으로 토큰 오버플로우 방지
        # Each column verification is independent, so no conversation history is needed
        # 각 컬럼 검증은 독립적이므로 대화 히스토리가 필요하지 않음
        # This keeps only the system prompt and discards previous user/assistant messages
        # 시스템 프롬프트만 유지하고 이전 사용자/어시스턴트 메시지는 버림
        session_id = f"stage1_{table_name}"
        session = self.session_manager.get_or_create_session(
            session_id=session_id,
            system_prompt=self.system_prompt,
            ttl_minutes=60,
            max_history=0  # No history - each column is independent
            # 히스토리 없음 - 각 컬럼이 독립적
        )

        self._log(f"[Search] [Stage 1] PII Identification for '{table_name}'", log_callback)
        self._log(f"Provider: {session.provider} | Session ID: {session_id}", log_callback)

        # Statistics
        pii_heuristic_count = 0
        non_pii_heuristic_count = 0
        llm_verification_count = 0
        learned_classification_count = 0

        # Columns that need LLM verification (will be processed in parallel batches)
        # LLM 검증이 필요한 컬럼들 (배치로 병렬 처리됨)
        llm_verification_columns: List[Tuple[str, Dict[str, Any]]] = []

        # Learned heuristic classification count
        learned_heuristic_count = 0

        # First pass: Apply heuristics and learned classifications (fast, sequential)
        # 첫 번째 패스: 휴리스틱과 학습된 분류 적용 (빠름, 순차적)
        for column, metadata in schema.items():
            sample_value = metadata.get("sample_value", "")
            data_type = metadata.get("type", "")
            column_comment = metadata.get("comment", "")

            # Step 0: Check learned PII classification (HIGHEST PRIORITY)
            # If user has previously corrected this column's classification, use that
            # Now supports RAG-based similarity search for similar columns
            # 사용자가 이전에 이 컬럼의 분류를 수정한 경우 해당 분류 사용
            # 이제 유사한 컬럼에 대한 RAG 기반 유사도 검색 지원
            learned_result = None
            if self.expert_preference_manager:
                learned_result = self.expert_preference_manager.get_learned_pii_classification(
                    column_name=column,
                    column_comment=column_comment,
                    data_type=data_type
                )

            if learned_result:
                learned_classification_count += 1
                classification = learned_result["classification"]
                match_type = learned_result.get("match_type", "exact")
                similar_column = learned_result.get("similar_column", column)
                similarity = learned_result.get("similarity", 1.0)
                rationale = learned_result.get("rationale", "")
                similar_comment = learned_result.get("similar_column_comment", "")

                is_pii = (classification == "PII")

                # Determine evidence source based on match type
                if match_type == "exact":
                    evidence_source = "EXPERT FEEDBACK (Learned)"
                    reasoning_prefix = f"User previously classified {column} as"
                else:
                    evidence_source = f"EXPERT FEEDBACK (RAG Similar: {similar_column}, similarity={similarity:.2f})"
                    reasoning_prefix = f"Similar column {similar_column} was classified as"

                if is_pii:
                    self._log(f"   [OK] Learned PII: {column} ({match_type} match" + (f", similar to {similar_column}" if match_type == "rag_similar" else "") + ")", log_callback)
                    if match_type == "rag_similar" and similar_comment:
                        self._log(f"     Similar column comment: {similar_comment}", log_callback)
                    if rationale:
                        self._log(f"     Rationale: {rationale}", log_callback)

                    results[column] = {
                        "is_pii": True,
                        "pii_type": self._infer_pii_type(column, column_comment),
                        "reasoning": f"{reasoning_prefix} PII" + (f" (rationale: {rationale})" if rationale else ""),
                        "evidence_source": evidence_source,
                        "detection_method": "learned",
                        "confidence": "Very High" if match_type == "exact" else "High",
                        "rationale": rationale or f"User correction from previous session",
                        "column_comment": column_comment,
                    }
                else:
                    self._log(f"   [X] Learned Non-PII: {column} ({match_type} match" + (f", similar to {similar_column}" if match_type == "rag_similar" else "") + ")", log_callback)
                    if match_type == "rag_similar" and similar_comment:
                        self._log(f"     Similar column comment: {similar_comment}", log_callback)

                    results[column] = {
                        "is_pii": False,
                        "pii_type": "Non-PII",
                        "reasoning": f"{reasoning_prefix} Non-PII",
                        "evidence_source": evidence_source,
                        "detection_method": "learned",
                        "confidence": "Very High" if match_type == "exact" else "High",
                        "rationale": rationale or f"User correction from previous session",
                        "column_comment": column_comment,
                    }
                continue

            # Step 0.5: Check learned heuristic patterns (second highest priority)
            # Fast pattern matching using regex patterns learned from expert feedback
            # 전문가 피드백으로부터 학습된 정규식 패턴을 사용한 빠른 패턴 매칭
            # This is faster than RAG similarity search (< 1ms vs ~100ms)
            heuristic_result = None
            if self.expert_preference_manager:
                heuristic_result = self.expert_preference_manager.classify_by_heuristics(
                    column_name=column,
                    column_comment=column_comment
                )

            if heuristic_result:
                learned_heuristic_count += 1
                classification = heuristic_result["classification"]
                match_type = heuristic_result.get("match_type", "learned_heuristic")
                pattern = heuristic_result.get("pattern", "")
                confidence = heuristic_result.get("confidence", 0.8)
                samples = heuristic_result.get("samples", [])

                is_pii = (classification == "PII")

                evidence_source = f"LEARNED HEURISTIC ({match_type}: {pattern})"

                if is_pii:
                    self._log(f"   [OK] Learned Heuristic PII: {column} (pattern: {pattern}, conf: {confidence:.2f})", log_callback)
                    if samples:
                        self._log(f"     Similar columns: {', '.join(samples[:3])}", log_callback)

                    results[column] = {
                        "is_pii": True,
                        "pii_type": self._infer_pii_type(column, column_comment),
                        "reasoning": f"Matched learned heuristic pattern: {pattern}",
                        "evidence_source": evidence_source,
                        "detection_method": "learned_heuristic",
                        "confidence": "High" if confidence >= 0.9 else "Medium",
                        "rationale": f"Pattern learned from expert feedback (similar columns: {', '.join(samples[:3])})" if samples else f"Pattern learned from expert feedback",
                        "column_comment": column_comment,
                    }
                else:
                    self._log(f"   [X] Learned Heuristic Non-PII: {column} (pattern: {pattern}, conf: {confidence:.2f})", log_callback)

                    results[column] = {
                        "is_pii": False,
                        "pii_type": "Non-PII",
                        "reasoning": f"Matched learned heuristic pattern: {pattern}",
                        "evidence_source": evidence_source,
                        "detection_method": "learned_heuristic",
                        "confidence": "High" if confidence >= 0.9 else "Medium",
                        "rationale": f"Pattern learned from expert feedback",
                        "column_comment": column_comment,
                    }
                continue

            # Step 1: Check PII heuristic (ALWAYS checked first)
            # This ensures PII columns are caught before Non-PII patterns
            # Example: 'phone_number' matches PII pattern first, never reaches Non-PII check
            if self._heuristic_pii(column, column_comment):
                pii_heuristic_count += 1
                self._log(f"   [OK] Heuristic PII: {column}", log_callback)

                results[column] = {
                    "is_pii": True,
                    "pii_type": self._infer_pii_type(column, column_comment),
                    "reasoning": f"Matched PII heuristic pattern for {column}" + (f" (Comment: {column_comment})" if column_comment else ""),
                    "evidence_source": "Column Naming Heuristic" + (" + Column Comment Heuristic" if column_comment else ""),
                    "detection_method": "heuristic",
                    "confidence": "High",
                    "rationale": f"Column name matches known PII pattern",
                    "column_comment": column_comment,
                }

            # Step 2: Check Non-PII heuristic (only if NOT PII)
            # Safe to use broad patterns (e.g., 'number') because PII already filtered out
            elif self._heuristic_non_pii(column, column_comment):
                non_pii_heuristic_count += 1
                self._log(f"   [X] Heuristic Non-PII: {column} (LLM skipped)", log_callback)

                results[column] = {
                    "is_pii": False,
                    "pii_type": "Non-PII",
                    "reasoning": f"Matched Non-PII heuristic pattern (system/metadata column)" + (f" (Comment: {column_comment})" if column_comment else ""),
                    "evidence_source": "Non-PII Column Heuristic",
                    "detection_method": "heuristic",
                    "confidence": "High",
                    "rationale": f"Column identified as system or metadata field",
                    "column_comment": column_comment,
                }

            # Step 3: Uncertain cases - queue for LLM verification
            # 불확실한 케이스 - LLM 검증 큐에 추가
            else:
                llm_verification_columns.append((column, metadata))

        # Second pass: Process LLM verification columns in parallel batches
        # 두 번째 패스: LLM 검증 컬럼들을 배치로 병렬 처리
        if llm_verification_columns:
            batch_size = Settings.LLM_BATCH_SIZE
            max_workers = Settings.MAX_CONCURRENT_LLM_REQUESTS
            total_llm_columns = len(llm_verification_columns)

            self._log(f"\n   [Parallel] Parallel LLM verification: {total_llm_columns} columns (batch_size={batch_size}, max_workers={max_workers})", log_callback)

            # Process in batches
            # 배치로 처리
            for batch_start in range(0, total_llm_columns, batch_size):
                batch_end = min(batch_start + batch_size, total_llm_columns)
                batch = llm_verification_columns[batch_start:batch_end]
                batch_num = (batch_start // batch_size) + 1
                total_batches = (total_llm_columns + batch_size - 1) // batch_size

                self._log(f"   [Batch] Processing batch {batch_num}/{total_batches} ({len(batch)} columns)", log_callback)

                # Process batch in parallel using ThreadPoolExecutor
                # ThreadPoolExecutor를 사용하여 배치를 병렬 처리
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks in this batch
                    future_to_column = {}
                    for column, metadata in batch:
                        future = executor.submit(
                            self._process_single_column_llm,
                            session=session,
                            column=column,
                            metadata=metadata,
                            table_name=table_name,
                            log_callback=log_callback
                        )
                        future_to_column[future] = column

                    # Collect results as they complete
                    for future in as_completed(future_to_column):
                        column = future_to_column[future]
                        try:
                            result = future.result()
                            results[column] = result
                            llm_verification_count += 1
                        except Exception as e:
                            self._log(f"   [ERROR] Error processing {column}: {str(e)}", log_callback)
                            # Fallback to safe default
                            results[column] = {
                                "is_pii": False,
                                "pii_type": "Unknown",
                                "reasoning": f"Error during LLM verification: {str(e)}",
                                "evidence_source": "Error",
                                "detection_method": "llm",
                                "confidence": "Low",
                                "column_comment": metadata.get("comment", ""),
                            }
                            llm_verification_count += 1

        # Get session stats
        stats = session.get_stats()

        # Enhanced logging with optimization stats
        total_columns = len(schema)
        llm_skipped = learned_classification_count + learned_heuristic_count + pii_heuristic_count + non_pii_heuristic_count
        optimization_rate = (llm_skipped / total_columns * 100) if total_columns > 0 else 0

        self._log(f"\n[Stats] Stage 1 Summary:", log_callback)
        self._log(f"   Total columns: {total_columns}", log_callback)
        if learned_classification_count > 0:
            self._log(f"   Learned (EXPERT FEEDBACK): {learned_classification_count} ", log_callback)
        if learned_heuristic_count > 0:
            self._log(f"   Learned Heuristic: {learned_heuristic_count} [Update]", log_callback)
        self._log(f"   PII (heuristic): {pii_heuristic_count}", log_callback)
        self._log(f"   Non-PII (heuristic): {non_pii_heuristic_count}", log_callback)
        self._log(f"   LLM verification: {llm_verification_count}", log_callback)
        self._log(f"   LLM calls avoided: {llm_skipped} ({optimization_rate:.1f}%)", log_callback)
        self._log(f"   Tokens saved: ~{stats['estimated_tokens_saved']}", log_callback)

        return results

    def _process_single_column_llm(self, session, column: str, metadata: Dict[str, Any], table_name: str, log_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Process a single column with LLM verification (used for parallel processing).
        단일 컬럼을 LLM으로 검증 (병렬 처리에 사용).

        Args:
            session: LLM session
            column: Column name
            metadata: Column metadata
            table_name: Table name
            log_callback: Optional logging callback

        Returns:
            PII analysis result for the column
        """
        sample_value = metadata.get("sample_value", "")
        data_type = metadata.get("type", "")
        column_comment = metadata.get("comment", "")

        self._log(f"   -> LLM verification: {column}", log_callback)

        llm_result = self._llm_verify_pii_with_session(
            session=session,
            column=column,
            sample_value=sample_value,
            table_name=table_name,
            data_type=data_type,
            column_comment=column_comment,
            log_callback=log_callback
        )

        # Extract LLM decision for comparison
        llm_is_pii = llm_result.get("is_pii", False)
        llm_pii_type = llm_result.get("pii_type", "Unknown")

        if llm_is_pii:
            pii_type = llm_pii_type

            # If LLM returns "Unknown", use fallback inference
            if pii_type == "Unknown":
                pii_type = self._infer_pii_type(column, column_comment)
                self._log(f"   [WARN]️  [PII Type Override]: LLM returned 'Unknown', using fallback inference: {pii_type}", log_callback)

            result = {
                "is_pii": True,
                "pii_type": pii_type,
                "evidence": llm_result.get("evidence", ""),
                "reasoning": llm_result.get("reasoning", ""),
                "evidence_source": llm_result.get("evidence_source", "LLM Verification"),
                "column_comment": column_comment,
                "detection_method": "llm",
                "confidence": llm_result.get("confidence", "Medium")
            }
            self._log(f"    [OK] {column}: PII ({pii_type})", log_callback)
            self._log(f"   [OK] [Final Classification]: is_pii=True, pii_type={pii_type} (matches LLM decision)", log_callback)
        else:
            result = {
                "is_pii": False,
                "pii_type": "Non-PII",
                "evidence": llm_result.get("evidence", ""),
                "reasoning": llm_result.get("reasoning", ""),
                "evidence_source": llm_result.get("evidence_source", "LLM Verification"),
                "column_comment": column_comment,
                "detection_method": "llm",
                "confidence": llm_result.get("confidence", "Medium")
            }
            self._log(f"    [X] {column}: Not PII", log_callback)
            self._log(f"   [OK] [Final Classification]: is_pii=False (matches LLM decision)", log_callback)

        return result

    def _heuristic_pii(self, column_name: str, column_comment: str = "") -> bool:
        """
        Fast heuristic check using regex patterns on column name and comment.
        컬럼 이름과 COMMENT에 대한 정규식 패턴을 사용한 빠른 휴리스틱 검사.

        Implements HEURISTIC_PII() from Algorithm 1 with COMMENT support.
        COMMENT 지원을 포함한 알고리즘 1의 HEURISTIC_PII()를 구현합니다.

        Args:
            column_name: Name of the column
            column_comment: Column description/comment from database schema
        """
        column_lower = column_name.lower()
        comment_lower = column_comment.lower() if column_comment else ""

        # Check environment-based patterns first (for backwards compatibility)
        for pii_type, pattern in self.heuristic_patterns.items():
            if re.search(pattern, column_lower) or re.search(pattern, comment_lower):
                return True

        # Check HeuristicManager patterns
        matched_heuristic = self.heuristic_manager.match_column(column_name)
        if matched_heuristic and matched_heuristic.get("pii_type") != "Non-PII":
            return True

        # Also check comment against heuristics if available
        if column_comment:
            matched_comment_heuristic = self.heuristic_manager.match_column(column_comment)
            if matched_comment_heuristic and matched_comment_heuristic.get("pii_type") != "Non-PII":
                return True

        return False

    def _infer_pii_type(self, column_name: str, column_comment: str = "") -> str:
        """
        Infer PII type from column name and comment
        컬럼 이름과 COMMENT에서 PII 유형을 추론합니다

        CRITICAL: This function is only called for columns that matched PII heuristics.
        중요: 이 함수는 PII 휴리스틱에 매칭된 컬럼에 대해서만 호출됩니다.
        Therefore, it should NEVER return "Non-PII".
        따라서 "Non-PII"를 반환해서는 안 됩니다.

        Args:
            column_name: Name of the column
            column_comment: Column description/comment from database schema

        Returns:
            PII type: "PII" (NEVER returns "Non-PII")
        """
        # First check HeuristicManager for more specific matches
        matched_heuristic = self.heuristic_manager.match_column(column_name)
        if matched_heuristic:
            pii_type = matched_heuristic.get("pii_type")
            # CRITICAL: Only accept valid PII types, reject "Non-PII"
            # This prevents the bug where is_pii=True but pii_type="Non-PII"
            if pii_type and pii_type == "PII":
                return pii_type

        # Check comment if available
        if column_comment:
            matched_comment_heuristic = self.heuristic_manager.match_column(column_comment)
            if matched_comment_heuristic:
                pii_type = matched_comment_heuristic.get("pii_type")
                # CRITICAL: Only accept valid PII types, reject "Non-PII"
                if pii_type and pii_type == "PII":
                    return pii_type

        # Default: All PII columns are classified as PII
        # PII로 감지된 모든 컬럼은 PII로 분류됩니다
        return "PII"

    def _heuristic_non_pii(self, column_name: str, column_comment: str = "") -> bool:
        """
        Fast heuristic check for clearly Non-PII columns
        명확하게 Non-PII인 컬럼에 대한 빠른 휴리스틱 검사
        
        Identifies system columns, metadata, timestamps, counters, etc.
        that are clearly not personal information.
        시스템 컬럼, 메타데이터, 타임스탬프, 카운터 등
        명백히 개인정보가 아닌 것들을 식별합니다.
        
        Args:
            column_name: Name of the column
            column_comment: Column description/comment from database schema
        
        Returns:
            True if column is clearly Non-PII (skip LLM verification)
            컬럼이 명확하게 Non-PII인 경우 True (LLM 검증 건너뜀)
        """
        column_lower = column_name.lower()
        comment_lower = column_comment.lower() if column_comment else ""

        # Check environment-based patterns first
        for pattern_name, pattern in self.non_pii_patterns.items():
            if re.search(pattern, column_lower) or re.search(pattern, comment_lower):
                return True

        # Check HeuristicManager patterns for Non-PII
        matched_heuristic = self.heuristic_manager.match_column(column_name)
        if matched_heuristic and matched_heuristic.get("pii_type") == "Non-PII":
            return True

        # Check comment against heuristics if available
        if column_comment:
            matched_comment_heuristic = self.heuristic_manager.match_column(column_comment)
            if matched_comment_heuristic and matched_comment_heuristic.get("pii_type") == "Non-PII":
                return True

        return False

    def _llm_verify_pii_with_session(self, session, column: str, sample_value: Any, table_name: str, data_type: str = "", column_comment: str = "", log_callback: Optional[Callable] = None) -> Dict[
        str, Any]:
        """
        LLM-based PII verification using universal session.
        범용 세션을 사용한 LLM 기반 PII 검증.
        
        Automatically uses provider-specific optimizations:
        - OpenAI: Prompt caching + Streaming (if enabled)
        - Ollama: KV cache with keep_alive
        """
        # Build column description with structured format
        # 구조화된 형식으로 컬럼 설명 구성
        column_desc = f"Column Name: {column}"
        column_desc += f"\nData Type: {data_type if data_type else 'Not specified'}"
        column_desc += f"\nDescription: {column_comment if column_comment else 'Not provided'}"

        # Retrieve context from RAG with detailed information (include data type in query)
        # RAG에서 상세 정보와 함께 컨텍스트 검색 (쿼리에 데이터 타입 포함)
        data_type = data_type if data_type else ""
        query = f"Does the column '{column}'({data_type}) with the description '{column_comment}' and sample '{sample_value}' constitute Personal Identifiable Information (PII)?"

        # Use retrieve_with_details if available, otherwise fall back to retrieve
        # Prioritize 'status_change' feedback for PII identification
        #
        # NOTE: The RAG system now performs DUAL-MODEL SEARCH:
        # 새로운 RAG 시스템은 이중 모델 검색을 수행합니다:
        # 1. Legal documents: Legal-BERT for legal context
        #    법률 문서: 법률 컨텍스트를 위한 Legal-BERT
        # 2. Feedback documents: SBERT for semantic similarity (finds SIMILAR columns, not exact match)
        #    피드백 문서: 의미적 유사도를 위한 SBERT (정확한 매칭이 아닌 유사 컬럼 검색)
        #
        # metadata_filter is passed for logging purposes but NOT used for strict filtering
        # metadata_filter는 로깅 목적으로 전달되지만 엄격한 필터링에는 사용되지 않음
        metadata_filter = {'table': table_name, 'column': column}

        # Check if RAG is available (not None)
        # RAG 사용 가능 여부 확인 (None이 아닌지)
        if self.rag is not None:
            if hasattr(self.rag, 'retrieve_with_details'):
                context, source_docs, doc_details = self.rag.retrieve_with_details(query, exclude_feedback=False, allowed_feedback_types=['status_change'], metadata_filter=metadata_filter)
            else:
                context, source_docs = self.rag.retrieve(query, exclude_feedback=False, allowed_feedback_types=['status_change'], metadata_filter=metadata_filter)
                doc_details = []

            # Log RAG context retrieval for debugging (summary only)
            self._log(f"\n   [RAG] [RAG Context for '{column}' ({data_type or 'unknown type'})]:", log_callback)
            self._log(f"      Query: {query}", log_callback)
            self._log(f"      Retrieved {len(doc_details)} document(s)", log_callback)
            # Note: Document details are not logged to avoid excessive output
        else:
            # RAG is disabled - use empty context
            # RAG 비활성화 - 빈 컨텍스트 사용
            context = ""
            source_docs = []
            doc_details = []
            self._log(f"\n   [LLM-ONLY] RAG disabled, using LLM knowledge only for '{column}'", log_callback)

        # Load user prompt template
        user_prompt_template = load_prompt("stage1_pii_identification", "user")
        user_prompt = user_prompt_template.format(context=context, table_name=table_name, column_desc=column_desc, data_type=data_type, sample_value=sample_value, )

        # Prepare streaming callback for real-time COT logging
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

            # Note: Streaming header removed to reduce log verbosity

        # Use session.chat() with streaming support
        response = session.chat(
            user_prompt=user_prompt,
            model=Settings.LLM_STAGE_1,
            temperature=0.1,
            response_format={"type": "json_object"},
            stream=stream_enabled,
            stream_callback=stream_callback
        )

        # Note: Streaming completion log removed to reduce verbosity

        # Safely parse JSON response (handles Ollama and other providers)
        content = response.choices[0].message.content
        result = safe_parse_json(
            content,
            default_response={"is_pii": False, "pii_type": "Unknown", "evidence": "Failed to parse LLM response", "reasoning": content[:500] if content else "Empty response"}
        )

        # Log LLM's decision for transparency
        llm_is_pii = result.get("is_pii", False)
        llm_pii_type = result.get("pii_type", "Unknown")
        llm_reasoning = result.get("reasoning", result.get("evidence", "No reasoning provided"))

        # Truncate reasoning for readability
        reasoning_display = str(llm_reasoning)[:200] + "..." if len(str(llm_reasoning)) > 200 else str(llm_reasoning)

        self._log(f"   [LLM] [LLM Decision for '{column}']:", log_callback)
        self._log(f"      Is PII: {llm_is_pii}", log_callback)
        self._log(f"      PII Type: {llm_pii_type}", log_callback)
        self._log(f"      Reasoning: {reasoning_display}", log_callback)

        # Log Chain-of-Thought reasoning if available
        if "chain_of_thought" in result:
            self._log_cot_reasoning(column, result["chain_of_thought"], log_callback)

            # Also log to audit logger if available
            if self.audit_logger:
                self.audit_logger.log_stage1_cot_reasoning(
                    table=table_name,
                    column=column,
                    chain_of_thought=result["chain_of_thought"],
                    is_pii=result.get("is_pii", False),
                    pii_type=result.get("pii_type", "Unknown")
                )

        # Add evidence source from RAG
        if doc_details:
            sources = [doc.get('source', 'Unknown') for doc in doc_details]
            # Remove duplicates while preserving order
            unique_sources = list(dict.fromkeys(sources))
            result["evidence_source"] = ", ".join(unique_sources)
        else:
            result["evidence_source"] = "LLM Knowledge (No RAG documents retrieved)"

        return result

    def _log_cot_reasoning(self, column: str, cot: Dict[str, Any], log_callback: Optional[Callable] = None):
        """
        Log chain-of-thought reasoning from LLM response.
        LLM 응답의 chain-of-thought 추론 로깅.

        Args:
            column: Column name
            cot: Chain-of-thought dictionary from LLM response
            log_callback: Optional callback for logging
        """
        self._log(f"\n   [CoT] [Chain-of-Thought] {column}:", log_callback)

        # Input Analysis
        if "input_analysis" in cot:
            self._log(f"      [Stats] Input: {cot['input_analysis']}", log_callback)

        # Key Features
        if "key_features" in cot and cot["key_features"]:
            self._log(f"      [Key] Key Features:", log_callback)
            for feature in cot["key_features"]:
                self._log(f"         - {feature}", log_callback)

        # Decision Steps
        if "decision_steps" in cot and cot["decision_steps"]:
            self._log(f"      [Info] Decision Steps:", log_callback)
            for i, step in enumerate(cot["decision_steps"], 1):
                self._log(f"         {i}. {step}", log_callback)

        # Legal References
        if "legal_references" in cot and cot["legal_references"]:
            self._log(f"        Legal References:", log_callback)
            for ref in cot["legal_references"]:
                self._log(f"         - {ref}", log_callback)

        # Final Justification
        if "final_justification" in cot:
            self._log(f"      [OK] Justification: {cot['final_justification']}", log_callback)

        self._log("", log_callback)  # Empty line for readability

    def set_audit_logger(self, audit_logger):
        """
        Set audit logger for COT logging.
        COT 로깅을 위한 감사 로거 설정.
        
        Args:
            audit_logger: AuditLogger instance
        """
        self.audit_logger = audit_logger

    def _log(self, message: str, callback: Optional[Callable] = None):
        """Helper for logging"""
        if callback:
            callback(message)
        else:
            print(message)

    def cleanup_sessions(self, log_callback: Optional[Callable] = None):
        """Clean up all sessions (call at end of pipeline)."""
        stats = self.session_manager.get_all_stats()
        self._log(f"[Stage 1] Session Stats: {json.dumps(stats, indent=2)}", log_callback)

        # Close all sessions
        for session_id in list(self.session_manager.sessions.keys()):
            self.session_manager.close_session(session_id)
