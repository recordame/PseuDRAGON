"""
Audit Logger
감사 로거

Implements audit trail system:
논문의 감사 추적 시스템 구현:
- Records final policy / 최종 정책 기록
- Logs all user edits and rationales / 모든 사용자 편집 및 근거 로깅
- Generates compliance report / 규정 준수 보고서 생성
- Creates audit trail for regulatory review / 규제 검토를 위한 감사 추적 생성

This module ensures accountability under GDPR Article 5(2) and PIPA Article 4.
이 모듈은 GDPR 제5조(2)항 및 PIPA 제4조에 따른 책임성을 보장합니다.
"""

import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional


class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for handling Decimal and datetime objects
    Decimal 및 datetime 객체를 처리하기 위한 커스텀 JSON 인코더
    """

    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)


class AuditLogger:
    """
    Audit Trail System for Regulatory Compliance
    규제 준수를 위한 감사 추적 시스템

    Implements audit logging requirements from paper Section 4.3.4.
    논문 Section 4.3.4의 감사 로깅 요구사항 구현.

    All events are logged in JSONL format for easy parsing and analysis.
    모든 이벤트는 쉬운 파싱 및 분석을 위해 JSONL 형식으로 로깅됩니다.

    Compliance reports are generated in Markdown format for human review.
    규정 준수 보고서는 사람이 검토할 수 있도록 Markdown 형식으로 생성됩니다.
    """

    def __init__(self, log_dir: str = "output", session_id: Optional[str] = None):
        """
        Initialize audit logger
        감사 로거 초기화

        Args:
            log_dir: Base directory to store audit logs
                    감사 로그를 저장할 기본 디렉토리
            session_id: Optional session ID (timestamp)
                       선택적 세션 ID (타임스탬프)
        """
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_output_dir = Path(log_dir)
        self.log_dir = self.base_output_dir / self.session_id / "audit_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / f"audit_{self.session_id}.jsonl"

        self._log_event("SESSION_START", {"session_id": self.session_id, "timestamp": datetime.now().isoformat()}, )

    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """
        Internal method to log an event
        이벤트를 로깅하는 내부 메서드

        Args:
            event_type: Type of event (USER_EDIT, POLICY_APPROVED, etc.)
                       이벤트 유형 (USER_EDIT, POLICY_APPROVED 등)
            data: Event data dictionary
                 이벤트 데이터 딕셔너리
        """
        event = {"timestamp": datetime.now().isoformat(), "session_id": self.session_id, "event_type": event_type, "data": data, }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, cls=CustomJSONEncoder) + "\n")

    def log_stage_start(self, stage: str, table: str):
        """
        Log the start of a pipeline stage
        파이프라인 단계 시작 로깅

        Args:
            stage: Stage name (Stage 1, Stage 2, etc.)
                  단계 이름 (Stage 1, Stage 2 등)
            table: Table name being processed
                  처리 중인 테이블 이름
        """
        self._log_event("STAGE_START", {"stage": stage, "table": table})

    def log_stage_complete(self, stage: str, table: str, result: Optional[Dict] = None):
        """
        Log the completion of a pipeline stage
        파이프라인 단계 완료 로깅

        Args:
            stage: Stage name
                  단계 이름
            table: Table name
                  테이블 이름
        """
        self._log_event("STAGE_COMPLETE", {"stage": stage, "table": table})

    def log_user_edit(
            self,
            table: str,
            column: str,
            old_action: str,
            new_action: str,
            rationale: str = "",
            old_parameters: Optional[Dict[str, Any]] = None,
            new_parameters: Optional[Dict[str, Any]] = None,
            old_code_snippet: str = "",
            new_code_snippet: str = "",
            legal_evidence: str = "", ):
        """
        Log user modification in Stage 3 HITL
        Stage 3 HITL에서 사용자 수정 로깅

        Implements user edit logging
        논문 Line 624의 사용자 편집 로깅 구현.

        Enhanced to include code snippets for RAG feedback learning.
        RAG 피드백 학습을 위해 코드 스니펫을 포함하도록 개선됨.

        Args:
            table: Table name
                  테이블 이름
            column: Column being modified
                   수정 중인 컬럼
            old_action: Previous action
                       이전 액션
            new_action: New action selected by user
                       사용자가 선택한 새 액션
            rationale: User's rationale for the change
                      변경에 대한 사용자의 근거
            old_parameters: Previous action parameters
                          이전 액션 매개변수
            new_parameters: New action parameters
                          새 액션 매개변수
            old_code_snippet: Previous code implementation
                            이전 코드 구현
            new_code_snippet: New code implementation
                            새 코드 구현
            legal_evidence: Legal basis for the change
                          변경에 대한 법적 근거
        """
        self._log_event(
            "USER_EDIT",
            {
                "table": table,
                "column": column,
                "old_action": old_action,
                "new_action": new_action,
                "rationale": rationale,
                "old_parameters": old_parameters or {},
                "new_parameters": new_parameters or {},
                "old_code_snippet": old_code_snippet,
                "new_code_snippet": new_code_snippet,
                "legal_evidence": legal_evidence,
                "feedback_quality": "expert_modified",
            }, )

    def log_policy_approval(self, table_name: str, policy: Dict[str, Any], approved_by: str = "user"):
        """
        Log final policy approval in Stage 3
        Stage 3에서 최종 정책 승인 로깅

        Implements policy approval logging
        논문 Line 625의 정책 승인 로깅 구현.

        Enhanced to extract and store code snippets for RAG feedback learning.
        RAG 피드백 학습을 위해 코드 스니펫을 추출하고 저장하도록 개선됨.

        Args:
            table_name: Name of the table
                       테이블 이름
            policy: Approved policy dictionary
                   승인된 정책 딕셔너리
            approved_by: User who approved the policy
                        정책을 승인한 사용자
        """
        policy_summary = {"table_name": table_name, "total_columns": len(policy) if isinstance(policy, dict) else 0, "approved_by": approved_by, }

        columns_with_code = []
        if isinstance(policy, dict):
            for col_name, col_policy in policy.items():
                if isinstance(col_policy, dict):
                    col_data = {
                        "column_name": col_name,
                        "pii_type": col_policy.get("pii_type", "unknown"),
                        "action": col_policy.get("action", "KEEP"),
                        "parameters": col_policy.get("parameters", {}),
                        "rationale": col_policy.get("rationale", ""),
                        "legal_evidence": col_policy.get("legal_evidence", ""),
                        "code_snippet": "",
                    }

                    if "recommended_methods" in col_policy and col_policy["recommended_methods"]:
                        first_method = col_policy["recommended_methods"][0]
                        if isinstance(first_method, dict):
                            col_data["code_snippet"] = first_method.get("example_implementation", "")

                    columns_with_code.append(col_data)

        policy_summary["columns_detail"] = columns_with_code
        policy_summary["feedback_quality"] = "expert_approved"

        self._log_event("POLICY_APPROVED", policy_summary)

    def log_validation_result(self, table: str, violations: List[Dict[str, Any]]):
        """
        Log validation results
        검증 결과 로깅

        Args:
            table: Table name
                  테이블 이름
            violations: List of violations found
                       발견된 위반 사항 목록
        """
        self._log_event("VALIDATION", {"table": table, "num_violations": len(violations), "violations": violations, }, )

    def log_code_generation(self, table: str, code_length: int, validation_passed: bool):
        """
        Log code generation event
        코드 생성 이벤트 로깅

        Args:
            table: Table name
                  테이블 이름
            code_length: Length of generated code
                        생성된 코드의 길이
            validation_passed: Whether code passed syntax validation
                             코드가 구문 검증을 통과했는지 여부
        """
        self._log_event("CODE_GENERATION", {"table": table, "code_length": code_length, "validation_passed": validation_passed, }, )

    def generate_compliance_report(self, policy_dict: Dict[str, Any], table_name: str, output_dir: Optional[str] = None, ) -> str:
        """
        Generate compliance report for regulatory review
        규제 검토를 위한 규정 준수 보고서 생성

        Implements compliance report generation
        논문 Line 625의 규정 준수 보고서 생성 구현.

        Args:
            policy_dict: Policy as dictionary
                        딕셔너리 형태의 정책
            table_name: Table name
                       테이블 이름
            output_dir: Optional directory to save the report
                       보고서를 저장할 선택적 디렉토리

        Returns:
            Path to generated report file
            생성된 보고서 파일 경로
        """
        if output_dir:
            report_dir = Path(output_dir)
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / f"compliance_report_{table_name}.md"
        else:
            report_path = (self.log_dir / f"compliance_report_{table_name}_{self.session_id}.md")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# Compliance Report\n\n")
            f.write(f"**Generated by:** PseuDRAGON Automated Pseudonymization System\n")
            f.write(f"**Session ID:** {self.session_id}\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Table:** {table_name}\n\n")

            f.write(f"---\n\n")

            f.write(f"## Executive Summary\n\n")

            referenced_docs = set()
            columns_dict = policy_dict.get("columns", policy_dict)

            for col_name, col_policy in columns_dict.items():
                if isinstance(col_policy, dict):
                    evidence = col_policy.get("evidence_source", "")
                    if evidence and evidence != "N/A" and evidence != "Unknown Source" and evidence != "Unknown":
                        for doc in evidence.split(", "):
                            if doc.strip() and not doc.strip().startswith("User Selection"):
                                referenced_docs.add(doc.strip())

            f.write(f"This report documents the pseudonymization policy applied to table `{table_name}` ")

            doc_list = ", ".join(sorted(referenced_docs))
            f.write(f"in accordance with the following legal documents: {doc_list}.\n\n")

            if "preferred_method" in policy_dict:
                f.write(f"**Preferred Method:** {policy_dict['preferred_method']}\n\n")

            f.write(f"## Policy Details\n\n")

            if columns_dict:
                f.write(f"### Column-Level Actions\n\n")
                f.write(f"| Column | PII Type | Action | Rationale | Legal Evidence |\n")
                f.write(f"|--------|----------|--------|-----------|----------------|\n")

                for col_name, col_policy in columns_dict.items():
                    if not isinstance(col_policy, dict):
                        continue

                    pii_type = col_policy.get("pii_type", "Unknown")

                    methods = col_policy.get("recommended_methods", [])
                    if methods and len(methods) > 0:
                        selected_method = methods[0]
                        action_type = selected_method.get("method", "KEEP")
                        rationale = selected_method.get("description", "N/A")
                    else:
                        action_type = "KEEP"
                        rationale = "N/A"

                    evidence = col_policy.get("evidence_source", "N/A")

                    f.write(f"| {col_name} | {pii_type} | {action_type} | {rationale} | {evidence} |\n")

            f.write(f"\n## Legal Compliance\n\n")
            f.write(f"This policy has been validated against the following documents:\n\n")
            for doc in sorted(referenced_docs):
                f.write(f"- **{doc}**\n")
            f.write(f"\n")

            f.write(f"## Audit Trail\n\n")
            f.write(f"Full audit trail available at: `{self.log_file}`\n\n")
            f.write(f"All user modifications and system decisions are logged for regulatory review.\n\n")

            f.write(f"## Certification\n\n")
            f.write(f"This policy was generated using PseuDRAGON's automated pseudonymization framework, ")
            f.write(f"which combines:\n\n")
            f.write("1. RAG-based legal knowledge retrieval from legal documents as mentioned in Legal Compliance.\n")
            f.write(f"2. LLM-based policy synthesis with legal citations\n")
            f.write(f"3. Human-in-the-loop review and approval\n")
            f.write(f"4. Automatic compliance validation\n\n")

            f.write(f"**Report Generated:** {datetime.now().isoformat()}\n")
            f.write(f"**System Version:** PseuDRAGON v1.0\n")

        return str(report_path)

    def log_heuristic_change(
            self, action: str, heuristic_id: str,
            old_data: Optional[Dict[str, Any]] = None,
            new_data: Optional[Dict[str, Any]] = None,
            user_id: str = "system"
    ):
        """
        Log heuristic pattern modification
        휴리스틱 패턴 수정 로깅
        
        Args:
            action: Action type (ADD, UPDATE, DELETE)
                   액션 유형
            heuristic_id: Heuristic ID
                         휴리스틱 ID
            old_data: Previous heuristic data (for UPDATE/DELETE)
                     이전 휴리스틱 데이터
            new_data: New heuristic data (for ADD/UPDATE)
                     새 휴리스틱 데이터
            user_id: User who made the change
                    변경한 사용자
        """
        self._log_event(
            "HEURISTIC_CHANGE", {
                "action": action,
                "heuristic_id": heuristic_id,
                "old_data": old_data,
                "new_data": new_data,
                "user_id": user_id
            }
        )

    def log_pii_classification_change(
            self, table: str, column: str,
            old_status: str, new_status: str,
            rationale: str = "",
            user_id: str = "user"
    ):
        """
        Log PII classification change in Stage 3
        Stage 3에서 PII 분류 변경 로깅
        
        Args:
            table: Table name
                  테이블 이름
            column: Column name
                   컬럼 이름
            old_status: Previous PII status
                       이전 PII 상태
            new_status: New PII status
                       새 PII 상태
            rationale: Reason for the change
                      변경 이유
            user_id: User who made the change
                    변경한 사용자
        """
        self._log_event(
            "PII_CLASSIFICATION_CHANGE", {
                "table": table,
                "column": column,
                "old_status": old_status,
                "new_status": new_status,
                "rationale": rationale,
                "user_id": user_id
            }
        )

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get summary of current session
        현재 세션 요약 가져오기

        Returns:
            Dictionary with session statistics
            세션 통계가 포함된 딕셔너리
        """
        if not self.log_file.exists():
            return {"error": "No log file found"}

        events = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                events.append(json.loads(line))

        event_counts = {}
        for event in events:
            event_type = event["event_type"]
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        return {"session_id": self.session_id, "total_events": len(events), "event_counts": event_counts, "log_file": str(self.log_file), }

    def _debug_log(self, msg: str) -> None:
        # try:
        #     with open("audit_debug.txt", "a", encoding="utf-8") as f:
        #         f.write(f"{datetime.now().isoformat()} - {msg}\n")
        # except:
        #     pass
        pass

    def get_all_events(
            self, event_type: Optional[str] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all audit events with optional filtering
        선택적 필터링을 사용하여 모든 감사 이벤트 가져오기
        
        Args:
            event_type: Filter by event type
                       이벤트 유형별 필터링
            start_date: Filter by start date (ISO format)
                       시작 날짜별 필터링
            end_date: Filter by end date (ISO format)
                     종료 날짜별 필터링
            user_id: Filter by user ID
                    사용자 ID별 필터링
        
        Returns:
            List of audit events
            감사 이벤트 목록
        """
        events = []

        # Search all audit logs in the base output directory recursively
        # 기본 출력 디렉토리의 모든 감사 로그를 재귀적으로 검색
        # Pattern: output/session_id/audit_logs/*.jsonl
        self._debug_log(f"Searching in {self.base_output_dir} with filters: type={event_type}, start={start_date}, end={end_date}")
        log_files = list(self.base_output_dir.rglob("*.jsonl"))
        self._debug_log(f"Found {len(log_files)} files: {[f.name for f in log_files]}")

        # Also check if current log file exists and is not in the list (edge case)
        if self.log_file.exists() and self.log_file not in log_files:
            log_files.append(self.log_file)

        for log_file in log_files:
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            event = json.loads(line)

                            # Apply filters
                            if event_type and event_type != "All Events" and event.get("event_type") != event_type:
                                # self._debug_log(f"Skipped {event.get('event_type')} due to type filter")
                                continue

                            event_ts = event.get("timestamp", "")
                            event_date = event_ts[:10]

                            if start_date:
                                # Compare date part only (YYYY-MM-DD)
                                if event_date < start_date:
                                    self._debug_log(f"Skipped {event_ts} < {start_date}")
                                    continue

                            if end_date:
                                # Compare date part only (YYYY-MM-DD)
                                if event_date > end_date:
                                    self._debug_log(f"Skipped {event_ts} > {end_date}")
                                    continue

                            if user_id:
                                event_user = event.get("data", {}).get("user_id")
                                if event_user != user_id:
                                    continue

                            events.append(event)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"Error reading log file {log_file}: {e}")
                continue

        # Sort events by timestamp descending (newest first)
        # 타임스탬프 내림차순 정렬 (최신순)
        events.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return events

    def log_stage1_cot_reasoning(
            self, table: str, column: str,
            chain_of_thought: Dict[str, Any],
            is_pii: bool, pii_type: str
    ):
        """
        Log Stage 1 Chain-of-Thought reasoning for PII detection.
        Stage 1 PII 탐지의 Chain-of-Thought 추론 로깅.
        
        Args:
            table: Table name
                  테이블 이름
            column: Column name
                   컬럼 이름
            chain_of_thought: Chain-of-thought reasoning from LLM
                            LLM의 chain-of-thought 추론
            is_pii: Whether column is PII
                   컬럼이 PII인지 여부
            pii_type: PII type classification
                     PII 유형 분류
        """
        self._log_event(
            "STAGE1_COT_REASONING", {
                "table": table,
                "column": column,
                "is_pii": is_pii,
                "pii_type": pii_type,
                "input_analysis": chain_of_thought.get("input_analysis", ""),
                "key_features": chain_of_thought.get("key_features", []),
                "decision_steps": chain_of_thought.get("decision_steps", []),
                "legal_references": chain_of_thought.get("legal_references", []),
                "final_justification": chain_of_thought.get("final_justification", "")
            }
        )

    def log_stage2_cot_reasoning(
            self, table: str, column: str, pii_type: str,
            chain_of_thought: Dict[str, Any],
            methods_count: int
    ):
        """
        Log Stage 2 Chain-of-Thought reasoning for policy synthesis.
        Stage 2 정책 합성의 Chain-of-Thought 추론 로깅.
        
        Args:
            table: Table name
                  테이블 이름
            column: Column name
                   컬럼 이름
            pii_type: PII type
                     PII 유형
            chain_of_thought: Chain-of-thought reasoning from LLM
                            LLM의 chain-of-thought 추론
            methods_count: Number of methods generated
                            생성된 메서드 수
        """
        self._log_event(
            "STAGE2_COT_REASONING", {
                "table": table,
                "column": column,
                "pii_type": pii_type,
                "methods_count": methods_count,
                "pii_analysis": chain_of_thought.get("pii_analysis", ""),
                "method_evaluation": chain_of_thought.get("method_evaluation", []),
                "legal_compliance": chain_of_thought.get("legal_compliance", []),
                "implementation_rationale": chain_of_thought.get("implementation_rationale", "")
            }
        )
