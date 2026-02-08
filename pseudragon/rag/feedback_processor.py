"""
Feedback Processor for PseuDRAGON Framework
PseuDRAGON 프레임워크를 위한 피드백 프로세서

This module processes audit logs from HITL stage and converts them into
searchable feedback knowledge base entries for the RAG system.
이 모듈은 HITL 단계의 감사 로그를 처리하고 RAG 시스템을 위한
검색 가능한 피드백 지식베이스 항목으로 변환합니다.
"""

# Standard library imports
# 표준 라이브러리 import
import json
import os
from pathlib import Path
from typing import Any, Dict, List


class FeedbackProcessor:
    """
    Audit Log Analyzer and Feedback Knowledge Base Generator
    감사 로그 분석 및 피드백 지식베이스 생성기
    
    Processes USER_EDIT events from audit logs and converts them into
    searchable feedback documents that can be integrated into the RAG system.
    감사 로그에서 USER_EDIT 이벤트를 처리하고 RAG 시스템에 통합할 수 있는
    검색 가능한 피드백 문서로 변환합니다.
    """

    def process_audit_logs(self, log_dir: str) -> List[Dict[str, Any]]:
        """
        Process all audit log files in the directory and extract feedback entries
        디렉토리의 모든 감사 로그 파일을 처리하고 피드백 항목을 추출합니다
        
        Args:
            log_dir: Path to directory containing audit logs (output directory)
                    감사 로그가 포함된 디렉토리 경로 (output 디렉토리)
        
        Returns:
            List of feedback documents ready for embedding
            임베딩 준비가 완료된 피드백 문서 목록
        """
        feedback_entries = []

        if not os.path.exists(log_dir):
            print(f"[INFO] Audit log directory not found: {log_dir}")
            return feedback_entries

        # Find all audit log files (*.jsonl) in all session subdirectories
        # 모든 세션 하위 디렉토리에서 모든 감사 로그 파일(*.jsonl) 검색
        log_dir_path = Path(log_dir)
        audit_files = []

        # Recursively find all audit log files
        for session_dir in log_dir_path.iterdir():
            if session_dir.is_dir():
                audit_logs_dir = session_dir / "audit_logs"
                if audit_logs_dir.exists():
                    audit_files.extend(audit_logs_dir.glob("*.jsonl"))

        if not audit_files:
            print(f"[INFO] No audit log files found in {log_dir}")
            return feedback_entries

        print(f"[INFO] Found {len(audit_files)} audit log files")

        # Process each audit log file
        # 각 감사 로그 파일 처리
        for log_file in audit_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            event = json.loads(line.strip())

                            # Extract USER_EDIT events
                            # USER_EDIT 이벤트 추출
                            if event.get("event_type") == "USER_EDIT":
                                data = event.get("data", {})
                                feedback_entry = self._create_feedback_entry(data)
                                if feedback_entry:
                                    feedback_entries.append(feedback_entry)
                        except json.JSONDecodeError:
                            continue  # Skip invalid JSON lines
            except Exception as e:
                print(f"[WARNING] Failed to process {log_file}: {e}")
                continue

        print(f"[INFO] Extracted {len(feedback_entries)} feedback entries from audit logs")
        return feedback_entries

    def _create_feedback_entry(self, edit_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a USER_EDIT event into a searchable feedback document
        USER_EDIT 이벤트를 검색 가능한 피드백 문서로 변환
        
        Args:
            edit_data: USER_EDIT event data containing table, column, actions, and rationale
                      테이블, 컬럼, 액션, 근거를 포함하는 USER_EDIT 이벤트 데이터
        
        Returns:
            Feedback document with formatted text and metadata
            포맷된 텍스트 및 메타데이터가 포함된 피드백 문서
        """
        # Extract and sanitize data from edit event
        # 편집 이벤트에서 데이터 추출 및 정제
        table_name = str(edit_data.get("table", "Unknown"))[:200]  # Limit length
        column_name = str(edit_data.get("column", "Unknown"))[:200]
        old_action = str(edit_data.get("old_action", "Unknown"))[:100]
        new_action = str(edit_data.get("new_action", "Unknown"))[:100]
        rationale = str(edit_data.get("rationale", "No rationale provided"))[:1000]

        # Basic sanitization: remove potential control characters
        # 기본 정제: 잠재적인 제어 문자 제거
        def sanitize(text: str) -> str:
            """Remove control characters but keep whitespace and newlines"""
            return ''.join(char for char in text if char.isprintable() or char in '\n\t ')

        table_name = sanitize(table_name)
        column_name = sanitize(column_name)
        old_action = sanitize(old_action)
        new_action = sanitize(new_action)
        rationale = sanitize(rationale)

        # Format feedback text for optimal embedding search
        # 최적의 임베딩 검색을 위한 피드백 텍스트 포맷
        feedback_text = f"""[피드백 기록]
테이블: {table_name}
컬럼: {column_name}
LLM 제안 기법: {old_action}
전문가 수정 기법: {new_action}
수정 근거: {rationale}

결론: {column_name} 유형의 컬럼에는 {new_action} 기법이 더 적합함.
이유: {rationale}"""

        return {
            "text": feedback_text, "source": "FEEDBACK_KNOWLEDGE_BASE", "priority_level": 0,  # Highest priority
            "priority_label": "EXPERT_FEEDBACK", "metadata": {"table": table_name, "column": column_name, "old_action": old_action, "new_action": new_action, "rationale": rationale}
        }
