"""Feedback Processor for PseuDRAGON Framework

This module processes audit logs from HITL stage and converts them into
searchable feedback knowledge base entries for the RAG system.
"""


# 표준 라이브러리 import
import json
import os
from pathlib import Path
from typing import Any, Dict, List


class FeedbackProcessor:
    """Audit Log Analyzer and Feedback Knowledge Base Generator

    Processes USER_EDIT events from audit logs and converts them into
    searchable feedback documents that can be integrated into the RAG system.
    """

    def process_audit_logs(self, log_dir: str) -> List[Dict[str, Any]]:
        """Process all audit log files in the directory and extract feedback entries

        Args:
            log_dir: Path to directory containing audit logs (output directory)

        Returns:
            List of feedback documents ready for embedding
        """
        feedback_entries = []

        if not os.path.exists(log_dir):
            print(f"[INFO] Audit log directory not found: {log_dir}")
            return feedback_entries

        # Find all audit log files (*.jsonl) in all session subdirectories

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

        for log_file in audit_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            event = json.loads(line.strip())

                            # Extract USER_EDIT events

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
        """Convert a USER_EDIT event into a searchable feedback document

        Args:
            edit_data: USER_EDIT event data containing table, column, actions, and rationale

        Returns:
            Feedback document with formatted text and metadata
        """
        # Extract and sanitize data from edit event

        table_name = str(edit_data.get("table", "Unknown"))[:200]  # Limit length
        column_name = str(edit_data.get("column", "Unknown"))[:200]
        old_action = str(edit_data.get("old_action", "Unknown"))[:100]
        new_action = str(edit_data.get("new_action", "Unknown"))[:100]
        rationale = str(edit_data.get("rationale", "No rationale provided"))[:1000]

        # Basic sanitization: remove potential control characters

        def sanitize(text: str) -> str:
            """Remove control characters but keep whitespace and newlines"""
            return ''.join(char for char in text if char.isprintable() or char in '\n\t ')

        table_name = sanitize(table_name)
        column_name = sanitize(column_name)
        old_action = sanitize(old_action)
        new_action = sanitize(new_action)
        rationale = sanitize(rationale)

        # Format feedback text for optimal embedding search

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
