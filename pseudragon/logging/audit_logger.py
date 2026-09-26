"""Audit Logger.

Implements audit trail system:

- Records final policy
- Logs all user edits and rationales
- Generates compliance report
- Creates audit trail for regulatory review

This module ensures accountability under GDPR Article 5(2) and PIPA Article 4.
"""

import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling Decimal and datetime objects."""

    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)


class AuditLogger:
    """Audit Trail System for Regulatory Compliance.

    Record pipeline decisions and reviewer actions for inspection.

    All events are logged in JSONL format for easy parsing and analysis.

    Compliance reports are generated in Markdown format for human review.
    """

    def __init__(self, log_dir: str = "output", session_id: Optional[str] = None):
        """Initialize audit logger.

        Args:
            log_dir: Base directory to store audit logs
            session_id: Optional session ID (timestamp)
        """
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_output_dir = Path(log_dir)
        self.log_dir = self.base_output_dir / self.session_id / "audit_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / f"audit_{self.session_id}.jsonl"

        self._log_event(
            "SESSION_START",
            {"session_id": self.session_id, "timestamp": datetime.now().isoformat()},
        )

    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Internal method to log an event.

        Args:
            event_type: Type of event (USER_EDIT, POLICY_APPROVED, etc.)
            data: Event data dictionary
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event_type": event_type,
            "data": data,
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, cls=CustomJSONEncoder) + "\n")

    def log_stage_start(self, stage: str, table: str):
        """Log the start of a pipeline stage.

        Args:
            stage: Stage name (Stage 1, Stage 2, etc.)
            table: Table name being processed
        """
        self._log_event("STAGE_START", {"stage": stage, "table": table})

    def log_stage_complete(self, stage: str, table: str, result: Optional[Dict] = None):
        """Log the completion of a pipeline stage.

        Args:
            stage: Stage name
            table: Table name
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
        legal_evidence: str = "",
    ):
        """Log user modification in Stage 3 HITL.

        Implements user edit logging

        Record code snippets together with feedback for reuse.

        Args:
            table: Table name
            column: Column being modified
            old_action: Previous action
            new_action: New action selected by user
            rationale: User's rationale for the change
            old_parameters: Previous action parameters
            new_parameters: New action parameters
            old_code_snippet: Previous code implementation
            new_code_snippet: New code implementation
            legal_evidence: Legal basis for the change
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
            },
        )

    def log_policy_approval(
        self, table_name: str, policy: Dict[str, Any], approved_by: str = "user"
    ):
        """Log final policy approval in Stage 3.

        Implements policy approval logging

        Store approved code snippets with their supporting feedback.

        Args:
            table_name: Name of the table
            policy: Approved policy dictionary
            approved_by: User who approved the policy
        """
        policy_summary = {
            "table_name": table_name,
            "total_columns": len(policy) if isinstance(policy, dict) else 0,
            "approved_by": approved_by,
        }

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
                            col_data["code_snippet"] = first_method.get(
                                "example_implementation", ""
                            )

                    columns_with_code.append(col_data)

        policy_summary["columns_detail"] = columns_with_code
        policy_summary["feedback_quality"] = "expert_approved"

        self._log_event("POLICY_APPROVED", policy_summary)

    def log_validation_result(self, table: str, violations: List[Dict[str, Any]]):
        """Log validation results.

        Args:
            table: Table name
            violations: List of violations found
        """
        self._log_event(
            "VALIDATION",
            {
                "table": table,
                "num_violations": len(violations),
                "violations": violations,
            },
        )

    def log_code_generation(self, table: str, code_length: int, validation_passed: bool):
        """Log code generation event.

        Args:
            table: Table name
            code_length: Length of generated code
            validation_passed: Whether code passed syntax validation
        """
        self._log_event(
            "CODE_GENERATION",
            {
                "table": table,
                "code_length": code_length,
                "validation_passed": validation_passed,
            },
        )

    def generate_compliance_report(
        self,
        policy_dict: Dict[str, Any],
        table_name: str,
        output_dir: Optional[str] = None,
    ) -> str:
        """Generate compliance report for regulatory review.

        Implements compliance report generation

        Args:
            policy_dict: Policy as dictionary
            table_name: Table name
            output_dir: Optional directory to save the report

        Returns:
            Path to generated report file
        """
        if output_dir:
            report_dir = Path(output_dir)
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / f"compliance_report_{table_name}.md"
        else:
            report_path = self.log_dir / f"compliance_report_{table_name}_{self.session_id}.md"

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
                    if (
                        evidence
                        and evidence != "N/A"
                        and evidence != "Unknown Source"
                        and evidence != "Unknown"
                    ):
                        for doc in evidence.split(", "):
                            if doc.strip() and not doc.strip().startswith("User Selection"):
                                referenced_docs.add(doc.strip())

            f.write(
                f"This report documents the pseudonymization policy applied to table `{table_name}` "
            )

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

                    f.write(
                        f"| {col_name} | {pii_type} | {action_type} | {rationale} | {evidence} |\n"
                    )

            f.write(f"\n## Legal Compliance\n\n")
            f.write(f"This policy has been validated against the following documents:\n\n")
            for doc in sorted(referenced_docs):
                f.write(f"- **{doc}**\n")
            f.write(f"\n")

            f.write(f"## Audit Trail\n\n")
            f.write(f"Full audit trail available at: `{self.log_file}`\n\n")
            f.write(
                f"All user modifications and system decisions are logged for regulatory review.\n\n"
            )

            f.write(f"## Certification\n\n")
            f.write(
                f"This policy was generated using PseuDRAGON's automated pseudonymization framework, "
            )
            f.write(f"which combines:\n\n")
            f.write(
                "1. RAG-based legal knowledge retrieval from legal documents as mentioned in Legal Compliance.\n"
            )
            f.write(f"2. LLM-based policy synthesis with legal citations\n")
            f.write(f"3. Human-in-the-loop review and approval\n")
            f.write(f"4. Automatic compliance validation\n\n")

            f.write(f"**Report Generated:** {datetime.now().isoformat()}\n")
            f.write(f"**System Version:** PseuDRAGON v1.0\n")

        return str(report_path)

    def log_heuristic_change(
        self,
        action: str,
        heuristic_id: str,
        old_data: Optional[Dict[str, Any]] = None,
        new_data: Optional[Dict[str, Any]] = None,
        user_id: str = "system",
    ):
        """Log heuristic pattern modification.

        Args:
            action: Action type (ADD, UPDATE, DELETE)
            heuristic_id: Heuristic ID
            old_data: Previous heuristic data (for UPDATE/DELETE)
            new_data: New heuristic data (for ADD/UPDATE)
            user_id: User who made the change
        """
        self._log_event(
            "HEURISTIC_CHANGE",
            {
                "action": action,
                "heuristic_id": heuristic_id,
                "old_data": old_data,
                "new_data": new_data,
                "user_id": user_id,
            },
        )

    def log_pii_classification_change(
        self,
        table: str,
        column: str,
        old_status: str,
        new_status: str,
        rationale: str = "",
        user_id: str = "user",
    ):
        """Log PII classification change in Stage 3.

        Args:
            table: Table name
            column: Column name
            old_status: Previous PII status
            new_status: New PII status
            rationale: Reason for the change
            user_id: User who made the change
        """
        self._log_event(
            "PII_CLASSIFICATION_CHANGE",
            {
                "table": table,
                "column": column,
                "old_status": old_status,
                "new_status": new_status,
                "rationale": rationale,
                "user_id": user_id,
            },
        )

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session.

        Returns:
            Dictionary with session statistics
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

        return {
            "session_id": self.session_id,
            "total_events": len(events),
            "event_counts": event_counts,
            "log_file": str(self.log_file),
        }

    def _debug_log(self, msg: str) -> None:
        """Compatibility hook; runtime events are recorded by the audit logger."""
        pass

    def get_all_events(
        self,
        event_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get all audit events with optional filtering.

        Args:
            event_type: Filter by event type
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            user_id: Filter by user ID

        Returns:
            List of audit events
        """
        events = []

        # Search all audit logs in the base output directory recursively

        # Pattern: output/session_id/audit_logs/*.jsonl
        self._debug_log(
            f"Searching in {self.base_output_dir} with filters: type={event_type}, start={start_date}, end={end_date}"
        )
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
                            if (
                                event_type
                                and event_type != "All Events"
                                and event.get("event_type") != event_type
                            ):
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

        events.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return events

    def log_stage1_cot_reasoning(
        self, table: str, column: str, chain_of_thought: Dict[str, Any], is_pii: bool, pii_type: str
    ):
        """Log Stage 1 Chain-of-Thought reasoning for PII detection.

        Args:
            table: Table name
            column: Column name
            chain_of_thought: Chain-of-thought reasoning from LLM
            is_pii: Whether column is PII
            pii_type: PII type classification
        """
        self._log_event(
            "STAGE1_COT_REASONING",
            {
                "table": table,
                "column": column,
                "is_pii": is_pii,
                "pii_type": pii_type,
                "input_analysis": chain_of_thought.get("input_analysis", ""),
                "key_features": chain_of_thought.get("key_features", []),
                "decision_steps": chain_of_thought.get("decision_steps", []),
                "legal_references": chain_of_thought.get("legal_references", []),
                "final_justification": chain_of_thought.get("final_justification", ""),
            },
        )

    def log_stage2_cot_reasoning(
        self,
        table: str,
        column: str,
        pii_type: str,
        chain_of_thought: Dict[str, Any],
        methods_count: int,
    ):
        """Log Stage 2 Chain-of-Thought reasoning for policy synthesis.

        Args:
            table: Table name
            column: Column name
            pii_type: PII type
            chain_of_thought: Chain-of-thought reasoning from LLM
            methods_count: Number of methods generated
        """
        self._log_event(
            "STAGE2_COT_REASONING",
            {
                "table": table,
                "column": column,
                "pii_type": pii_type,
                "methods_count": methods_count,
                "pii_analysis": chain_of_thought.get("pii_analysis", ""),
                "method_evaluation": chain_of_thought.get("method_evaluation", []),
                "legal_compliance": chain_of_thought.get("legal_compliance", []),
                "implementation_rationale": chain_of_thought.get("implementation_rationale", ""),
            },
        )
