"""PseuDRAGON Web Application.

Flask-based web interface for the PseuDRAGON pseudonymization pipeline.

This application provides:

- Interactive UI for pipeline execution
- Real-time progress streaming
- Human-in-the-loop policy refinement
- Code generation and download
"""

import json
import logging
import copy
import ipaddress
import os
import sys
import time
import uuid
from datetime import date, datetime
from decimal import Decimal
from threading import Lock, Thread
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

# Add parent directory to Python path BEFORE any project imports

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Use UTF-8 for Unicode output on Windows.

if sys.platform == "win32":
    try:
        # Reconfigure stdout and stderr to use UTF-8 encoding

        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception as e:
        print(f"Warning: Could not set UTF-8 encoding: {e}")


from flask import Flask, Response, jsonify, render_template, request, send_file, g


from config.config import (
    DEFAULT_LLM_CLIENT,
    DatabaseConfig,
    DirectoryConfig,
    LLM_CLIENT_STAGE_1,
    LLM_CLIENT_STAGE_2,
    RAGConfig,
    Settings,
)
from pseudragon.database.adapters import DatabaseManager
from pseudragon.rag.heuristic_pattern_learner import HeuristicPatternLearner
from pseudragon.rag.retriever import RAGSystem
from pseudragon.rag.expert_preference_manager import ExpertPreferenceManager
from pseudragon.rag.knowledge_store import KNOWLEDGE_LOCK, revision
from pseudragon.domain.policy_dsl import ActionType, ColumnPolicy, Policy, PolicyAction
from pseudragon.heuristics.heuristic_manager import HeuristicManager
from pseudragon.logging.audit_logger import AuditLogger
from pseudragon.logging.file_logger import reinitialize_logger, PseuDRAGONLogger
from pseudragon.llm.session_manager import UniversalLLMSession
from pseudragon.pipeline import PseuDRAGONPipeline
from pseudragon.validation.compliance_checker import ComplianceValidator
from pseudragon.validation.method_contract import MethodContractError, prepare_method_selection

# Use BASE_DIR from DirectoryConfig for consistency

BASE_DIR = DirectoryConfig.BASE_DIR
sys.path.insert(0, BASE_DIR)


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling Decimal and datetime objects."""

    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)

        if isinstance(obj, (datetime, date)):
            return obj.isoformat()

        return super().default(obj)


class AppConfig:
    """Application configuration constants.

    Centralizes all application-level configuration values.
    """

    DB_PATH = DatabaseConfig.SAMPLE_DB_PATH
    DOCS_DIR = DirectoryConfig.DOCS_DIR
    OUTPUT_DIR = DirectoryConfig.OUTPUT_DIR
    REPORTS_DIR = DirectoryConfig.OUTPUT_DIR
    LOGS_DIR = os.path.join(DirectoryConfig.BASE_DIR, "logs")
    HOST = os.getenv("FLASK_HOST", "127.0.0.1")
    PORT = int(os.getenv("FLASK_PORT", 5001))
    DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    PAPER_TITLE = (
        "PseuDRAGON: Knowledge-Guided Local LLMs for Financial-Data "
        "Pseudonymization in Closed Networks"
    )


class PipelineStep:
    """Pipeline execution step constants.

    Defines all possible states of the pipeline execution.
    """

    IDLE = "IDLE"
    STAGE1_2_PROCESSING = "STAGE1_2_PROCESSING"
    STAGE3_READY = "STAGE3_READY"
    STAGE4_PROCESSING = "STAGE4_PROCESSING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class PipelineState:
    """Global state manager for pipeline execution.

    Maintains all runtime state including:

    - Pipeline execution status
    - Intermediate results
    - Logs and progress

    Thread-safe using locks for concurrent access.
    """

    def __init__(self) -> None:
        """Initialize pipeline state with default values."""
        self.engine: Optional[PseuDRAGONPipeline] = None
        self.rag: Optional[RAGSystem] = None
        self.expert_preference_manager: Optional[ExpertPreferenceManager] = None
        self.validator: Optional[ComplianceValidator] = None
        self.audit_logger: Optional[AuditLogger] = None
        self.file_logger: Optional[PseuDRAGONLogger] = None  # File logger for persistent logging
        self.is_running: bool = False
        self.should_stop: bool = False
        self.logs: List[str] = []
        self.pii_data: Dict[str, Any] = {}
        self.policy_data: Dict[str, Any] = {}
        self.policies: Dict[str, Policy] = {}
        self.schemas: Dict[str, Any] = {}
        self.generated_code: Dict[str, str] = {}
        self.report_files: Dict[str, str] = {}
        self.progress_lock: Lock = Lock()
        self.session_id: Optional[str] = None
        self.session_dir: Optional[str] = None
        self.current_step: str = PipelineStep.IDLE
        self.purpose_goal: str = ""  # User's intended purpose for data pseudonymization
        # Immutable model/RAG output captured before human review.
        self.model_initial_pii_data: Dict[str, Any] = {}
        self.model_initial_policy_data: Dict[str, Any] = {}
        self.manual_corrections: List[Dict[str, Any]] = []
        self.pipeline_started_at: Optional[str] = None
        self.model_completed_at: Optional[str] = None
        self.hitl_window_opened_at: Optional[str] = None

    def reset(self) -> None:
        """Reset pipeline state to initial values.

        Thread-safe reset of all state variables.
        """
        with self.progress_lock:
            self.should_stop = False
            self.logs = []
            self.pii_data = {}
            self.policy_data = {}
            self.policies = {}
            self.schemas = {}
            self.generated_code = {}
            self.report_files = {}
            self.session_id = None
            self.session_dir = None
            self.current_step = PipelineStep.IDLE
            self.purpose_goal = ""
            self.model_initial_pii_data = {}
            self.model_initial_policy_data = {}
            self.manual_corrections = []
            self.pipeline_started_at = None
            self.model_completed_at = None
            self.hitl_window_opened_at = None


class SessionManager:
    """Session manager for multi-client support.

    Each client gets an independent session with its own PipelineState.
    """

    # Session expiry time in seconds (30 minutes)

    SESSION_EXPIRY = 1800

    def __init__(self) -> None:
        """Initialize session manager."""
        self._sessions: Dict[str, PipelineState] = {}
        self._session_timestamps: Dict[str, float] = {}
        self._lock: Lock = Lock()
        # Shared backend components (RAG, engine) - initialized once

        self._rag: Optional[RAGSystem] = None
        self._engine: Optional[PseuDRAGONPipeline] = None
        self._expert_preference_manager: Optional[ExpertPreferenceManager] = None
        self._validator: Optional[ComplianceValidator] = None
        self._backend_initialized: bool = False
        self._backend_lock: Lock = Lock()

    def create_session(self) -> str:
        """Create a new session and return its ID.

        Returns:
            Unique session ID
        """
        session_id = str(uuid.uuid4())
        with self._lock:
            self._sessions[session_id] = PipelineState()
            self._session_timestamps[session_id] = time.time()
        return session_id

    def get_session(self, session_id: str) -> Optional[PipelineState]:
        """Get session state by ID.

        Args:
            session_id: Session ID to look up

        Returns:
            PipelineState if found, None otherwise
        """
        with self._lock:
            if session_id in self._sessions:
                # Update timestamp on access
                self._session_timestamps[session_id] = time.time()
                return self._sessions[session_id]
        return None

    def remove_session(self, session_id: str) -> bool:
        """Remove a session.

        Args:
            session_id: Session ID to remove

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                del self._session_timestamps[session_id]
                return True
        return False

    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions.

        Returns:
            Number of sessions removed
        """
        current_time = time.time()
        expired = []
        with self._lock:
            for sid, ts in self._session_timestamps.items():
                if current_time - ts > self.SESSION_EXPIRY:
                    # Don't remove sessions that are currently running

                    if sid in self._sessions and not self._sessions[sid].is_running:
                        expired.append(sid)

            for sid in expired:
                del self._sessions[sid]
                del self._session_timestamps[sid]

        return len(expired)

    def get_active_session_count(self) -> int:
        """Get count of active sessions.

        Returns:
            Number of active sessions
        """
        with self._lock:
            return len(self._sessions)

    def get_all_sessions_info(self) -> List[Dict[str, Any]]:
        """Get information about all active sessions for monitoring.

        Returns:
            List of session info dictionaries
        """
        sessions_info = []
        current_time = time.time()

        with self._lock:
            for session_id, session_state in self._sessions.items():
                # Get table names being processed
                tables = list(session_state.schemas.keys()) if session_state.schemas else []

                # Calculate session duration
                start_time = self._session_timestamps.get(session_id, current_time)
                duration_seconds = int(current_time - start_time)

                session_info = {
                    "session_id": session_id,
                    "session_id_short": session_id[:8],
                    "is_running": session_state.is_running,
                    "current_step": session_state.current_step,
                    "tables": tables,
                    "table_count": len(tables),
                    "purpose_goal": session_state.purpose_goal[:50] + "..."
                    if len(session_state.purpose_goal) > 50
                    else session_state.purpose_goal,
                    "log_count": len(session_state.logs),
                    "duration_seconds": duration_seconds,
                    "has_pii_data": bool(session_state.pii_data),
                    "has_policy_data": bool(session_state.policy_data),
                    "has_generated_code": bool(session_state.generated_code),
                }
                sessions_info.append(session_info)

        # Sort by running status first, then by duration (most recent first)

        sessions_info.sort(key=lambda x: (not x["is_running"], -x["duration_seconds"]))

        return sessions_info

    def get_session_logs(
        self, session_id: str, offset: int = 0, limit: int = 100
    ) -> Dict[str, Any]:
        """Get logs from a specific session for monitoring.

        Args:
            session_id: Session ID
            offset: Starting index for logs
            limit: Maximum number of logs to return

        Returns:
            Dictionary with logs and session info
        """
        with self._lock:
            if session_id not in self._sessions:
                return {"error": "Session not found", "logs": [], "total": 0}

            session_state = self._sessions[session_id]
            total_logs = len(session_state.logs)
            logs = session_state.logs[offset : offset + limit]

            return {
                "session_id": session_id,
                "session_id_short": session_id[:8],
                "is_running": session_state.is_running,
                "current_step": session_state.current_step,
                "tables": list(session_state.schemas.keys()),
                "logs": logs,
                "total": total_logs,
                "offset": offset,
                "has_more": offset + limit < total_logs,
            }

    def get_shared_backend(
        self,
    ) -> Tuple[
        Optional[RAGSystem],
        Optional[PseuDRAGONPipeline],
        Optional[ExpertPreferenceManager],
        Optional[ComplianceValidator],
    ]:
        """Get shared backend components.

        Returns:
            Tuple of (RAG, engine, expert_preference_manager, validator)
        """
        return self._rag, self._engine, self._expert_preference_manager, self._validator

    def set_shared_backend(
        self,
        rag: RAGSystem,
        engine: PseuDRAGONPipeline,
        expert_preference_manager: ExpertPreferenceManager,
        validator: ComplianceValidator,
    ) -> None:
        """Set shared backend components.

        Args:
            rag: RAG system instance
            engine: PseuDRAGON pipeline instance
            expert_preference_manager: User preference manager instance
            validator: Compliance validator instance
        """
        with self._backend_lock:
            self._rag = rag
            self._engine = engine
            self._expert_preference_manager = expert_preference_manager
            self._validator = validator
            self._backend_initialized = True

    def is_backend_initialized(self) -> bool:
        """Check if backend is initialized.

        Returns:
            True if initialized
        """
        return self._backend_initialized

    def get_all_audit_events(
        self,
        event_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get audit events from all sessions.

        Args:
            event_type: Filter by event type
            start_date: Filter by start date
            end_date: Filter by end date
            user_id: Filter by user ID

        Returns:
            List of audit events from all sessions
        """
        all_events = []
        with self._lock:
            for session_id, session_state in self._sessions.items():
                if session_state.audit_logger:
                    try:
                        events = session_state.audit_logger.get_all_events(
                            event_type=event_type,
                            start_date=start_date,
                            end_date=end_date,
                            user_id=user_id,
                        )
                        # Add session ID to each event for tracking
                        for event in events:
                            event["session_id"] = session_id[:8]  # Short session ID
                        all_events.extend(events)
                    except Exception:
                        pass  # Skip sessions with errors

        # Sort by timestamp (newest first)
        all_events.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return all_events

    def get_all_audit_summary(self) -> Dict[str, Any]:
        """Get combined audit summary from all sessions.

        Returns:
            Combined summary statistics
        """
        combined_summary = {
            "total_events": 0,
            "events_by_type": {},
            "session_count": 0,
            "sessions": [],
        }

        with self._lock:
            combined_summary["session_count"] = len(self._sessions)
            for session_id, session_state in self._sessions.items():
                if session_state.audit_logger:
                    try:
                        summary = session_state.audit_logger.get_session_summary()
                        combined_summary["total_events"] += summary.get("total_events", 0)

                        # Merge events by type
                        for event_type, count in summary.get("events_by_type", {}).items():
                            if event_type not in combined_summary["events_by_type"]:
                                combined_summary["events_by_type"][event_type] = 0
                            combined_summary["events_by_type"][event_type] += count

                        # Add session info
                        combined_summary["sessions"].append(
                            {
                                "session_id": session_id[:8],
                                "total_events": summary.get("total_events", 0),
                                "start_time": summary.get("start_time", ""),
                            }
                        )
                    except Exception:
                        pass

        return combined_summary


app = Flask(__name__, template_folder="templates", static_folder="static")
app.json_encoder = CustomJSONEncoder

# Session manager for multi-client support

session_manager = SessionManager()

# Shared state for callers without a session identifier.

state = PipelineState()

# Global audit logger for reading all logs (regardless of session)

global_audit_logger = AuditLogger(log_dir=AppConfig.REPORTS_DIR, session_id="global")

# Initialize heuristic manager
heuristic_manager = HeuristicManager()


def init_backend(
    session_state: Optional[PipelineState] = None, log_callback: Optional[Any] = None
) -> Tuple[RAGSystem, PseuDRAGONPipeline, ExpertPreferenceManager, ComplianceValidator]:
    """Initialize backend components (RAG system and PseuDRAGON engine).

    This function:

    - Checks for required files and directories
    - Initializes RAG system with legal documents
    - Creates PseuDRAGON engine instance

    Backend components are shared across all sessions (singleton pattern).

    Args:
        session_state: Optional session state for logging
        log_callback: Optional log callback function

    Returns:
        Tuple of (RAG, engine, expert_preference_manager, validator)
    """

    # Use session-specific or global logging
    def _log(msg: str) -> None:
        if log_callback:
            log_callback(msg)
        else:
            log_producer(msg, session_state)

    # Check if already initialized in session manager
    if session_manager.is_backend_initialized():
        rag, engine, user_pref, validator = session_manager.get_shared_backend()
        _log("Backend already initialized, reusing existing components...")
        return rag, engine, user_pref, validator

    _log("=" * 60)
    _log("Initializing Backend...")
    _log("=" * 60)

    if not os.path.exists(AppConfig.DB_PATH):
        _log(f"[WARN]️  WARNING: Database not found at {AppConfig.DB_PATH}")
        _log(f"   Please ensure database exists at: {AppConfig.DB_PATH}")

    docs_dir = Settings.DOCS_DIR
    if not os.path.exists(docs_dir):
        _log(f"   Creating directory: {docs_dir}")
        os.makedirs(docs_dir, exist_ok=True)

        # Create 3-tier priority subdirectories
        for subdir in ["institutional_policy", "national_law", "international_regulation"]:
            subdir_path = os.path.join(docs_dir, subdir)
            os.makedirs(subdir_path, exist_ok=True)

        _log(f"   [WARN]️  Please add applicable evidence to these directories:")
        _log(f"      - {docs_dir}/institutional_policy (institutional policy)")
        _log(f"      - {docs_dir}/national_law (national law/guidance)")
        _log(f"      - {docs_dir}/international_regulation (international regulation/guidance)")
        _log(f"   Without documents, the RAG system cannot provide legal guidance.")

    os.makedirs(AppConfig.REPORTS_DIR, exist_ok=True)

    _log("\n[RAG] Initializing RAG System...")
    retrieval_profile = _get_runtime_retrieval_profile()
    _log(f"   [Config] {retrieval_profile['label']}: {retrieval_profile['summary']}")
    if retrieval_profile["retrieval_mode"] == "source_order":
        _log("   [Config] Institutional policy -> National law -> International guidance")
    else:
        _log(f"   [Config] {retrieval_profile['formula']}")
    _log(f"   [Config] {retrieval_profile['review_note']}")
    rag = RAGSystem()

    # Check for PDF files in all 3-tier subdirectories
    total_pdf_count = 0
    priority_dirs = ["institutional_policy", "national_law", "international_regulation"]

    for subdir in priority_dirs:
        subdir_path = os.path.join(docs_dir, subdir)

        if os.path.exists(subdir_path):
            pdf_files = [f for f in os.listdir(subdir_path) if f.lower().endswith(".pdf")]
            total_pdf_count += len(pdf_files)

            if pdf_files:
                _log(f"   Found {len(pdf_files)} PDF file(s) in {subdir}/")

    if total_pdf_count > 0:
        _log(f"   Total: {total_pdf_count} PDF file(s) found")
        _log(f"   Loading documents from {docs_dir}...")
        rag.load_documents(docs_dir)
        _log("   [OK] RAG System initialized successfully!")
    else:
        _log(f"   [WARN]️  No PDF files found in any priority directory")
        _log("   [WARN]️  RAG System initialized but without documents.")
        _log("   [WARN]️  Pipeline will work but without legal guidance.")
        rag.is_initialized = True

    _log("\n[Stats] Initializing Approved Expert Feedback Manager...")
    expert_preference_manager = ExpertPreferenceManager()
    stats = expert_preference_manager.get_statistics()
    _log(
        f"   [OK] Approved Expert Feedback Manager initialized! ({stats['unique_columns']} method preferences)"
    )
    if stats.get("pii_classifications", 0) > 0:
        _log(
            f"   [OK] Learned {stats['pii_classifications']} PII classification(s) from previous sessions "
        )

    _log("\nPseuDRAGON Initializing PseuDRAGON Engine...")
    engine = PseuDRAGONPipeline(
        rag,
        DEFAULT_LLM_CLIENT,
        LLM_CLIENT_STAGE_1,
        LLM_CLIENT_STAGE_2,
        expert_preference_manager=expert_preference_manager,
    )
    _log("   [OK] PseuDRAGON Engine initialized!")

    _log("\n[Search] Initializing Compliance Validator...")
    validator = ComplianceValidator(rag)
    _log("   [OK] Compliance Validator initialized!")

    _log("\n" + "=" * 60)
    _log("   [OK] Backend Initialization Complete!")
    _log("=" * 60 + "\n")

    # Store in session manager for sharing across sessions
    session_manager.set_shared_backend(rag, engine, expert_preference_manager, validator)

    # Keep the shared state synchronized with the session state.
    state.rag = rag
    state.engine = engine
    state.expert_preference_manager = expert_preference_manager
    state.validator = validator

    return rag, engine, expert_preference_manager, validator


def log_producer(message: str, session_state: Optional[PipelineState] = None) -> None:
    """Thread-safe logging function with timestamps.

    Adds timestamped messages to the session-specific or global log buffer for streaming to clients.
    Also writes to the file logger for persistent logging.

    Args:
        message: Log message to record
        session_state: Optional session state to log to
    """
    timestamp = time.strftime("[%H:%M:%S]")
    formatted_msg = f"{timestamp} {message}"
    print(formatted_msg, flush=True)

    # Use session-specific state or fall back to global state

    target_state = session_state if session_state else state

    # Write to file logger if available

    if target_state.file_logger:
        target_state.file_logger.info(message)

    with target_state.progress_lock:
        target_state.logs.append(formatted_msg)


def run_pipeline_task(
    target_tables: List[str], purpose_goal: str, preferred_method: str, client_session_id: str
) -> None:
    """Execute Stages 1 and 2 of the pipeline in a background thread.

    This function runs asynchronously and:

    - Stage 1: Identifies PII in selected tables
    - Stage 2: Generates policy recommendations
    - Waits for user input (Stage 3)

    Args:
        target_tables: List of table names to process
        purpose_goal: Purpose/intent of data pseudonymization
        preferred_method: Preferred pseudonymization method
        client_session_id: Unique session ID for this client
    """
    # Get session-specific state

    session_state = session_manager.get_session(client_session_id)
    if not session_state:
        print(f"ERROR: Session {client_session_id} not found!", flush=True)
        return

    # Create session-specific log callback

    def session_log(msg: str) -> None:
        log_producer(msg, session_state)

    logging.getLogger(__name__).debug(
        f"DEBUG: run_pipeline_task started for {target_tables} (session: {client_session_id})"
    )
    session_state.is_running = True

    # Store purpose_goal in state for use in policy synthesis

    session_state.purpose_goal = purpose_goal

    # Store target_tables for use in get_stage3_data (filter to selected tables only)

    session_state.target_tables = target_tables

    # Initialize session directory with unique ID (UUID-based to avoid collisions)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_state.session_id = f"{timestamp}_{client_session_id[:8]}"
    session_state.session_dir = os.path.join(AppConfig.OUTPUT_DIR, session_state.session_id)
    os.makedirs(session_state.session_dir, exist_ok=True)
    session_state.pipeline_started_at = datetime.now().isoformat()

    # Initialize file logger for this session

    os.makedirs(AppConfig.LOGS_DIR, exist_ok=True)
    session_state.file_logger = reinitialize_logger(session_id=session_state.session_id)
    session_state.file_logger.info(f"Pipeline started for tables: {target_tables}")
    if purpose_goal:
        session_state.file_logger.info(f"Pseudonymization Purpose: {purpose_goal}")
    if preferred_method:
        session_state.file_logger.info(f"Preferred Method: {preferred_method}")

    # Re-initialize AuditLogger for this specific run
    session_state.audit_logger = AuditLogger(
        log_dir=AppConfig.OUTPUT_DIR, session_id=session_state.session_id
    )

    with session_state.progress_lock:
        session_state.logs = []

    session_state.current_step = PipelineStep.STAGE1_2_PROCESSING

    try:
        session_log("Starting backend initialization...")
        rag, engine, user_pref_manager, validator = init_backend(session_state, session_log)

        # Store references in session state for later use
        session_state.engine = engine
        session_state.rag = rag
        session_state.expert_preference_manager = user_pref_manager
        session_state.validator = validator

        for table_name in target_tables:
            # Check if stop was requested
            if session_state.should_stop:
                session_log("⛔ Pipeline stopped by user")
                session_state.current_step = PipelineStep.IDLE
                session_state.is_running = False
                return

            session_log(f"--- Processing Table: {table_name} ---")

            # Create table-specific subdirectory
            table_dir = os.path.join(session_state.session_dir, table_name)
            os.makedirs(table_dir, exist_ok=True)

            # Use DatabaseManager to get enhanced schema with COMMENT
            db_manager = DatabaseManager(AppConfig.DB_PATH)
            schema = db_manager.get_enhanced_schema(table_name)

            if not schema:
                session_log(f"Skipping {table_name}: No data.")
                continue

            session_state.schemas[table_name] = schema

            # Check if stop was requested before Stage 1
            if session_state.should_stop:
                session_log("⛔ Pipeline stopped by user")
                session_state.current_step = PipelineStep.IDLE
                session_state.is_running = False
                return

            # Stage 1: PII Identification with timing

            stage1_name = f"Stage 1 - PII Identification ({table_name})"
            session_log(f"Starting Stage 1: PII Identification for {table_name}")
            if session_state.file_logger:
                session_state.file_logger.stage_start(stage1_name)

            if session_state.audit_logger:
                session_state.audit_logger.log_stage_start(
                    "Stage 1: PII Identification", table_name
                )

            try:
                pii_res = engine.identify_pii(schema, table_name, log_callback=session_log)
                session_state.pii_data[table_name] = pii_res
            finally:
                # Ensure stage_end is always called even if exception occurs

                if session_state.file_logger:
                    session_state.file_logger.stage_end(stage1_name)

            if session_state.audit_logger:
                session_state.audit_logger.log_stage_complete(
                    "Stage 1: PII Identification", table_name
                )

            # Log table statistics after PII identification

            total_columns = len(pii_res)
            pii_columns = sum(1 for col_data in pii_res.values() if col_data.get("is_pii", False))
            session_state.file_logger.log_table_stats(table_name, total_columns, pii_columns)
            session_log(
                f"[Stats] Table Stats: {total_columns} columns, {pii_columns} PII columns identified"
            )

            # Check if stop was requested before Stage 2
            if session_state.should_stop:
                session_log("⛔ Pipeline stopped by user")
                session_state.current_step = PipelineStep.IDLE
                session_state.is_running = False
                return

            # Stage 2: Policy Synthesis with timing

            stage2_name = f"Stage 2 - Policy Synthesis ({table_name})"
            session_log(f"Starting Stage 2: Policy Synthesis for {table_name}")

            if session_state.file_logger:
                session_state.file_logger.stage_start(stage2_name)

            if session_state.audit_logger:
                session_state.audit_logger.log_stage_start("Stage 2: Policy Synthesis", table_name)

            try:
                policy_res = engine.synthesize_policy(
                    pii_res,
                    table_name,
                    preferred_method,
                    purpose_goal=purpose_goal,
                    log_callback=session_log,
                    output_dir=table_dir,
                )
                session_state.policy_data[table_name] = policy_res
            finally:
                # Ensure stage_end is always called even if exception occurs

                if session_state.file_logger:
                    session_state.file_logger.stage_end(stage2_name)

            if session_state.audit_logger:
                session_state.audit_logger.log_stage_complete(
                    "Stage 2: Policy Synthesis", table_name
                )

            # Apply user preferences to reorder methods

            if user_pref_manager:
                session_log(f"[Stats] Applying user preferences for {table_name}...")
                for col_name, col_data in session_state.policy_data[table_name].items():
                    if col_data.get("is_pii", False):
                        # PII column - reorder anonymization methods
                        pii_type = col_data.get("pii_type", "Unknown")
                        methods = col_data.get("recommended_methods", [])

                        # Reorder methods based on user's past selections
                        reordered_methods = user_pref_manager.reorder_methods(
                            col_name, pii_type, methods
                        )

                        # Apply user's custom code snippet if available (exact match or RAG similar)

                        # Get column metadata for RAG search
                        col_comment = (
                            session_state.pii_data.get(table_name, {})
                            .get(col_name, {})
                            .get("column_comment", "")
                        )
                        col_data_type = ""
                        if isinstance(
                            session_state.schemas.get(table_name, {}).get(col_name), dict
                        ):
                            col_data_type = session_state.schemas[table_name][col_name].get(
                                "type", ""
                            )

                        # Try RAG-based preference lookup (includes exact match first, then similar)

                        rag_preference = user_pref_manager.get_preference_from_similar(
                            col_name, pii_type, col_comment, col_data_type
                        )
                        if rag_preference:
                            preferred_method = rag_preference["method"]
                            custom_snippet = rag_preference.get("code_snippet")
                            custom_description = rag_preference.get("description")
                            match_type = rag_preference.get("match_type", "unknown")
                            similar_col = rag_preference.get("similar_column", "")

                            if len(reordered_methods) > 0:
                                # Apply custom code snippet if available
                                if custom_snippet:
                                    reordered_methods[0]["code_snippet"] = custom_snippet
                                # Apply custom description if available (ensures code and description match)

                                if custom_description:
                                    reordered_methods[0]["description"] = custom_description

                                if custom_snippet or custom_description:
                                    if match_type == "rag_similar":
                                        print(
                                            f"[RAG] Applied custom snippet/description from similar column {similar_col} -> {col_name}"
                                        )
                                        session_log(
                                            f"[RAG] RAG: Applied {preferred_method} from {similar_col}"
                                        )
                                    else:
                                        print(
                                            f"[INFO] Applied custom code/description for PII column {col_name} ({preferred_method})"
                                        )

                        # Update policy data with reordered methods
                        session_state.policy_data[table_name][col_name]["recommended_methods"] = (
                            reordered_methods
                        )
                    else:
                        # Non-PII column - reorder methods based on user preference
                        # But preserve ALL LLM-generated methods (don't replace them!)
                        # Get column metadata for RAG search
                        col_comment = (
                            session_state.pii_data.get(table_name, {})
                            .get(col_name, {})
                            .get("column_comment", "")
                        )
                        col_data_type = ""
                        if isinstance(
                            session_state.schemas.get(table_name, {}).get(col_name), dict
                        ):
                            col_data_type = session_state.schemas[table_name][col_name].get(
                                "type", ""
                            )

                        # Try RAG-based preference lookup (includes exact match first, then similar)

                        rag_preference = user_pref_manager.get_preference_from_similar(
                            col_name, "Non-PII", col_comment, col_data_type
                        )

                        if rag_preference and col_name in session_state.policy_data[table_name]:
                            preferred_method = rag_preference["method"]
                            custom_snippet = rag_preference.get("code_snippet")
                            custom_description = rag_preference.get("description")
                            match_type = rag_preference.get("match_type", "unknown")
                            similar_col = rag_preference.get("similar_column", "")

                            # Get existing methods from policy_data
                            existing_methods = session_state.policy_data[table_name][col_name].get(
                                "recommended_methods", []
                            )

                            if len(existing_methods) > 2:
                                # LLM generated rich methods - reorder them with user's preferred method first
                                reordered = user_pref_manager.reorder_methods(
                                    col_name, "Non-PII", existing_methods
                                )

                                # Apply custom code snippet and description if available
                                if len(reordered) > 0:
                                    if custom_snippet:
                                        reordered[0]["code_snippet"] = custom_snippet
                                    # Apply custom description if available (ensures code and description match)

                                    if custom_description:
                                        reordered[0]["description"] = custom_description

                                    if custom_snippet or custom_description:
                                        if match_type == "rag_similar":
                                            print(
                                                f"[RAG] Applied custom snippet/description from similar column {similar_col} -> {col_name}"
                                            )
                                            session_log(
                                                f"[RAG] RAG: Applied {preferred_method} from {similar_col}"
                                            )
                                        else:
                                            print(
                                                f"[INFO] Applied custom code/description for Non-PII column {col_name} ({preferred_method})"
                                            )

                                # Update with reordered methods
                                session_state.policy_data[table_name][col_name][
                                    "recommended_methods"
                                ] = reordered
                                print(
                                    f"[INFO] Reordered Non-PII methods for {col_name}: {preferred_method} prioritized (kept all {len(reordered)} methods)"
                                )
                            else:
                                # Leave KEEP/DELETE-only policies unchanged.
                                print(
                                    f"[INFO] Non-PII column {col_name} has only basic methods, skipping reorder"
                                )

            # Store integrated report filename (relative to OUTPUT_DIR)

            if hasattr(engine, "_last_integrated_report"):
                rel_path = os.path.relpath(engine._last_integrated_report, AppConfig.OUTPUT_DIR)
                session_state.report_files[f"{table_name}_integrated"] = rel_path

        session_log("Stage 1 & 2 Completed for all tables.")
        session_log("WAITING_USER_INPUT")

        # Freeze the unedited RAG+LLM output before any Stage-3 display or
        # human correction can mutate policy_data. This is the primary
        # cold-start record used to count model errors.
        session_state.model_initial_pii_data = copy.deepcopy(session_state.pii_data)
        session_state.model_initial_policy_data = copy.deepcopy(session_state.policy_data)
        session_state.model_completed_at = datetime.now().isoformat()
        session_state.hitl_window_opened_at = None
        model_elapsed_seconds = None
        if session_state.pipeline_started_at:
            model_elapsed_seconds = (
                datetime.fromisoformat(session_state.model_completed_at)
                - datetime.fromisoformat(session_state.pipeline_started_at)
            ).total_seconds()
        initial_output_path = os.path.join(
            session_state.session_dir, "model_initial_predictions.json"
        )
        with open(initial_output_path, "w", encoding="utf-8") as initial_file:
            json.dump(
                {
                    "tables": target_tables,
                    "pipeline_started_at": session_state.pipeline_started_at,
                    "model_completed_at": session_state.model_completed_at,
                    "model_elapsed_seconds": model_elapsed_seconds,
                    "timing_definition": "model_completed_at is the end of all RAG+LLM judgments, immediately before HITL review becomes available",
                    "pii_data": session_state.model_initial_pii_data,
                    "policy_data": session_state.model_initial_policy_data,
                },
                initial_file,
                ensure_ascii=False,
                indent=2,
                cls=CustomJSONEncoder,
            )
        with open(
            os.path.join(session_state.session_dir, "timing.json"), "w", encoding="utf-8"
        ) as timing_file:
            json.dump(
                {
                    "pipeline_started_at": session_state.pipeline_started_at,
                    "model_completed_at": session_state.model_completed_at,
                    "model_elapsed_seconds": model_elapsed_seconds,
                    "hitl_window_opened_at": None,
                    "timing_definition": "system comparison endpoint is the completion of all model judgments; human review starts afterward",
                },
                timing_file,
                ensure_ascii=False,
                indent=2,
            )
        session_log(f"[Audit] Saved immutable model/RAG output: {initial_output_path}")

        session_state.current_step = PipelineStep.STAGE3_READY

        # Log stage timing summary for Stage 1 & 2

        if session_state.file_logger:
            session_state.file_logger.info("=" * 60)
            session_state.file_logger.info("Stage 1 & 2 Timing Summary")
            session_state.file_logger.info("=" * 60)
            for stage_name, duration in session_state.file_logger.get_all_stage_durations().items():
                if duration < 60:
                    duration_str = f"{duration:.2f}s"
                else:
                    minutes = int(duration // 60)
                    seconds = duration % 60
                    duration_str = f"{minutes}m {seconds:.2f}s"
                session_state.file_logger.info(f"  {stage_name}: {duration_str}")
                session_log(f"[Time] {stage_name}: {duration_str}")

    except Exception as e:
        import traceback

        traceback.print_exc()
        session_log(f"ERROR: {str(e)}")
        session_state.current_step = PipelineStep.ERROR
        if session_state.file_logger:
            session_state.file_logger.error(f"Pipeline error: {str(e)}")
    finally:
        session_state.is_running = False


def _count_evidence_pdfs(source_family: str) -> int:
    """Count indexed PDF inputs without exposing confidential filenames."""
    source_dir = os.path.join(Settings.DOCS_DIR, source_family)
    if not os.path.isdir(source_dir):
        return 0

    return sum(
        1
        for filename in os.listdir(source_dir)
        if filename.lower().endswith(".pdf") and os.path.isfile(os.path.join(source_dir, filename))
    )


def _classify_endpoint_scope(base_url: Optional[str]) -> str:
    """Classify an LLM endpoint without exposing its hostname in the UI.

    This is a configuration check, not proof that an entire deployment is
    air-gapped. Hostname-based network endpoints remain "configured-network"
    because their network boundary must be verified operationally.
    """
    if not base_url:
        return "provider-default"

    hostname = urlparse(base_url).hostname
    if not hostname:
        return "provider-default"

    normalized_host = hostname.lower()
    if normalized_host in {"localhost", "localhost.localdomain"}:
        return "loopback"

    try:
        address = ipaddress.ip_address(normalized_host)
    except ValueError:
        return "configured-network"

    if address.is_loopback:
        return "loopback"
    if address.is_private:
        return "private-network"
    return "configured-network"


def _get_runtime_retrieval_profile() -> dict:
    """Describe the active retrieval mode without initializing a model."""
    config = RAGConfig.get_retrieval_config()
    mode = config["retrieval_mode"]
    families = [
        {
            "priority_level": level,
            "key": key,
            "label": label,
            "document_count": _count_evidence_pdfs(key),
        }
        for level, key, label in [
            (1, "institutional_policy", "Institutional policy"),
            (2, "national_law", "National law/guidance"),
            (3, "international_regulation", "International regulation/guidance"),
        ]
    ]
    profile = {
        "retrieval_mode": mode,
        "source_families": families,
        "review_note": "Expert review confirms applicability before treatment approval.",
    }
    if mode == "source_order":
        profile.update(
            label="Source-ordered retrieval",
            summary="Use relevant passages; otherwise try the next source.",
            source_order=list(config["source_order"]),
            stop_condition=config["stop_condition"],
            automatic_treatment_approval=config["automatic_treatment_approval"],
            legal_applicability_review_required=config["legal_applicability_review_required"],
        )
        return profile

    weights = config["authority_source_weights"]
    alpha = config["semantic_similarity_weight"]
    if mode == "lexical":
        gap = RAGConfig.get_lexical_source_gap()
        profile.update(
            label="Lexical retrieval",
            summary=f"Lexical relevance with source gap {gap:g}; semantic fallback for unmatched terms.",
            formula="score = 0.7 * relevance + 0.3 * (1 - source_gap * source_level)",
            source_gap=gap,
            feedback_ranking={
                "alpha_semantic_similarity": alpha,
                "authority_mixing_weight": config["authority_mixing_weight"],
                "source_weight": weights[0],
            },
        )
        return profile

    for family in families:
        family["weight"] = weights[family["priority_level"]]
    families.insert(
        0,
        {
            "priority_level": 0,
            "key": "approved_expert_feedback",
            "label": "Approved expert feedback",
            "weight": weights[0],
            "document_count": None,
        },
    )
    profile.update(
        label="Weighted retrieval (legacy)",
        summary=f"Similarity {alpha:g}; source preference {config['authority_mixing_weight']:g}.",
        formula="score = alpha * similarity + (1 - alpha) * source_authority",
        alpha_semantic_similarity=alpha,
        authority_mixing_weight=config["authority_mixing_weight"],
    )
    return profile


@app.route("/")
def index() -> str:
    """Serve the main web interface.

    Returns:
        Rendered HTML template
    """
    return render_template("index.html")


@app.route("/api/runtime_config")
def get_runtime_config() -> Tuple:
    """Return the audited, non-secret runtime profile shown in the web UI.

    The retriever and this endpoint read the same mode-aware RAGConfig methods.
    Inactive legacy coefficients are not reported for source-ordered retrieval.
    """
    try:
        retrieval_profile = _get_runtime_retrieval_profile()
        endpoint_scope = _classify_endpoint_scope(Settings.STAGE_1_OPENAI_BASE_URL)

        return jsonify(
            {
                "status": "success",
                "title": AppConfig.PAPER_TITLE,
                "workflow": [
                    "Schema metadata",
                    "Approved feedback and regex heuristics",
                    f"Local LLM with {retrieval_profile['label'].lower()}",
                    "HITL approval",
                    "Template-based code generation",
                ],
                "features": {
                    "rag_enabled": Settings.RAG_ENABLED,
                    "heuristics_enabled": Settings.HEURISTICS_ENABLED,
                    "llm_audit_enabled": Settings.LLM_AUDIT_ENABLED,
                    "raw_records_accessed": False,
                },
                "ranking": retrieval_profile,
                "retrieval": {
                    "chunk_size": Settings.CHUNK_SIZE,
                    "chunk_overlap": Settings.CHUNK_OVERLAP,
                    "top_k": Settings.TOP_K_RESULTS,
                    "legal_embedding_model": Settings.LEGAL_EMBEDDING_MODEL,
                    "feedback_embedding_model": Settings.FEEDBACK_EMBEDDING_MODEL,
                },
                "knowledge_update": {
                    "minimum_consistent_corrections": HeuristicPatternLearner.MIN_PATTERN_OCCURRENCES,
                    "maximum_conflict_rate": HeuristicPatternLearner.MAX_CONFLICT_RATE,
                },
                "models": {
                    "stage_1": Settings.LLM_STAGE_1,
                    "stage_2": Settings.LLM_STAGE_2,
                    "temperature": Settings.LLM_TEMPERATURE,
                    "endpoint_scope": endpoint_scope,
                    "network_boundary_verification": "operational",
                },
            }
        )
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/get_tables")
def get_tables_route() -> Tuple:
    """API endpoint to retrieve available database tables.

    Returns:
        JSON response with table names or error
    """
    try:
        if not os.path.exists(AppConfig.DB_PATH):
            return jsonify({"error": f"Database not found at {AppConfig.DB_PATH}"}), 500

        from pseudragon.database.adapters import get_table_names

        tables = get_table_names(AppConfig.DB_PATH)

        return jsonify({"tables": tables})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_table_schema/<table_name>")
def get_table_schema_route(table_name: str) -> Tuple:
    """API endpoint to retrieve schema for a specific table.

    Args:
        table_name: Name of the table

    Returns:
        JSON response with table schema or error
    """
    try:
        if not os.path.exists(AppConfig.DB_PATH):
            return jsonify({"error": f"Database not found at {AppConfig.DB_PATH}"}), 500

        # Get schema from database
        db_manager = DatabaseManager(AppConfig.DB_PATH)
        schema = db_manager.get_table_schema(table_name)

        if not schema:
            return jsonify({"error": f"Table '{table_name}' not found"}), 404

        return jsonify({"table_name": table_name, "schema": schema})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/execute_ddl", methods=["POST"])
def execute_ddl() -> Tuple:
    """Execute SQL statements directly (CREATE, INSERT, etc.).

    Request body:
        ddl: SQL statements

    Returns:
        JSON response with success status or error
    """
    try:
        data = request.get_json()
        sql_input = data.get("ddl", "").strip()

        if not sql_input:
            return jsonify({"error": "SQL statement is required"}), 400

        import re

        # Block dangerous and query commands

        blocked_patterns = [
            # Dangerous commands
            (r"\bDROP\s+DATABASE\b", "DROP DATABASE is not allowed"),
            (r"\bDELETE\s+FROM\b", "DELETE is not allowed"),
            (r"\bTRUNCATE\b", "TRUNCATE is not allowed"),
            (r"\bALTER\s+DATABASE\b", "ALTER DATABASE is not allowed"),
            (r"\bDROP\s+SCHEMA\b", "DROP SCHEMA is not allowed"),
            # Query commands (require output)
            (r"\bSELECT\b", "SELECT is not allowed (use for data modification only)"),
            (r"\bSHOW\b", "SHOW is not allowed"),
            (r"\bDESCRIBE\b", "DESCRIBE is not allowed"),
            (r"\bEXPLAIN\b", "EXPLAIN is not allowed"),
            (r"\bPRAGMA\b", "PRAGMA is not allowed"),
        ]

        sql_upper = sql_input.upper()
        for pattern, message in blocked_patterns:
            if re.search(pattern, sql_upper):
                return jsonify({"error": message}), 400

        # Execute SQL directly using DuckDB's execute method which handles multiple statements

        db_path = AppConfig.DB_PATH

        try:
            if db_path.endswith(".duckdb"):
                # Use DuckDB - execute the entire SQL script
                import duckdb

                conn = duckdb.connect(db_path)
                # DuckDB can execute multiple statements separated by semicolons

                conn.execute(sql_input)
                conn.close()
            else:
                # Use SQLite - executescript for multiple statements
                import sqlite3

                conn = sqlite3.connect(db_path)
                conn.executescript(sql_input)
                conn.commit()
                conn.close()
        except Exception as e:
            return jsonify({"error": f"SQL Error: {str(e)}"}), 400

        return jsonify(
            {
                "status": "success",
                "message": "SQL executed successfully",
                "table_name": None,
                "tables": [],
                "statement_count": sql_input.count(";"),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/drop_table", methods=["POST"])
def drop_table() -> Tuple:
    """Drop a table from the database.

    Request body:
        table_name: Name of the table to drop

    Returns:
        JSON response with success status or error
    """
    try:
        data = request.get_json()
        table_name = data.get("table_name", "").strip()

        if not table_name:
            return jsonify({"error": "Table name is required"}), 400

        # Validate table name (prevent SQL injection)

        import re

        if not re.match(r"^[\w가-힣]+$", table_name):
            return jsonify({"error": "Invalid table name"}), 400

        db_path = AppConfig.DB_PATH

        try:
            if db_path.endswith(".duckdb"):
                # Use DuckDB
                import duckdb

                conn = duckdb.connect(db_path)
                conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                conn.close()
            else:
                # Use SQLite
                import sqlite3

                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                conn.commit()
                conn.close()
        except Exception as e:
            return jsonify({"error": f"SQL Error: {str(e)}"}), 400

        return jsonify(
            {
                "status": "success",
                "message": f"Table '{table_name}' has been dropped successfully",
                "table_name": table_name,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/stream")
def stream() -> Response:
    """Server-Sent Events (SSE) endpoint for real-time log streaming.

    Streams pipeline logs and status updates to the client in real-time.

    Query Parameters:
        session_id: Client session ID for session-specific logging

    Returns:
        SSE stream with log messages and events
    """
    # Get session ID from query parameter

    client_session_id = request.args.get("session_id", "")
    logging.getLogger(__name__).debug(
        f"DEBUG: /stream endpoint called (session: {client_session_id})"
    )

    # Get session-specific state or fall back to global state

    session_state = session_manager.get_session(client_session_id) if client_session_id else None
    target_state = session_state if session_state else state

    def generate() -> Any:
        """Generator function for SSE stream.

        Yields log messages and stage completion events.
        """
        yield ": connected\n\n"
        last_idx = 0
        stage3_signaled = False
        while True:
            with target_state.progress_lock:
                if last_idx < len(target_state.logs):
                    for i in range(last_idx, len(target_state.logs)):
                        yield f"data: {target_state.logs[i]}\n\n"

                    last_idx = len(target_state.logs)

                if target_state.current_step == PipelineStep.STAGE3_READY and not stage3_signaled:
                    yield f"event: stage2_complete\ndata: ready\n\n"
                    stage3_signaled = True

            time.sleep(0.5)

    return Response(generate(), mimetype="text/event-stream")


@app.route("/start", methods=["POST"])
def start_pipeline() -> Tuple:
    """API endpoint to start pipeline execution (Stages 1 & 2).

    Accepts:

    - tables: List of table names to process
    - purpose_goal: Purpose/intent of data pseudonymization
    - preferred_method: Preferred pseudonymization method
    - session_id: (Optional) Existing session ID to resume

    Returns:
        JSON response with status and session_id
    """
    req_data = request.json or {}

    # Get or create session ID

    client_session_id = req_data.get("session_id", "")
    if not client_session_id:
        # Create new session for this client

        client_session_id = session_manager.create_session()
        logging.getLogger(__name__).debug(f"DEBUG: Created new session: {client_session_id}")
    else:
        # Verify existing session exists

        existing_session = session_manager.get_session(client_session_id)
        if not existing_session:
            # Session expired or invalid, create new one

            client_session_id = session_manager.create_session()
            logging.getLogger(__name__).debug(
                f"DEBUG: Session not found, created new session: {client_session_id}"
            )

    # Get session state
    session_state = session_manager.get_session(client_session_id)
    if not session_state:
        return jsonify({"status": "error", "message": "Failed to create session"}), 500

    # Check if THIS session is already running (not global state)

    if session_state.is_running:
        return jsonify({"status": "already_running", "session_id": client_session_id}), 400

    tables = req_data.get("tables", ["users"])
    purpose_goal = req_data.get("purpose_goal", "")
    preferred_method = req_data.get("preferred_method", "")

    # Cleanup expired sessions periodically

    cleaned = session_manager.cleanup_expired_sessions()
    if cleaned > 0:
        logging.getLogger(__name__).debug(f"DEBUG: Cleaned up {cleaned} expired sessions")

    session_state.knowledge_invalidated = False
    session_state.is_running = True

    thread = Thread(
        target=run_pipeline_task, args=(tables, purpose_goal, preferred_method, client_session_id)
    )
    thread.start()

    return jsonify({"status": "started", "session_id": client_session_id})


@app.route("/stop", methods=["POST"])
def stop_pipeline() -> Tuple:
    """API endpoint to stop pipeline execution.

    Sets the stop flag to gracefully terminate the running pipeline.

    Accepts:
        - session_id: Client session ID

    Returns:
        JSON response with status
    """
    req_data = request.json or {}
    client_session_id = req_data.get("session_id", "")

    # Get session-specific state or fall back to global state

    session_state = session_manager.get_session(client_session_id) if client_session_id else None
    target_state = session_state if session_state else state

    if not target_state.is_running:
        return jsonify({"status": "not_running"}), 400

    # Set stop flag
    with target_state.progress_lock:
        target_state.should_stop = True

    log_producer("⛔ Pipeline stop requested by user", target_state)
    log_producer("⏸️  Stopping pipeline gracefully...", target_state)

    return jsonify({"status": "stopping"})


@app.route("/get_stage3_data")
def get_stage3_data() -> Tuple:
    """API endpoint to retrieve Stage 3 data for human review.

    Query Parameters:
        session_id: Client session ID

    Returns policy recommendations with PII analysis for user selection.

    Returns:
        JSON with PII types, methods, evidence sources, and report filenames
    """
    # Get session ID from query parameter

    client_session_id = request.args.get("session_id", "")

    # Get session-specific state or fall back to global state

    session_state = session_manager.get_session(client_session_id) if client_session_id else None
    target_state = session_state if session_state else state

    if target_state.current_step != PipelineStep.STAGE3_READY:
        return jsonify({"error": "Not ready for Stage 3"}), 400

    # Record when the HITL review window is first made available to the user.
    # The model-side endpoint for the 62x comparison remains model_completed_at.
    if target_state.hitl_window_opened_at is None:
        target_state.hitl_window_opened_at = datetime.now().isoformat()
        if target_state.session_dir:
            timing_path = os.path.join(target_state.session_dir, "timing.json")
            timing = {
                "pipeline_started_at": target_state.pipeline_started_at,
                "model_completed_at": target_state.model_completed_at,
                "hitl_window_opened_at": target_state.hitl_window_opened_at,
                "model_elapsed_seconds": (
                    datetime.fromisoformat(target_state.model_completed_at)
                    - datetime.fromisoformat(target_state.pipeline_started_at)
                ).total_seconds()
                if target_state.pipeline_started_at and target_state.model_completed_at
                else None,
                "timing_definition": "model_completed_at ends all RAG+LLM judgments; hitl_window_opened_at is the first review-data response",
            }
            with open(timing_path, "w", encoding="utf-8") as timing_file:
                json.dump(timing, timing_file, ensure_ascii=False, indent=2)

    display_data: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # Use target_tables if available, otherwise fall back to all schemas

    tables_to_process = getattr(target_state, "target_tables", None)
    if not tables_to_process:
        tables_to_process = list(target_state.schemas.keys())
    logging.getLogger(__name__).debug(
        f"[DEBUG /get_stage3_data] Processing tables: {tables_to_process}"
    )

    for table in tables_to_process:
        if table not in target_state.schemas:
            print(f"[WARNING] Table {table} not found in schemas, skipping")
            continue
        display_data[table] = {}

        # Process ALL columns from schema (both PII and non-PII)
        for col in target_state.schemas[table].keys():
            # Check if column is in policy_data (could be PII or Non-PII)
            if table in target_state.policy_data and col in target_state.policy_data[table]:
                # Column has policy data - check if it's PII or Non-PII
                policy = target_state.policy_data[table][col]
                is_pii = policy.get("is_pii", False)

                if is_pii:
                    # PII column - use existing policy
                    pii_info = target_state.pii_data[table].get(col, {})

                    # Validate pii_type consistency
                    # If is_pii=True but pii_type="Non-PII", this is a data inconsistency bug
                    # Fix it by defaulting to "PII"
                    pii_type = pii_info.get("pii_type", "Unknown")
                    if pii_type == "Non-PII":
                        print(
                            f"[WARNING] Data inconsistency detected for {col}: is_pii=True but pii_type='Non-PII'"
                        )
                        print(f"[FIX] Correcting pii_type to 'PII' for {col}")
                        pii_type = "PII"
                        # Also update the stored data
                        if table in target_state.pii_data and col in target_state.pii_data[table]:
                            target_state.pii_data[table][col]["pii_type"] = "PII"

                    display_data[table][col] = {
                        "pii_type": pii_type,
                        "is_pii": True,
                        "id_evidence": pii_info.get("evidence_source", "Unknown"),
                        "methods": policy["recommended_methods"],
                        "tech_evidence": policy.get("legal_source", "Unknown"),
                        "column_comment": pii_info.get("column_comment", ""),
                    }
                else:
                    # Non-PII column with existing policy data

                    cached_methods = policy.get("recommended_methods", [])
                    cached_method_names = {m.get("method", "").upper() for m in cached_methods}

                    pii_info = target_state.pii_data[table].get(col, {})
                    column_comment = pii_info.get("column_comment", "")

                    # If not in pii_data, try to get comment from schema
                    if not column_comment and isinstance(target_state.schemas[table][col], dict):
                        column_comment = target_state.schemas[table][col].get("comment", "")

                    # Check if cached methods are incomplete (only KEEP, missing DELETE)

                    needs_regeneration = (
                        len(cached_methods) == 1
                        and "KEEP" in cached_method_names
                        and "DELETE" not in cached_method_names
                    )

                    if needs_regeneration:
                        # Regenerate methods for Non-PII column

                        print(
                            f"[INFO] Regenerating Non-PII methods for {col} (cached methods incomplete: {[m.get('method') for m in cached_methods]})"
                        )

                        # Get data type for better method recommendations
                        data_type = ""
                        if isinstance(target_state.schemas[table][col], dict):
                            data_type = target_state.schemas[table][col].get("type", "")

                        non_pii_methods = _generate_non_pii_methods(col, column_comment, data_type)
                        logging.getLogger(__name__).debug(
                            f"[DEBUG] Regenerated {len(non_pii_methods)} methods for {col}: {[t['method'] for t in non_pii_methods]}"
                        )

                        # Update policy_data with regenerated methods
                        target_state.policy_data[table][col]["recommended_methods"] = (
                            non_pii_methods
                        )

                        display_data[table][col] = {
                            "pii_type": "Non-PII",
                            "is_pii": False,
                            "id_evidence": "Not identified as PII",
                            "methods": non_pii_methods,
                            "tech_evidence": "LLM-generated recommendations for non-PII data",
                            "column_comment": column_comment,
                        }
                    else:
                        # Use cached methods
                        print(
                            f"[INFO] Using cached Non-PII policy for {col} ({len(cached_methods)} methods)"
                        )

                        display_data[table][col] = {
                            "pii_type": "Non-PII",
                            "is_pii": False,
                            "id_evidence": "Not identified as PII",
                            "methods": cached_methods,
                            "tech_evidence": policy.get(
                                "evidence_source", "Data Minimization Principle"
                            ),
                            "column_comment": column_comment,
                        }
            else:
                # Non-PII column - generate methods using LLM (similar to PII columns)
                pii_info = target_state.pii_data[table].get(col, {})
                column_comment = pii_info.get("column_comment", "")

                # If not in pii_data, try to get comment from schema
                if not column_comment and isinstance(target_state.schemas[table][col], dict):
                    column_comment = target_state.schemas[table][col].get("comment", "")

                # Get data type for better method recommendations
                data_type = ""
                if isinstance(target_state.schemas[table][col], dict):
                    data_type = target_state.schemas[table][col].get("type", "")

                # Generate Non-PII methods using LLM
                logging.getLogger(__name__).debug(
                    f"[DEBUG /get_stage3_data] Generating Non-PII methods for {col}"
                )
                logging.getLogger(__name__).debug(f"  - Column: {col}")
                logging.getLogger(__name__).debug(f"  - Data Type: {data_type}")
                logging.getLogger(__name__).debug(f"  - Comment: {column_comment}")
                non_pii_methods = _generate_non_pii_methods(col, column_comment, data_type)
                logging.getLogger(__name__).debug(
                    f"[DEBUG /get_stage3_data] Generated {len(non_pii_methods)} methods for {col}"
                )
                logging.getLogger(__name__).debug(
                    f"  - Methods: {[t['method'] for t in non_pii_methods]}"
                )

                # Store Non-PII methods in policy_data for later retrieval in submit_stage3

                if table not in target_state.policy_data:
                    target_state.policy_data[table] = {}
                target_state.policy_data[table][col] = {
                    "is_pii": False,
                    "pii_type": "Non-PII",
                    "recommended_methods": non_pii_methods,
                    "evidence_source": "LLM-generated recommendations for non-PII data",
                }

                display_data[table][col] = {
                    "pii_type": "Non-PII",
                    "is_pii": False,
                    "id_evidence": "Not identified as PII",
                    "methods": non_pii_methods,
                    "tech_evidence": "LLM-generated recommendations for non-PII data",
                    "column_comment": column_comment,
                }

    # Include report filenames

    response_data = {"tables": display_data, "reports": target_state.report_files}

    return jsonify(response_data)


@app.route("/submit_stage3", methods=["POST"])
def submit_stage3() -> Tuple:
    """API endpoint to submit Stage 3 user selections and execute Stage 4.

    Processes user-selected methods and generates final pseudonymization code.

    Accepts:

    - choices: Dictionary of user-selected method indices

    - session_id: Client session ID

    Returns:
        JSON with generated code for each table
    """
    req_data = request.json or {}
    client_session_id = req_data.get("session_id", "")
    choices = {k: v for k, v in req_data.items() if k != "session_id"}

    # Get session-specific state or fall back to global state

    session_state = session_manager.get_session(client_session_id) if client_session_id else None
    target_state = session_state if session_state else state

    # Validate all submitted edits before state, audit logs or preferences change.
    # Unsupported custom edits need an explicit declaration; do not guess semantics.
    normalized_selections = {}
    try:
        for key, selection_data in choices.items():
            if not key.startswith("method_"):
                continue
            remainder = key[7:]
            for table_name in sorted(target_state.schemas, key=len, reverse=True):
                if not remainder.startswith(f"{table_name}_"):
                    continue
                column_name = remainder[len(table_name) + 1 :]
                methods = (
                    target_state.policy_data.get(table_name, {})
                    .get(column_name, {})
                    .get("recommended_methods", [])
                )
                index = int(
                    selection_data.get("index", 0)
                    if isinstance(selection_data, dict)
                    else selection_data
                )
                if methods and not 0 <= index < len(methods):
                    raise MethodContractError(
                        f"Invalid selected method for {table_name}.{column_name}."
                    )
                if methods:
                    normalized_selections[key] = prepare_method_selection(
                        methods[index], column_name, selection_data
                    )
                break
    except (MethodContractError, TypeError, ValueError) as exc:
        return jsonify({"status": "validation_error", "error": str(exc)}), 400

    target_state.current_step = PipelineStep.STAGE4_PROCESSING

    if target_state.audit_logger:
        target_state.audit_logger.log_stage_start("Stage 3: HITL Approval", "all_tables")

    # Process user selections for ALL columns (PII and non-PII)
    logging.getLogger(__name__).debug(f"[DEBUG submit_stage3] Processing {len(choices)} selections")
    logging.getLogger(__name__).debug(
        f"[DEBUG submit_stage3] Keys received: {list(choices.keys())}"
    )

    for key, selection_data in choices.items():
        # Key format: "method_TABLENAME_COLUMNNAME"
        # Table names can contain underscores (e.g., "withdraw_transfers")
        # So we need to be smarter about parsing

        if not key.startswith("method_"):
            print(f"[WARNING] Invalid key format: {key}")
            continue

        # Remove "method_" prefix
        remainder = key[7:]  # Remove "method_"

        # Find table name by checking against known tables
        table = None
        col = None

        for table_name in sorted(target_state.schemas, key=len, reverse=True):
            if remainder.startswith(f"{table_name}_"):
                table = table_name
                col = remainder[len(table_name) + 1 :]  # +1 for the underscore
                break

        if not table or not col:
            print(f"[WARNING] Could not parse key: {key}, remainder: {remainder}")
            print(f"[WARNING] Known tables: {list(target_state.schemas.keys())}")
            continue

        # Accept either a selection index or an explicit selection payload.
        if isinstance(selection_data, dict):
            idx = int(selection_data.get("index", 0))
            edited_code = selection_data.get("code_snippet", "")
            code_modified = selection_data.get("code_modified", False)
            edited_description = selection_data.get("description", "")
            description_modified = selection_data.get("description_modified", False)
        else:
            idx = int(selection_data)
            edited_code = None
            code_modified = False
            edited_description = None
            description_modified = False

        logging.getLogger(__name__).debug(
            f"[DEBUG] Parsed: key={key} -> table={table}, col={col}, idx={idx}, code_modified={code_modified}, desc_modified={description_modified}"
        )

        # A non-default method index or explicit code/description edit is a
        # human correction to the model's proposal. Keep this separate from
        # the immutable cold-start output captured at STAGE3_READY.
        if idx != 0 or code_modified or description_modified:
            target_state.manual_corrections.append(
                {
                    "kind": "policy_selection",
                    "table": table,
                    "column": col,
                    "selected_index": idx,
                    "code_modified": bool(code_modified),
                    "description_modified": bool(description_modified),
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # Check if column is in policy_data AND is still PII
        # User may have changed PII status, so check is_pii flag
        in_policy_data = (
            table in target_state.policy_data and col in target_state.policy_data[table]
        )
        is_pii_flag = (
            target_state.policy_data[table][col].get("is_pii", True) if in_policy_data else False
        )

        logging.getLogger(__name__).debug(
            f"[DEBUG] Column {col}: in_policy_data={in_policy_data}, is_pii_flag={is_pii_flag}"
        )

        is_current_pii = in_policy_data and is_pii_flag

        if is_current_pii:
            # PII column - reorder methods based on user selection
            techs = target_state.policy_data[table][col]["recommended_methods"]
            if 0 <= idx < len(techs):
                # Save old action object BEFORE modifying the list
                old_action_obj = techs[0].copy() if techs else {}
                old_action = old_action_obj.get("method", "UNKNOWN")

                # Get and remove selected method from list
                selected = normalized_selections.get(key, techs[idx])
                techs.pop(idx)
                new_action = selected.get("method", "UNKNOWN")

                # Update code snippet if user edited it
                if code_modified and edited_code:
                    old_code = selected.get("code_snippet", "")
                    selected["code_snippet"] = edited_code
                    logging.getLogger(__name__).debug(
                        f"[DEBUG] Code modified for {col}: User edited code snippet"
                    )
                    log_producer(f"✏️ User edited code for {col}", target_state)

                # Update description if user edited it
                if description_modified and edited_description:
                    old_desc = selected.get("description", "")
                    selected["description"] = edited_description
                    logging.getLogger(__name__).debug(
                        f"[DEBUG] Description modified for {col}: User edited description"
                    )
                    log_producer(f"✏️ User edited description for {col}", target_state)

                if code_modified and edited_code:
                    if target_state.audit_logger:
                        target_state.audit_logger.log_user_edit(
                            table=table,
                            column=col,
                            old_action=old_action,
                            new_action=new_action,
                            rationale=f"User modified code snippet. {selected.get('description', '')}",
                            old_parameters=old_action_obj.get("parameters", {}),
                            new_parameters=selected.get("parameters", {}),
                            old_code_snippet=old_action_obj.get("code_snippet", ""),
                            new_code_snippet=edited_code,
                            legal_evidence=selected.get("legal_evidence", ""),
                        )

                # Insert selected method at the beginning
                techs.insert(0, selected)
                target_state.policy_data[table][col]["action"] = new_action

                logging.getLogger(__name__).debug(
                    f"[DEBUG] PII Column {col}: User selected idx={idx}, Method changed from '{old_action}' to '{new_action}'"
                )
                log_producer(
                    f"✏️ User selection for {col}: {new_action} (was: {old_action})", target_state
                )

                # Record user preference for future sessions (including default selection idx=0)

                if target_state.expert_preference_manager:
                    pii_type = target_state.policy_data[table][col].get("pii_type", "Unknown")
                    all_methods = [t.get("method", "") for t in techs]
                    # Pass code snippet and description if user edited them

                    final_code_snippet = selected.get("code_snippet", "") if code_modified else None
                    final_description = (
                        selected.get("description", "") if description_modified else None
                    )

                    # Get column_comment from schema or pii_data for RAG-based similarity search

                    col_comment = ""
                    if table in target_state.schemas and col in target_state.schemas[table]:
                        schema_col = target_state.schemas[table][col]
                        if isinstance(schema_col, dict):
                            col_comment = schema_col.get("comment", "")
                    if (
                        not col_comment
                        and table in target_state.pii_data
                        and col in target_state.pii_data[table]
                    ):
                        col_comment = target_state.pii_data[table][col].get("column_comment", "")

                    target_state.expert_preference_manager.record_selection(
                        table_name=table,
                        column_name=col,
                        pii_type=pii_type,
                        is_pii=True,
                        selected_method=new_action,
                        available_methods=all_methods,
                        selected_index=idx,
                        code_snippet=final_code_snippet,
                        code_modified=code_modified,
                        description=final_description,
                        description_modified=description_modified,
                        column_comment=col_comment,
                    )

                if target_state.audit_logger and not code_modified:
                    target_state.audit_logger.log_user_edit(
                        table=table,
                        column=col,
                        old_action=old_action,
                        new_action=new_action,
                        rationale=selected.get("description", ""),
                        old_code_snippet=old_action_obj.get("code_snippet", ""),
                        new_code_snippet=selected.get("code_snippet", ""),
                        old_parameters=old_action_obj.get("parameters", {}),
                        new_parameters=selected.get("parameters", {}),
                        legal_evidence=selected.get("legal_evidence", ""),
                    )
        else:
            # Non-PII column - create or update policy entry for user selection
            # This handles:
            # 1. Originally Non-PII columns with user selections
            # 2. PII -> Non-PII reclassified columns (is_pii=False was set in change_pii_status)
            if table not in target_state.policy_data:
                target_state.policy_data[table] = {}
            selected = {}

            # Get the actual selected method from policy_data (could be ROUND, GENERALIZE, HASH, KEEP, DELETE)
            # Don't hardcode idx->method mapping; use actual method at that index
            if table in target_state.policy_data and col in target_state.policy_data[table]:
                techs = target_state.policy_data[table][col].get("recommended_methods", [])

                if 0 <= idx < len(techs):
                    # Use actual method at selected index
                    selected = normalized_selections.get(key, techs[idx])
                    method = selected.get("method", "Keep")
                    description = selected.get("description", "Keep the column unchanged")
                    code_snippet = selected.get("code_snippet", "# No transformation needed")

                    # Apply user-edited code if modified

                    if code_modified and edited_code:
                        old_code = code_snippet
                        code_snippet = edited_code
                        logging.getLogger(__name__).debug(
                            f"[DEBUG] Non-PII Column {col}: User edited code snippet"
                        )
                        logging.getLogger(__name__).debug(f"[DEBUG]   Original: {old_code[:50]}...")
                        logging.getLogger(__name__).debug(
                            f"[DEBUG]   Edited: {edited_code[:50]}..."
                        )
                        log_producer(f"✏️ User edited code for Non-PII column {col}", target_state)

                    # Apply user-edited description if modified

                    if description_modified and edited_description:
                        old_desc = description
                        description = edited_description
                        logging.getLogger(__name__).debug(
                            f"[DEBUG] Non-PII Column {col}: User edited description"
                        )
                        log_producer(
                            f"✏️ User edited description for Non-PII column {col}", target_state
                        )

                    logging.getLogger(__name__).debug(
                        f"[DEBUG] Non-PII Column {col}: User selected idx={idx}, Method='{method}' from {len(techs)} methods"
                    )
                    logging.getLogger(__name__).debug(
                        f"[DEBUG]   Available methods: {[t.get('method') for t in techs]}"
                    )
                else:
                    # Invalid index - fallback to Keep
                    print(
                        f"[WARNING] Non-PII Column {col}: Invalid idx={idx} (only {len(techs)} methods available)"
                    )
                    method = "Keep"
                    description = "Keep the column unchanged"
                    code_snippet = "# No transformation needed"
            else:
                # No policy data - use simple fallback for Keep/Delete only
                print(
                    f"[WARNING] Non-PII Column {col}: No policy data found, using fallback for idx={idx}"
                )
                if idx == 1:
                    method = "Delete"
                    description = "Remove the column if not needed for analysis"
                    code_snippet = f"record.pop('{col}', None)"
                else:
                    method = "Keep"
                    description = "Keep the column unchanged"
                    code_snippet = "# No transformation needed"

            # Record the selected non-PII operation at debug level.
            logging.getLogger(__name__).debug(
                f"[DEBUG] Non-PII Column {col}: Final selection - Method='{method}'"
            )
            log_producer(f"✏️ Non-PII selection for {col}: {method}", target_state)

            # Record user preference for Non-PII columns (if user changed from default)

            if target_state.expert_preference_manager:
                # Get all available methods from the method list
                if table in target_state.policy_data and col in target_state.policy_data[table]:
                    available_methods = [
                        t.get("method")
                        for t in target_state.policy_data[table][col].get("recommended_methods", [])
                    ]
                else:
                    # Fallback to Keep/Delete only
                    available_methods = ["Keep", "Delete"]

                # Pass code snippet and description if user edited them

                final_code_snippet = code_snippet if code_modified else None
                final_description = description if description_modified else None

                # Get column_comment from schema or pii_data for RAG-based similarity search

                col_comment = ""
                if table in target_state.schemas and col in target_state.schemas[table]:
                    schema_col = target_state.schemas[table][col]
                    if isinstance(schema_col, dict):
                        col_comment = schema_col.get("comment", "")
                if (
                    not col_comment
                    and table in target_state.pii_data
                    and col in target_state.pii_data[table]
                ):
                    col_comment = target_state.pii_data[table][col].get("column_comment", "")

                target_state.expert_preference_manager.record_selection(
                    table_name=table,
                    column_name=col,
                    pii_type="Non-PII",
                    is_pii=False,
                    selected_method=method,
                    available_methods=available_methods,
                    selected_index=idx,
                    code_snippet=final_code_snippet,
                    code_modified=code_modified,
                    description=final_description,
                    description_modified=description_modified,
                    column_comment=col_comment,
                )

            # Create or update policy for non-PII column with the actual selected method
            # This overwrites any existing policy (including PII -> Non-PII reclassified ones)
            target_state.policy_data[table][col] = {
                "is_pii": False,
                "pii_type": "Non-PII",
                "action": method,  # Add action field for audit logging
                "recommended_methods": [
                    {
                        **selected,
                        "method": method,
                        "applicability": "High",
                        "description": description,
                        "code_snippet": code_snippet,
                        "example_implementation": code_snippet,
                        "parameters": copy.deepcopy(selected.get("parameters", {})),
                    }
                ],
                "evidence_source": target_state.policy_data[table]
                .get(col, {})
                .get("evidence_source", "User Selection (Non-PII)"),
            }

            # Log user selection for non-PII column
            if target_state.audit_logger:
                target_state.audit_logger.log_user_edit(
                    table=table,
                    column=col,
                    old_action="N/A (Non-PII)",
                    new_action=method,
                    rationale=f"User selected {method} for non-PII column",
                    old_parameters={},
                    new_parameters=selected.get("parameters", {}),
                    old_code_snippet="",
                    new_code_snippet=code_snippet,
                    legal_evidence="",
                )

    # Regenerate integrated report AFTER user modifications in Stage 3

    # This ensures that the report reflects all user changes (PII reclassification, method selection, code edits)

    log_producer(
        "\n[Stats] Regenerating integrated reports with user modifications...", target_state
    )

    # Extract selected tables from choices to only process those tables

    selected_tables = set()
    for key in choices.keys():
        if key.startswith("method_"):
            remainder = key[7:]
            for table_name in sorted(target_state.schemas, key=len, reverse=True):
                if remainder.startswith(f"{table_name}_"):
                    selected_tables.add(table_name)
                    break

    # Clear previous generated_code and report_files for clean generation

    target_state.generated_code = {}
    target_state.report_files = {}

    log_producer(
        f"[Info] Processing {len(selected_tables)} selected table(s): {list(selected_tables)}",
        target_state,
    )

    for table in selected_tables:
        if table not in target_state.policy_data:
            continue

        try:
            # Convert policy_data to Policy object for report generation

            from pseudragon.domain.policy_dsl import Policy, ColumnPolicy, PolicyAction, ActionType
            from pseudragon.reporting.compliance_reporter import ReportGenerator

            policy_obj = Policy(table_name=table)

            for col_name, col_data in target_state.policy_data[table].items():
                is_pii = col_data.get("is_pii", False)
                pii_type = col_data.get("pii_type", "Non-PII")

                # Get the first (selected) method
                techs = col_data.get("recommended_methods", [])
                if techs:
                    selected_tech = techs[0]
                    action_type = ActionType[selected_tech["method"]]

                    # Extract rationale from PII classification change if available

                    evidence_source = col_data.get("evidence_source", "")
                    user_rationale = ""
                    legal_evidence = selected_tech.get("legal_evidence", "")

                    # Check if this was a user-defined PII classification change

                    if evidence_source.startswith("User-defined PII"):
                        # Extract the rationale from evidence_source

                        if "Rationale:" in evidence_source:
                            parts = evidence_source.split("Rationale:", 1)
                            if len(parts) == 2:
                                user_rationale = parts[1].strip()

                        # Set legal_evidence to just "User-defined PII"

                        legal_evidence = "User-defined PII"

                    # Use user rationale if available, otherwise use method description

                    final_rationale = (
                        user_rationale if user_rationale else selected_tech.get("description", "")
                    )

                    action = PolicyAction(
                        action=action_type,
                        rationale=final_rationale,
                        code_snippet=selected_tech.get("code_snippet", ""),
                        legal_evidence=legal_evidence,
                        parameters=selected_tech.get("parameters", {}),
                    )

                    col_policy = ColumnPolicy(
                        column_name=col_name, pii_type=pii_type, is_pii=is_pii, action=action
                    )

                    policy_obj.columns[col_name] = col_policy
                else:
                    # Handle columns without methods (e.g., Non-PII columns or PII->Non-PII reclassified)
                    # Default to KEEP action
                    action = PolicyAction(
                        action=ActionType.KEEP,
                        rationale="No transformation applied",
                        code_snippet="",
                        legal_evidence="",
                        parameters={},
                    )

                    col_policy = ColumnPolicy(
                        column_name=col_name, pii_type=pii_type, is_pii=is_pii, action=action
                    )

                    policy_obj.columns[col_name] = col_policy

            # Ensure pii_data has all columns with updated information

            if table not in target_state.pii_data:
                target_state.pii_data[table] = {}

            for col_name, col_data in target_state.policy_data[table].items():
                if col_name not in target_state.pii_data[table]:
                    # Add missing column to pii_data
                    target_state.pii_data[table][col_name] = {
                        "is_pii": col_data.get("is_pii", False),
                        "pii_type": col_data.get("pii_type", "Non-PII"),
                        "reasoning": "User modified",
                        "column_comment": col_data.get("column_comment", ""),
                    }
                else:
                    # Update existing entry with current PII status
                    target_state.pii_data[table][col_name]["is_pii"] = col_data.get("is_pii", False)
                    target_state.pii_data[table][col_name]["pii_type"] = col_data.get(
                        "pii_type", "Non-PII"
                    )

            # Regenerate integrated report with user-modified policy

            table_dir = os.path.join(target_state.session_dir, table)
            os.makedirs(table_dir, exist_ok=True)

            report_gen = ReportGenerator()
            report_path = report_gen.generate_integrated_report(
                table_name=table,
                pii_analysis=target_state.pii_data.get(table, {}),
                policy=policy_obj,
                output_dir=table_dir,
            )

            # Update target_state.report_files so the frontend shows the updated report

            rel_report_path = os.path.relpath(report_path, AppConfig.OUTPUT_DIR)
            report_key = f"{table}_integrated"
            target_state.report_files[report_key] = rel_report_path

            log_producer(f"[OK] Updated integrated report for {table}: {report_path}", target_state)
            log_producer(f"  Report key: {report_key}, Path: {rel_report_path}", target_state)

        except Exception as e:
            log_producer(f"[WARN]️ Failed to regenerate report for {table}: {e}", target_state)
            print(f"[ERROR] Report regeneration failed for {table}: {e}")
            import traceback

            traceback.print_exc()

    log_producer("[OK] All integrated reports updated with user modifications\n", target_state)

    # Persist the user's final Stage-3 selections as the experiment oracle
    # before Stage 4 code generation begins. This keeps oracle labels
    # recoverable even if code generation later fails or is interrupted.
    oracle_root = target_state.session_dir or AppConfig.OUTPUT_DIR
    oracle_path = os.path.join(oracle_root, "oracle_labels.json")
    with open(oracle_path, "w", encoding="utf-8") as oracle_file:
        json.dump(
            {
                "oracle_source": "user_final_stage3_selections",
                "oracle_definition": "The user's final PII status/type and selected policy are the oracle labels for this experiment.",
                "model_completed_at": target_state.model_completed_at,
                "hitl_window_opened_at": target_state.hitl_window_opened_at,
                "choices": choices,
                "oracle_pii_data": target_state.pii_data,
                "oracle_policy_data": target_state.policy_data,
            },
            oracle_file,
            ensure_ascii=False,
            indent=2,
            cls=CustomJSONEncoder,
        )
    log_producer(f"[Audit] Saved user oracle labels: {oracle_path}", target_state)

    if target_state.audit_logger:
        target_state.audit_logger.log_stage_complete("Stage 3: HITL Approval", "all_tables")

    # Trigger heuristic pattern learning at Stage 3 completion

    heuristic_learning_stats = None
    if target_state.expert_preference_manager:
        log_producer("[Update] Learning heuristic patterns from expert feedback...", target_state)
        pattern_stats = target_state.expert_preference_manager.trigger_pattern_learning()
        if pattern_stats:
            heuristic_learning_stats = pattern_stats
            log_producer(
                f"[OK] Learned {pattern_stats['total_patterns']} heuristic patterns", target_state
            )
            log_producer(
                f"  - PII patterns: {pattern_stats['pii_patterns']['suffixes']} suffixes, {pattern_stats['pii_patterns']['prefixes']} prefixes, {pattern_stats['pii_patterns']['keywords']} keywords",
                target_state,
            )
            log_producer(
                f"  - Non-PII patterns: {pattern_stats['non_pii_patterns']['suffixes']} suffixes, {pattern_stats['non_pii_patterns']['prefixes']} prefixes, {pattern_stats['non_pii_patterns']['keywords']} keywords",
                target_state,
            )

    if not os.path.exists(AppConfig.OUTPUT_DIR):
        os.makedirs(AppConfig.OUTPUT_DIR)

    # Only generate code for selected tables (not all tables in schemas)

    for table in selected_tables:
        if table not in target_state.policy_data or table not in target_state.schemas:
            continue
        schema = target_state.schemas[table]

        if target_state.audit_logger:
            target_state.audit_logger.log_policy_approval(table, target_state.policy_data[table])

        # Table directory already exists from Stage 1/2
        table_dir = os.path.join(target_state.session_dir, table)
        os.makedirs(table_dir, exist_ok=True)

        # Stage 4: Code Generation with timing

        stage4_name = f"Stage 4 - Code Generation ({table})"
        duration = 0.0  # Initialize duration variable
        if target_state.file_logger:
            target_state.file_logger.stage_start(stage4_name)

        if target_state.audit_logger:
            target_state.audit_logger.log_stage_start(stage4_name, table)

        # Create session-specific log callback for code generation
        def session_log_callback(msg: str) -> None:
            log_producer(msg, target_state)

        try:
            code = target_state.engine.generate_python_code(
                schema,
                target_state.policy_data[table],
                target_state.pii_data[table],
                table_name=table,
                db_config=AppConfig.DB_PATH,  # Can be string path or dict config
                log_callback=session_log_callback,
            )

            output_file = os.path.join(table_dir, f"pseudonymize_{table}.py")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(code)

            target_state.generated_code[table] = code
            log_producer(f"Code for {table} generated and saved to {output_file}", target_state)
        finally:
            # Ensure stage_end is always called even if exception occurs

            if target_state.file_logger:
                duration = target_state.file_logger.stage_end(stage4_name)

        if target_state.audit_logger:
            target_state.audit_logger.log_stage_complete(stage4_name, table)

        if duration < 60:
            duration_str = f"{duration:.2f}s"
        else:
            minutes = int(duration // 60)
            seconds = duration % 60
            duration_str = f"{minutes}m {seconds:.2f}s"

        log_producer(f"[Time] {stage4_name}: {duration_str}", target_state)

        if target_state.audit_logger:
            target_state.audit_logger.log_code_generation(
                table=table, code_length=len(code), validation_passed=True
            )

            report_path = target_state.audit_logger.generate_compliance_report(
                policy_dict=target_state.policy_data[table],
                table_name=table,
                output_dir=table_dir,
            )

            # Store relative path for web interface
            rel_report_path = os.path.relpath(report_path, AppConfig.OUTPUT_DIR)
            target_state.report_files[f"{table}_compliance"] = rel_report_path
            log_producer(f"[File] Compliance report generated: {report_path}", target_state)

    # Persist the human-review corrections separately from the immutable
    # model output. Every entry here is a model-error candidate for the
    # primary cold-start analysis; the resulting report is Loop 2/HITL output.
    correction_root = target_state.session_dir or AppConfig.OUTPUT_DIR
    corrections_path = os.path.join(correction_root, "manual_corrections.json")
    with open(corrections_path, "w", encoding="utf-8") as corrections_file:
        json.dump(
            {
                "model_completed_at": target_state.model_completed_at,
                "hitl_window_opened_at": target_state.hitl_window_opened_at,
                "correction_count": len(target_state.manual_corrections),
                "corrections": target_state.manual_corrections,
                "interpretation": "human changes to the initial RAG+LLM proposal are counted as model errors; final policy is reported separately as HITL/Loop 2",
            },
            corrections_file,
            ensure_ascii=False,
            indent=2,
        )

    target_state.current_step = PipelineStep.COMPLETED

    # Log final summary

    if target_state.file_logger:
        target_state.file_logger.log_summary()
        log_producer(
            f"[Dir] Log file saved: {target_state.file_logger.get_log_file_path()}", target_state
        )

    # Return updated report files so frontend can show the regenerated integrated reports

    return jsonify(
        {
            "status": "completed",
            "codes": target_state.generated_code,
            "reports": target_state.report_files,  # Include updated report files
            "heuristic_learning": heuristic_learning_stats,  # Include heuristic learning stats for notification
        }
    )


@app.route("/validate_policy", methods=["POST"])
def validate_policy() -> Tuple:
    """API endpoint for real-time policy validation.

    Implements real-time validation

    Accepts:
        - session_id: Client session ID
        - table: Table name
        - policies: Dictionary of user-modified policies

    Returns:
        JSON with validation results and violations
    """
    try:
        data = request.json
        client_session_id = data.get("session_id", "")
        table = data.get("table")
        user_policies = data.get("policies", {})

        # Get session-specific state or fall back to global state

        session_state = (
            session_manager.get_session(client_session_id) if client_session_id else None
        )
        target_state = session_state if session_state else state

        if not table or table not in target_state.policy_data:
            return jsonify({"error": "Table not found"}), 404

        if not target_state.validator:
            return jsonify({"error": "Validator not initialized"}), 500

        temp_policy = Policy(table_name=table, preferred_method="")

        for col_name, col_data in target_state.policy_data[table].items():
            is_pii = col_data.get("is_pii", False)
            pii_type = col_data.get("pii_type", "Non-PII")

            if col_name in user_policies:
                action_name = user_policies[col_name].get("action", "KEEP")
                rationale = user_policies[col_name].get("rationale", "")

                try:
                    action_type = ActionType[action_name]
                except KeyError:
                    action_type = ActionType.KEEP

                # Extract legal evidence and check for user-defined PII

                evidence_source = col_data.get("evidence_source", "")
                legal_evidence = col_data.get("legal_evidence", "Unknown")

                if evidence_source.startswith("User-defined PII"):
                    legal_evidence = "User-defined PII"

                action = PolicyAction(
                    action=action_type,
                    rationale=rationale,
                    legal_evidence=legal_evidence,
                )
            else:
                techs = col_data.get("recommended_methods", [])
                if techs:
                    first_tech = techs[0]

                    # Extract rationale from PII classification change if available

                    evidence_source = col_data.get("evidence_source", "")
                    user_rationale = ""
                    legal_evidence = col_data.get("legal_evidence", "Unknown")

                    # Check if this was a user-defined PII classification change

                    if evidence_source.startswith("User-defined PII"):
                        # Extract the rationale from evidence_source

                        if "Rationale:" in evidence_source:
                            parts = evidence_source.split("Rationale:", 1)
                            if len(parts) == 2:
                                user_rationale = parts[1].strip()

                        # Set legal_evidence to just "User-defined PII"

                        legal_evidence = "User-defined PII"

                    # Use user rationale if available, otherwise use method description

                    final_rationale = (
                        user_rationale if user_rationale else first_tech.get("description", "")
                    )

                    action = PolicyAction(
                        action=ActionType[first_tech.get("method", "KEEP")],
                        rationale=final_rationale,
                        legal_evidence=legal_evidence,
                    )
                else:
                    action = PolicyAction(action=ActionType.KEEP)

            col_policy = ColumnPolicy(
                column_name=col_name, pii_type=pii_type, is_pii=is_pii, action=action
            )
            temp_policy.add_column_policy(col_policy)

        violations = target_state.validator.validate_policy(
            temp_policy, target_state.schemas[table]
        )

        if target_state.audit_logger:
            target_state.audit_logger.log_validation_result(
                table,
                [
                    {"severity": v.severity, "column": v.column, "message": v.message}
                    for v in violations
                ],
            )

        summary = target_state.validator.get_validation_summary(violations)

        return jsonify(
            {
                "status": "validated",
                "violations": [
                    {
                        "severity": v.severity,
                        "column": v.column,
                        "message": v.message,
                        "suggestion": v.suggestion,
                        "check_type": v.check_type,
                    }
                    for v in violations
                ],
                "summary": summary,
            }
        )

    except Exception as e:
        import traceback

        traceback.print_exc()

        return jsonify({"error": str(e)}), 500


@app.route("/get_report/<path:filename>")
def get_report(filename: str) -> Tuple:
    """API endpoint to retrieve markdown report content.

    Args:
        filename: Report filename or path

    Returns:
        Markdown content as text
    """
    try:
        report_path = os.path.join(AppConfig.REPORTS_DIR, filename)
        if not os.path.exists(report_path):
            return "Report not found", 404

        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()

        return Response(content, mimetype="text/plain")
    except Exception as e:
        return f"Error loading report: {str(e)}", 500


@app.route("/download_report/<path:filename>")
def download_report(filename: str) -> Tuple:
    """API endpoint to download markdown report.

    Args:
        filename: Report filename

    Returns:
        File download response
    """
    try:
        report_path = os.path.join(AppConfig.REPORTS_DIR, filename)
        if not os.path.exists(report_path):
            return "Report not found", 404

        return send_file(report_path, as_attachment=True, download_name=filename)
    except Exception as e:
        return f"Error downloading report: {str(e)}", 500


# Heuristics Management API Endpoints


@app.route("/api/heuristics", methods=["GET"])
def get_heuristics() -> Tuple:
    """Get all heuristics.

    Query params:
        enabled_only: "true" to get only enabled heuristics
        source: "manual" or "auto_learned" to filter by source

    Returns:
        JSON with all heuristics (includes source field: "manual" or "auto_learned")
    """
    try:
        # Reload from file to get latest data (including auto-learned patterns)

        heuristic_manager.load()

        enabled_only = request.args.get("enabled_only", "false").lower() == "true"
        source = request.args.get("source", None)

        if source:
            # Filter by source
            heuristics = heuristic_manager.get_by_source(source, enabled_only=enabled_only)
        else:
            heuristics = heuristic_manager.get_all(enabled_only=enabled_only)

        return jsonify({"status": "success", "heuristics": heuristics})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/heuristics/statistics", methods=["GET"])
def get_heuristics_statistics() -> Tuple:
    """Get heuristics statistics.

    Returns:
        JSON with heuristics statistics (total, manual, auto-learned counts)
    """
    try:
        # Reload from file to get latest data

        heuristic_manager.load()

        stats = heuristic_manager.get_statistics()
        return jsonify({"status": "success", "statistics": stats})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/heuristics/<heuristic_id>", methods=["GET"])
def get_heuristic(heuristic_id: str) -> Tuple:
    """Get specific heuristic by ID.

    Args:
        heuristic_id: Heuristic ID

    Returns:
        JSON with heuristic data
    """
    try:
        heuristic = heuristic_manager.get_by_id(heuristic_id)
        if not heuristic:
            return jsonify({"status": "error", "message": "Heuristic not found"}), 404

        return jsonify({"status": "success", "heuristic": heuristic})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/heuristics", methods=["POST"])
def add_heuristic() -> Tuple:
    """Add new heuristic.

    Accepts:
        - name: Heuristic name
        - regex: Regular expression pattern
        - pii_type: PII type
        - rationale: Reason for this heuristic
        - priority: Priority (optional, default 50)
        - enabled: Whether enabled (optional, default true)
        - pattern_type: Pattern type (optional, e.g., "suffix", "prefix", "keyword", "custom")

    Returns:
        JSON with created heuristic
    """
    try:
        data = request.json

        # Validate required fields
        required_fields = ["name", "regex", "pii_type", "rationale"]
        for field in required_fields:
            if field not in data:
                return (
                    jsonify({"status": "error", "message": f"Missing required field: {field}"}),
                    400,
                )

        # Add heuristic
        heuristic = heuristic_manager.add(
            name=data["name"],
            regex=data["regex"],
            pii_type=data["pii_type"],
            rationale=data["rationale"],
            priority=data.get("priority", 50),
            enabled=data.get("enabled", True),
            pattern_type=data.get("pattern_type", ""),
        )

        # Log heuristic change
        if state.audit_logger:
            state.audit_logger.log_heuristic_change(
                action="ADD",
                heuristic_id=heuristic["id"],
                new_data=heuristic,
                user_id=request.remote_addr or "unknown",
            )

        return jsonify({"status": "success", "heuristic": heuristic})
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/heuristics/<heuristic_id>", methods=["PUT"])
def update_heuristic(heuristic_id: str) -> Tuple:
    """Update existing heuristic.

    Args:
        heuristic_id: Heuristic ID to update

    Returns:
        JSON with updated heuristic
    """
    try:
        data = request.json

        # Get old data for audit
        old_heuristic = heuristic_manager.get_by_id(heuristic_id)
        if not old_heuristic:
            return jsonify({"status": "error", "message": "Heuristic not found"}), 404

        # Update heuristic
        heuristic = heuristic_manager.update(heuristic_id, **data)
        if not heuristic:
            return jsonify({"status": "error", "message": "Heuristic not found"}), 404

        # Log heuristic change
        if state.audit_logger:
            state.audit_logger.log_heuristic_change(
                action="UPDATE",
                heuristic_id=heuristic_id,
                old_data=old_heuristic,
                new_data=heuristic,
                user_id=request.remote_addr or "unknown",
            )

        return jsonify({"status": "success", "heuristic": heuristic})
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/heuristics/<heuristic_id>", methods=["DELETE"])
def delete_heuristic(heuristic_id: str) -> Tuple:
    """Delete heuristic.

    Args:
        heuristic_id: Heuristic ID to delete

    Returns:
        JSON with status
    """
    try:
        # Get old data for audit
        old_heuristic = heuristic_manager.get_by_id(heuristic_id)

        # Delete heuristic
        success = heuristic_manager.delete(heuristic_id)
        if not success:
            return jsonify({"status": "error", "message": "Heuristic not found"}), 404

        # Log heuristic change
        if state.audit_logger and old_heuristic:
            state.audit_logger.log_heuristic_change(
                action="DELETE",
                heuristic_id=heuristic_id,
                old_data=old_heuristic,
                user_id=request.remote_addr or "unknown",
            )

        return jsonify({"status": "success", "message": "Heuristic deleted"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/heuristics/test", methods=["POST"])
def test_heuristic() -> Tuple:
    """Test regex pattern against sample strings.

    Accepts:
        - regex: Regular expression pattern
        - test_strings: List of strings to test

    Returns:
        JSON with test results
    """
    try:
        data = request.json

        if "regex" not in data or "test_strings" not in data:
            return jsonify({"status": "error", "message": "Missing required fields"}), 400

        result = heuristic_manager.test_pattern(
            regex=data["regex"], test_strings=data["test_strings"]
        )

        return jsonify({"status": "success", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# Stage 3 User Validation API Endpoints


@app.route("/change_pii_status", methods=["POST"])
@app.route("/api/stage3/change_pii_status", methods=["POST"])
def change_pii_status() -> Tuple:
    """Handle PII status change and query LLM for anonymization methods.

    Accepts:
        - session_id: Client session ID
        - table: Table name
        - column: Column name
        - old_status: Original PII status ("PII" or "Non-PII")
        - new_status: New PII status ("PII" or "Non-PII")
        - rationale: User's rationale for the change (required for Non-PII -> PII)

    Returns:
        JSON with anonymization methods from LLM (for Non-PII -> PII)
        or restricted methods (for PII -> Non-PII)
    """
    try:
        data = request.json
        client_session_id = data.get("session_id", "")
        table = data.get("table")
        column = data.get("column")
        old_status = data.get("old_status")
        new_status = data.get("new_status")
        rationale = data.get("rationale", "")
        pii_type = data.get("pii_type", "PII")  # Default PII type

        # Get session-specific state or fall back to global state

        session_state = (
            session_manager.get_session(client_session_id) if client_session_id else None
        )
        target_state = session_state if session_state else state

        logging.getLogger(__name__).debug(f"[DEBUG /change_pii_status] Received request:")
        logging.getLogger(__name__).debug(f"  - session_id: {client_session_id}")
        logging.getLogger(__name__).debug(f"  - table: {table}")
        logging.getLogger(__name__).debug(f"  - column: {column}")
        logging.getLogger(__name__).debug(f"  - old_status: {old_status}")
        logging.getLogger(__name__).debug(f"  - new_status: {new_status}")
        logging.getLogger(__name__).debug(f"  - pii_type: {pii_type}")
        logging.getLogger(__name__).debug(f"  - rationale: {rationale}")

        # If old_status is not provided, infer it from new_status
        if not old_status and new_status:
            old_status = "Non-PII" if new_status == "PII" else "PII"
            logging.getLogger(__name__).debug(f"[DEBUG] Inferred old_status: {old_status}")

        # Validate inputs
        if not all([table, column, new_status]):
            return jsonify({"status": "error", "message": "Missing required fields"}), 400

        # Any PII status change during Stage 3 is a human correction to the
        # immutable model/RAG classification captured before review.
        target_state.manual_corrections.append(
            {
                "kind": "pii_classification",
                "table": table,
                "column": column,
                "old_status": old_status,
                "new_status": new_status,
                "rationale": rationale,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Log the PII classification change
        if target_state.audit_logger:
            target_state.audit_logger.log_pii_classification_change(
                table=table,
                column=column,
                old_status=old_status,
                new_status=new_status,
                rationale=rationale,
                user_id=request.remote_addr or "user",
            )

        # Record PII classification change in user preference manager for future learning

        if target_state.expert_preference_manager:
            # Get column_comment from schema or pii_data for RAG-based similarity search

            col_comment = ""
            if table in target_state.schemas and column in target_state.schemas[table]:
                schema_col = target_state.schemas[table][column]
                if isinstance(schema_col, dict):
                    col_comment = schema_col.get("comment", "")
            if (
                not col_comment
                and table in target_state.pii_data
                and column in target_state.pii_data[table]
            ):
                col_comment = target_state.pii_data[table][column].get("column_comment", "")

            target_state.expert_preference_manager.record_pii_classification_change(
                column_name=column,
                old_classification=old_status,
                new_classification=new_status,
                rationale=rationale,
                column_comment=col_comment,
            )
            log_producer(
                f"[RAG] Recorded PII classification learning: {column} ({old_status} -> {new_status})",
                target_state,
            )

        methods = []

        logging.getLogger(__name__).debug(f"[DEBUG] Checking status change conditions:")
        logging.getLogger(__name__).debug(f"  - old_status == 'PII': {old_status == 'PII'}")
        logging.getLogger(__name__).debug(f"  - new_status == 'Non-PII': {new_status == 'Non-PII'}")
        logging.getLogger(__name__).debug(f"  - old_status == 'Non-PII': {old_status == 'Non-PII'}")
        logging.getLogger(__name__).debug(f"  - new_status == 'PII': {new_status == 'PII'}")

        if old_status == "PII" and new_status == "Non-PII":
            logging.getLogger(__name__).debug(f"[DEBUG] Executing branch: PII -> Non-PII")
            # PII -> Non-PII: Generate Non-PII methods using LLM (same as regular Non-PII columns)
            # This allows ROUND, GENERALIZATION for numeric/date columns

            # Get column metadata
            column_comment = ""
            data_type = ""

            if table in target_state.schemas and column in target_state.schemas[table]:
                schema_col = target_state.schemas[table][column]
                if isinstance(schema_col, dict):
                    column_comment = schema_col.get("comment", "")
                    data_type = schema_col.get("type", "")

            # If not in schemas, try pii_data
            if (
                not column_comment
                and table in target_state.pii_data
                and column in target_state.pii_data[table]
            ):
                column_comment = target_state.pii_data[table][column].get("column_comment", "")

            # Generate Non-PII methods using LLM (includes ROUND, GENERALIZATION for applicable types)
            methods = _generate_non_pii_methods(column, column_comment, data_type)

            print(
                f"[INFO] Generated {len(methods)} Non-PII methods for {column} (type: {data_type})"
            )
            log_producer(
                f"[Note] Generated Non-PII methods for {column}: {', '.join([m['method'] for m in methods])}",
                target_state,
            )

            # Update target_state.policy_data AND target_state.pii_data to reflect PII -> Non-PII change
            # This ensures that the old PII policy (e.g., MASK) is replaced with Non-PII methods
            logging.getLogger(__name__).debug(
                f"[DEBUG] Updating state for {table}.{column}: PII -> Non-PII"
            )
            log_producer(f"[Note] State updated for {column}: PII -> Non-PII", target_state)

            # Remove from policy_data (since it's no longer PII)
            if table in target_state.policy_data and column in target_state.policy_data[table]:
                del target_state.policy_data[table][column]
                logging.getLogger(__name__).debug(f"[DEBUG] Removed {column} from policy_data")

            # Update pii_data to mark as Non-PII
            if table not in target_state.pii_data:
                target_state.pii_data[table] = {}

            target_state.pii_data[table][column] = {
                "is_pii": False,
                "pii_type": "Non-PII",
                "evidence_source": f"User reclassified from PII. {rationale}"
                if rationale
                else "User reclassified from PII",
                "column_comment": column_comment,
            }
            logging.getLogger(__name__).debug(f"[DEBUG] Updated pii_data for {column} to Non-PII")
        elif old_status == "Non-PII" and new_status == "PII":
            logging.getLogger(__name__).debug(f"[DEBUG] Executing branch: Non-PII -> PII")
            # Non-PII -> PII: Query LLM for anonymization methods
            if not rationale:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "Rationale is required for Non-PII to PII conversion",
                        }
                    ),
                    400,
                )

            # Get column metadata for better method generation
            column_comment = ""
            data_type = ""

            if table in target_state.schemas and column in target_state.schemas[table]:
                schema_col = target_state.schemas[table][column]
                if isinstance(schema_col, dict):
                    column_comment = schema_col.get("comment", "")
                    data_type = schema_col.get("type", "")

            # If not in schemas, try pii_data
            if (
                not column_comment
                and table in target_state.pii_data
                and column in target_state.pii_data[table]
            ):
                column_comment = target_state.pii_data[table][column].get("column_comment", "")

            # Get LLM-generated anonymization methods with column description
            logging.getLogger(__name__).debug(
                f"[DEBUG] Calling _query_llm_for_anonymization_methods for {column}"
            )
            logging.getLogger(__name__).debug(f"  - pii_type: {pii_type}")
            logging.getLogger(__name__).debug(f"  - rationale: {rationale}")
            logging.getLogger(__name__).debug(f"  - column_comment: {column_comment}")
            logging.getLogger(__name__).debug(f"  - data_type: {data_type}")
            methods = _query_llm_for_anonymization_methods(
                column, pii_type, rationale, column_comment, data_type
            )
            logging.getLogger(__name__).debug(
                f"[DEBUG] _query_llm_for_anonymization_methods returned {len(methods)} methods"
            )

            # Update target_state.policy_data AND target_state.pii_data to reflect Non-PII -> PII change
            # This ensures that the new PII policy is stored for code generation
            logging.getLogger(__name__).debug(
                f"[DEBUG] Updating state for {table}.{column}: Non-PII -> PII"
            )
            log_producer(f"[Note] State updated for {column}: Non-PII -> PII", target_state)

            # Add to policy_data
            if table not in target_state.policy_data:
                target_state.policy_data[table] = {}

            target_state.policy_data[table][column] = {
                "is_pii": True,
                "pii_type": pii_type,
                "recommended_methods": methods,
                "evidence_source": f"User-defined PII. Rationale: {rationale}",
            }
            logging.getLogger(__name__).debug(f"[DEBUG] Added {column} to policy_data as PII")

            # Update pii_data to mark as PII
            if table not in target_state.pii_data:
                target_state.pii_data[table] = {}

            # Get column comment
            column_comment = ""
            if table in target_state.schemas and column in target_state.schemas[table]:
                schema_col = target_state.schemas[table][column]
                if isinstance(schema_col, dict):
                    column_comment = schema_col.get("comment", "")

            target_state.pii_data[table][column] = {
                "is_pii": True,
                "pii_type": pii_type,
                "evidence_source": f"User-defined PII. Rationale: {rationale}",
                "column_comment": column_comment,
            }
            logging.getLogger(__name__).debug(
                f"[DEBUG] Updated pii_data for {column} to PII ({pii_type})"
            )
        else:
            return jsonify({"status": "error", "message": "Invalid status change"}), 400

        # Get column comment if available
        column_comment = ""
        if table in target_state.schemas and column in target_state.schemas[table]:
            schema_col = target_state.schemas[table][column]
            if isinstance(schema_col, dict):
                column_comment = schema_col.get("comment", "")
        # Also try to get from pii_data if available
        if (
            not column_comment
            and table in target_state.pii_data
            and column in target_state.pii_data[table]
        ):
            column_comment = target_state.pii_data[table][column].get("column_comment", "")

        # Prepare response data
        response_pii_type = pii_type if new_status == "PII" else "Non-PII"

        logging.getLogger(__name__).debug(f"[DEBUG] Response data:")
        logging.getLogger(__name__).debug(f"  - column: {column}")
        logging.getLogger(__name__).debug(f"  - old_status: {old_status}")
        logging.getLogger(__name__).debug(f"  - new_status: {new_status}")
        logging.getLogger(__name__).debug(f"  - pii_type (from request): {pii_type}")
        logging.getLogger(__name__).debug(f"  - response pii_type: {response_pii_type}")
        logging.getLogger(__name__).debug(f"  - methods count: {len(methods)}")

        return jsonify(
            {
                "status": "success",
                "message": f"Successfully changed {column} from {old_status} to {new_status}",
                "pii_type": response_pii_type,
                "methods": methods,
                "column_comment": column_comment,
                "column": column,
                "old_status": old_status,
                "new_status": new_status,
            }
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


def _query_llm_for_anonymization_methods(
    column_name: str, pii_type: str, rationale: str, column_comment: str = "", data_type: str = ""
) -> List[Dict[str, Any]]:
    """Query LLM to generate anonymization methods for a column with RAG integration.

    This function integrates with RAG to retrieve legal context and uses Stage 2
    prompts for consistent policy generation across the system.

    Args:
        column_name: Column name
        pii_type: PII type (PII or Non-PII)
        rationale: User's rationale for why this is PII
        column_comment: Column description/comment (more reliable than column name)
        data_type: Column data type (e.g., INTEGER, VARCHAR)

    Returns:
        List of anonymization methods with code snippets and legal evidence
    """
    try:
        logging.getLogger(__name__).debug(
            f"[DEBUG _query_llm_for_anonymization_methods] Starting for column: {column_name}"
        )
        logging.getLogger(__name__).debug(f"  - pii_type: {pii_type}")
        logging.getLogger(__name__).debug(f"  - rationale: {rationale}")
        logging.getLogger(__name__).debug(f"  - column_comment: {column_comment}")
        logging.getLogger(__name__).debug(f"  - data_type: {data_type}")

        # Use shared backend components from session_manager
        shared_rag, shared_engine, _, _ = session_manager.get_shared_backend()

        logging.getLogger(__name__).debug(f"[DEBUG] shared_engine: {shared_engine}")
        logging.getLogger(__name__).debug(f"[DEBUG] shared_rag: {shared_rag}")

        if not shared_engine or not shared_engine.llm_client_stage_2:
            # Fallback to default methods if LLM is not available
            logging.getLogger(__name__).debug(
                f"[DEBUG] LLM client not available, using fallback methods"
            )
            return _get_default_anonymization_methods(column_name, column_comment, data_type)

        if not shared_rag:
            # Fallback if RAG is not initialized
            logging.getLogger(__name__).debug(
                f"[DEBUG] RAG not initialized, using fallback methods"
            )
            return _get_default_anonymization_methods(column_name, column_comment, data_type)

        llm_client = shared_engine.llm_client_stage_2

        # Step 1: Retrieve legal context using RAG
        # Include column description in query for better context
        description_text = f"Description: {column_comment}. " if column_comment else ""
        type_text = f"Data type: {data_type}. " if data_type else ""
        query = f"PII classification and anonymization methods for {pii_type} column '{column_name}'. {description_text}{type_text}User context: {rationale}"
        context, source_docs = shared_rag.retrieve(query)

        # Step 2: Load Stage 2 prompts for consistency
        from pseudragon.stages.stage2_policy_synthesis import load_prompt

        system_prompt = load_prompt("stage2_policy_synthesis", "system")
        user_prompt_template = load_prompt("stage2_policy_synthesis", "user")

        # Step 3: Build user prompt with retrieved legal context
        # Prioritize column description over column name
        column_description_text = column_comment if column_comment else "No description available"
        data_type_text = data_type if data_type else "Unknown"
        user_prompt = user_prompt_template.format(
            context=context,
            column=column_name,
            pii_type=pii_type,
            column_desc=f"{column_description_text}. User rationale: {rationale}",
            data_type=data_type_text,
            purpose_goal="Pseudonymization for privacy protection",
            preferred_method="General Pseudonymization",
        )

        # Step 5: Query LLM with legal context
        # Create a temporary session for this query
        session = UniversalLLMSession(
            client=llm_client,
            system_prompt=system_prompt,
            session_id="temp_anonymization_query",
            max_history=0,
        )

        response = session.chat(
            user_prompt=user_prompt,
            model=Settings.LLM_STAGE_2,
            temperature=Settings.LLM_TEMPERATURE,
        )

        # Step 6: Parse response
        content = response.choices[0].message.content

        # Try to extract JSON from response
        import json
        import re
        from pseudragon.llm.json_parser import safe_parse_json

        result = safe_parse_json(
            content,
            default_response={
                "recommended_methods": [],
                "evidence_source": "Failed to parse LLM response",
            },
        )

        methods_data = result.get("recommended_methods", [])

        # Build legal evidence from source documents
        legal_evidence = (
            ", ".join(source_docs)
            if source_docs
            else result.get("evidence_source", "User-defined PII")
        )

        # Convert to format expected by web interface
        methods = []
        for tech in methods_data:
            method_name = tech.get("method", "HASH").upper()

            # Skip KEEP and DELETE for user-triggered PII classification
            if method_name in ["KEEP", "DELETE"]:
                continue

            method_dict = {
                "method": method_name,
                "applicability": "High",  # Default to High
                "description": tech.get("description", ""),
                "code_snippet": tech.get("code_snippet", ""),
                "example_implementation": tech.get("code_snippet", ""),
                "legal_evidence": tech.get("legal_source", legal_evidence),
            }
            methods.append(method_dict)

        # Ensure we have at least 2 methods - add fallback if needed
        MIN_METHODS = 2
        if len(methods) < MIN_METHODS:
            fallback_methods = _get_default_anonymization_methods(
                column_name, column_comment, data_type
            )
            for fallback in fallback_methods:
                if len(methods) >= MIN_METHODS:
                    break
                # Check if method type already exists
                existing_methods = {m["method"].upper() for m in methods}
                if fallback["method"].upper() not in existing_methods:
                    # Create a copy to avoid modifying the original
                    fallback_copy = fallback.copy()
                    fallback_copy["legal_evidence"] = legal_evidence
                    methods.append(fallback_copy)

        return methods[:5]  # Return at most 5 methods

    except Exception as e:
        print(f"Error querying LLM for anonymization methods: {e}")
        import traceback

        traceback.print_exc()
        # Fallback to default methods
        return _get_default_anonymization_methods(column_name, column_comment, data_type)


def _infer_pii_type_from_context(column_name: str, rationale: str) -> str:
    """Infer PII type from column name and user rationale.

    Args:
        column_name: Column name
        rationale: User's rationale for PII classification

    Returns:
        PII type (PII)
    """
    # All PII columns are classified as PII

    return "PII"


def _analyze_column_semantics(
    column_name: str, column_comment: str = "", data_type: str = ""
) -> Dict[str, Any]:
    """Analyze column semantics from description and name to determine data type.

    Args:
        column_name: Column name
        column_comment: Column description/comment (more reliable than column name)
        data_type: Column data type

    Returns:
        Dictionary containing semantic analysis results
    """
    col_name_upper = column_name.upper()
    col_comment_upper = (column_comment or "").upper()

    # Check if numeric type
    is_numeric = any(
        t in (data_type or "").upper()
        for t in ["INT", "DECIMAL", "NUMERIC", "FLOAT", "DOUBLE", "REAL", "NUMBER"]
    )

    # PRIORITY 1: Check column description/comment for semantic meaning (more reliable)
    # PRIORITY 2: Check column name as fallback

    # === PII-related keywords ===
    # Phone number
    is_phone_by_comment = any(
        kw in col_comment_upper
        for kw in ["전화", "핸드폰", "휴대폰", "PHONE", "MOBILE", "TEL", "CONTACT"]
    )
    is_phone_by_name = any(kw in col_name_upper for kw in ["PHONE", "TEL", "MOBILE", "CONTACT"])
    is_phone = is_phone_by_comment or is_phone_by_name

    # Email
    is_email_by_comment = any(
        kw in col_comment_upper for kw in ["이메일", "EMAIL", "E-MAIL", "MAIL"]
    )
    is_email_by_name = any(kw in col_name_upper for kw in ["EMAIL", "MAIL"])
    is_email = is_email_by_comment or is_email_by_name

    # Name
    is_name_by_comment = any(
        kw in col_comment_upper for kw in ["이름", "성명", "NAME", "고객명", "CUSTOMER NAME"]
    )
    is_name_by_name = any(kw in col_name_upper for kw in ["NAME", "NM"])
    is_name = is_name_by_comment or is_name_by_name

    # ID/identifier
    is_id_by_comment = any(
        kw in col_comment_upper
        for kw in ["아이디", "식별자", "ID", "IDENTIFIER", "고유번호", "UNIQUE"]
    )
    is_id_by_name = any(kw in col_name_upper for kw in ["ID", "_ID", "IDENTIFIER", "UNIQUE_ID"])
    is_id = is_id_by_comment or is_id_by_name

    # === Non-PII related keywords ===
    # Date/time
    is_date_by_comment = any(
        kw in col_comment_upper
        for kw in [
            "날짜",
            "생년월일",
            "DATE",
            "BIRTH",
            "TRANSACTION DATE",
            "CREATED",
            "UPDATED",
            "YYYY",
            "MM",
            "DD",
            "년",
            "월",
            "일",
            "DATETIME",
            "TIME",
            "TIMESTAMP",
            "INQUIRY DATETIME",
            "HHMMSS",
            "STANDARD DATE",
            "STD DATE",
            "일자",
            "기준일",
            "YYYYMMDD",
        ]
    )
    is_date_by_name = any(
        kw in col_name_upper
        for kw in [
            "DT",
            "DATE",
            "BIRTH",
            "TRAN_DT",
            "CREATE_DT",
            "UPDATE_DT",
            "DTIME",
            "DATETIME",
            "TIME",
            "TIMESTAMP",
            "STD_DT",
            "STDT",
        ]
    )
    is_date_column = is_date_by_comment or is_date_by_name

    # Birth date (specific type of date)
    is_birth_date = (
        any(kw in col_comment_upper for kw in ["생년월일", "BIRTH"]) or "BIRTH" in col_name_upper
    )

    # Amount/balance
    is_amount_by_comment = any(
        kw in col_comment_upper
        for kw in [
            "금액",
            "잔액",
            "BALANCE",
            "AMOUNT",
            "PRICE",
            "COST",
            "FEE",
            "SALARY",
            "RATE",
            "CHARGE",
            "PAYMENT",
            "원",
            "달러",
            "DOLLAR",
            "WON",
        ]
    )
    is_amount_by_name = any(
        kw in col_name_upper
        for kw in ["AMT", "AMOUNT", "BLNC", "BALANCE", "PRICE", "COST", "FEE", "RATE"]
    )

    # Count/quantity
    is_count_by_comment = any(
        kw in col_comment_upper
        for kw in ["개수", "수량", "COUNT", "QUANTITY", "QTY", "NUMBER OF", "건수"]
    )
    is_count_by_name = any(kw in col_name_upper for kw in ["QTY", "QUANTITY", "CNT", "COUNT"])

    # Combine amount and count
    is_amount_column = (
        is_amount_by_comment or is_count_by_comment or is_amount_by_name or is_count_by_name
    )

    # Age
    is_age_by_comment = any(
        kw in col_comment_upper for kw in ["나이", "연령", "AGE", "세", "YEARS OLD"]
    )
    is_age_column = is_age_by_comment

    # If it's an amount column but data_type is not detected as numeric, treat it as numeric
    if is_amount_column and not is_numeric:
        is_numeric = True

    return {
        "is_numeric": is_numeric,
        # PII-related
        "is_phone": is_phone,
        "is_phone_by_comment": is_phone_by_comment,
        "is_phone_by_name": is_phone_by_name,
        "is_email": is_email,
        "is_email_by_comment": is_email_by_comment,
        "is_email_by_name": is_email_by_name,
        "is_name": is_name,
        "is_name_by_comment": is_name_by_comment,
        "is_name_by_name": is_name_by_name,
        "is_id": is_id,
        "is_id_by_comment": is_id_by_comment,
        "is_id_by_name": is_id_by_name,
        # Non-PII related
        "is_date_column": is_date_column,
        "is_date_by_comment": is_date_by_comment,
        "is_date_by_name": is_date_by_name,
        "is_birth_date": is_birth_date,
        "is_amount_column": is_amount_column,
        "is_amount_by_comment": is_amount_by_comment,
        "is_amount_by_name": is_amount_by_name,
        "is_age_column": is_age_column,
    }


def _get_default_anonymization_methods(
    column_name: str, column_comment: str = "", data_type: str = ""
) -> List[Dict[str, Any]]:
    """Get default anonymization methods when LLM is not available.

    Args:
        column_name: Column name
        column_comment: Column description/comment (more reliable than column name)
        data_type: Column data type

    Returns:
        List of default anonymization methods
    """
    methods = []

    logging.getLogger(__name__).debug(
        f"[DEBUG _get_default_anonymization_methods] column={column_name}, comment={column_comment}, data_type={data_type}"
    )

    # Use unified semantic analysis
    semantics = _analyze_column_semantics(column_name, column_comment, data_type)

    logging.getLogger(__name__).debug(
        f"[DEBUG] is_phone={semantics['is_phone']} (comment:{semantics['is_phone_by_comment']}, name:{semantics['is_phone_by_name']})"
    )
    logging.getLogger(__name__).debug(
        f"[DEBUG] is_email={semantics['is_email']} (comment:{semantics['is_email_by_comment']}, name:{semantics['is_email_by_name']})"
    )
    logging.getLogger(__name__).debug(
        f"[DEBUG] is_name={semantics['is_name']} (comment:{semantics['is_name_by_comment']}, name:{semantics['is_name_by_name']})"
    )
    logging.getLogger(__name__).debug(
        f"[DEBUG] is_id={semantics['is_id']} (comment:{semantics['is_id_by_comment']}, name:{semantics['is_id_by_name']})"
    )

    # Provide context-specific methods based on column semantics
    if semantics["is_phone"]:
        # Phone numbers: MASK (hide the last 4 characters), HASH, ENCRYPT
        methods.append(
            {
                "method": "MASK",
                "applicability": "High",
                "description": f"Mask the last 4 characters of {column_name}; preserve the prefix (e.g., 010-1234-5678 -> 010-1234-****)",
                "parameters": {"mask_char": "*", "mask_last": 4, "short_value_mask_length": 4},
                "code_snippet": f"masked_{column_name} = str(record['{column_name}'])[:-4] + '****' if len(str(record['{column_name}'])) > 4 else '****'",
                "example_implementation": f"masked_{column_name} = str(record['{column_name}'])[:-4] + '****' if len(str(record['{column_name}'])) > 4 else '****'",
                "legal_evidence": "Phone number masking - recommended for PII protection",
            }
        )
    elif semantics["is_email"]:
        # Email: MASK (show domain only), HASH
        methods.append(
            {
                "method": "MASK",
                "applicability": "High",
                "description": f"Mask {column_name} - show domain only (e.g., user@example.com -> ****@example.com)",
                "code_snippet": f"masked_{column_name} = '****@' + str(record['{column_name}']).split('@')[1] if '@' in str(record['{column_name}']) else '****'",
                "example_implementation": f"masked_{column_name} = '****@' + str(record['{column_name}']).split('@')[1] if '@' in str(record['{column_name}']) else '****'",
                "legal_evidence": "Email masking - recommended for PII protection",
            }
        )
    elif semantics["is_name"]:
        # Names: GENERALIZATION (first character only), MASK, TOKENIZATION
        methods.append(
            {
                "method": "GENERALIZATION",
                "applicability": "High",
                "description": f"Generalize {column_name} - show only first character (e.g., 홍길동 -> 홍**, John Doe -> J***)",
                "code_snippet": f"generalized_{column_name} = str(record['{column_name}'])[0] + '*' * (len(str(record['{column_name}'])) - 1) if len(str(record['{column_name}'])) > 0 else ''",
                "example_implementation": f"generalized_{column_name} = str(record['{column_name}'])[0] + '*' * (len(str(record['{column_name}'])) - 1) if len(str(record['{column_name}'])) > 0 else "
                f"''",
                "legal_evidence": "Name generalization - recommended for PII protection",
            }
        )
    else:
        # Default: MASK (partial)
        methods.append(
            {
                "method": "MASK",
                "applicability": "High",
                "description": "Partial masking - hide sensitive parts while keeping some information visible",
                "code_snippet": f"masked_{column_name} = str(record['{column_name}'])[:2] + '****' if len(str(record['{column_name}'])) > 2 else '****'",
                "example_implementation": f"masked_{column_name} = str(record['{column_name}'])[:2] + '****' if len(str(record['{column_name}'])) > 2 else '****'",
                "legal_evidence": "User-defined PII - Masking recommended",
            }
        )

    # Always add HASH (universal method)
    methods.append(
        {
            "method": "HASH",
            "applicability": "High",
            "description": "Irreversible SHA-256 hashing - ensures data cannot be reversed to original form",
            "code_snippet": f"import hashlib\nhashed_{column_name} = hashlib.sha256(str(record['{column_name}']).encode()).hexdigest()",
            "example_implementation": f"import hashlib\nhashed_{column_name} = hashlib.sha256(str(record['{column_name}']).encode()).hexdigest()",
            "legal_evidence": "User-defined PII - Hashing recommended",
        }
    )

    # Add TOKENIZATION for reversible needs
    methods.append(
        {
            "method": "TOKENIZATION",
            "applicability": "Medium",
            "description": "Reversible token assignment - replace with random token that can be mapped back",
            "code_snippet": f"import uuid\ntoken_map = {{}}\nif record['{column_name}'] not in token_map:\n    token_map[record['{column_name}']] = str(uuid.uuid4())\ntokenized_{column_name} = token_map[record['{column_name}']]",
            "example_implementation": f"import uuid\ntoken_map = {{}}\nif record['{column_name}'] not in token_map:\n    token_map[record['{column_name}']] = str(uuid.uuid4())\ntokenized_"
            f"{column_name} = token_map[record['{column_name}']]",
            "legal_evidence": "User-defined PII - Tokenization recommended",
        }
    )

    return methods


def _generate_non_pii_methods(
    column_name: str, column_comment: str, data_type: str
) -> List[Dict[str, Any]]:
    """Generate pseudonymization methods for Non-PII columns using LLM.

    For numeric columns (e.g., BLNC/Balance), this may recommend:

    - ROUND: Round to nearest value (e.g., round to thousands)
    - GENERALIZATION: Generalize to ranges (e.g., 0-1000, 1001-5000)
    - KEEP: Keep unchanged
    - DELETE: Remove if not needed

    Args:
        column_name: Column name
        column_comment: Column description/comment
        data_type: Column data type (e.g., INTEGER, DECIMAL, VARCHAR)

    Returns:
        List of pseudonymization methods sorted by recommendation priority
    """
    try:
        # Use shared backend components from session_manager
        _, shared_engine, _, _ = session_manager.get_shared_backend()

        if not shared_engine or not shared_engine.llm_client_stage_2:
            # Fallback to default if LLM not available
            return _get_default_non_pii_methods(column_name, data_type, column_comment)

        llm_client = shared_engine.llm_client_stage_2

        # Load prompts from files
        from pseudragon.stages.stage2_policy_synthesis import load_prompt

        try:
            system_prompt = load_prompt("non_pii_methods", "system")
            user_prompt_template = load_prompt("non_pii_methods", "user")
        except FileNotFoundError:
            # Fallback to default if prompt files not found
            print(f"[WARNING] Non-PII prompt files not found, using default methods")
            return _get_default_non_pii_methods(column_name, data_type, column_comment)

        # Build user prompt with column information
        user_prompt = user_prompt_template.format(
            column=column_name,
            column_desc=column_comment if column_comment else "No description available",
            data_type=data_type if data_type else "Unknown",
        )

        logging.getLogger(__name__).debug(
            f"[DEBUG _generate_non_pii_methods] Querying LLM for {column_name}"
        )
        logging.getLogger(__name__).debug(f"  - Column: {column_name}")
        logging.getLogger(__name__).debug(f"  - Description: {column_comment}")
        logging.getLogger(__name__).debug(f"  - Data Type: {data_type}")

        # Query LLM
        session = UniversalLLMSession(
            client=llm_client,
            system_prompt=system_prompt,
            session_id="temp_non_pii_methods",
            max_history=0,
        )

        response = session.chat(
            user_prompt=user_prompt,
            model=Settings.LLM_STAGE_2,
            temperature=Settings.LLM_TEMPERATURE,
        )

        content = response.choices[0].message.content

        logging.getLogger(__name__).debug(
            f"[DEBUG _generate_non_pii_methods] LLM response length: {len(content)} characters"
        )
        logging.getLogger(__name__).debug(f"[DEBUG _generate_non_pii_methods] LLM FULL response:")
        print("=" * 80)
        logging.getLogger(__name__).debug("Model response received (%d characters)", len(content))
        print("=" * 80)

        # Parse JSON response
        from pseudragon.llm.json_parser import safe_parse_json

        result = safe_parse_json(content, default_response={"recommended_methods": []})

        methods_data = result.get("recommended_methods", [])

        logging.getLogger(__name__).debug(
            f"[DEBUG _generate_non_pii_methods] Parsed {len(methods_data)} methods from LLM"
        )

        # Convert to expected format
        methods = []
        for tech in methods_data:
            method_name = tech.get("method", "KEEP").upper()

            method_dict = {
                "method": method_name,
                "applicability": tech.get("applicability", "Medium"),
                "description": tech.get("description", ""),
                "code_snippet": tech.get("code_snippet", ""),
                "example_implementation": tech.get("code_snippet", ""),
                "legal_evidence": tech.get("rationale", "LLM-generated recommendation for Non-PII"),
            }
            methods.append(method_dict)
            logging.getLogger(__name__).debug(f"  - Added method: {method_name}")

        # Ensure KEEP and DELETE are always available (add if not present)
        existing_methods = {m["method"].upper() for m in methods}

        if "KEEP" not in existing_methods:
            methods.append(
                {
                    "method": "Keep",
                    "applicability": "High",
                    "description": "Keep the column unchanged for analysis",
                    "code_snippet": "# No transformation needed",
                    "example_implementation": "# No transformation needed",
                    "legal_evidence": "Standard option for Non-PII data",
                }
            )

        if "DELETE" not in existing_methods:
            methods.append(
                {
                    "method": "Delete",
                    "applicability": "Low",
                    "description": "Remove the column if not needed for analysis",
                    "code_snippet": f"record.pop('{column_name}', None)",
                    "example_implementation": f"record.pop('{column_name}', None)",
                    "legal_evidence": "Standard option for Non-PII data",
                }
            )

        return methods

    except Exception as e:
        print(f"Error generating Non-PII methods for {column_name}: {e}")
        import traceback

        traceback.print_exc()
        # Fallback to default
        return _get_default_non_pii_methods(column_name, data_type, column_comment)


def _get_default_non_pii_methods(
    column_name: str, data_type: str, column_comment: str = ""
) -> List[Dict[str, Any]]:
    """Get default Non-PII methods when LLM is not available.

    Args:
        column_name: Column name
        data_type: Column data type
        column_comment: Column description/comment (more reliable than column name)

    Returns:
        List of default Non-PII methods
    """
    methods = []

    logging.getLogger(__name__).debug(
        f"[DEBUG _get_default_non_pii_methods] column={column_name}, data_type={data_type}, comment={column_comment}"
    )

    # Use unified semantic analysis
    semantics = _analyze_column_semantics(column_name, column_comment, data_type)

    logging.getLogger(__name__).debug(
        f"[DEBUG] is_numeric={semantics['is_numeric']}, is_date_column={semantics['is_date_column']} (comment:{semantics['is_date_by_comment']}, name:{semantics['is_date_by_name']})"
    )
    logging.getLogger(__name__).debug(
        f"[DEBUG] is_amount_column={semantics['is_amount_column']} (comment:{semantics['is_amount_by_comment']}, name:{semantics['is_amount_by_name']})"
    )
    logging.getLogger(__name__).debug(f"[DEBUG] is_age_column={semantics['is_age_column']}")

    # Get uppercase versions for checking
    col_name_upper = column_name.upper()
    col_comment_upper = (column_comment or "").upper()

    if semantics["is_numeric"]:
        if semantics["is_date_column"]:
            # Date column (YYYYMMDD format)
            # Check if it's a birth date (by comment or name) - prioritize age range for birth dates
            is_birth_date = semantics["is_birth_date"]

            if is_birth_date or semantics["is_age_column"]:
                methods.append(
                    {
                        "method": "Generalization",
                        "applicability": "High",
                        "description": f"Convert {column_name} to age range (e.g., 19900115 -> '30s')",
                        "code_snippet": f'lambda x: f"{{((datetime.now().year - int(str(x)[:4]))//10)*10}}s" if pd.notna(x) and len(str(x)) >= 8 and str(x).isdigit() else str(x)',
                        "example_implementation": f'lambda x: f"{{((datetime.now().year - int(str(x)[:4]))//10)*10}}s" if pd.notna(x) and len(str(x)) >= 8 and str(x).isdigit() else str(x)',
                        "legal_evidence": "Age range generalization for privacy protection",
                    }
                )

            # General date methods
            # Check if it's a datetime (YYYYMMDDhhmmss - 14 digits) or just date (YYYYMMDD - 8 digits)
            is_datetime = (
                "DATETIME" in col_comment_upper
                or "HHMMSS" in col_comment_upper
                or "DTIME" in col_name_upper
            )

            if is_datetime:
                # DateTime-specific methods (YYYYMMDDhhmmss -> YYYYMMDD or YYYYMM or YYYY)
                methods.append(
                    {
                        "method": "Generalization",
                        "applicability": "High",
                        "description": f"Extract date only from {column_name} (e.g., 20240115143025 -> 20240115)",
                        "code_snippet": f"lambda x: str(x)[:8] if pd.notna(x) and len(str(x)) >= 14 and str(x).isdigit() else str(x)",
                        "example_implementation": f"lambda x: str(x)[:8] if pd.notna(x) and len(str(x)) >= 14 and str(x).isdigit() else str(x)",
                        "legal_evidence": "Temporal generalization for privacy protection (remove time)",
                    }
                )

                methods.append(
                    {
                        "method": "Generalization",
                        "applicability": "High",
                        "description": f"Extract year-month from {column_name} (e.g., 20240115143025 -> 202401)",
                        "code_snippet": f"lambda x: str(x)[:6] if pd.notna(x) and len(str(x)) >= 14 and str(x).isdigit() else str(x)",
                        "example_implementation": f"lambda x: str(x)[:6] if pd.notna(x) and len(str(x)) >= 14 and str(x).isdigit() else str(x)",
                        "legal_evidence": "Temporal generalization for privacy protection (year-month only)",
                    }
                )

                methods.append(
                    {
                        "method": "Generalization",
                        "applicability": "Medium",
                        "description": f"Extract year only from {column_name} (e.g., 20240115143025 -> 2024)",
                        "code_snippet": f"lambda x: str(x)[:4] if pd.notna(x) and len(str(x)) >= 14 and str(x).isdigit() else str(x)",
                        "example_implementation": f"lambda x: str(x)[:4] if pd.notna(x) and len(str(x)) >= 14 and str(x).isdigit() else str(x)",
                        "legal_evidence": "Temporal generalization for privacy protection (year only)",
                    }
                )
            else:
                # Date-only methods (YYYYMMDD)
                methods.append(
                    {
                        "method": "Generalization",
                        "applicability": "High" if not is_birth_date else "Medium",
                        "description": f"Extract year only from {column_name} (e.g., 20240115 -> 2024)",
                        "code_snippet": f"lambda x: str(x)[:4] if pd.notna(x) and len(str(x)) >= 8 and str(x).isdigit() else str(x)",
                        "example_implementation": f"lambda x: str(x)[:4] if pd.notna(x) and len(str(x)) >= 8 and str(x).isdigit() else str(x)",
                        "legal_evidence": "Temporal generalization for privacy protection",
                    }
                )

                methods.append(
                    {
                        "method": "Generalization",
                        "applicability": "Medium",
                        "description": f"Extract year-month from {column_name} (e.g., 20240115 -> 202401)",
                        "code_snippet": f"lambda x: str(x)[:6] if pd.notna(x) and len(str(x)) >= 8 and str(x).isdigit() else str(x)",
                        "example_implementation": f"lambda x: str(x)[:6] if pd.notna(x) and len(str(x)) >= 8 and str(x).isdigit() else str(x)",
                        "legal_evidence": "Temporal generalization for privacy protection",
                    }
                )
        else:
            # Regular numeric column (amount, balance, count, or age value)
            # Special handling for age columns (already age values, not birth dates)
            if semantics["is_age_column"]:
                methods.append(
                    {
                        "method": "Generalization",
                        "applicability": "High",
                        "description": f"Convert {column_name} to age range (e.g., 34 -> '30s')",
                        "code_snippet": f"lambda x: f\"{{(int(x)//10)*10}}s\" if pd.notna(x) and str(x).replace('.','').isdigit() else str(x)",
                        "example_implementation": f"lambda x: f\"{{(int(x)//10)*10}}s\" if pd.notna(x) and str(x).replace('.','').isdigit() else str(x)",
                        "legal_evidence": "Age range generalization for privacy protection",
                    }
                )
                methods.append(
                    {
                        "method": "Generalization",
                        "applicability": "Medium",
                        "description": f"Generalize {column_name} into broader age ranges (e.g., 34 -> '30-39')",
                        "code_snippet": f"lambda x: f\"{{(int(x)//10)*10}}-{{(int(x)//10)*10+9}}\" if pd.notna(x) and str(x).replace('.','').isdigit() else str(x)",
                        "example_implementation": f"lambda x: f\"{{(int(x)//10)*10}}-{{(int(x)//10)*10+9}}\" if pd.notna(x) and str(x).replace('.','').isdigit() else str(x)",
                        "legal_evidence": "Age range generalization for privacy protection",
                    }
                )
            else:
                # Amount, balance, count columns
                methods.append(
                    {
                        "method": "Round",
                        "applicability": "High",
                        "description": f"Round {column_name} to reduce precision (e.g., round to nearest 1000)",
                        "code_snippet": f"lambda x: round(float(x), -3) if pd.notna(x) and str(x).replace('.','').replace('-','').isdigit() else x",
                        "example_implementation": f"lambda x: round(float(x), -3) if pd.notna(x) and str(x).replace('.','').replace('-','').isdigit() else x",
                        "legal_evidence": "Numeric rounding for privacy protection",
                    }
                )

                methods.append(
                    {
                        "method": "Generalization",
                        "applicability": "Medium",
                        "description": f"Generalize {column_name} into ranges (e.g., 0-1000, 1001-5000)",
                        "code_snippet": f"lambda x: '0-1000' if pd.notna(x) and float(x) < 1000 else ('1001-5000' if float(x) < 5000 else ('5001-10000' if float(x) < 10000 else '10000+')) if "
                        f"pd.notna(x) and str(x).replace('.','').replace('-','').isdigit() else str(x)",
                        "example_implementation": f"lambda x: '0-1000' if pd.notna(x) and float(x) < 1000 else ('1001-5000' if float(x) < 5000 else ('5001-10000' if float(x) < 10000 else '10000+')) "
                        f"if pd.notna(x) and str(x).replace('.','').replace('-','').isdigit() else str(x)",
                        "legal_evidence": "Range generalization for privacy protection",
                    }
                )

    # Always add KEEP
    methods.append(
        {
            "method": "Keep",
            "applicability": "High",
            "description": "Keep the column unchanged for analysis",
            "code_snippet": "# No transformation needed",
            "example_implementation": "# No transformation needed",
            "legal_evidence": "Standard option for Non-PII data",
        }
    )

    # Always add DELETE
    methods.append(
        {
            "method": "Delete",
            "applicability": "Low",
            "description": "Remove the column if not needed for analysis",
            "code_snippet": f"record.pop('{column_name}', None)",
            "example_implementation": f"record.pop('{column_name}', None)",
            "legal_evidence": "Standard option for Non-PII data",
        }
    )

    return methods


# Expert Feedback API Endpoints


@app.before_request
def guard_knowledge_updates():
    guarded = request.path in {
        "/start",
        "/submit_stage3",
        "/change_pii_status",
        "/api/stage3/change_pii_status",
        "/api/expert_feedback/manage",
    }
    guarded = (
        guarded
        or request.path.startswith("/api/heuristics")
        or request.path == "/api/expert_feedback"
    )
    if guarded:
        KNOWLEDGE_LOCK.acquire()
        g.knowledge_locked = True
        if request.path in {
            "/submit_stage3",
            "/change_pii_status",
            "/api/stage3/change_pii_status",
        }:
            payload = request.get_json(silent=True) or {}
            target = session_manager.get_session(payload.get("session_id", "")) or state
            if getattr(target, "knowledge_invalidated", False):
                return jsonify(
                    status="error",
                    message="Knowledge changed. Start a new analysis before approval.",
                ), 409


@app.teardown_request
def release_knowledge_update_lock(error=None):
    if getattr(g, "knowledge_locked", False):
        g.knowledge_locked = False
        KNOWLEDGE_LOCK.release()


@app.route("/api/expert_feedback/manage", methods=["POST"])
def manage_expert_feedback():
    payload = request.get_json(silent=True) or {}
    with session_manager._lock:
        sessions = list(session_manager._sessions.values())
    if any(item.is_running for item in sessions + [state]):
        return jsonify(
            status="error", message="Wait for active analyses to finish before changing knowledge."
        ), 409
    manager = session_manager.get_shared_backend()[2] or state.expert_preference_manager
    if manager is None:
        manager = ExpertPreferenceManager()
    try:
        result = manager.manage_knowledge(
            payload.get("action"),
            payload.get("revision"),
            payload.get("reason"),
            payload.get("section"),
            payload.get("key"),
        )
    except ValueError as exc:
        return jsonify(status="error", message=str(exc)), 409
    except Exception:
        logging.exception("Knowledge update failed")
        return jsonify(
            status="error", message="Knowledge update failed; inspect the server log."
        ), 500
    for item in sessions + [state]:
        item.knowledge_invalidated = True
    return jsonify(status="success", **result)


@app.route("/api/expert_feedback", methods=["GET"])
def get_expert_feedback() -> Tuple:
    """Get expert feedback data.

    Returns all accumulated expert feedback from Stage 3 HITL sessions.

    Returns:
        JSON with expert feedback data
    """
    try:
        # Load expert feedback from file
        expert_pref_manager = ExpertPreferenceManager()

        # Get all feedback data
        feedback_data = {
            "status": "success",
            "revision": revision(expert_pref_manager.preference_file),
            "history": expert_pref_manager.preferences.get("knowledge_history", []),
            "method_preferences": [],
            "pii_classifications": [],
            "statistics": expert_pref_manager.get_statistics(),
        }

        # Get column exact matches (method preferences)
        column_exact = expert_pref_manager.preferences.get("column_exact", {})
        for key, value in column_exact.items():
            # Find the most preferred method
            method_counts = value.get("method_counts", {})
            preferred_method = None
            if method_counts:
                preferred_method = max(method_counts.items(), key=lambda x: x[1])[0]

            feedback_data["method_preferences"].append(
                {
                    "key": key,
                    "section": "column_exact",
                    "column_name": value.get("column_name", ""),
                    "pii_type": value.get("pii_type", ""),
                    "method_counts": method_counts,
                    "total_selections": value.get("total_selections", 0),
                    "preferred_method": preferred_method,
                }
            )

        # Get PII classification changes
        pii_classifications = expert_pref_manager.preferences.get("pii_classification", {})
        for key, value in pii_classifications.items():
            feedback_data["pii_classifications"].append(
                {
                    "key": key,
                    "section": "pii_classification",
                    "column_name": value.get("column_name", ""),
                    "classification": value.get("classification", ""),
                    "change_count": value.get("change_count", 0),
                    "rationale": value.get("rationale", ""),
                    "history": value.get("history", []),
                }
            )

        # Sort by total_selections descending
        feedback_data["method_preferences"].sort(
            key=lambda x: x.get("total_selections", 0), reverse=True
        )
        feedback_data["pii_classifications"].sort(
            key=lambda x: x.get("change_count", 0), reverse=True
        )

        return jsonify(feedback_data)
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/expert_feedback/summary", methods=["GET"])
def get_expert_feedback_summary() -> Tuple:
    """Get expert feedback summary statistics.

    Returns:
        JSON with summary statistics
    """
    try:
        expert_pref_manager = ExpertPreferenceManager()
        stats = expert_pref_manager.get_statistics()

        return jsonify({"status": "success", "summary": stats})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# Audit Log API Endpoints


@app.route("/api/audit_log", methods=["GET"])
def get_audit_log() -> Tuple:
    """Get audit log entries with optional filtering.

    Query parameters:
        - session_id: Client session ID (optional, if not provided returns all logs)

        - event_type: Filter by event type
        - start_date: Filter by start date (ISO format)
        - end_date: Filter by end date (ISO format)
        - user_id: Filter by user ID

    Returns:
        JSON with audit log entries
    """
    try:
        # Get filter parameters
        event_type = request.args.get("event_type")
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        user_id = request.args.get("user_id")

        # Get session ID from query parameter

        client_session_id = request.args.get("session_id", "")

        # If session_id is provided, try to get events from that specific session

        if client_session_id:
            session_state = session_manager.get_session(client_session_id)
            if session_state and session_state.audit_logger:
                events = session_state.audit_logger.get_all_events(
                    event_type=event_type, start_date=start_date, end_date=end_date, user_id=user_id
                )
                return jsonify({"status": "success", "events": events, "count": len(events)})

        # Use global audit logger to read ALL log files from disk

        # This works even after page refresh or server restart

        events = global_audit_logger.get_all_events(
            event_type=event_type, start_date=start_date, end_date=end_date, user_id=user_id
        )

        return jsonify({"status": "success", "events": events, "count": len(events)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/audit_log/summary", methods=["GET"])
def get_audit_log_summary() -> Tuple:
    """Get audit log summary.

    Query parameters:
        - session_id: Client session ID (optional, if not provided returns combined summary)

    Returns:
        JSON with summary statistics
    """
    try:
        # Get session ID from query parameter

        client_session_id = request.args.get("session_id", "")

        # If session_id is provided, get summary from that specific session

        if client_session_id:
            session_state = session_manager.get_session(client_session_id)
            if session_state and session_state.audit_logger:
                summary = session_state.audit_logger.get_session_summary()
                return jsonify({"status": "success", "summary": summary})

        # Use global audit logger to get summary from ALL log files on disk

        summary = global_audit_logger.get_session_summary()
        return jsonify({"status": "success", "summary": summary})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# Session Monitoring API Endpoints


@app.route("/api/sessions", methods=["GET"])
def get_all_sessions() -> Tuple:
    """Get list of all active sessions for monitoring.

    Returns:
        JSON with list of session info
    """
    try:
        sessions = session_manager.get_all_sessions_info()
        return jsonify({"status": "success", "sessions": sessions, "count": len(sessions)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/sessions/<session_id>/logs", methods=["GET"])
def get_session_logs_api(session_id: str) -> Tuple:
    """Get logs from a specific session for monitoring (read-only).

    Args:
        session_id: Session ID to get logs from

    Query parameters:
        offset: Starting index (default 0)
        limit: Max logs to return (default 100)

    Returns:
        JSON with logs and session info
    """
    try:
        offset = int(request.args.get("offset", 0))
        limit = int(request.args.get("limit", 100))

        result = session_manager.get_session_logs(session_id, offset, limit)

        if "error" in result:
            return jsonify({"status": "error", "message": result["error"]}), 404

        return jsonify({"status": "success", **result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/sessions/<session_id>/status", methods=["GET"])
def get_session_status_api(session_id: str) -> Tuple:
    """Get current status of a specific session.

    Args:
        session_id: Session ID

    Returns:
        JSON with session status
    """
    try:
        session_state = session_manager.get_session(session_id)

        if not session_state:
            return jsonify({"status": "error", "message": "Session not found"}), 404

        return jsonify(
            {
                "status": "success",
                "session_id": session_id,
                "session_id_short": session_id[:8],
                "is_running": session_state.is_running,
                "current_step": session_state.current_step,
                "tables": list(session_state.schemas.keys()),
                "purpose_goal": session_state.purpose_goal,
                "has_pii_data": bool(session_state.pii_data),
                "has_policy_data": bool(session_state.policy_data),
                "has_generated_code": bool(session_state.generated_code),
                "log_count": len(session_state.logs),
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    """
    Application entry point
    애플리케이션 진입점

    Performs startup checks and launches the Flask development server.
    시작 검사를 수행하고 Flask 개발 서버를 시작합니다.
    """
    print("=" * 60)
    print(AppConfig.PAPER_TITLE)
    print("=" * 60)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Database Path: {AppConfig.DB_PATH}")
    print(f"Documents Path: {Settings.DOCS_DIR}")
    print(f"Reports Path: {AppConfig.REPORTS_DIR}")
    print(f"Templates Path: {os.path.join(os.path.dirname(__file__), 'templates')}")
    print("=" * 60)

    if not os.path.exists(Settings.DOCS_DIR):
        print(f"\nWARNING: Docs directory not found!")
        print(f"Creating: {Settings.DOCS_DIR}")
        os.makedirs(Settings.DOCS_DIR, exist_ok=True)
        print("Please add legal documents to this directory.\n")

    if not os.path.exists(AppConfig.DB_PATH):
        print(f"\nWARNING: Database not found at {AppConfig.DB_PATH}")
        print("Please ensure the database file exists.\n")

    print(f"\nStarting Flask server on http://{AppConfig.HOST}:{AppConfig.PORT}")
    print("Press Ctrl+C to stop the server\n")

    app.run(
        host=AppConfig.HOST,
        port=AppConfig.PORT,
        debug=AppConfig.DEBUG,
        use_reloader=False,
        threaded=True,
    )
