"""
PseuDRAGON Web Application
PseuDRAGON 웹 애플리케이션

Flask-based web interface for the PseuDRAGON pseudonymization pipeline.
PseuDRAGON 가명처리 파이프라인을 위한 Flask 기반 웹 인터페이스.

This application provides:
이 애플리케이션은 다음을 제공합니다:
- Interactive UI for pipeline execution / 파이프라인 실행을 위한 대화형 UI
- Real-time progress streaming / 실시간 진행 상황 스트리밍
- Human-in-the-loop policy refinement / 사람-루프 정책 개선
- Code generation and download / 코드 생성 및 다운로드
"""

# Standard library imports
# 표준 라이브러리 import
import json
import os
import sys
import time
import uuid
from datetime import date, datetime
from decimal import Decimal
from threading import Lock, Thread
from typing import Any, Dict, List, Optional, Tuple

# CRITICAL: Add parent directory to Python path BEFORE any project imports
# 중요: 프로젝트 import 전에 부모 디렉토리를 Python 경로에 추가
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Fix Windows console encoding for emoji support
# Windows 콘솔 인코딩 수정 (이모지 지원)
if sys.platform == "win32":
    try:
        # Reconfigure stdout and stderr to use UTF-8 encoding
        # stdout와 stderr를 UTF-8 인코딩으로 재설정
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception as e:
        print(f"Warning: Could not set UTF-8 encoding: {e}")

# Third-party imports
# 서드파티 라이브러리 import
from flask import Flask, Response, jsonify, render_template, request, send_file

# Project-specific imports
# 프로젝트 관련 import
from config.config import (
    DEFAULT_LLM_CLIENT,
    DatabaseConfig,
    DirectoryConfig,
    LLM_CLIENT_STAGE_1,
    LLM_CLIENT_STAGE_2,
    Settings,
)
from pseudragon.database.adapters import DatabaseManager
from pseudragon.rag.retriever import RAGSystem
from pseudragon.rag.expert_preference_manager import ExpertPreferenceManager
from pseudragon.domain.policy_dsl import ActionType, ColumnPolicy, Policy, PolicyAction
from pseudragon.heuristics.heuristic_manager import HeuristicManager
from pseudragon.logging.audit_logger import AuditLogger
from pseudragon.logging.file_logger import reinitialize_logger, PseuDRAGONLogger
from pseudragon.llm.session_manager import UniversalLLMSession
from pseudragon.pipeline import PseuDRAGONPipeline
from pseudragon.validation.compliance_checker import ComplianceValidator

# Use BASE_DIR from DirectoryConfig for consistency
# 일관성을 위해 DirectoryConfig의 BASE_DIR 사용
BASE_DIR = DirectoryConfig.BASE_DIR
sys.path.insert(0, BASE_DIR)


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


class AppConfig:
    """
    Application configuration constants
    애플리케이션 설정 상수

    Centralizes all application-level configuration values.
    모든 애플리케이션 수준 설정 값을 중앙화합니다.
    """

    DB_PATH = DatabaseConfig.SAMPLE_DB_PATH
    DOCS_DIR = DirectoryConfig.DOCS_DIR
    OUTPUT_DIR = DirectoryConfig.OUTPUT_DIR
    REPORTS_DIR = DirectoryConfig.OUTPUT_DIR
    LOGS_DIR = os.path.join(DirectoryConfig.BASE_DIR, "logs")
    HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    PORT = int(os.getenv("FLASK_PORT", 5001))


class PipelineStep:
    """
    Pipeline execution step constants
    파이프라인 실행 단계 상수

    Defines all possible states of the pipeline execution.
    파이프라인 실행의 모든 가능한 상태를 정의합니다.
    """

    IDLE = "IDLE"
    STAGE1_2_PROCESSING = "STAGE1_2_PROCESSING"
    STAGE3_READY = "STAGE3_READY"
    STAGE4_PROCESSING = "STAGE4_PROCESSING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class PipelineState:
    """
    Global state manager for pipeline execution
    파이프라인 실행을 위한 전역 상태 관리자

    Maintains all runtime state including:
    다음을 포함한 모든 런타임 상태를 유지합니다:
    - Pipeline execution status / 파이프라인 실행 상태
    - Intermediate results / 중간 결과
    - Logs and progress / 로그 및 진행 상황

    Thread-safe using locks for concurrent access.
    동시 접근을 위해 락을 사용하여 스레드 안전합니다.
    """

    def __init__(self) -> None:
        """Initialize pipeline state with default values / 기본값으로 파이프라인 상태 초기화"""
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

    def reset(self) -> None:
        """
        Reset pipeline state to initial values
        파이프라인 상태를 초기값으로 재설정

        Thread-safe reset of all state variables.
        모든 상태 변수의 스레드 안전 재설정.
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


class SessionManager:
    """
    Session manager for multi-client support
    다중 클라이언트 지원을 위한 세션 관리자

    Each client gets an independent session with its own PipelineState.
    각 클라이언트는 자신만의 PipelineState를 가진 독립 세션을 얻습니다.
    """

    # Session expiry time in seconds (30 minutes)
    # 세션 만료 시간 (30분)
    SESSION_EXPIRY = 1800

    def __init__(self) -> None:
        """Initialize session manager / 세션 관리자 초기화"""
        self._sessions: Dict[str, PipelineState] = {}
        self._session_timestamps: Dict[str, float] = {}
        self._lock: Lock = Lock()
        # Shared backend components (RAG, engine) - initialized once
        # 공유 백엔드 컴포넌트 (RAG, 엔진) - 한 번만 초기화
        self._rag: Optional[RAGSystem] = None
        self._engine: Optional[PseuDRAGONPipeline] = None
        self._expert_preference_manager: Optional[ExpertPreferenceManager] = None
        self._validator: Optional[ComplianceValidator] = None
        self._backend_initialized: bool = False
        self._backend_lock: Lock = Lock()

    def create_session(self) -> str:
        """
        Create a new session and return its ID
        새 세션을 생성하고 ID를 반환

        Returns:
            Unique session ID / 고유 세션 ID
        """
        session_id = str(uuid.uuid4())
        with self._lock:
            self._sessions[session_id] = PipelineState()
            self._session_timestamps[session_id] = time.time()
        return session_id

    def get_session(self, session_id: str) -> Optional[PipelineState]:
        """
        Get session state by ID
        ID로 세션 상태 가져오기

        Args:
            session_id: Session ID to look up / 조회할 세션 ID

        Returns:
            PipelineState if found, None otherwise / 찾으면 PipelineState, 아니면 None
        """
        with self._lock:
            if session_id in self._sessions:
                # Update timestamp on access
                self._session_timestamps[session_id] = time.time()
                return self._sessions[session_id]
        return None

    def remove_session(self, session_id: str) -> bool:
        """
        Remove a session
        세션 제거

        Args:
            session_id: Session ID to remove / 제거할 세션 ID

        Returns:
            True if removed, False if not found / 제거되면 True, 찾지 못하면 False
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                del self._session_timestamps[session_id]
                return True
        return False

    def cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions
        만료된 세션 제거

        Returns:
            Number of sessions removed / 제거된 세션 수
        """
        current_time = time.time()
        expired = []
        with self._lock:
            for sid, ts in self._session_timestamps.items():
                if current_time - ts > self.SESSION_EXPIRY:
                    # Don't remove sessions that are currently running
                    # 현재 실행 중인 세션은 제거하지 않음
                    if sid in self._sessions and not self._sessions[sid].is_running:
                        expired.append(sid)

            for sid in expired:
                del self._sessions[sid]
                del self._session_timestamps[sid]

        return len(expired)

    def get_active_session_count(self) -> int:
        """
        Get count of active sessions
        활성 세션 수 가져오기

        Returns:
            Number of active sessions / 활성 세션 수
        """
        with self._lock:
            return len(self._sessions)

    def get_all_sessions_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all active sessions for monitoring
        모니터링을 위한 모든 활성 세션 정보 가져오기

        Returns:
            List of session info dictionaries / 세션 정보 딕셔너리 목록
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
                    "purpose_goal": session_state.purpose_goal[:50] + "..." if len(session_state.purpose_goal) > 50 else session_state.purpose_goal,
                    "log_count": len(session_state.logs),
                    "duration_seconds": duration_seconds,
                    "has_pii_data": bool(session_state.pii_data),
                    "has_policy_data": bool(session_state.policy_data),
                    "has_generated_code": bool(session_state.generated_code),
                }
                sessions_info.append(session_info)

        # Sort by running status first, then by duration (most recent first)
        # 실행 상태 우선, 그 다음 기간 (최신 순) 정렬
        sessions_info.sort(key=lambda x: (not x["is_running"], -x["duration_seconds"]))

        return sessions_info

    def get_session_logs(self, session_id: str, offset: int = 0, limit: int = 100) -> Dict[str, Any]:
        """
        Get logs from a specific session for monitoring
        모니터링을 위한 특정 세션의 로그 가져오기

        Args:
            session_id: Session ID / 세션 ID
            offset: Starting index for logs / 로그 시작 인덱스
            limit: Maximum number of logs to return / 반환할 최대 로그 수

        Returns:
            Dictionary with logs and session info / 로그와 세션 정보가 포함된 딕셔너리
        """
        with self._lock:
            if session_id not in self._sessions:
                return {"error": "Session not found", "logs": [], "total": 0}

            session_state = self._sessions[session_id]
            total_logs = len(session_state.logs)
            logs = session_state.logs[offset:offset + limit]

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

    def get_shared_backend(self) -> Tuple[Optional[RAGSystem], Optional[PseuDRAGONPipeline], Optional[ExpertPreferenceManager], Optional[ComplianceValidator]]:
        """
        Get shared backend components
        공유 백엔드 컴포넌트 가져오기

        Returns:
            Tuple of (RAG, engine, expert_preference_manager, validator)
        """
        return self._rag, self._engine, self._expert_preference_manager, self._validator

    def set_shared_backend(
            self, rag: RAGSystem, engine: PseuDRAGONPipeline,
            expert_preference_manager: ExpertPreferenceManager,
            validator: ComplianceValidator
    ) -> None:
        """
        Set shared backend components
        공유 백엔드 컴포넌트 설정

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
        """
        Check if backend is initialized
        백엔드가 초기화되었는지 확인

        Returns:
            True if initialized / 초기화되었으면 True
        """
        return self._backend_initialized

    def get_all_audit_events(
            self, event_type: Optional[str] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get audit events from all sessions
        모든 세션의 감사 이벤트 가져오기

        Args:
            event_type: Filter by event type / 이벤트 유형 필터
            start_date: Filter by start date / 시작 날짜 필터
            end_date: Filter by end date / 종료 날짜 필터
            user_id: Filter by user ID / 사용자 ID 필터

        Returns:
            List of audit events from all sessions / 모든 세션의 감사 이벤트 목록
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
                            user_id=user_id
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
        """
        Get combined audit summary from all sessions
        모든 세션의 통합 감사 요약 가져오기

        Returns:
            Combined summary statistics / 통합 요약 통계
        """
        combined_summary = {
            "total_events": 0,
            "events_by_type": {},
            "session_count": 0,
            "sessions": []
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
                                "start_time": summary.get("start_time", "")
                            }
                        )
                    except Exception:
                        pass

        return combined_summary


app = Flask(__name__, template_folder="templates", static_folder="static")
app.json_encoder = CustomJSONEncoder

# Session manager for multi-client support
# 다중 클라이언트 지원을 위한 세션 관리자
session_manager = SessionManager()

# Legacy global state for backward compatibility (will be deprecated)
# 하위 호환성을 위한 레거시 전역 상태 (향후 제거 예정)
state = PipelineState()

# Global audit logger for reading all logs (regardless of session)
# 모든 로그를 읽기 위한 전역 감사 로거 (세션과 무관)
global_audit_logger = AuditLogger(log_dir=AppConfig.REPORTS_DIR, session_id="global")

# Initialize heuristic manager
heuristic_manager = HeuristicManager()


def init_backend(session_state: Optional[PipelineState] = None, log_callback: Optional[Any] = None) -> Tuple[RAGSystem, PseuDRAGONPipeline, ExpertPreferenceManager, ComplianceValidator]:
    """
    Initialize backend components (RAG system and PseuDRAGON engine)
    백엔드 컴포넌트 초기화 (RAG 시스템 및 PseuDRAGON 엔진)

    This function:
    이 함수는:
    - Checks for required files and directories / 필요한 파일 및 디렉토리 확인
    - Initializes RAG system with legal documents / 법률 문서로 RAG 시스템 초기화
    - Creates PseuDRAGON engine instance / PseuDRAGON 엔진 인스턴스 생성

    Backend components are shared across all sessions (singleton pattern).
    백엔드 컴포넌트는 모든 세션에서 공유됩니다 (싱글톤 패턴).

    Args:
        session_state: Optional session state for logging / 로깅을 위한 선택적 세션 상태
        log_callback: Optional log callback function / 선택적 로그 콜백 함수

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

        _log(f"   [WARN]️  Please add legal documents to these directories:")
        _log(f"      - {docs_dir}/institutional_policy (사내 법률 - HIGHEST PRIORITY)")
        _log(f"      - {docs_dir}/national_law (국내 법률 - MEDIUM PRIORITY)")
        _log(f"      - {docs_dir}/international_regulation (국제 법률 - LOW PRIORITY)")
        _log(f"   Without documents, the RAG system cannot provide legal guidance.")

    os.makedirs(AppConfig.REPORTS_DIR, exist_ok=True)

    _log("\n[RAG] Initializing RAG System...")
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

    _log("\n[Stats] Initializing User Preference Manager...")
    expert_preference_manager = ExpertPreferenceManager()
    stats = expert_preference_manager.get_statistics()
    _log(f"   [OK] User Preference Manager initialized! ({stats['unique_columns']} method preferences)")
    if stats.get('pii_classifications', 0) > 0:
        _log(f"   [OK] Learned {stats['pii_classifications']} PII classification(s) from previous sessions ")

    _log("\nPseuDRAGON Initializing PseuDRAGON Engine...")
    engine = PseuDRAGONPipeline(
        rag,
        DEFAULT_LLM_CLIENT,
        LLM_CLIENT_STAGE_1,
        LLM_CLIENT_STAGE_2,
        expert_preference_manager=expert_preference_manager
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

    # Also update legacy global state for backward compatibility
    state.rag = rag
    state.engine = engine
    state.expert_preference_manager = expert_preference_manager
    state.validator = validator

    return rag, engine, expert_preference_manager, validator


def log_producer(message: str, session_state: Optional[PipelineState] = None) -> None:
    """
    Thread-safe logging function with timestamps
    타임스탬프가 있는 스레드 안전 로깅 함수

    Adds timestamped messages to the session-specific or global log buffer for streaming to clients.
    Also writes to the file logger for persistent logging.
    클라이언트로 스트리밍하기 위해 세션별 또는 전역 로그 버퍼에 타임스탬프가 있는 메시지를 추가합니다.
    또한 영구 로깅을 위해 파일 로거에 기록합니다.

    Args:
        message: Log message to record / 기록할 로그 메시지
        session_state: Optional session state to log to / 로그를 기록할 선택적 세션 상태
    """
    timestamp = time.strftime("[%H:%M:%S]")
    formatted_msg = f"{timestamp} {message}"
    print(formatted_msg, flush=True)

    # Use session-specific state or fall back to global state
    # 세션별 상태를 사용하거나 전역 상태로 폴백
    target_state = session_state if session_state else state

    # Write to file logger if available
    # 파일 로거가 있으면 파일에도 기록
    if target_state.file_logger:
        target_state.file_logger.info(message)

    with target_state.progress_lock:
        target_state.logs.append(formatted_msg)


def run_pipeline_task(target_tables: List[str], purpose_goal: str, preferred_method: str, client_session_id: str) -> None:
    """
    Execute Stages 1 and 2 of the pipeline in a background thread
    백그라운드 스레드에서 파이프라인의 1단계와 2단계 실행

    This function runs asynchronously and:
    이 함수는 비동기적으로 실행되며:
    - Stage 1: Identifies PII in selected tables / 1단계: 선택된 테이블에서 PII 식별
    - Stage 2: Generates policy recommendations / 2단계: 정책 권장사항 생성
    - Waits for user input (Stage 3) / 사용자 입력 대기 (3단계)

    Args:
        target_tables: List of table names to process / 처리할 테이블 이름 목록
        purpose_goal: Purpose/intent of data pseudonymization / 데이터 가명처리 의도(목적)
        preferred_method: Preferred pseudonymization method / 선호하는 가명화 방법
        client_session_id: Unique session ID for this client / 이 클라이언트의 고유 세션 ID
    """
    # Get session-specific state
    # 세션별 상태 가져오기
    session_state = session_manager.get_session(client_session_id)
    if not session_state:
        print(f"ERROR: Session {client_session_id} not found!", flush=True)
        return

    # Create session-specific log callback
    # 세션별 로그 콜백 생성
    def session_log(msg: str) -> None:
        log_producer(msg, session_state)

    print(f"DEBUG: run_pipeline_task started for {target_tables} (session: {client_session_id})", flush=True)
    session_state.is_running = True

    # Store purpose_goal in state for use in policy synthesis
    # 정책 합성에서 사용할 수 있도록 목적을 state에 저장
    session_state.purpose_goal = purpose_goal

    # Store target_tables for use in get_stage3_data (filter to selected tables only)
    # get_stage3_data에서 선택된 테이블만 필터링하기 위해 target_tables 저장
    session_state.target_tables = target_tables

    # Initialize session directory with unique ID (UUID-based to avoid collisions)
    # 고유 ID로 세션 디렉토리 초기화 (충돌 방지를 위해 UUID 기반)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_state.session_id = f"{timestamp}_{client_session_id[:8]}"
    session_state.session_dir = os.path.join(AppConfig.OUTPUT_DIR, session_state.session_id)
    os.makedirs(session_state.session_dir, exist_ok=True)

    # Initialize file logger for this session
    # 이 세션을 위한 파일 로거 초기화
    os.makedirs(AppConfig.LOGS_DIR, exist_ok=True)
    session_state.file_logger = reinitialize_logger(session_id=session_state.session_id)
    session_state.file_logger.info(f"Pipeline started for tables: {target_tables}")
    if purpose_goal:
        session_state.file_logger.info(f"Pseudonymization Purpose: {purpose_goal}")
    if preferred_method:
        session_state.file_logger.info(f"Preferred Method: {preferred_method}")

    # Re-initialize AuditLogger for this specific run
    session_state.audit_logger = AuditLogger(log_dir=AppConfig.OUTPUT_DIR, session_id=session_state.session_id)

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

            # Add sample values to schema
            sample_data = db_manager.get_table_sample(table_name)
            for col in schema:
                schema[col]["sample_value"] = sample_data.get(col, "")

            session_state.schemas[table_name] = schema

            # Check if stop was requested before Stage 1
            if session_state.should_stop:
                session_log("⛔ Pipeline stopped by user")
                session_state.current_step = PipelineStep.IDLE
                session_state.is_running = False
                return

            # Stage 1: PII Identification with timing
            # 1단계: 수행시간 측정을 포함한 PII 식별
            stage1_name = f"Stage 1 - PII Identification ({table_name})"
            session_log(f"Starting Stage 1: PII Identification for {table_name}")
            if session_state.file_logger:
                session_state.file_logger.stage_start(stage1_name)

            if session_state.audit_logger:
                session_state.audit_logger.log_stage_start("Stage 1: PII Identification", table_name)

            try:
                pii_res = engine.identify_pii(schema, table_name, log_callback=session_log)
                session_state.pii_data[table_name] = pii_res
            finally:
                # Ensure stage_end is always called even if exception occurs
                # 예외가 발생해도 stage_end가 항상 호출되도록 보장
                if session_state.file_logger:
                    session_state.file_logger.stage_end(stage1_name)

            if session_state.audit_logger:
                session_state.audit_logger.log_stage_complete("Stage 1: PII Identification", table_name)

            # Log table statistics after PII identification
            # PII 식별 후 테이블 통계 기록
            total_columns = len(pii_res)
            pii_columns = sum(1 for col_data in pii_res.values() if col_data.get("is_pii", False))
            session_state.file_logger.log_table_stats(table_name, total_columns, pii_columns)
            session_log(f"[Stats] Table Stats: {total_columns} columns, {pii_columns} PII columns identified")

            # Check if stop was requested before Stage 2
            if session_state.should_stop:
                session_log("⛔ Pipeline stopped by user")
                session_state.current_step = PipelineStep.IDLE
                session_state.is_running = False
                return

            # Stage 2: Policy Synthesis with timing
            # 2단계: 수행시간 측정을 포함한 정책 합성
            stage2_name = f"Stage 2 - Policy Synthesis ({table_name})"
            session_log(f"Starting Stage 2: Policy Synthesis for {table_name}")

            if session_state.file_logger:
                session_state.file_logger.stage_start(stage2_name)

            if session_state.audit_logger:
                session_state.audit_logger.log_stage_start("Stage 2: Policy Synthesis", table_name)

            try:
                policy_res = engine.synthesize_policy(pii_res, table_name, preferred_method, purpose_goal=purpose_goal, log_callback=session_log, output_dir=table_dir)
                session_state.policy_data[table_name] = policy_res
            finally:
                # Ensure stage_end is always called even if exception occurs
                # 예외가 발생해도 stage_end가 항상 호출되도록 보장
                if session_state.file_logger:
                    session_state.file_logger.stage_end(stage2_name)

            if session_state.audit_logger:
                session_state.audit_logger.log_stage_complete("Stage 2: Policy Synthesis", table_name)

            # Apply user preferences to reorder methods
            # 사용자 선호도를 적용하여 기법 재정렬
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
                        # 사용자 정의 코드 스니펫이 있으면 적용 (정확한 매칭 또는 RAG 유사 매칭)
                        # Get column metadata for RAG search
                        col_comment = session_state.pii_data.get(table_name, {}).get(col_name, {}).get("column_comment", "")
                        col_data_type = ""
                        if isinstance(session_state.schemas.get(table_name, {}).get(col_name), dict):
                            col_data_type = session_state.schemas[table_name][col_name].get("type", "")

                        # Try RAG-based preference lookup (includes exact match first, then similar)
                        # RAG 기반 선호도 조회 (정확한 매칭 먼저, 그 다음 유사 매칭)
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
                                # 커스텀 설명이 있으면 적용 (코드와 설명이 일치하도록 보장)
                                if custom_description:
                                    reordered_methods[0]["description"] = custom_description

                                if custom_snippet or custom_description:
                                    if match_type == "rag_similar":
                                        print(f"[RAG] Applied custom snippet/description from similar column {similar_col} -> {col_name}")
                                        session_log(f"[RAG] RAG: Applied {preferred_method} from {similar_col}")
                                    else:
                                        print(f"[INFO] Applied custom code/description for PII column {col_name} ({preferred_method})")

                        # Update policy data with reordered methods
                        session_state.policy_data[table_name][col_name]["recommended_methods"] = reordered_methods
                    else:
                        # Non-PII column - reorder methods based on user preference
                        # But preserve ALL LLM-generated methods (don't replace them!)
                        # Get column metadata for RAG search
                        col_comment = session_state.pii_data.get(table_name, {}).get(col_name, {}).get("column_comment", "")
                        col_data_type = ""
                        if isinstance(session_state.schemas.get(table_name, {}).get(col_name), dict):
                            col_data_type = session_state.schemas[table_name][col_name].get("type", "")

                        # Try RAG-based preference lookup (includes exact match first, then similar)
                        # RAG 기반 선호도 조회 시도 (정확한 매칭 먼저, 그 다음 유사 매칭)
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
                            existing_methods = session_state.policy_data[table_name][col_name].get("recommended_methods", [])

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
                                    # 커스텀 설명이 있으면 적용 (코드와 설명이 일치하도록 보장)
                                    if custom_description:
                                        reordered[0]["description"] = custom_description

                                    if custom_snippet or custom_description:
                                        if match_type == "rag_similar":
                                            print(f"[RAG] Applied custom snippet/description from similar column {similar_col} -> {col_name}")
                                            session_log(f"[RAG] RAG: Applied {preferred_method} from {similar_col}")
                                        else:
                                            print(f"[INFO] Applied custom code/description for Non-PII column {col_name} ({preferred_method})")

                                # Update with reordered methods
                                session_state.policy_data[table_name][col_name]["recommended_methods"] = reordered
                                print(f"[INFO] Reordered Non-PII methods for {col_name}: {preferred_method} prioritized (kept all {len(reordered)} methods)")
                            else:
                                # Only KEEP/DELETE exist - this is an old policy, don't touch it
                                print(f"[INFO] Non-PII column {col_name} has only basic methods, skipping reorder")

            # Store integrated report filename (relative to OUTPUT_DIR)
            # 통합 보고서 파일명 저장 (OUTPUT_DIR 기준 상대 경로)
            if hasattr(engine, "_last_integrated_report"):
                rel_path = os.path.relpath(engine._last_integrated_report, AppConfig.OUTPUT_DIR)
                session_state.report_files[f"{table_name}_integrated"] = rel_path

        session_log("Stage 1 & 2 Completed for all tables.")
        session_log("WAITING_USER_INPUT")

        session_state.current_step = PipelineStep.STAGE3_READY

        # Log stage timing summary for Stage 1 & 2
        # Stage 1 & 2 수행시간 요약 로깅
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


@app.route("/")
def index() -> str:
    """
    Serve the main web interface
    메인 웹 인터페이스 제공

    Returns:
        Rendered HTML template / 렌더링된 HTML 템플릿
    """
    return render_template("index.html")


@app.route("/get_tables")
def get_tables_route() -> Tuple:
    """
    API endpoint to retrieve available database tables
    사용 가능한 데이터베이스 테이블을 검색하는 API 엔드포인트

    Returns:
        JSON response with table names or error / 테이블 이름 또는 오류가 포함된 JSON 응답
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
    """
    API endpoint to retrieve schema for a specific table
    특정 테이블의 스키마를 검색하는 API 엔드포인트

    Args:
        table_name: Name of the table / 테이블 이름

    Returns:
        JSON response with table schema or error / 테이블 스키마 또는 오류가 포함된 JSON 응답
    """
    try:
        if not os.path.exists(AppConfig.DB_PATH):
            return jsonify({"error": f"Database not found at {AppConfig.DB_PATH}"}), 500

        # Get schema from database
        db_manager = DatabaseManager(AppConfig.DB_PATH)
        schema = db_manager.get_table_schema(table_name)

        if not schema:
            return jsonify({"error": f"Table '{table_name}' not found"}), 404

        return jsonify(
            {
                "table_name": table_name,
                "schema": schema
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/execute_ddl", methods=["POST"])
def execute_ddl() -> Tuple:
    """
    Execute SQL statements directly (CREATE, INSERT, etc.)
    SQL 문을 직접 실행 (CREATE, INSERT 등)

    Request body:
        ddl: SQL statements / SQL 문

    Returns:
        JSON response with success status or error / 성공 상태 또는 오류가 포함된 JSON 응답
    """
    try:
        data = request.get_json()
        sql_input = data.get("ddl", "").strip()

        if not sql_input:
            return jsonify({"error": "SQL statement is required"}), 400

        import re

        # Block dangerous and query commands
        # 위험한 명령어 및 조회 명령어 차단
        blocked_patterns = [
            # Dangerous commands / 위험한 명령어
            (r'\bDROP\s+DATABASE\b', "DROP DATABASE is not allowed"),
            (r'\bDELETE\s+FROM\b', "DELETE is not allowed"),
            (r'\bTRUNCATE\b', "TRUNCATE is not allowed"),
            (r'\bALTER\s+DATABASE\b', "ALTER DATABASE is not allowed"),
            (r'\bDROP\s+SCHEMA\b', "DROP SCHEMA is not allowed"),
            # Query commands (require output) / 조회 명령어 (출력 필요)
            (r'\bSELECT\b', "SELECT is not allowed (use for data modification only)"),
            (r'\bSHOW\b', "SHOW is not allowed"),
            (r'\bDESCRIBE\b', "DESCRIBE is not allowed"),
            (r'\bEXPLAIN\b', "EXPLAIN is not allowed"),
            (r'\bPRAGMA\b', "PRAGMA is not allowed"),
        ]

        sql_upper = sql_input.upper()
        for pattern, message in blocked_patterns:
            if re.search(pattern, sql_upper):
                return jsonify({"error": message}), 400

        # Execute SQL directly using DuckDB's execute method which handles multiple statements
        # DuckDB의 execute 메서드를 사용하여 SQL을 직접 실행 (여러 문장 처리)
        db_path = AppConfig.DB_PATH

        try:
            if db_path.endswith('.duckdb'):
                # Use DuckDB - execute the entire SQL script
                import duckdb
                conn = duckdb.connect(db_path)
                # DuckDB can execute multiple statements separated by semicolons
                # DuckDB는 세미콜론으로 구분된 여러 문장을 실행할 수 있음
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
                "statement_count": sql_input.count(';')
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/drop_table", methods=["POST"])
def drop_table() -> Tuple:
    """
    Drop a table from the database
    데이터베이스에서 테이블 삭제

    Request body:
        table_name: Name of the table to drop / 삭제할 테이블 이름

    Returns:
        JSON response with success status or error / 성공 상태 또는 오류가 포함된 JSON 응답
    """
    try:
        data = request.get_json()
        table_name = data.get("table_name", "").strip()

        if not table_name:
            return jsonify({"error": "Table name is required"}), 400

        # Validate table name (prevent SQL injection)
        # 테이블 이름 검증 (SQL 인젝션 방지)
        import re
        if not re.match(r'^[\w가-힣]+$', table_name):
            return jsonify({"error": "Invalid table name"}), 400

        db_path = AppConfig.DB_PATH

        try:
            if db_path.endswith('.duckdb'):
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
                "table_name": table_name
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/stream")
def stream() -> Response:
    """
    Server-Sent Events (SSE) endpoint for real-time log streaming
    실시간 로그 스트리밍을 위한 Server-Sent Events (SSE) 엔드포인트

    Streams pipeline logs and status updates to the client in real-time.
    파이프라인 로그 및 상태 업데이트를 실시간으로 클라이언트에 스트리밍합니다.

    Query Parameters:
        session_id: Client session ID for session-specific logging / 세션별 로깅을 위한 클라이언트 세션 ID

    Returns:
        SSE stream with log messages and events / 로그 메시지 및 이벤트가 포함된 SSE 스트림
    """
    # Get session ID from query parameter
    # 쿼리 파라미터에서 세션 ID 가져오기
    client_session_id = request.args.get("session_id", "")
    print(f"DEBUG: /stream endpoint called (session: {client_session_id})", flush=True)

    # Get session-specific state or fall back to global state
    # 세션별 상태를 가져오거나 전역 상태로 폴백
    session_state = session_manager.get_session(client_session_id) if client_session_id else None
    target_state = session_state if session_state else state

    def generate() -> Any:
        """
        Generator function for SSE stream
        SSE 스트림을 위한 제너레이터 함수

        Yields log messages and stage completion events.
        로그 메시지 및 단계 완료 이벤트를 생성합니다.
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
    """
    API endpoint to start pipeline execution (Stages 1 & 2)
    파이프라인 실행을 시작하는 API 엔드포인트 (1단계 및 2단계)

    Accepts:
    수락:
    - tables: List of table names to process / 처리할 테이블 이름 목록
    - purpose_goal: Purpose/intent of data pseudonymization / 데이터 가명처리 의도(목적)
    - preferred_method: Preferred pseudonymization method / 선호하는 가명화 방법
    - session_id: (Optional) Existing session ID to resume / (선택) 재개할 기존 세션 ID

    Returns:
        JSON response with status and session_id / 상태 및 세션 ID가 포함된 JSON 응답
    """
    req_data = request.json or {}

    # Get or create session ID
    # 세션 ID 가져오기 또는 생성
    client_session_id = req_data.get("session_id", "")
    if not client_session_id:
        # Create new session for this client
        # 이 클라이언트를 위한 새 세션 생성
        client_session_id = session_manager.create_session()
        print(f"DEBUG: Created new session: {client_session_id}", flush=True)
    else:
        # Verify existing session exists
        # 기존 세션이 존재하는지 확인
        existing_session = session_manager.get_session(client_session_id)
        if not existing_session:
            # Session expired or invalid, create new one
            # 세션이 만료되었거나 유효하지 않음, 새로 생성
            client_session_id = session_manager.create_session()
            print(f"DEBUG: Session not found, created new session: {client_session_id}", flush=True)

    # Get session state
    session_state = session_manager.get_session(client_session_id)
    if not session_state:
        return jsonify({"status": "error", "message": "Failed to create session"}), 500

    # Check if THIS session is already running (not global state)
    # 이 세션이 이미 실행 중인지 확인 (전역 상태가 아님)
    if session_state.is_running:
        return jsonify({"status": "already_running", "session_id": client_session_id}), 400

    tables = req_data.get("tables", ["users"])
    purpose_goal = req_data.get("purpose_goal", "")
    preferred_method = req_data.get("preferred_method", "")

    # Cleanup expired sessions periodically
    # 만료된 세션 주기적으로 정리
    cleaned = session_manager.cleanup_expired_sessions()
    if cleaned > 0:
        print(f"DEBUG: Cleaned up {cleaned} expired sessions", flush=True)

    thread = Thread(target=run_pipeline_task, args=(tables, purpose_goal, preferred_method, client_session_id))
    thread.start()

    return jsonify({"status": "started", "session_id": client_session_id})


@app.route("/stop", methods=["POST"])
def stop_pipeline() -> Tuple:
    """
    API endpoint to stop pipeline execution
    파이프라인 실행을 중단하는 API 엔드포인트

    Sets the stop flag to gracefully terminate the running pipeline.
    중단 플래그를 설정하여 실행 중인 파이프라인을 안전하게 종료합니다.

    Accepts:
        - session_id: Client session ID / 클라이언트 세션 ID

    Returns:
        JSON response with status / 상태가 포함된 JSON 응답
    """
    req_data = request.json or {}
    client_session_id = req_data.get("session_id", "")

    # Get session-specific state or fall back to global state
    # 세션별 상태를 가져오거나 전역 상태로 폴백
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
    """
    API endpoint to retrieve Stage 3 data for human review
    사람 검토를 위한 3단계 데이터를 검색하는 API 엔드포인트

    Query Parameters:
        session_id: Client session ID / 클라이언트 세션 ID

    Returns policy recommendations with PII analysis for user selection.
    사용자 선택을 위해 PII 분석과 함께 정책 권장사항을 반환합니다.

    Returns:
        JSON with PII types, methods, evidence sources, and report filenames
        PII 유형, 기법, 증거 출처 및 보고서 파일명이 포함된 JSON
    """
    # Get session ID from query parameter
    # 쿼리 파라미터에서 세션 ID 가져오기
    client_session_id = request.args.get("session_id", "")

    # Get session-specific state or fall back to global state
    # 세션별 상태를 가져오거나 전역 상태로 폴백
    session_state = session_manager.get_session(client_session_id) if client_session_id else None
    target_state = session_state if session_state else state

    if target_state.current_step != PipelineStep.STAGE3_READY:
        return jsonify({"error": "Not ready for Stage 3"}), 400

    display_data: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # Use target_tables if available, otherwise fall back to all schemas
    # target_tables가 있으면 사용, 없으면 모든 스키마로 폴백
    tables_to_process = getattr(target_state, 'target_tables', None)
    if not tables_to_process:
        tables_to_process = list(target_state.schemas.keys())
    print(f"[DEBUG /get_stage3_data] Processing tables: {tables_to_process}")

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

                    # CRITICAL FIX: Validate pii_type consistency
                    # If is_pii=True but pii_type="Non-PII", this is a data inconsistency bug
                    # Fix it by defaulting to "PII"
                    pii_type = pii_info.get("pii_type", "Unknown")
                    if pii_type == "Non-PII":
                        print(f"[WARNING] Data inconsistency detected for {col}: is_pii=True but pii_type='Non-PII'")
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
                    # Non-PII 컬럼에 기존 정책 데이터가 있음
                    cached_methods = policy.get('recommended_methods', [])
                    cached_method_names = {m.get('method', '').upper() for m in cached_methods}

                    pii_info = target_state.pii_data[table].get(col, {})
                    column_comment = pii_info.get("column_comment", "")

                    # If not in pii_data, try to get comment from schema
                    if not column_comment and isinstance(target_state.schemas[table][col], dict):
                        column_comment = target_state.schemas[table][col].get("comment", "")

                    # Check if cached methods are incomplete (only KEEP, missing DELETE)
                    # 캐시된 메서드가 불완전한지 확인 (KEEP만 있고 DELETE가 없는 경우)
                    needs_regeneration = len(cached_methods) == 1 and 'KEEP' in cached_method_names and 'DELETE' not in cached_method_names

                    if needs_regeneration:
                        # Regenerate methods for Non-PII column
                        # Non-PII 컬럼의 메서드 재생성
                        print(f"[INFO] Regenerating Non-PII methods for {col} (cached methods incomplete: {[m.get('method') for m in cached_methods]})")

                        # Get data type for better method recommendations
                        data_type = ""
                        if isinstance(target_state.schemas[table][col], dict):
                            data_type = target_state.schemas[table][col].get("type", "")

                        non_pii_methods = _generate_non_pii_methods(col, column_comment, data_type)
                        print(f"[DEBUG] Regenerated {len(non_pii_methods)} methods for {col}: {[t['method'] for t in non_pii_methods]}")

                        # Update policy_data with regenerated methods
                        target_state.policy_data[table][col]["recommended_methods"] = non_pii_methods

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
                        print(f"[INFO] Using cached Non-PII policy for {col} ({len(cached_methods)} methods)")

                        display_data[table][col] = {
                            "pii_type": "Non-PII",
                            "is_pii": False,
                            "id_evidence": "Not identified as PII",
                            "methods": cached_methods,
                            "tech_evidence": policy.get("evidence_source", "Data Minimization Principle"),
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
                print(f"[DEBUG /get_stage3_data] Generating Non-PII methods for {col}")
                print(f"  - Column: {col}")
                print(f"  - Data Type: {data_type}")
                print(f"  - Comment: {column_comment}")
                non_pii_methods = _generate_non_pii_methods(col, column_comment, data_type)
                print(f"[DEBUG /get_stage3_data] Generated {len(non_pii_methods)} methods for {col}")
                print(f"  - Methods: {[t['method'] for t in non_pii_methods]}")

                # CRITICAL: Store Non-PII methods in policy_data for later retrieval in submit_stage3
                # submit_stage3에서 나중에 검색할 수 있도록 Non-PII 기술을 policy_data에 저장
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
    # 보고서 파일명 포함
    response_data = {"tables": display_data, "reports": target_state.report_files}

    return jsonify(response_data)


@app.route("/submit_stage3", methods=["POST"])
def submit_stage3() -> Tuple:
    """
    API endpoint to submit Stage 3 user selections and execute Stage 4
    3단계 사용자 선택을 제출하고 4단계를 실행하는 API 엔드포인트

    Processes user-selected methods and generates final pseudonymization code.
    사용자가 선택한 기법을 처리하고 최종 가명처리 코드를 생성합니다.

    Accepts:
    수락:
    - choices: Dictionary of user-selected method indices
              사용자가 선택한 방법 인덱스의 딕셔너리
    - session_id: Client session ID / 클라이언트 세션 ID

    Returns:
        JSON with generated code for each table
        각 테이블에 대해 생성된 코드가 포함된 JSON
    """
    req_data = request.json or {}
    client_session_id = req_data.get("session_id", "")
    choices = {k: v for k, v in req_data.items() if k != "session_id"}

    # Get session-specific state or fall back to global state
    # 세션별 상태를 가져오거나 전역 상태로 폴백
    session_state = session_manager.get_session(client_session_id) if client_session_id else None
    target_state = session_state if session_state else state

    target_state.current_step = PipelineStep.STAGE4_PROCESSING

    if target_state.audit_logger:
        target_state.audit_logger.log_stage_start("Stage 3: HITL Approval", "all_tables")

    # Process user selections for ALL columns (PII and non-PII)
    print(f"[DEBUG submit_stage3] Processing {len(choices)} selections")
    print(f"[DEBUG submit_stage3] Keys received: {list(choices.keys())}")

    for key, selection_data in choices.items():
        # Key format: "method_TABLENAME_COLUMNNAME"
        # CRITICAL: Table names can contain underscores (e.g., "withdraw_transfers")
        # So we need to be smarter about parsing

        if not key.startswith("method_"):
            print(f"[WARNING] Invalid key format: {key}")
            continue

        # Remove "method_" prefix
        remainder = key[7:]  # Remove "method_"

        # Find table name by checking against known tables
        table = None
        col = None

        for table_name in target_state.schemas.keys():
            if remainder.startswith(f"{table_name}_"):
                table = table_name
                col = remainder[len(table_name) + 1:]  # +1 for the underscore
                break

        if not table or not col:
            print(f"[WARNING] Could not parse key: {key}, remainder: {remainder}")
            print(f"[WARNING] Known tables: {list(target_state.schemas.keys())}")
            continue

        # Handle both old format (int) and new format (dict)
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

        print(f"[DEBUG] Parsed: key={key} -> table={table}, col={col}, idx={idx}, code_modified={code_modified}, desc_modified={description_modified}")

        # Check if column is in policy_data AND is still PII
        # IMPORTANT: User may have changed PII status, so check is_pii flag
        in_policy_data = table in target_state.policy_data and col in target_state.policy_data[table]
        is_pii_flag = target_state.policy_data[table][col].get("is_pii", True) if in_policy_data else False

        print(f"[DEBUG] Column {col}: in_policy_data={in_policy_data}, is_pii_flag={is_pii_flag}")

        is_current_pii = in_policy_data and is_pii_flag

        if is_current_pii:
            # PII column - reorder methods based on user selection
            techs = target_state.policy_data[table][col]["recommended_methods"]
            if 0 <= idx < len(techs):
                # Save old action object BEFORE modifying the list
                old_action_obj = techs[0].copy() if techs else {}
                old_action = old_action_obj.get("method", "UNKNOWN")

                # Get and remove selected method from list
                selected = techs.pop(idx)
                new_action = selected.get("method", "UNKNOWN")

                # Update code snippet if user edited it
                if code_modified and edited_code:
                    old_code = selected.get("code_snippet", "")
                    selected["code_snippet"] = edited_code
                    print(f"[DEBUG] Code modified for {col}: User edited code snippet")
                    log_producer(f"✏️ User edited code for {col}", target_state)

                # Update description if user edited it
                if description_modified and edited_description:
                    old_desc = selected.get("description", "")
                    selected["description"] = edited_description
                    print(f"[DEBUG] Description modified for {col}: User edited description")
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

                print(f"[DEBUG] PII Column {col}: User selected idx={idx}, Method changed from '{old_action}' to '{new_action}'")
                log_producer(f"✏️ User selection for {col}: {new_action} (was: {old_action})", target_state)

                # Record user preference for future sessions (including default selection idx=0)
                # 향후 세션을 위한 사용자 선호도 기록 (기본 선택 idx=0 포함)
                if target_state.expert_preference_manager:
                    pii_type = target_state.policy_data[table][col].get("pii_type", "Unknown")
                    all_methods = [t.get("method", "") for t in techs]
                    # Pass code snippet and description if user edited them
                    # 사용자가 수정한 경우 커스텀 코드와 설명을 저장하도록 전달
                    final_code_snippet = selected.get("code_snippet", "") if code_modified else None
                    final_description = selected.get("description", "") if description_modified else None

                    # Get column_comment from schema or pii_data for RAG-based similarity search
                    # RAG 기반 유사성 검색을 위해 스키마 또는 pii_data에서 column_comment 가져오기
                    col_comment = ""
                    if table in target_state.schemas and col in target_state.schemas[table]:
                        schema_col = target_state.schemas[table][col]
                        if isinstance(schema_col, dict):
                            col_comment = schema_col.get("comment", "")
                    if not col_comment and table in target_state.pii_data and col in target_state.pii_data[table]:
                        col_comment = target_state.pii_data[table][col].get("column_comment", "")

                    target_state.expert_preference_manager.record_selection(
                        table_name=table,
                        column_name=col,
                        pii_type=pii_type,
                        selected_method=new_action,
                        available_methods=all_methods,
                        selected_index=idx,
                        code_snippet=final_code_snippet,
                        code_modified=code_modified,
                        description=final_description,
                        description_modified=description_modified,
                        column_comment=col_comment
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

            # Get the actual selected method from policy_data (could be ROUND, GENERALIZE, HASH, KEEP, DELETE)
            # IMPORTANT: Don't hardcode idx->method mapping; use actual method at that index
            if table in target_state.policy_data and col in target_state.policy_data[table]:
                techs = target_state.policy_data[table][col].get("recommended_methods", [])

                if 0 <= idx < len(techs):
                    # Use actual method at selected index
                    selected = techs[idx]
                    method = selected.get("method", "Keep")
                    description = selected.get("description", "Keep the column unchanged")
                    code_snippet = selected.get("code_snippet", "# No transformation needed")

                    # CRITICAL: Apply user-edited code if modified
                    # 사용자가 수정한 코드가 있으면 적용
                    if code_modified and edited_code:
                        old_code = code_snippet
                        code_snippet = edited_code
                        print(f"[DEBUG] Non-PII Column {col}: User edited code snippet")
                        print(f"[DEBUG]   Original: {old_code[:50]}...")
                        print(f"[DEBUG]   Edited: {edited_code[:50]}...")
                        log_producer(f"✏️ User edited code for Non-PII column {col}", target_state)

                    # Apply user-edited description if modified
                    # 사용자가 수정한 설명이 있으면 적용
                    if description_modified and edited_description:
                        old_desc = description
                        description = edited_description
                        print(f"[DEBUG] Non-PII Column {col}: User edited description")
                        log_producer(f"✏️ User edited description for Non-PII column {col}", target_state)

                    print(f"[DEBUG] Non-PII Column {col}: User selected idx={idx}, Method='{method}' from {len(techs)} methods")
                    print(f"[DEBUG]   Available methods: {[t.get('method') for t in techs]}")
                else:
                    # Invalid index - fallback to Keep
                    print(f"[WARNING] Non-PII Column {col}: Invalid idx={idx} (only {len(techs)} methods available)")
                    method = "Keep"
                    description = "Keep the column unchanged"
                    code_snippet = "# No transformation needed"
            else:
                # No policy data - use simple fallback for Keep/Delete only
                print(f"[WARNING] Non-PII Column {col}: No policy data found, using fallback for idx={idx}")
                if idx == 1:
                    method = "Delete"
                    description = "Remove the column if not needed for analysis"
                    code_snippet = f"record.pop('{col}', None)"
                else:
                    method = "Keep"
                    description = "Keep the column unchanged"
                    code_snippet = "# No transformation needed"

            # CRITICAL DEBUG LOG for non-PII
            print(f"[DEBUG] Non-PII Column {col}: Final selection - Method='{method}'")
            log_producer(f"✏️ Non-PII selection for {col}: {method}", target_state)

            # Record user preference for Non-PII columns (if user changed from default)
            # Non-PII 컬럼에 대한 사용자 선호도 기록 (기본값에서 변경한 경우)
            if target_state.expert_preference_manager:
                # Get all available methods from the method list
                if table in target_state.policy_data and col in target_state.policy_data[table]:
                    available_methods = [t.get("method") for t in target_state.policy_data[table][col].get("recommended_methods", [])]
                else:
                    # Fallback to Keep/Delete only
                    available_methods = ["Keep", "Delete"]

                # Pass code snippet and description if user edited them
                # 사용자가 수정한 경우 커스텀 코드와 설명을 저장하도록 전달
                final_code_snippet = code_snippet if code_modified else None
                final_description = description if description_modified else None

                # Get column_comment from schema or pii_data for RAG-based similarity search
                # RAG 기반 유사성 검색을 위해 스키마 또는 pii_data에서 column_comment 가져오기
                col_comment = ""
                if table in target_state.schemas and col in target_state.schemas[table]:
                    schema_col = target_state.schemas[table][col]
                    if isinstance(schema_col, dict):
                        col_comment = schema_col.get("comment", "")
                if not col_comment and table in target_state.pii_data and col in target_state.pii_data[table]:
                    col_comment = target_state.pii_data[table][col].get("column_comment", "")

                target_state.expert_preference_manager.record_selection(
                    table_name=table,
                    column_name=col,
                    pii_type="Non-PII",
                    selected_method=method,
                    available_methods=available_methods,
                    selected_index=idx,
                    code_snippet=final_code_snippet,
                    code_modified=code_modified,
                    description=final_description,
                    description_modified=description_modified,
                    column_comment=col_comment
                )

            # Create or update policy for non-PII column with the actual selected method
            # IMPORTANT: This overwrites any existing policy (including PII -> Non-PII reclassified ones)
            target_state.policy_data[table][col] = {
                "is_pii": False,
                "pii_type": "Non-PII",
                "action": method,  # Add action field for audit logging
                "recommended_methods": [
                    {
                        "method": method,
                        "applicability": "High",
                        "description": description,
                        "code_snippet": code_snippet,
                        "example_implementation": code_snippet,
                    }
                ],
                "evidence_source": target_state.policy_data[table].get(col, {}).get("evidence_source", "User Selection (Non-PII)"),
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
                    new_parameters={},
                    old_code_snippet="",
                    new_code_snippet=code_snippet,
                    legal_evidence="",
                )

    # CRITICAL: Regenerate integrated report AFTER user modifications in Stage 3
    # 중요: Stage 3에서 사용자 수정 후 통합 보고서 재생성
    # This ensures that the report reflects all user changes (PII reclassification, method selection, code edits)
    # 이를 통해 보고서가 모든 사용자 변경사항을 반영하도록 보장 (PII 재분류, 기법 선택, 코드 편집)
    log_producer("\n[Stats] Regenerating integrated reports with user modifications...", target_state)

    # Extract selected tables from choices to only process those tables
    # 선택된 테이블만 처리하기 위해 choices에서 테이블 목록 추출
    selected_tables = set()
    for key in choices.keys():
        if key.startswith("method_"):
            remainder = key[7:]
            for table_name in target_state.schemas.keys():
                if remainder.startswith(f"{table_name}_"):
                    selected_tables.add(table_name)
                    break

    # Clear previous generated_code and report_files for clean generation
    # 깨끗한 생성을 위해 이전 generated_code와 report_files 클리어
    target_state.generated_code = {}
    target_state.report_files = {}

    log_producer(f"[Info] Processing {len(selected_tables)} selected table(s): {list(selected_tables)}", target_state)

    for table in selected_tables:
        if table not in target_state.policy_data:
            continue

        try:
            # Convert policy_data to Policy object for report generation
            # 보고서 생성을 위해 policy_data를 Policy 객체로 변환
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
                    # PII 분류 변경에서 근거 추출 (가능한 경우)
                    evidence_source = col_data.get("evidence_source", "")
                    user_rationale = ""
                    legal_evidence = selected_tech.get("legal_evidence", "")

                    # Check if this was a user-defined PII classification change
                    # 사용자 정의 PII 분류 변경인지 확인
                    if evidence_source.startswith("User-defined PII"):
                        # Extract the rationale from evidence_source
                        # evidence_source에서 근거 추출
                        if "Rationale:" in evidence_source:
                            parts = evidence_source.split("Rationale:", 1)
                            if len(parts) == 2:
                                user_rationale = parts[1].strip()

                        # Set legal_evidence to just "User-defined PII"
                        # legal_evidence를 "User-defined PII"로만 설정
                        legal_evidence = "User-defined PII"

                    # Use user rationale if available, otherwise use method description
                    # 사용자 근거가 있으면 사용, 없으면 기법 설명 사용
                    final_rationale = user_rationale if user_rationale else selected_tech.get("description", "")

                    action = PolicyAction(
                        action=action_type,
                        rationale=final_rationale,
                        code_snippet=selected_tech.get("code_snippet", ""),
                        legal_evidence=legal_evidence,
                        parameters=selected_tech.get("parameters", {})
                    )

                    col_policy = ColumnPolicy(
                        column_name=col_name,
                        pii_type=pii_type,
                        is_pii=is_pii,
                        action=action
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
                        parameters={}
                    )

                    col_policy = ColumnPolicy(
                        column_name=col_name,
                        pii_type=pii_type,
                        is_pii=is_pii,
                        action=action
                    )

                    policy_obj.columns[col_name] = col_policy

            # Ensure pii_data has all columns with updated information
            # pii_data가 업데이트된 정보를 가진 모든 컬럼을 포함하도록 보장
            if table not in target_state.pii_data:
                target_state.pii_data[table] = {}

            for col_name, col_data in target_state.policy_data[table].items():
                if col_name not in target_state.pii_data[table]:
                    # Add missing column to pii_data
                    target_state.pii_data[table][col_name] = {
                        "is_pii": col_data.get("is_pii", False),
                        "pii_type": col_data.get("pii_type", "Non-PII"),
                        "reasoning": "User modified",
                        "column_comment": col_data.get("column_comment", "")
                    }
                else:
                    # Update existing entry with current PII status
                    target_state.pii_data[table][col_name]["is_pii"] = col_data.get("is_pii", False)
                    target_state.pii_data[table][col_name]["pii_type"] = col_data.get("pii_type", "Non-PII")

            # Regenerate integrated report with user-modified policy
            # 사용자가 수정한 정책으로 통합 보고서 재생성
            table_dir = os.path.join(target_state.session_dir, table)
            os.makedirs(table_dir, exist_ok=True)

            report_gen = ReportGenerator()
            report_path = report_gen.generate_integrated_report(
                table_name=table,
                pii_analysis=target_state.pii_data.get(table, {}),
                policy=policy_obj,
                output_dir=table_dir
            )

            # Update target_state.report_files so the frontend shows the updated report
            # target_state.report_files를 업데이트하여 프론트엔드에서 업데이트된 보고서를 표시하도록 함
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

    if target_state.audit_logger:
        target_state.audit_logger.log_stage_complete("Stage 3: HITL Approval", "all_tables")

    # Trigger heuristic pattern learning at Stage 3 completion
    # Stage 3 완료 시점에 휴리스틱 패턴 학습 트리거
    heuristic_learning_stats = None
    if target_state.expert_preference_manager:
        log_producer("[Update] Learning heuristic patterns from expert feedback...", target_state)
        pattern_stats = target_state.expert_preference_manager.trigger_pattern_learning()
        if pattern_stats:
            heuristic_learning_stats = pattern_stats
            log_producer(f"[OK] Learned {pattern_stats['total_patterns']} heuristic patterns", target_state)
            log_producer(f"  - PII patterns: {pattern_stats['pii_patterns']['suffixes']} suffixes, {pattern_stats['pii_patterns']['prefixes']} prefixes, {pattern_stats['pii_patterns']['keywords']} keywords", target_state)
            log_producer(f"  - Non-PII patterns: {pattern_stats['non_pii_patterns']['suffixes']} suffixes, {pattern_stats['non_pii_patterns']['prefixes']} prefixes, {pattern_stats['non_pii_patterns']['keywords']} keywords", target_state)

    if not os.path.exists(AppConfig.OUTPUT_DIR):
        os.makedirs(AppConfig.OUTPUT_DIR)

    # Only generate code for selected tables (not all tables in schemas)
    # 선택된 테이블에 대해서만 코드 생성 (schemas의 모든 테이블이 아님)
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
        # 4단계: 수행시간 측정을 포함한 코드 생성
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
                schema, target_state.policy_data[table], target_state.pii_data[table], table_name=table, db_config=AppConfig.DB_PATH,  # Can be string path or dict config
                log_callback=session_log_callback, )

            output_file = os.path.join(table_dir, f"pseudonymize_{table}.py")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(code)

            target_state.generated_code[table] = code
            log_producer(f"Code for {table} generated and saved to {output_file}", target_state)
        finally:
            # Ensure stage_end is always called even if exception occurs
            # 예외가 발생해도 stage_end가 항상 호출되도록 보장
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
            target_state.audit_logger.log_code_generation(table=table, code_length=len(code), validation_passed=True)

            report_path = target_state.audit_logger.generate_compliance_report(policy_dict=target_state.policy_data[table], table_name=table, output_dir=table_dir, )

            # Store relative path for web interface
            rel_report_path = os.path.relpath(report_path, AppConfig.OUTPUT_DIR)
            target_state.report_files[f"{table}_compliance"] = rel_report_path
            log_producer(f"[File] Compliance report generated: {report_path}", target_state)

    target_state.current_step = PipelineStep.COMPLETED

    # Log final summary
    # 최종 요약 로깅
    if target_state.file_logger:
        target_state.file_logger.log_summary()
        log_producer(f"[Dir] Log file saved: {target_state.file_logger.get_log_file_path()}", target_state)

    # Return updated report files so frontend can show the regenerated integrated reports
    # 프론트엔드에서 재생성된 통합 보고서를 표시할 수 있도록 업데이트된 보고서 파일 반환
    return jsonify(
        {
            "status": "completed",
            "codes": target_state.generated_code,
            "reports": target_state.report_files,  # Include updated report files
            "heuristic_learning": heuristic_learning_stats  # Include heuristic learning stats for notification
        }
    )


@app.route("/validate_policy", methods=["POST"])
def validate_policy() -> Tuple:
    """
    API endpoint for real-time policy validation
    실시간 정책 검증을 위한 API 엔드포인트

    Implements real-time validation
    논문의 실시간 검증 구현

    Accepts:
        - session_id: Client session ID / 클라이언트 세션 ID
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
        # 세션별 상태를 가져오거나 전역 상태로 폴백
        session_state = session_manager.get_session(client_session_id) if client_session_id else None
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
                # 법적 근거 추출 및 사용자 정의 PII 확인
                evidence_source = col_data.get("evidence_source", "")
                legal_evidence = col_data.get("legal_evidence", "Unknown")

                if evidence_source.startswith("User-defined PII"):
                    legal_evidence = "User-defined PII"

                action = PolicyAction(action=action_type, rationale=rationale, legal_evidence=legal_evidence, )
            else:
                techs = col_data.get("recommended_methods", [])
                if techs:
                    first_tech = techs[0]

                    # Extract rationale from PII classification change if available
                    # PII 분류 변경에서 근거 추출 (가능한 경우)
                    evidence_source = col_data.get("evidence_source", "")
                    user_rationale = ""
                    legal_evidence = col_data.get("legal_evidence", "Unknown")

                    # Check if this was a user-defined PII classification change
                    # 사용자 정의 PII 분류 변경인지 확인
                    if evidence_source.startswith("User-defined PII"):
                        # Extract the rationale from evidence_source
                        # evidence_source에서 근거 추출
                        if "Rationale:" in evidence_source:
                            parts = evidence_source.split("Rationale:", 1)
                            if len(parts) == 2:
                                user_rationale = parts[1].strip()

                        # Set legal_evidence to just "User-defined PII"
                        # legal_evidence를 "User-defined PII"로만 설정
                        legal_evidence = "User-defined PII"

                    # Use user rationale if available, otherwise use method description
                    # 사용자 근거가 있으면 사용, 없으면 방법 설명 사용
                    final_rationale = user_rationale if user_rationale else first_tech.get("description", "")

                    action = PolicyAction(action=ActionType[first_tech.get("method", "KEEP")], rationale=final_rationale, legal_evidence=legal_evidence, )
                else:
                    action = PolicyAction(action=ActionType.KEEP)

            col_policy = ColumnPolicy(column_name=col_name, pii_type=pii_type, is_pii=is_pii, action=action)
            temp_policy.add_column_policy(col_policy)

        violations = target_state.validator.validate_policy(temp_policy, target_state.schemas[table])

        if target_state.audit_logger:
            target_state.audit_logger.log_validation_result(table, [{"severity": v.severity, "column": v.column, "message": v.message} for v in violations], )

        summary = target_state.validator.get_validation_summary(violations)

        return jsonify(
            {
                "status": "validated",
                "violations": [{"severity": v.severity, "column": v.column, "message": v.message, "suggestion": v.suggestion, "check_type": v.check_type, } for v in violations],
                "summary": summary,
            }
        )

    except Exception as e:
        import traceback

        traceback.print_exc()

        return jsonify({"error": str(e)}), 500


@app.route("/get_report/<path:filename>")
def get_report(filename: str) -> Tuple:
    """
    API endpoint to retrieve markdown report content
    마크다운 리포트 내용을 검색하는 API 엔드포인트

    Args:
        filename: Report filename or path / 리포트 파일명 또는 경로

    Returns:
        Markdown content as text / 텍스트 형태의 마크다운 내용
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
    """
    API endpoint to download markdown report
    마크다운 리포트를 다운로드하는 API 엔드포인트

    Args:
        filename: Report filename / 리포트 파일명

    Returns:
        File download response / 파일 다운로드 응답
    """
    try:
        report_path = os.path.join(AppConfig.REPORTS_DIR, filename)
        if not os.path.exists(report_path):
            return "Report not found", 404

        return send_file(report_path, as_attachment=True, download_name=filename)
    except Exception as e:
        return f"Error downloading report: {str(e)}", 500


# ============================================================================
# Heuristics Management API Endpoints
# 휴리스틱 관리 API 엔드포인트
# ============================================================================

@app.route("/api/heuristics", methods=["GET"])
def get_heuristics() -> Tuple:
    """
    Get all heuristics
    모든 휴리스틱 가져오기

    Query params:
        enabled_only: "true" to get only enabled heuristics
        source: "manual" or "auto_learned" to filter by source

    Returns:
        JSON with all heuristics (includes source field: "manual" or "auto_learned")
        모든 휴리스틱이 포함된 JSON (source 필드 포함: "manual" 또는 "auto_learned")
    """
    try:
        # Reload from file to get latest data (including auto-learned patterns)
        # 파일에서 다시 로드하여 최신 데이터 가져오기 (자동 학습된 패턴 포함)
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
    """
    Get heuristics statistics
    휴리스틱 통계 가져오기

    Returns:
        JSON with heuristics statistics (total, manual, auto-learned counts)
        휴리스틱 통계가 포함된 JSON (전체, 수동, 자동학습 수)
    """
    try:
        # Reload from file to get latest data
        # 파일에서 다시 로드하여 최신 데이터 가져오기
        heuristic_manager.load()

        stats = heuristic_manager.get_statistics()
        return jsonify({"status": "success", "statistics": stats})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/heuristics/<heuristic_id>", methods=["GET"])
def get_heuristic(heuristic_id: str) -> Tuple:
    """
    Get specific heuristic by ID
    ID로 특정 휴리스틱 가져오기
    
    Args:
        heuristic_id: Heuristic ID
                     휴리스틱 ID
    
    Returns:
        JSON with heuristic data
        휴리스틱 데이터가 포함된 JSON
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
    """
    Add new heuristic
    새 휴리스틱 추가
    
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
        생성된 휴리스틱이 포함된 JSON
    """
    try:
        data = request.json

        # Validate required fields
        required_fields = ["name", "regex", "pii_type", "rationale"]
        for field in required_fields:
            if field not in data:
                return jsonify({"status": "error", "message": f"Missing required field: {field}"}), 400

        # Add heuristic
        heuristic = heuristic_manager.add(
            name=data["name"],
            regex=data["regex"],
            pii_type=data["pii_type"],
            rationale=data["rationale"],
            priority=data.get("priority", 50),
            enabled=data.get("enabled", True),
            pattern_type=data.get("pattern_type", "")
        )

        # Log heuristic change
        if state.audit_logger:
            state.audit_logger.log_heuristic_change(
                action="ADD",
                heuristic_id=heuristic["id"],
                new_data=heuristic,
                user_id=request.remote_addr or "unknown"
            )

        return jsonify({"status": "success", "heuristic": heuristic})
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/heuristics/<heuristic_id>", methods=["PUT"])
def update_heuristic(heuristic_id: str) -> Tuple:
    """
    Update existing heuristic
    기존 휴리스틱 업데이트
    
    Args:
        heuristic_id: Heuristic ID to update
                     업데이트할 휴리스틱 ID
    
    Returns:
        JSON with updated heuristic
        업데이트된 휴리스틱이 포함된 JSON
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
                user_id=request.remote_addr or "unknown"
            )

        return jsonify({"status": "success", "heuristic": heuristic})
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/heuristics/<heuristic_id>", methods=["DELETE"])
def delete_heuristic(heuristic_id: str) -> Tuple:
    """
    Delete heuristic
    휴리스틱 삭제
    
    Args:
        heuristic_id: Heuristic ID to delete
                     삭제할 휴리스틱 ID
    
    Returns:
        JSON with status
        상태가 포함된 JSON
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
                user_id=request.remote_addr or "unknown"
            )

        return jsonify({"status": "success", "message": "Heuristic deleted"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/heuristics/test", methods=["POST"])
def test_heuristic() -> Tuple:
    """
    Test regex pattern against sample strings
    샘플 문자열에 대해 정규식 패턴 테스트
    
    Accepts:
        - regex: Regular expression pattern
        - test_strings: List of strings to test
    
    Returns:
        JSON with test results
        테스트 결과가 포함된 JSON
    """
    try:
        data = request.json

        if "regex" not in data or "test_strings" not in data:
            return jsonify({"status": "error", "message": "Missing required fields"}), 400

        result = heuristic_manager.test_pattern(
            regex=data["regex"],
            test_strings=data["test_strings"]
        )

        return jsonify({"status": "success", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================================================
# Stage 3 User Validation API Endpoints
# Stage 3 사용자 검증 API 엔드포인트
# ============================================================================

@app.route("/change_pii_status", methods=["POST"])
@app.route("/api/stage3/change_pii_status", methods=["POST"])
def change_pii_status() -> Tuple:
    """
    Handle PII status change and query LLM for anonymization methods
    PII 상태 변경 처리 및 익명화 방법에 대한 LLM 쿼리

    Accepts:
        - session_id: Client session ID / 클라이언트 세션 ID
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
        # 세션별 상태를 가져오거나 전역 상태로 폴백
        session_state = session_manager.get_session(client_session_id) if client_session_id else None
        target_state = session_state if session_state else state

        print(f"[DEBUG /change_pii_status] Received request:")
        print(f"  - session_id: {client_session_id}")
        print(f"  - table: {table}")
        print(f"  - column: {column}")
        print(f"  - old_status: {old_status}")
        print(f"  - new_status: {new_status}")
        print(f"  - pii_type: {pii_type}")
        print(f"  - rationale: {rationale}")

        # If old_status is not provided, infer it from new_status
        if not old_status and new_status:
            old_status = "Non-PII" if new_status == "PII" else "PII"
            print(f"[DEBUG] Inferred old_status: {old_status}")

        # Validate inputs
        if not all([table, column, new_status]):
            return jsonify({"status": "error", "message": "Missing required fields"}), 400

        # Log the PII classification change
        if target_state.audit_logger:
            target_state.audit_logger.log_pii_classification_change(
                table=table,
                column=column,
                old_status=old_status,
                new_status=new_status,
                rationale=rationale,
                user_id=request.remote_addr or "user"
            )

        # Record PII classification change in user preference manager for future learning
        # 향후 학습을 위해 사용자 선호도 관리자에 PII 분류 변경 기록
        if target_state.expert_preference_manager:
            # Get column_comment from schema or pii_data for RAG-based similarity search
            # RAG 기반 유사성 검색을 위해 스키마 또는 pii_data에서 column_comment 가져오기
            col_comment = ""
            if table in target_state.schemas and column in target_state.schemas[table]:
                schema_col = target_state.schemas[table][column]
                if isinstance(schema_col, dict):
                    col_comment = schema_col.get("comment", "")
            if not col_comment and table in target_state.pii_data and column in target_state.pii_data[table]:
                col_comment = target_state.pii_data[table][column].get("column_comment", "")

            target_state.expert_preference_manager.record_pii_classification_change(
                column_name=column,
                old_classification=old_status,
                new_classification=new_status,
                rationale=rationale,
                column_comment=col_comment
            )
            log_producer(f"[RAG] Recorded PII classification learning: {column} ({old_status} -> {new_status})", target_state)

        methods = []

        print(f"[DEBUG] Checking status change conditions:")
        print(f"  - old_status == 'PII': {old_status == 'PII'}")
        print(f"  - new_status == 'Non-PII': {new_status == 'Non-PII'}")
        print(f"  - old_status == 'Non-PII': {old_status == 'Non-PII'}")
        print(f"  - new_status == 'PII': {new_status == 'PII'}")

        if old_status == "PII" and new_status == "Non-PII":
            print(f"[DEBUG] Executing branch: PII -> Non-PII")
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
            if not column_comment and table in target_state.pii_data and column in target_state.pii_data[table]:
                column_comment = target_state.pii_data[table][column].get("column_comment", "")

            # Generate Non-PII methods using LLM (includes ROUND, GENERALIZATION for applicable types)
            methods = _generate_non_pii_methods(column, column_comment, data_type)

            print(f"[INFO] Generated {len(methods)} Non-PII methods for {column} (type: {data_type})")
            log_producer(f"[Note] Generated Non-PII methods for {column}: {', '.join([m['method'] for m in methods])}", target_state)

            # CRITICAL: Update target_state.policy_data AND target_state.pii_data to reflect PII -> Non-PII change
            # This ensures that the old PII policy (e.g., MASK) is replaced with Non-PII methods
            print(f"[DEBUG] Updating state for {table}.{column}: PII -> Non-PII")
            log_producer(f"[Note] State updated for {column}: PII -> Non-PII", target_state)

            # Remove from policy_data (since it's no longer PII)
            if table in target_state.policy_data and column in target_state.policy_data[table]:
                del target_state.policy_data[table][column]
                print(f"[DEBUG] Removed {column} from policy_data")

            # Update pii_data to mark as Non-PII
            if table not in target_state.pii_data:
                target_state.pii_data[table] = {}

            target_state.pii_data[table][column] = {
                "is_pii": False,
                "pii_type": "Non-PII",
                "evidence_source": f"User reclassified from PII. {rationale}" if rationale else "User reclassified from PII",
                "column_comment": column_comment
            }
            print(f"[DEBUG] Updated pii_data for {column} to Non-PII")
        elif old_status == "Non-PII" and new_status == "PII":
            print(f"[DEBUG] Executing branch: Non-PII -> PII")
            # Non-PII -> PII: Query LLM for anonymization methods
            if not rationale:
                return jsonify({"status": "error", "message": "Rationale is required for Non-PII to PII conversion"}), 400

            # Get column metadata for better method generation
            column_comment = ""
            data_type = ""

            if table in target_state.schemas and column in target_state.schemas[table]:
                schema_col = target_state.schemas[table][column]
                if isinstance(schema_col, dict):
                    column_comment = schema_col.get("comment", "")
                    data_type = schema_col.get("type", "")

            # If not in schemas, try pii_data
            if not column_comment and table in target_state.pii_data and column in target_state.pii_data[table]:
                column_comment = target_state.pii_data[table][column].get("column_comment", "")

            # Get LLM-generated anonymization methods with column description
            print(f"[DEBUG] Calling _query_llm_for_anonymization_methods for {column}")
            print(f"  - pii_type: {pii_type}")
            print(f"  - rationale: {rationale}")
            print(f"  - column_comment: {column_comment}")
            print(f"  - data_type: {data_type}")
            methods = _query_llm_for_anonymization_methods(column, pii_type, rationale, column_comment, data_type)
            print(f"[DEBUG] _query_llm_for_anonymization_methods returned {len(methods)} methods")

            # CRITICAL: Update target_state.policy_data AND target_state.pii_data to reflect Non-PII -> PII change
            # This ensures that the new PII policy is stored for code generation
            print(f"[DEBUG] Updating state for {table}.{column}: Non-PII -> PII")
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
            print(f"[DEBUG] Added {column} to policy_data as PII")

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
                "column_comment": column_comment
            }
            print(f"[DEBUG] Updated pii_data for {column} to PII ({pii_type})")
        else:
            return jsonify({"status": "error", "message": "Invalid status change"}), 400

        # Get column comment if available
        column_comment = ""
        if table in target_state.schemas and column in target_state.schemas[table]:
            schema_col = target_state.schemas[table][column]
            if isinstance(schema_col, dict):
                column_comment = schema_col.get("comment", "")
        # Also try to get from pii_data if available
        if not column_comment and table in target_state.pii_data and column in target_state.pii_data[table]:
            column_comment = target_state.pii_data[table][column].get("column_comment", "")

        # Prepare response data
        response_pii_type = pii_type if new_status == "PII" else "Non-PII"

        print(f"[DEBUG] Response data:")
        print(f"  - column: {column}")
        print(f"  - old_status: {old_status}")
        print(f"  - new_status: {new_status}")
        print(f"  - pii_type (from request): {pii_type}")
        print(f"  - response pii_type: {response_pii_type}")
        print(f"  - methods count: {len(methods)}")

        return jsonify(
            {
                "status": "success",
                "message": f"Successfully changed {column} from {old_status} to {new_status}",
                "pii_type": response_pii_type,
                "methods": methods,
                "column_comment": column_comment,
                "column": column,
                "old_status": old_status,
                "new_status": new_status
            }
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


def _query_llm_for_anonymization_methods(column_name: str, pii_type: str, rationale: str, column_comment: str = "", data_type: str = "") -> List[Dict[str, Any]]:
    """
    Query LLM to generate anonymization methods for a column with RAG integration
    RAG 통합을 통해 컬럼에 대한 익명화 방법을 생성하기 위해 LLM에 쿼리

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
        print(f"[DEBUG _query_llm_for_anonymization_methods] Starting for column: {column_name}")
        print(f"  - pii_type: {pii_type}")
        print(f"  - rationale: {rationale}")
        print(f"  - column_comment: {column_comment}")
        print(f"  - data_type: {data_type}")

        # Use shared backend components from session_manager
        shared_rag, shared_engine, _, _ = session_manager.get_shared_backend()

        print(f"[DEBUG] shared_engine: {shared_engine}")
        print(f"[DEBUG] shared_rag: {shared_rag}")

        if not shared_engine or not shared_engine.llm_client_stage_2:
            # Fallback to default methods if LLM is not available
            print(f"[DEBUG] LLM client not available, using fallback methods")
            return _get_default_anonymization_methods(column_name, column_comment, data_type)

        if not shared_rag:
            # Fallback if RAG is not initialized
            print(f"[DEBUG] RAG not initialized, using fallback methods")
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
        # IMPORTANT: Prioritize column description over column name
        column_description_text = column_comment if column_comment else "No description available"
        data_type_text = data_type if data_type else "Unknown"
        user_prompt = user_prompt_template.format(
            context=context,
            column=column_name,
            pii_type=pii_type,
            column_desc=f"{column_description_text}. User rationale: {rationale}",
            data_type=data_type_text,
            purpose_goal="Pseudonymization for privacy protection",
            preferred_method="General Pseudonymization"
        )

        # Step 5: Query LLM with legal context
        # Create a temporary session for this query
        session = UniversalLLMSession(
            client=llm_client,
            system_prompt=system_prompt,
            session_id="temp_anonymization_query",
            max_history=0
        )

        response = session.chat(
            user_prompt=user_prompt,
            model=Settings.LLM_STAGE_2,
            temperature=0.1
        )

        # Step 6: Parse response
        content = response.choices[0].message.content

        # Try to extract JSON from response
        import json
        import re
        from pseudragon.llm.json_parser import safe_parse_json

        result = safe_parse_json(
            content, default_response={
                "recommended_methods": [],
                "evidence_source": "Failed to parse LLM response"
            }
        )

        methods_data = result.get("recommended_methods", [])

        # Build legal evidence from source documents
        legal_evidence = ", ".join(source_docs) if source_docs else result.get("evidence_source", "User-defined PII")

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
                "legal_evidence": tech.get("legal_source", legal_evidence)
            }
            methods.append(method_dict)

        # Ensure we have at least 2 methods - add fallback if needed
        MIN_METHODS = 2
        if len(methods) < MIN_METHODS:
            fallback_methods = _get_default_anonymization_methods(column_name, column_comment, data_type)
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
    """
    Infer PII type from column name and user rationale
    컬럼 이름과 사용자 근거에서 PII 유형 추론

    Args:
        column_name: Column name
        rationale: User's rationale for PII classification

    Returns:
        PII type (PII)
    """
    # All PII columns are classified as PII
    # 모든 PII 컬럼은 PII로 분류됩니다
    return "PII"


def _analyze_column_semantics(column_name: str, column_comment: str = "", data_type: str = "") -> Dict[str, Any]:
    """
    Analyze column semantics from description and name to determine data type
    컬럼 설명과 이름으로부터 의미를 분석하여 데이터 타입 결정

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
    is_numeric = any(t in (data_type or "").upper() for t in ["INT", "DECIMAL", "NUMERIC", "FLOAT", "DOUBLE", "REAL", "NUMBER"])

    # PRIORITY 1: Check column description/comment for semantic meaning (more reliable)
    # PRIORITY 2: Check column name as fallback

    # === PII-related keywords ===
    # Phone number
    is_phone_by_comment = any(kw in col_comment_upper for kw in ["전화", "핸드폰", "휴대폰", "PHONE", "MOBILE", "TEL", "CONTACT"])
    is_phone_by_name = any(kw in col_name_upper for kw in ["PHONE", "TEL", "MOBILE", "CONTACT"])
    is_phone = is_phone_by_comment or is_phone_by_name

    # Email
    is_email_by_comment = any(kw in col_comment_upper for kw in ["이메일", "EMAIL", "E-MAIL", "MAIL"])
    is_email_by_name = any(kw in col_name_upper for kw in ["EMAIL", "MAIL"])
    is_email = is_email_by_comment or is_email_by_name

    # Name
    is_name_by_comment = any(kw in col_comment_upper for kw in ["이름", "성명", "NAME", "고객명", "CUSTOMER NAME"])
    is_name_by_name = any(kw in col_name_upper for kw in ["NAME", "NM"])
    is_name = is_name_by_comment or is_name_by_name

    # ID/identifier
    is_id_by_comment = any(kw in col_comment_upper for kw in ["아이디", "식별자", "ID", "IDENTIFIER", "고유번호", "UNIQUE"])
    is_id_by_name = any(kw in col_name_upper for kw in ["ID", "_ID", "IDENTIFIER", "UNIQUE_ID"])
    is_id = is_id_by_comment or is_id_by_name

    # === Non-PII related keywords ===
    # Date/time
    is_date_by_comment = any(
        kw in col_comment_upper for kw in [
            "날짜", "생년월일", "DATE", "BIRTH", "TRANSACTION DATE", "CREATED", "UPDATED",
            "YYYY", "MM", "DD", "년", "월", "일",
            "DATETIME", "TIME", "TIMESTAMP", "INQUIRY DATETIME", "HHMMSS",
            "STANDARD DATE", "STD DATE", "일자", "기준일", "YYYYMMDD"
        ]
    )
    is_date_by_name = any(
        kw in col_name_upper for kw in [
            "DT", "DATE", "BIRTH", "TRAN_DT", "CREATE_DT", "UPDATE_DT",
            "DTIME", "DATETIME", "TIME", "TIMESTAMP", "STD_DT", "STDT"
        ]
    )
    is_date_column = is_date_by_comment or is_date_by_name

    # Birth date (specific type of date)
    is_birth_date = any(kw in col_comment_upper for kw in ["생년월일", "BIRTH"]) or "BIRTH" in col_name_upper

    # Amount/balance
    is_amount_by_comment = any(
        kw in col_comment_upper for kw in [
            "금액", "잔액", "BALANCE", "AMOUNT", "PRICE", "COST", "FEE", "SALARY",
            "RATE", "CHARGE", "PAYMENT", "원", "달러", "DOLLAR", "WON"
        ]
    )
    is_amount_by_name = any(kw in col_name_upper for kw in ["AMT", "AMOUNT", "BLNC", "BALANCE", "PRICE", "COST", "FEE", "RATE"])

    # Count/quantity
    is_count_by_comment = any(
        kw in col_comment_upper for kw in [
            "개수", "수량", "COUNT", "QUANTITY", "QTY", "NUMBER OF", "건수"
        ]
    )
    is_count_by_name = any(kw in col_name_upper for kw in ["QTY", "QUANTITY", "CNT", "COUNT"])

    # Combine amount and count
    is_amount_column = is_amount_by_comment or is_count_by_comment or is_amount_by_name or is_count_by_name

    # Age
    is_age_by_comment = any(
        kw in col_comment_upper for kw in [
            "나이", "연령", "AGE", "세", "YEARS OLD"
        ]
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


def _get_default_anonymization_methods(column_name: str, column_comment: str = "", data_type: str = "") -> List[Dict[str, Any]]:
    """
    Get default anonymization methods when LLM is not available
    LLM을 사용할 수 없을 때 기본 익명화 방법 가져오기

    Args:
        column_name: Column name
        column_comment: Column description/comment (more reliable than column name)
        data_type: Column data type

    Returns:
        List of default anonymization methods
    """
    methods = []

    print(f"[DEBUG _get_default_anonymization_methods] column={column_name}, comment={column_comment}, data_type={data_type}")

    # Use unified semantic analysis
    semantics = _analyze_column_semantics(column_name, column_comment, data_type)

    print(f"[DEBUG] is_phone={semantics['is_phone']} (comment:{semantics['is_phone_by_comment']}, name:{semantics['is_phone_by_name']})")
    print(f"[DEBUG] is_email={semantics['is_email']} (comment:{semantics['is_email_by_comment']}, name:{semantics['is_email_by_name']})")
    print(f"[DEBUG] is_name={semantics['is_name']} (comment:{semantics['is_name_by_comment']}, name:{semantics['is_name_by_name']})")
    print(f"[DEBUG] is_id={semantics['is_id']} (comment:{semantics['is_id_by_comment']}, name:{semantics['is_id_by_name']})")

    # Provide context-specific methods based on column semantics
    if semantics['is_phone']:
        # Phone numbers: MASK (show last 4 digits), HASH, ENCRYPT
        methods.append(
            {
                "method": "MASK",
                "applicability": "High",
                "description": f"Mask {column_name} - show only last 4 digits (e.g., 010-1234-5678 -> ***-****-5678)",
                "code_snippet": f"masked_{column_name} = str(record['{column_name}'])[:-4] + '****' if len(str(record['{column_name}'])) > 4 else '****'",
                "example_implementation": f"masked_{column_name} = str(record['{column_name}'])[:-4] + '****' if len(str(record['{column_name}'])) > 4 else '****'",
                "legal_evidence": "Phone number masking - recommended for PII protection"
            }
        )
    elif semantics['is_email']:
        # Email: MASK (show domain only), HASH
        methods.append(
            {
                "method": "MASK",
                "applicability": "High",
                "description": f"Mask {column_name} - show domain only (e.g., user@example.com -> ****@example.com)",
                "code_snippet": f"masked_{column_name} = '****@' + str(record['{column_name}']).split('@')[1] if '@' in str(record['{column_name}']) else '****'",
                "example_implementation": f"masked_{column_name} = '****@' + str(record['{column_name}']).split('@')[1] if '@' in str(record['{column_name}']) else '****'",
                "legal_evidence": "Email masking - recommended for PII protection"
            }
        )
    elif semantics['is_name']:
        # Names: GENERALIZATION (first character only), MASK, TOKENIZATION
        methods.append(
            {
                "method": "GENERALIZATION",
                "applicability": "High",
                "description": f"Generalize {column_name} - show only first character (e.g., 홍길동 -> 홍**, John Doe -> J***)",
                "code_snippet": f"generalized_{column_name} = str(record['{column_name}'])[0] + '*' * (len(str(record['{column_name}'])) - 1) if len(str(record['{column_name}'])) > 0 else ''",
                "example_implementation": f"generalized_{column_name} = str(record['{column_name}'])[0] + '*' * (len(str(record['{column_name}'])) - 1) if len(str(record['{column_name}'])) > 0 else "
                                          f"''",
                "legal_evidence": "Name generalization - recommended for PII protection"
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
                "legal_evidence": "User-defined PII - Masking recommended"
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
            "legal_evidence": "User-defined PII - Hashing recommended"
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
            "legal_evidence": "User-defined PII - Tokenization recommended"
        }
    )

    return methods


def _generate_non_pii_methods(column_name: str, column_comment: str, data_type: str) -> List[Dict[str, Any]]:
    """
    Generate pseudonymization methods for Non-PII columns using LLM
    Non-PII 컬럼에 대한 가명처리 기법을 LLM을 사용하여 생성

    For numeric columns (e.g., BLNC/Balance), this may recommend:
    숫자 컬럼(예: BLNC/잔액)의 경우 다음을 추천할 수 있습니다:
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
            data_type=data_type if data_type else "Unknown"
        )

        print(f"[DEBUG _generate_non_pii_methods] Querying LLM for {column_name}")
        print(f"  - Column: {column_name}")
        print(f"  - Description: {column_comment}")
        print(f"  - Data Type: {data_type}")

        # Query LLM
        session = UniversalLLMSession(
            client=llm_client,
            system_prompt=system_prompt,
            session_id="temp_non_pii_methods",
            max_history=0
        )

        response = session.chat(
            user_prompt=user_prompt,
            model=Settings.LLM_STAGE_2,
            temperature=0.3  # Slightly higher for more variety
        )

        content = response.choices[0].message.content

        print(f"[DEBUG _generate_non_pii_methods] LLM response length: {len(content)} characters")
        print(f"[DEBUG _generate_non_pii_methods] LLM FULL response:")
        print("=" * 80)
        print(content)
        print("=" * 80)

        # Parse JSON response
        from pseudragon.llm.json_parser import safe_parse_json

        result = safe_parse_json(
            content, default_response={
                "recommended_methods": []
            }
        )

        methods_data = result.get("recommended_methods", [])

        print(f"[DEBUG _generate_non_pii_methods] Parsed {len(methods_data)} methods from LLM")

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
                "legal_evidence": tech.get("rationale", "LLM-generated recommendation for Non-PII")
            }
            methods.append(method_dict)
            print(f"  - Added method: {method_name}")

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
                    "legal_evidence": "Standard option for Non-PII data"
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
                    "legal_evidence": "Standard option for Non-PII data"
                }
            )

        return methods

    except Exception as e:
        print(f"Error generating Non-PII methods for {column_name}: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to default
        return _get_default_non_pii_methods(column_name, data_type, column_comment)


def _get_default_non_pii_methods(column_name: str, data_type: str, column_comment: str = "") -> List[Dict[str, Any]]:
    """
    Get default Non-PII methods when LLM is not available
    LLM을 사용할 수 없을 때 기본 Non-PII 기법 가져오기

    Args:
        column_name: Column name
        data_type: Column data type
        column_comment: Column description/comment (more reliable than column name)

    Returns:
        List of default Non-PII methods
    """
    methods = []

    print(f"[DEBUG _get_default_non_pii_methods] column={column_name}, data_type={data_type}, comment={column_comment}")

    # Use unified semantic analysis
    semantics = _analyze_column_semantics(column_name, column_comment, data_type)

    print(f"[DEBUG] is_numeric={semantics['is_numeric']}, is_date_column={semantics['is_date_column']} (comment:{semantics['is_date_by_comment']}, name:{semantics['is_date_by_name']})")
    print(f"[DEBUG] is_amount_column={semantics['is_amount_column']} (comment:{semantics['is_amount_by_comment']}, name:{semantics['is_amount_by_name']})")
    print(f"[DEBUG] is_age_column={semantics['is_age_column']}")

    # Get uppercase versions for checking
    col_name_upper = column_name.upper()
    col_comment_upper = (column_comment or "").upper()

    if semantics['is_numeric']:
        if semantics['is_date_column']:
            # Date column (YYYYMMDD format)
            # Check if it's a birth date (by comment or name) - prioritize age range for birth dates
            is_birth_date = semantics['is_birth_date']

            if is_birth_date or semantics['is_age_column']:
                methods.append(
                    {
                        "method": "Generalization",
                        "applicability": "High",
                        "description": f"Convert {column_name} to age range (e.g., 19900115 -> '30s')",
                        "code_snippet": f"lambda x: f\"{{((datetime.now().year - int(str(x)[:4]))//10)*10}}s\" if pd.notna(x) and len(str(x)) >= 8 and str(x).isdigit() else str(x)",
                        "example_implementation": f"lambda x: f\"{{((datetime.now().year - int(str(x)[:4]))//10)*10}}s\" if pd.notna(x) and len(str(x)) >= 8 and str(x).isdigit() else str(x)",
                        "legal_evidence": "Age range generalization for privacy protection"
                    }
                )

            # General date methods
            # Check if it's a datetime (YYYYMMDDhhmmss - 14 digits) or just date (YYYYMMDD - 8 digits)
            is_datetime = "DATETIME" in col_comment_upper or "HHMMSS" in col_comment_upper or "DTIME" in col_name_upper

            if is_datetime:
                # DateTime-specific methods (YYYYMMDDhhmmss -> YYYYMMDD or YYYYMM or YYYY)
                methods.append(
                    {
                        "method": "Generalization",
                        "applicability": "High",
                        "description": f"Extract date only from {column_name} (e.g., 20240115143025 -> 20240115)",
                        "code_snippet": f"lambda x: str(x)[:8] if pd.notna(x) and len(str(x)) >= 14 and str(x).isdigit() else str(x)",
                        "example_implementation": f"lambda x: str(x)[:8] if pd.notna(x) and len(str(x)) >= 14 and str(x).isdigit() else str(x)",
                        "legal_evidence": "Temporal generalization for privacy protection (remove time)"
                    }
                )

                methods.append(
                    {
                        "method": "Generalization",
                        "applicability": "High",
                        "description": f"Extract year-month from {column_name} (e.g., 20240115143025 -> 202401)",
                        "code_snippet": f"lambda x: str(x)[:6] if pd.notna(x) and len(str(x)) >= 14 and str(x).isdigit() else str(x)",
                        "example_implementation": f"lambda x: str(x)[:6] if pd.notna(x) and len(str(x)) >= 14 and str(x).isdigit() else str(x)",
                        "legal_evidence": "Temporal generalization for privacy protection (year-month only)"
                    }
                )

                methods.append(
                    {
                        "method": "Generalization",
                        "applicability": "Medium",
                        "description": f"Extract year only from {column_name} (e.g., 20240115143025 -> 2024)",
                        "code_snippet": f"lambda x: str(x)[:4] if pd.notna(x) and len(str(x)) >= 14 and str(x).isdigit() else str(x)",
                        "example_implementation": f"lambda x: str(x)[:4] if pd.notna(x) and len(str(x)) >= 14 and str(x).isdigit() else str(x)",
                        "legal_evidence": "Temporal generalization for privacy protection (year only)"
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
                        "legal_evidence": "Temporal generalization for privacy protection"
                    }
                )

                methods.append(
                    {
                        "method": "Generalization",
                        "applicability": "Medium",
                        "description": f"Extract year-month from {column_name} (e.g., 20240115 -> 202401)",
                        "code_snippet": f"lambda x: str(x)[:6] if pd.notna(x) and len(str(x)) >= 8 and str(x).isdigit() else str(x)",
                        "example_implementation": f"lambda x: str(x)[:6] if pd.notna(x) and len(str(x)) >= 8 and str(x).isdigit() else str(x)",
                        "legal_evidence": "Temporal generalization for privacy protection"
                    }
                )
        else:
            # Regular numeric column (amount, balance, count, or age value)
            # Special handling for age columns (already age values, not birth dates)
            if semantics['is_age_column']:
                methods.append(
                    {
                        "method": "Generalization",
                        "applicability": "High",
                        "description": f"Convert {column_name} to age range (e.g., 34 -> '30s')",
                        "code_snippet": f"lambda x: f\"{{(int(x)//10)*10}}s\" if pd.notna(x) and str(x).replace('.','').isdigit() else str(x)",
                        "example_implementation": f"lambda x: f\"{{(int(x)//10)*10}}s\" if pd.notna(x) and str(x).replace('.','').isdigit() else str(x)",
                        "legal_evidence": "Age range generalization for privacy protection"
                    }
                )
                methods.append(
                    {
                        "method": "Generalization",
                        "applicability": "Medium",
                        "description": f"Generalize {column_name} into broader age ranges (e.g., 34 -> '30-39')",
                        "code_snippet": f"lambda x: f\"{{(int(x)//10)*10}}-{{(int(x)//10)*10+9}}\" if pd.notna(x) and str(x).replace('.','').isdigit() else str(x)",
                        "example_implementation": f"lambda x: f\"{{(int(x)//10)*10}}-{{(int(x)//10)*10+9}}\" if pd.notna(x) and str(x).replace('.','').isdigit() else str(x)",
                        "legal_evidence": "Age range generalization for privacy protection"
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
                        "legal_evidence": "Numeric rounding for privacy protection"
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
                        "legal_evidence": "Range generalization for privacy protection"
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
            "legal_evidence": "Standard option for Non-PII data"
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
            "legal_evidence": "Standard option for Non-PII data"
        }
    )

    return methods


# ============================================================================
# Expert Feedback API Endpoints
# 전문가 피드백 API 엔드포인트
# ============================================================================

@app.route("/api/expert_feedback", methods=["GET"])
def get_expert_feedback() -> Tuple:
    """
    Get expert feedback data
    전문가 피드백 데이터 가져오기

    Returns all accumulated expert feedback from Stage 3 HITL sessions.
    Stage 3 HITL 세션에서 축적된 모든 전문가 피드백을 반환합니다.

    Returns:
        JSON with expert feedback data
        전문가 피드백 데이터가 포함된 JSON
    """
    try:
        # Load expert feedback from file
        expert_pref_manager = ExpertPreferenceManager()

        # Get all feedback data
        feedback_data = {
            "status": "success",
            "method_preferences": [],
            "pii_classifications": [],
            "statistics": expert_pref_manager.get_statistics()
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
                    "column_name": value.get("column_name", ""),
                    "pii_type": value.get("pii_type", ""),
                    "method_counts": method_counts,
                    "total_selections": value.get("total_selections", 0),
                    "preferred_method": preferred_method
                }
            )

        # Get PII classification changes
        pii_classifications = expert_pref_manager.preferences.get("pii_classification", {})
        for key, value in pii_classifications.items():
            feedback_data["pii_classifications"].append(
                {
                    "column_name": value.get("column_name", ""),
                    "classification": value.get("classification", ""),
                    "change_count": value.get("change_count", 0),
                    "rationale": value.get("rationale", ""),
                    "history": value.get("history", [])
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
    """
    Get expert feedback summary statistics
    전문가 피드백 요약 통계 가져오기

    Returns:
        JSON with summary statistics
        요약 통계가 포함된 JSON
    """
    try:
        expert_pref_manager = ExpertPreferenceManager()
        stats = expert_pref_manager.get_statistics()

        return jsonify(
            {
                "status": "success",
                "summary": stats
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================================================
# Audit Log API Endpoints
# 감사 로그 API 엔드포인트
# ============================================================================

@app.route("/api/audit_log", methods=["GET"])
def get_audit_log() -> Tuple:
    """
    Get audit log entries with optional filtering
    선택적 필터링을 사용하여 감사 로그 항목 가져오기

    Query parameters:
        - session_id: Client session ID (optional, if not provided returns all logs)
                      클라이언트 세션 ID (선택사항, 없으면 모든 로그 반환)
        - event_type: Filter by event type
        - start_date: Filter by start date (ISO format)
        - end_date: Filter by end date (ISO format)
        - user_id: Filter by user ID

    Returns:
        JSON with audit log entries
        감사 로그 항목이 포함된 JSON
    """
    try:
        # Get filter parameters
        event_type = request.args.get("event_type")
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        user_id = request.args.get("user_id")

        # Get session ID from query parameter
        # 쿼리 파라미터에서 세션 ID 가져오기
        client_session_id = request.args.get("session_id", "")

        # If session_id is provided, try to get events from that specific session
        # 세션 ID가 제공되면 해당 세션의 이벤트만 가져오기 시도
        if client_session_id:
            session_state = session_manager.get_session(client_session_id)
            if session_state and session_state.audit_logger:
                events = session_state.audit_logger.get_all_events(
                    event_type=event_type,
                    start_date=start_date,
                    end_date=end_date,
                    user_id=user_id
                )
                return jsonify({"status": "success", "events": events, "count": len(events)})

        # Use global audit logger to read ALL log files from disk
        # 전역 감사 로거를 사용하여 디스크에서 모든 로그 파일 읽기
        # This works even after page refresh or server restart
        # 페이지 새로고침이나 서버 재시작 후에도 동작
        events = global_audit_logger.get_all_events(
            event_type=event_type,
            start_date=start_date,
            end_date=end_date,
            user_id=user_id
        )

        return jsonify({"status": "success", "events": events, "count": len(events)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/audit_log/summary", methods=["GET"])
def get_audit_log_summary() -> Tuple:
    """
    Get audit log summary
    감사 로그 요약 가져오기

    Query parameters:
        - session_id: Client session ID (optional, if not provided returns combined summary)
                      클라이언트 세션 ID (선택사항, 없으면 전체 요약 반환)

    Returns:
        JSON with summary statistics
        요약 통계가 포함된 JSON
    """
    try:
        # Get session ID from query parameter
        # 쿼리 파라미터에서 세션 ID 가져오기
        client_session_id = request.args.get("session_id", "")

        # If session_id is provided, get summary from that specific session
        # 세션 ID가 제공되면 해당 세션의 요약만 가져오기
        if client_session_id:
            session_state = session_manager.get_session(client_session_id)
            if session_state and session_state.audit_logger:
                summary = session_state.audit_logger.get_session_summary()
                return jsonify({"status": "success", "summary": summary})

        # Use global audit logger to get summary from ALL log files on disk
        # 전역 감사 로거를 사용하여 디스크의 모든 로그 파일에서 요약 가져오기
        summary = global_audit_logger.get_session_summary()
        return jsonify({"status": "success", "summary": summary})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================================================
# Session Monitoring API Endpoints
# 세션 모니터링 API 엔드포인트
# ============================================================================

@app.route("/api/sessions", methods=["GET"])
def get_all_sessions() -> Tuple:
    """
    Get list of all active sessions for monitoring
    모니터링을 위한 모든 활성 세션 목록 가져오기

    Returns:
        JSON with list of session info / 세션 정보 목록이 포함된 JSON
    """
    try:
        sessions = session_manager.get_all_sessions_info()
        return jsonify(
            {
                "status": "success",
                "sessions": sessions,
                "count": len(sessions)
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/sessions/<session_id>/logs", methods=["GET"])
def get_session_logs_api(session_id: str) -> Tuple:
    """
    Get logs from a specific session for monitoring (read-only)
    모니터링을 위한 특정 세션의 로그 가져오기 (읽기 전용)

    Args:
        session_id: Session ID to get logs from / 로그를 가져올 세션 ID

    Query parameters:
        offset: Starting index (default 0) / 시작 인덱스
        limit: Max logs to return (default 100) / 반환할 최대 로그 수

    Returns:
        JSON with logs and session info / 로그와 세션 정보가 포함된 JSON
    """
    try:
        offset = int(request.args.get("offset", 0))
        limit = int(request.args.get("limit", 100))

        result = session_manager.get_session_logs(session_id, offset, limit)

        if "error" in result:
            return jsonify({"status": "error", "message": result["error"]}), 404

        return jsonify(
            {
                "status": "success",
                **result
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/sessions/<session_id>/status", methods=["GET"])
def get_session_status_api(session_id: str) -> Tuple:
    """
    Get current status of a specific session
    특정 세션의 현재 상태 가져오기

    Args:
        session_id: Session ID / 세션 ID

    Returns:
        JSON with session status / 세션 상태가 포함된 JSON
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
    print("PseuDRAGON Web Application")
    print("=" * 60)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Database Path: {AppConfig.DB_PATH}")
    print(f"Documents Path: {Settings.DOCS_DIR}")
    print(f"Reports Path: {AppConfig.REPORTS_DIR}")
    print(f"Templates Path: {os.path.join(BASE_DIR, 'templates')}")
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

    app.run(host=AppConfig.HOST, port=AppConfig.PORT, debug=True, use_reloader=False, threaded=True, )
