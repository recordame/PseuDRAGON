"""PseuDRAGON File Logger

Provides file-based logging with timestamp-based log file creation.
"""

import logging
import os
from datetime import datetime
from typing import Optional


class PseuDRAGONLogger:
    """Centralized logger for PseuDRAGON pipeline

    Features:

    - Console output with emoji support
    - File logging with timestamp-based filenames
    - Stage timing measurement
    """

    _instance: Optional["PseuDRAGONLogger"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        """Singleton pattern / 싱글톤 패턴"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, log_dir: Optional[str] = None, session_id: Optional[str] = None):
        """Initialize the logger

        Args:
            log_dir: Directory for log files
            session_id: Optional session identifier
        """
        if PseuDRAGONLogger._initialized:
            return

        # Determine log directory (default: code/logs)

        if log_dir is None:
            # code/pseudragon/logging -> code
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            log_dir = os.path.join(base_dir, "logs")

        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # Generate session ID if not provided
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = session_id

        # Create log filename with timestamp
        log_filename = f"pseudragon_{session_id}.log"
        self.log_file_path = os.path.join(self.log_dir, log_filename)

        # Setup logging
        self._setup_logging()

        # Stage timing tracking
        self._stage_start_times: dict[str, float] = {}
        self._stage_durations: dict[str, float] = {}

        # Table statistics tracking

        self._table_stats: dict[str, dict] = {}

        PseuDRAGONLogger._initialized = True

    def _setup_logging(self) -> None:
        """Configure logging handlers"""
        # Create logger
        self.logger = logging.getLogger("PseuDRAGON")
        self.logger.setLevel(logging.DEBUG)

        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Create formatters
        file_formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        console_formatter = logging.Formatter("%(message)s")

        # File handler - UTF-8 encoding for emoji support
        file_handler = logging.FileHandler(self.log_file_path, encoding="utf-8", mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler - DISABLED to prevent duplicate logging

        # The web interface's log_producer() already handles console output

        # console_handler = logging.StreamHandler(sys.stdout)
        # console_handler.setLevel(logging.INFO)
        # console_handler.setFormatter(console_formatter)
        # self.logger.addHandler(console_handler)

        # Log initialization
        self.logger.info(f"=" * 80)
        self.logger.info(f"PseuDRAGON Session Started: {self.session_id}")
        self.logger.info(f"Log file: {self.log_file_path}")
        self.logger.info(f"=" * 80)

    def reinitialize(self, session_id: Optional[str] = None) -> None:
        """Reinitialize logger with new session

        Args:
            session_id: New session identifier
        """
        PseuDRAGONLogger._initialized = False

        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = session_id

        # Create new log filename
        log_filename = f"pseudragon_{session_id}.log"
        self.log_file_path = os.path.join(self.log_dir, log_filename)

        # Clear existing handlers
        if hasattr(self, 'logger'):
            self.logger.handlers.clear()

        # Reset timing and stats
        self._stage_start_times = {}
        self._stage_durations = {}
        self._table_stats = {}

        # Setup logging again
        self._setup_logging()

        PseuDRAGONLogger._initialized = True

    def info(self, message: str) -> None:
        """Log info message / INFO 메시지 로깅"""
        self.logger.info(message)

    def debug(self, message: str) -> None:
        """Log debug message / DEBUG 메시지 로깅"""
        self.logger.debug(message)

    def warning(self, message: str) -> None:
        """Log warning message / WARNING 메시지 로깅"""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message / ERROR 메시지 로깅"""
        self.logger.error(message)

    def stage_start(self, stage_name: str) -> None:
        """Record stage start time

        Args:
            stage_name: Name of the pipeline stage
        """
        import time
        self._stage_start_times[stage_name] = time.time()
        self.info(f"[TIMER START] {stage_name}")

    def stage_end(self, stage_name: str) -> float:
        """Record stage end time and calculate duration

        Args:
            stage_name: Name of the pipeline stage

        Returns:
            Duration in seconds
        """
        import time
        end_time = time.time()

        if stage_name in self._stage_start_times:
            duration = end_time - self._stage_start_times[stage_name]
            self._stage_durations[stage_name] = duration

            # Format duration
            if duration < 60:
                duration_str = f"{duration:.2f} seconds"
            elif duration < 3600:
                minutes = int(duration // 60)
                seconds = duration % 60
                duration_str = f"{minutes}m {seconds:.2f}s"
            else:
                hours = int(duration // 3600)
                minutes = int((duration % 3600) // 60)
                seconds = duration % 60
                duration_str = f"{hours}h {minutes}m {seconds:.2f}s"

            self.info(f"[TIMER END] {stage_name} - Duration: {duration_str}")
            return duration
        else:
            self.warning(f"[TIMER END] {stage_name} - Start time not found")
            return 0.0

    def get_stage_duration(self, stage_name: str) -> Optional[float]:
        """Get duration for a completed stage

        Args:
            stage_name: Name of the pipeline stage

        Returns:
            Duration in seconds or None
        """
        return self._stage_durations.get(stage_name)

    def get_all_stage_durations(self) -> dict[str, float]:
        """Get all recorded stage durations

        Returns:
            Dictionary of stage names to durations
        """
        return self._stage_durations.copy()

    def log_table_stats(self, table_name: str, total_columns: int, pii_columns: int) -> None:
        """Record table statistics for summary

        Args:
            table_name: Name of the table
            total_columns: Total number of columns
            pii_columns: Number of PII columns identified
        """
        self._table_stats[table_name] = {"total_columns": total_columns, "pii_columns": pii_columns, "non_pii_columns": total_columns - pii_columns}
        self.info(f"[TABLE STATS] {table_name}: {total_columns} columns total, {pii_columns} PII columns")

    def get_table_stats(self, table_name: str) -> Optional[dict]:
        """Get statistics for a specific table

        Args:
            table_name: Name of the table

        Returns:
            Dictionary with table statistics or None
        """
        return self._table_stats.get(table_name)

    def get_all_table_stats(self) -> dict[str, dict]:
        """Get all recorded table statistics

        Returns:
            Dictionary of table names to statistics
        """
        return self._table_stats.copy()

    def log_summary(self) -> None:
        """Log summary of all stage durations and table statistics"""
        self.info("=" * 80)
        self.info("PIPELINE EXECUTION SUMMARY")
        self.info("=" * 80)

        # Table Statistics Section

        if self._table_stats:
            self.info("")
            self.info("[TABLE STATISTICS]")
            self.info("-" * 40)

            total_tables = len(self._table_stats)
            total_all_columns = 0
            total_pii_columns = 0

            for table_name, stats in self._table_stats.items():
                total_cols = stats["total_columns"]
                pii_cols = stats["pii_columns"]
                non_pii_cols = stats["non_pii_columns"]
                pii_ratio = (pii_cols / total_cols * 100) if total_cols > 0 else 0

                self.info(f"  {table_name}:")
                self.info(f"    - Total Columns: {total_cols}")
                self.info(f"    - PII Columns: {pii_cols} ({pii_ratio:.1f}%)")
                self.info(f"    - Non-PII Columns: {non_pii_cols}")

                total_all_columns += total_cols
                total_pii_columns += pii_cols

            if total_tables > 1:
                self.info("-" * 40)
                overall_pii_ratio = (total_pii_columns / total_all_columns * 100) if total_all_columns > 0 else 0
                self.info(f"  OVERALL ({total_tables} tables):")
                self.info(f"    - Total Columns: {total_all_columns}")
                self.info(f"    - Total PII Columns: {total_pii_columns} ({overall_pii_ratio:.1f}%)")

        # Stage Timing Section

        self.info("")
        self.info("[STAGE TIMING]")
        self.info("-" * 40)

        total_duration = 0.0
        for stage_name, duration in self._stage_durations.items():
            total_duration += duration

            if duration < 60:
                duration_str = f"{duration:.2f}s"
            elif duration < 3600:
                minutes = int(duration // 60)
                seconds = duration % 60
                duration_str = f"{minutes}m {seconds:.2f}s"
            else:
                hours = int(duration // 3600)
                minutes = int((duration % 3600) // 60)
                seconds = duration % 60
                duration_str = f"{hours}h {minutes}m {seconds:.2f}s"

            self.info(f"  {stage_name}: {duration_str}")

        self.info("-" * 40)

        if total_duration < 60:
            total_str = f"{total_duration:.2f}s"
        elif total_duration < 3600:
            minutes = int(total_duration // 60)
            seconds = total_duration % 60
            total_str = f"{minutes}m {seconds:.2f}s"
        else:
            hours = int(total_duration // 3600)
            minutes = int((total_duration % 3600) // 60)
            seconds = total_duration % 60
            total_str = f"{hours}h {minutes}m {seconds:.2f}s"

        self.info(f"  TOTAL EXECUTION TIME: {total_str}")
        self.info("=" * 80)

    def get_log_file_path(self) -> str:
        """Get the current log file path

        Returns:
            Path to the log file
        """
        return self.log_file_path


# Global logger instance

_logger: Optional[PseuDRAGONLogger] = None


def get_logger(log_dir: Optional[str] = None, session_id: Optional[str] = None) -> PseuDRAGONLogger:
    """Get or create the global logger instance

    Args:
        log_dir: Directory for log files
        session_id: Optional session identifier

    Returns:
        PseuDRAGONLogger instance
    """
    global _logger
    if _logger is None:
        _logger = PseuDRAGONLogger(log_dir=log_dir, session_id=session_id)
    return _logger


def reinitialize_logger(session_id: Optional[str] = None) -> PseuDRAGONLogger:
    """Reinitialize the global logger with a new session

    Args:
        session_id: New session identifier

    Returns:
        PseuDRAGONLogger instance
    """
    global _logger
    if _logger is None:
        _logger = PseuDRAGONLogger(session_id=session_id)
    else:
        _logger.reinitialize(session_id=session_id)
    return _logger
