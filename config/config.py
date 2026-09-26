"""PseuDRAGON Configuration Module

This module manages all configuration settings for the PseuDRAGON framework.

Configuration includes:

- OpenAI API settings
- Model configurations
- Directory paths
- RAG system parameters
- Database configurations
"""


# 표준 라이브러리 import
import os
import ssl
from typing import Optional


# 서드파티 라이브러리 import
import httpx
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables

load_dotenv()


class ModelConfig:
    """Model Configuration

    Defines the AI models used for embeddings and language generation.
    """

    # Embedding Configuration


    # Provider: "openai" or "local"

    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")

    # Model Name for Legal Document RAG

    # OpenAI: "text-embedding-3-small"
    # Legal-BERT (English): "nlpaueb/legal-bert-base-uncased"
    # Ko-Legal-SBERT (Korean, default): "woong0322/ko-legal-sbert-finetuned"
    LEGAL_EMBEDDING_MODEL = os.getenv(
        "LEGAL_EMBEDDING_MODEL",
        "woong0322/ko-legal-sbert-finetuned",
    )

    # Model Name for PII Classification Similarity Search

    # General-purpose models are better for database column description matching

    #
    # Note: Legal-BERT produces high false positives for PII classification similarity search
    # because it finds superficial structural similarities rather than semantic PII-related meanings.


    FEEDBACK_EMBEDDING_MODEL = os.getenv("FEEDBACK_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # LLM Settings
    DEFAULT_LLM = os.getenv("DEFAULT_LLM", "gpt-oss:120b")
    STAGE_1_LLM = os.getenv("STAGE1_LLM", DEFAULT_LLM)
    STAGE_2_LLM = os.getenv("STAGE_2_LLM", DEFAULT_LLM)
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))


class DirectoryConfig:
    """Directory Configuration

    Defines paths for documentation, output, and other directories.
    """

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DOCS_DIR = os.path.join(BASE_DIR, "resources", "legal_documents")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    SAMPLE_DB_DIR = os.path.join(BASE_DIR, "resources", "sample_db")

    @classmethod
    def ensure_directories_exist(cls) -> None:
        """Create necessary directories if they don't exist"""
        os.makedirs(cls.DOCS_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.SAMPLE_DB_DIR, exist_ok=True)


class HeuristicConfig:
    """Heuristic Pattern Configuration for Stage 1 PII Detection

    Allows domain experts to customize PII detection patterns for their organization.
    """

    @classmethod
    def get_heuristic_patterns(cls) -> dict:
        """Load heuristic patterns from environment variables.

        Returns:
            Dictionary mapping pattern names to regex patterns
        """
        patterns = {}
        prefix = "HEURISTIC_PATTERN_"

        for key, value in os.environ.items():
            if key.startswith(prefix) and value.strip():
                pattern_name = key[len(prefix):].lower()
                patterns[pattern_name] = value.strip()

        return patterns

    @classmethod
    def get_non_pii_patterns(cls) -> dict:
        """Load Non-PII heuristic patterns from environment variables.

        These patterns identify columns that are clearly NOT PII,
        allowing the system to skip expensive LLM verification.

        Returns:
            Dictionary mapping pattern names to regex patterns
        """
        patterns = {}
        prefix = "HEURISTIC_NON_PII_"

        for key, value in os.environ.items():
            if key.startswith(prefix) and value.strip():
                pattern_name = key[len(prefix):].lower()
                patterns[pattern_name] = value.strip()

        return patterns

    @classmethod
    def get_pii_type_keywords(cls) -> dict:
        """Load PII type inference keywords from environment variables.

        Returns:
            Dictionary mapping PII types to lists of keywords
        """
        keywords = {}
        prefix = "PII_TYPE_"
        suffix = "_KEYWORDS"

        for key, value in os.environ.items():
            if key.startswith(prefix) and key.endswith(suffix) and value.strip():
                type_name = key[len(prefix):-len(suffix)]
                # Format type name: replace underscores with spaces and title case

                type_name = type_name.replace("_", " ").title()
                keywords[type_name] = [kw.strip() for kw in value.split(",") if kw.strip()]

        return keywords


class DatabaseConfig:
    """Database Configuration

    Manages database connection settings for both sample and production databases.
    """

    SAMPLE_DB_PATH = os.path.join(
        DirectoryConfig.SAMPLE_DB_DIR,
        "institution_a_synthetic_schema.duckdb",
    )

    USE_SAMPLE_DB = os.getenv("USE_SAMPLE_DB", "true").lower() == "true"

    DB_TYPE = os.getenv("DB_TYPE", "duckdb")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "3306")
    DB_NAME = os.getenv("DB_NAME", "")
    DB_USER = os.getenv("DB_USER", "")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")

    @classmethod
    def get_db_path(cls) -> str:
        """Get the appropriate database path based on configuration

        Returns:
            Database path or connection string
        """
        if cls.USE_SAMPLE_DB:
            return cls.SAMPLE_DB_PATH
        else:
            return {
                "type": cls.DB_TYPE,
                "host": cls.DB_HOST,
                "port": cls.DB_PORT,
                "database": cls.DB_NAME,
                "user": cls.DB_USER,
                "password": cls.DB_PASSWORD,
            }

    @classmethod
    def validate_config(cls) -> bool:
        """Validate database configuration

        Returns:
            True if configuration is valid
        """
        if cls.USE_SAMPLE_DB:
            if not os.path.exists(cls.SAMPLE_DB_PATH):
                print(f"[WARN] Sample database not found at {cls.SAMPLE_DB_PATH}")
                print("   Please run setup_test_db_duckdb_for_1st_loop.py to create it.")
                print(f"   경고: 샘플 데이터베이스를 찾을 수 없습니다: {cls.SAMPLE_DB_PATH}")
                print("   setup_test_db_duckdb_for_1st_loop.py를 실행하여 생성하세요.")
                return False
            return True
        else:
            if not cls.DB_NAME or not cls.DB_USER:
                print("[WARN] Production database configuration incomplete")
                print("   Please set DB_NAME and DB_USER in .env file")
                print("   경고: 프로덕션 데이터베이스 설정이 불완전합니다")
                print("   .env 파일에 DB_NAME과 DB_USER를 설정하세요")
                return False
            return True


class PolicyConfig:
    """Policy Configuration

    Defines default actions for PII and Non-PII columns.
    """

    # Default action for PII columns (identified as containing personal information)

    DEFAULT_PII_ACTION = "DELETE"

    # Default action for Non-PII columns (identified as not containing personal information)

    DEFAULT_NON_PII_ACTION = "KEEP"

    # Available action types

    AVAILABLE_ACTIONS = [
        "KEEP",  # Keep the data as-is
        "DELETE",  # Remove the column
        "HASH",  # Hash the values
        "MASK",  # Mask the values
        "TOKENIZE",  # Tokenize the values
        "GENERALIZE",  # Generalize the values
        "ENCRYPT",  # Encrypt the values
    ]

    @classmethod
    def validate_action(cls, action: str) -> bool:
        """Validate if an action is available

        Args:
            action: Action type to validate

        Returns:
            True if action is valid
        """
        return action.upper() in cls.AVAILABLE_ACTIONS

    @classmethod
    def get_default_action(cls, is_pii: bool) -> str:
        """Get default action based on PII classification

        Args:
            is_pii: Whether the column is classified as PII

        Returns:
            Default action type
        """
        return cls.DEFAULT_PII_ACTION if is_pii else cls.DEFAULT_NON_PII_ACTION


class RAGConfig:
    """RAG System Configuration

    Defines parameters for the Retrieval-Augmented Generation system.
    """

    # RAG System Enable/Disable Switch

    # When disabled, Stage 1 PII detection will use LLM-only mode without RAG context

    RAG_ENABLED = os.getenv("RAG_ENABLED", "true").lower() == "true"

    # Heuristics Enable/Disable Switch

    # When disabled, Stage 1 skips all heuristic pattern matching (learned + manual)

    # This includes: HeuristicManager patterns, ExpertPreferenceManager, env-based patterns

    HEURISTICS_ENABLED = os.getenv("HEURISTICS_ENABLED", "true").lower() == "true"

    # Defaults match the configuration reported in the paper. Environment
    # variables may still override them for a separately documented run.
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "128"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "30"))
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))
    MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "30000"))

    # Feedback document settings for SBERT embedding

    # SBERT (all-MiniLM-L6-v2) supports max 256 tokens, ~200 words is safe
    FEEDBACK_MAX_LENGTH = int(os.getenv("FEEDBACK_MAX_LENGTH", "200"))  # Max characters per feedback doc

    # Retained only for explicitly selected legacy runs and their reproduction.

    DEFAULT_AUTHORITY_WEIGHTS = {
        0: 1.0,  # Approved expert feedback
        1: 0.8,  # Institutional policy
        2: 0.6,  # Applicable national law/guidance
        3: 0.4,  # Applicable international regulation/guidance
    }

    @staticmethod
    def _read_unit_interval(
            env_name: str,
            default: float,
    ) -> float:
        """Read and validate a score weight in the closed interval [0, 1]."""
        raw_value = os.getenv(env_name)
        if raw_value is None:
            raw_value = str(default)

        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{env_name} must be a number between 0 and 1; got {raw_value!r}"
            ) from exc

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{env_name} must be between 0 and 1; got {value}"
            )

        return value

    @classmethod
    def get_semantic_similarity_weight(cls) -> float:
        """Return alpha in score = alpha * similarity + (1-alpha) * authority."""
        return cls._read_unit_interval(
            "SEMANTIC_SIMILARITY_WEIGHT",
            0.7,
        )

    @classmethod
    def get_authority_weights(cls) -> dict:
        """Return the audited source-authority hierarchy used for reranking."""
        return {
            0: cls._read_unit_interval(
                "EXPERT_FEEDBACK_WEIGHT",
                cls.DEFAULT_AUTHORITY_WEIGHTS[0],
            ),
            1: cls._read_unit_interval(
                "INTERNAL_POLICY_WEIGHT",
                cls.DEFAULT_AUTHORITY_WEIGHTS[1],
            ),
            2: cls._read_unit_interval(
                "NATIONAL_LAW_WEIGHT",
                cls.DEFAULT_AUTHORITY_WEIGHTS[2],
            ),
            3: cls._read_unit_interval(
                "INTERNATIONAL_REGULATIONS_WEIGHT",
                cls.DEFAULT_AUTHORITY_WEIGHTS[3],
            ),
        }

    @classmethod
    def get_retrieval_mode(cls) -> str:
        mode = os.getenv("RAG_RETRIEVAL_MODE", "source_order").strip().lower()
        if mode not in {"source_order", "legacy", "lexical"}:
            raise ValueError("RAG_RETRIEVAL_MODE must be source_order, legacy or lexical")
        return mode

    @classmethod
    def get_retrieval_config(cls) -> dict:
        """Describe the active route without reading irrelevant legacy weights."""
        mode = cls.get_retrieval_mode()
        if mode == "source_order":
            return {
                "retrieval_mode": mode,
                "source_order": [1, 2, 3],
                "ranking": "description_bm25_with_adjacent_source_windows",
                "stop_condition": "validated_relevance_selection",
                "automatic_treatment_approval": False,
                "legal_applicability_review_required": True,
            }
        return {"retrieval_mode": mode, **cls.get_ranking_config()}

    @classmethod
    def get_lexical_source_gap(cls) -> float:
        return cls._read_unit_interval("RAG_LEXICAL_SOURCE_GAP", 0.0)

    @classmethod
    def get_ranking_config(cls) -> dict:
        """Describe the retained legacy score. Use get_retrieval_config for the
        active route; source_order does not use these coefficients.
        """
        alpha = cls.get_semantic_similarity_weight()
        return {
            "semantic_similarity_weight": alpha,
            "authority_mixing_weight": round(1.0 - alpha, 10),
            "authority_source_weights": cls.get_authority_weights(),
        }


class Settings:
    """Main Configuration Class

    Aggregates all configuration settings and provides initialization.
    """

    # Prompt Mode (full or lite)
    PROMPT_MODE: str = os.getenv("PROMPT_MODE", "full")

    # Streaming Configuration

    ENABLE_STREAMING: bool = os.getenv("ENABLE_STREAMING", "true").lower() == "true"
    STREAMING_LOG_CHUNKS: bool = os.getenv("STREAMING_LOG_CHUNKS", "false").lower() == "true"
    TLS_VERIFY: bool = os.getenv("TLS_VERIFY", "true").lower() == "true"
    TLS_CA_BUNDLE: Optional[str] = os.getenv("TLS_CA_BUNDLE")

    # Local audit cache for the exact LLM prompts and responses used in a run.

    LLM_AUDIT_ENABLED: bool = os.getenv("LLM_AUDIT_ENABLED", "true").lower() == "true"
    LLM_AUDIT_DIR: str = os.getenv(
        "LLM_AUDIT_DIR",
        os.path.join(DirectoryConfig.OUTPUT_DIR, "llm_audit"),
    )

    # Batch Processing Configuration for LLM calls

    LLM_BATCH_SIZE: int = int(os.getenv("LLM_BATCH_SIZE", "4"))  # Number of columns to process in parallel
    MAX_CONCURRENT_LLM_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_LLM_REQUESTS", "4"))  # Rate limit semaphore
    LLM_PARSE_MAX_ATTEMPTS: int = max(
        1, int(os.getenv("LLM_PARSE_MAX_ATTEMPTS", "3"))
    )
    STAGE_1_MAX_TOKENS: int = max(
        1, int(os.getenv("STAGE_1_MAX_TOKENS", "4096"))
    )

    EMBEDDING_OPENAI_API_KEY = os.getenv("EMBEDDING_OPENAI_API_KEY")

    DEFAULT_LLM_OPENAI_API_KEY = os.getenv("DEFAULT_LLM_OPENAI_API_KEY")
    DEFAULT_LLM_OPENAI_BASE_URL = os.getenv("DEFAULT_LLM_OPENAI_BASE_URL")

    STAGE_1_LLM_OPENAI_API_KEY = os.getenv("STAGE_1_LLM_OPENAI_API_KEY")
    STAGE_1_OPENAI_BASE_URL = os.getenv("STAGE_1_OPENAI_BASE_URL")

    STAGE_2_LLM_OPENAI_API_KEY = os.getenv("STAGE_2_LLM_OPENAI_API_KEY")
    STAGE_2_OPENAI_BASE_URL = os.getenv("STAGE_2_OPENAI_BASE_URL")

    EMBEDDING_PROVIDER = ModelConfig.EMBEDDING_PROVIDER
    LEGAL_EMBEDDING_MODEL = ModelConfig.LEGAL_EMBEDDING_MODEL
    FEEDBACK_EMBEDDING_MODEL = ModelConfig.FEEDBACK_EMBEDDING_MODEL  # For PII classification similarity search
    DEFAULT_LLM = ModelConfig.DEFAULT_LLM
    LLM_STAGE_1 = ModelConfig.STAGE_1_LLM
    LLM_STAGE_2 = ModelConfig.STAGE_2_LLM
    LLM_TEMPERATURE = ModelConfig.LLM_TEMPERATURE

    BASE_DIR = DirectoryConfig.BASE_DIR
    DOCS_DIR = DirectoryConfig.DOCS_DIR
    OUTPUT_DIR = DirectoryConfig.OUTPUT_DIR

    USE_SAMPLE_DB = DatabaseConfig.USE_SAMPLE_DB
    SAMPLE_DB_PATH = DatabaseConfig.SAMPLE_DB_PATH
    DB_TYPE = DatabaseConfig.DB_TYPE

    RAG_ENABLED = RAGConfig.RAG_ENABLED
    HEURISTICS_ENABLED = RAGConfig.HEURISTICS_ENABLED
    CHUNK_SIZE = RAGConfig.CHUNK_SIZE
    CHUNK_OVERLAP = RAGConfig.CHUNK_OVERLAP
    TOP_K_RESULTS = RAGConfig.TOP_K_RESULTS
    MAX_CONTEXT_TOKENS = RAGConfig.MAX_CONTEXT_TOKENS
    FEEDBACK_MAX_LENGTH = RAGConfig.FEEDBACK_MAX_LENGTH

    DEFAULT_PII_ACTION = PolicyConfig.DEFAULT_PII_ACTION
    DEFAULT_NON_PII_ACTION = PolicyConfig.DEFAULT_NON_PII_ACTION

    @classmethod
    def validate_api_key(cls) -> bool:
        """Check if OpenAI API key is set

        Returns:
            True if API key is set, False otherwise
        """
        isTrue = True
        isTrue &= cls.EMBEDDING_OPENAI_API_KEY is not None and cls.EMBEDDING_OPENAI_API_KEY != ""
        isTrue &= cls.DEFAULT_LLM_OPENAI_API_KEY is not None and cls.DEFAULT_LLM_OPENAI_API_KEY != ""
        isTrue &= cls.STAGE_1_LLM_OPENAI_API_KEY is not None and cls.STAGE_1_LLM_OPENAI_API_KEY != ""
        isTrue &= cls.STAGE_2_LLM_OPENAI_API_KEY is not None and cls.STAGE_2_LLM_OPENAI_API_KEY != ""

        return isTrue

    @staticmethod
    def get_prompt_path(stage: str, prompt_type: str) -> str:
        """Get prompt file path based on PROMPT_MODE setting

        Args:
            stage: "stage1" or "stage2"
            prompt_type: "system" or "user"

        Returns:
            Path to the appropriate prompt file
        """
        base_path = "resources/prompts"
        mode_suffix = "_lite" if Settings.PROMPT_MODE == "lite" else ""

        prompt_files = {
            ("stage1", "system"): f"stage1_pii_identification_system{mode_suffix}.txt",
            ("stage1", "user"): f"stage1_pii_identification_user.txt",
            ("stage2", "system"): f"stage2_policy_synthesis_system{mode_suffix}.txt",
            ("stage2", "user"): f"stage2_policy_synthesis_user.txt",
        }

        return os.path.join(base_path, prompt_files.get((stage, prompt_type), ""))

    @classmethod
    def initialize(cls) -> None:
        """Initialize configuration

        Creates necessary directories and validates settings.
        """
        DirectoryConfig.ensure_directories_exist()

        if not cls.validate_api_key():
            print(
                "[WARN] OPENAI_API_KEY not found in environment variables.\n"
                "   Please set it in your .env file or environment.\n"
                "   경고: 환경 변수에서 OPENAI_API_KEY를 찾을 수 없습니다.\n"
                "   .env 파일 또는 환경에 설정하세요."
            )

        DatabaseConfig.validate_config()


class OpenAIClientFactory:
    """Factory for creating OpenAI client instances"""

    @staticmethod
    def create_client(model_type: str) -> Optional[OpenAI]:
        """Create an OpenAI client

        Args:
            model_type: Client type (EMBEDDING_CLIENT, DEFAULT_LLM_CLIENT, etc.)

        Returns:
            OpenAI client instance or None if API key is not set
        """
        verify: object = Settings.TLS_VERIFY
        if Settings.TLS_CA_BUNDLE:
            verify = ssl.create_default_context(
                cafile=Settings.TLS_CA_BUNDLE,
            )

        if model_type == "EMBEDDING_CLIENT":
            if not Settings.EMBEDDING_OPENAI_API_KEY:
                print("[WARN] Cannot create OpenAI client - API key not set.")
                return None

            return OpenAI(
                api_key=Settings.EMBEDDING_OPENAI_API_KEY,
                http_client=httpx.Client(verify=verify),
            )

        elif model_type == "DEFAULT_LLM_CLIENT":
            if not Settings.DEFAULT_LLM_OPENAI_API_KEY:
                print("[WARN] Cannot create OpenAI client - API key not set.")
                return None

            return OpenAI(
                api_key=Settings.DEFAULT_LLM_OPENAI_API_KEY,
                base_url=Settings.DEFAULT_LLM_OPENAI_BASE_URL if Settings.DEFAULT_LLM_OPENAI_BASE_URL else None,
                http_client=httpx.Client(verify=verify),
            )

        elif model_type == "LLM_CLIENT_STAGE_1":
            if not Settings.STAGE_1_LLM_OPENAI_API_KEY:
                print("[WARN] Cannot create OpenAI client - API key not set.")
                return None

            return OpenAI(
                api_key=Settings.STAGE_1_LLM_OPENAI_API_KEY,
                base_url=Settings.STAGE_1_OPENAI_BASE_URL if Settings.STAGE_1_OPENAI_BASE_URL else None,
                http_client=httpx.Client(verify=verify),
            )

        elif model_type == "LLM_CLIENT_STAGE_2":
            if not Settings.STAGE_2_LLM_OPENAI_API_KEY:
                print("[WARN] Cannot create OpenAI client - API key not set.")
                return None

            return OpenAI(
                api_key=Settings.STAGE_2_LLM_OPENAI_API_KEY,
                base_url=Settings.STAGE_2_OPENAI_BASE_URL if Settings.STAGE_2_OPENAI_BASE_URL else None,
                http_client=httpx.Client(verify=verify),
            )

        return None


# Initialize settings and create client instances

Settings.initialize()
EMBEDDING_CLIENT = OpenAIClientFactory.create_client("EMBEDDING_CLIENT")
DEFAULT_LLM_CLIENT = OpenAIClientFactory.create_client("DEFAULT_LLM_CLIENT")
LLM_CLIENT_STAGE_1 = OpenAIClientFactory.create_client("LLM_CLIENT_STAGE_1")
LLM_CLIENT_STAGE_2 = OpenAIClientFactory.create_client("LLM_CLIENT_STAGE_2")
