"""
PseuDRAGON Configuration Module
PseuDRAGON 설정 모듈

This module manages all configuration settings for the PseuDRAGON framework.
이 모듈은 PseuDRAGON 프레임워크의 모든 설정을 관리합니다.

Configuration includes:
설정 항목:
- OpenAI API settings / OpenAI API 설정
- Model configurations / 모델 설정
- Directory paths / 디렉토리 경로
- RAG system parameters / RAG 시스템 매개변수
- Database configurations / 데이터베이스 설정
"""

# Standard library imports
# 표준 라이브러리 import
import os
import ssl
from typing import Optional

# Third-party imports
# 서드파티 라이브러리 import
import httpx
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
# 환경 변수 로드
load_dotenv()


class ModelConfig:
    """
    Model Configuration
    모델 설정

    Defines the AI models used for embeddings and language generation.
    임베딩 및 언어 생성에 사용되는 AI 모델을 정의합니다.
    """

    # Embedding Configuration
    # 임베딩 설정

    # Provider: "openai" or "local"
    # 공급자: "openai" 또는 "local"
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")

    # Model Name for Legal Document RAG
    # 법률 문서 RAG용 모델명
    # OpenAI: "text-embedding-3-small"
    # Legal-BERT (English): "nlpaueb/legal-bert-base-uncased" (Recommended for legal docs)
    # Ko-Legal-BERT (Korean): "woong0322/ko-legal-sbert-finetuned"
    LEGAL_EMBEDDING_MODEL = os.getenv("LEGAL_EMBEDDING_MODEL", "nlpaueb/legal-bert-base-uncased")

    # Model Name for PII Classification Similarity Search
    # PII 분류 유사도 검색용 모델명
    # General-purpose models are better for database column description matching
    # 범용 모델이 데이터베이스 컬럼 설명 매칭에 더 적합합니다
    #
    # Note: Legal-BERT produces high false positives for PII classification similarity search
    # because it finds superficial structural similarities rather than semantic PII-related meanings.
    # 참고: Legal-BERT는 PII 분류 유사도 검색에서 높은 거짓양성률을 보입니다.
    # 이는 실제 의미적 PII 관련 의미가 아닌 표면적인 구조적 유사성을 찾기 때문입니다.
    FEEDBACK_EMBEDDING_MODEL = os.getenv("FEEDBACK_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # LLM Settings
    DEFAULT_LLM = os.getenv("DEFAULT_LLM", "gpt-4o")
    STAGE_1_LLM = os.getenv("STAGE1_LLM", DEFAULT_LLM)
    STAGE_2_LLM = os.getenv("STAGE_2_LLM", DEFAULT_LLM)


class DirectoryConfig:
    """
    Directory Configuration
    디렉토리 설정

    Defines paths for documentation, output, and other directories.
    문서, 출력 및 기타 디렉토리의 경로를 정의합니다.
    """

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DOCS_DIR = os.path.join(BASE_DIR, "resources", "legal_documents")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    SAMPLE_DB_DIR = os.path.join(BASE_DIR, "resources", "sample_db")

    @classmethod
    def ensure_directories_exist(cls) -> None:
        """
        Create necessary directories if they don't exist
        필요한 디렉토리가 없으면 생성합니다
        """
        os.makedirs(cls.DOCS_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.SAMPLE_DB_DIR, exist_ok=True)


class HeuristicConfig:
    """
    Heuristic Pattern Configuration for Stage 1 PII Detection
    Stage 1 PII 탐지를 위한 휴리스틱 패턴 설정

    Allows domain experts to customize PII detection patterns for their organization.
    도메인 전문가가 조직에 맞는 PII 탐지 패턴을 커스터마이징할 수 있습니다.
    """

    @classmethod
    def get_heuristic_patterns(cls) -> dict:
        """
        Load heuristic patterns from environment variables.
        환경 변수에서 휴리스틱 패턴을 로드합니다.

        Returns:
            Dictionary mapping pattern names to regex patterns
            패턴 이름을 정규식 패턴에 매핑하는 딕셔너리
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
        """
        Load Non-PII heuristic patterns from environment variables.
        환경 변수에서 Non-PII 휴리스틱 패턴을 로드합니다.
        
        These patterns identify columns that are clearly NOT PII,
        allowing the system to skip expensive LLM verification.
        이 패턴들은 명확하게 PII가 아닌 컬럼을 식별하여,
        비용이 많이 드는 LLM 검증을 건너뛸 수 있게 합니다.
        
        Returns:
            Dictionary mapping pattern names to regex patterns
            패턴 이름을 정규식 패턴에 매핑하는 딕셔너리
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
        """
        Load PII type inference keywords from environment variables.
        환경 변수에서 PII 유형 추론 키워드를 로드합니다.

        Returns:
            Dictionary mapping PII types to lists of keywords
            PII 유형을 키워드 리스트에 매핑하는 딕셔너리
        """
        keywords = {}
        prefix = "PII_TYPE_"
        suffix = "_KEYWORDS"

        for key, value in os.environ.items():
            if key.startswith(prefix) and key.endswith(suffix) and value.strip():
                type_name = key[len(prefix):-len(suffix)]
                # Format type name: replace underscores with spaces and title case
                # 유형명 포맷: 언더스코어를 공백으로 변환하고 제목 케이스 적용
                type_name = type_name.replace("_", " ").title()
                keywords[type_name] = [kw.strip() for kw in value.split(",") if kw.strip()]

        return keywords


class DatabaseConfig:
    """
    Database Configuration
    데이터베이스 설정

    Manages database connection settings for both sample and production databases.
    샘플 및 프로덕션 데이터베이스의 연결 설정을 관리합니다.
    """

    SAMPLE_DB_PATH = os.path.join(DirectoryConfig.SAMPLE_DB_DIR, "KFTC_sample_table_schemas.duckdb")

    USE_SAMPLE_DB = os.getenv("USE_SAMPLE_DB", "true").lower() == "true"

    DB_TYPE = os.getenv("DB_TYPE", "duckdb")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "3306")
    DB_NAME = os.getenv("DB_NAME", "")
    DB_USER = os.getenv("DB_USER", "")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")

    @classmethod
    def get_db_path(cls) -> str:
        """
        Get the appropriate database path based on configuration
        설정에 따라 적절한 데이터베이스 경로를 가져옵니다

        Returns:
            Database path or connection string
            데이터베이스 경로 또는 연결 문자열
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
        """
        Validate database configuration
        데이터베이스 설정 검증

        Returns:
            True if configuration is valid
            설정이 유효하면 True
        """
        if cls.USE_SAMPLE_DB:
            if not os.path.exists(cls.SAMPLE_DB_PATH):
                print(f"[WARN] Sample database not found at {cls.SAMPLE_DB_PATH}")
                print("   Please run setup_test_db_duckdb.py to create it.")
                print(f"   경고: 샘플 데이터베이스를 찾을 수 없습니다: {cls.SAMPLE_DB_PATH}")
                print("   setup_test_db_duckdb.py를 실행하여 생성하세요.")
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
    """
    Policy Configuration
    정책 설정

    Defines default actions for PII and Non-PII columns.
    PII 및 Non-PII 컬럼에 대한 기본 액션을 정의합니다.
    """

    # Default action for PII columns (identified as containing personal information)
    # PII 컬럼에 대한 기본 액션 (개인정보를 포함하는 것으로 식별된 컬럼)
    DEFAULT_PII_ACTION = "DELETE"

    # Default action for Non-PII columns (identified as not containing personal information)
    # Non-PII 컬럼에 대한 기본 액션 (개인정보를 포함하지 않는 것으로 식별된 컬럼)
    DEFAULT_NON_PII_ACTION = "KEEP"

    # Available action types
    # 사용 가능한 액션 타입
    AVAILABLE_ACTIONS = [
        "KEEP",  # Keep the data as-is / 데이터를 그대로 유지
        "DELETE",  # Remove the column / 컬럼 삭제
        "HASH",  # Hash the values / 값을 해시화
        "MASK",  # Mask the values / 값을 마스킹
        "TOKENIZE",  # Tokenize the values / 값을 토큰화
        "GENERALIZE",  # Generalize the values / 값을 일반화
        "ENCRYPT",  # Encrypt the values / 값을 암호화
    ]

    @classmethod
    def validate_action(cls, action: str) -> bool:
        """
        Validate if an action is available
        액션이 사용 가능한지 검증합니다

        Args:
            action: Action type to validate
                   검증할 액션 타입

        Returns:
            True if action is valid
            액션이 유효하면 True
        """
        return action.upper() in cls.AVAILABLE_ACTIONS

    @classmethod
    def get_default_action(cls, is_pii: bool) -> str:
        """
        Get default action based on PII classification
        PII 분류에 따른 기본 액션을 가져옵니다

        Args:
            is_pii: Whether the column is classified as PII
                   컬럼이 PII로 분류되었는지 여부

        Returns:
            Default action type
            기본 액션 타입
        """
        return cls.DEFAULT_PII_ACTION if is_pii else cls.DEFAULT_NON_PII_ACTION


class RAGConfig:
    """
    RAG System Configuration
    RAG 시스템 설정

    Defines parameters for the Retrieval-Augmented Generation system.
    검색 증강 생성 시스템의 매개변수를 정의합니다.
    """

    # RAG System Enable/Disable Switch
    # RAG 시스템 활성화/비활성화 스위치
    # When disabled, Stage 1 PII detection will use LLM-only mode without RAG context
    # 비활성화 시 Stage 1 PII 탐지가 RAG 컨텍스트 없이 LLM만 사용합니다
    RAG_ENABLED = os.getenv("RAG_ENABLED", "true").lower() == "true"

    # Heuristics Enable/Disable Switch
    # 휴리스틱 활성화/비활성화 스위치
    # When disabled, Stage 1 skips all heuristic pattern matching (learned + manual)
    # 비활성화 시 Stage 1이 모든 휴리스틱 패턴 매칭을 건너뜁니다 (학습된 + 수동)
    # This includes: HeuristicManager patterns, ExpertPreferenceManager, env-based patterns
    # 포함 항목: HeuristicManager 패턴, ExpertPreferenceManager, 환경 변수 기반 패턴
    HEURISTICS_ENABLED = os.getenv("HEURISTICS_ENABLED", "true").lower() == "true"

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "2"))
    MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "30000"))

    # Feedback document settings for SBERT embedding
    # SBERT 임베딩을 위한 피드백 문서 설정
    # SBERT (all-MiniLM-L6-v2) supports max 256 tokens, ~200 words is safe
    FEEDBACK_MAX_LENGTH = int(os.getenv("FEEDBACK_MAX_LENGTH", "500"))  # Max characters per feedback doc


class Settings:
    """
    Main Configuration Class
    메인 설정 클래스

    Aggregates all configuration settings and provides initialization.
    모든 설정을 집계하고 초기화를 제공합니다.
    """

    # Prompt Mode (full or lite)
    PROMPT_MODE: str = os.getenv("PROMPT_MODE", "full")

    # Streaming Configuration
    # OpenAI 스트리밍 설정
    ENABLE_STREAMING: bool = os.getenv("ENABLE_STREAMING", "true").lower() == "true"
    STREAMING_LOG_CHUNKS: bool = os.getenv("STREAMING_LOG_CHUNKS", "true").lower() == "true"

    # Batch Processing Configuration for LLM calls
    # LLM 호출을 위한 배치 처리 설정
    LLM_BATCH_SIZE: int = int(os.getenv("LLM_BATCH_SIZE", "4"))  # Number of columns to process in parallel / 병렬 처리할 컬럼 수
    MAX_CONCURRENT_LLM_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_LLM_REQUESTS", "4"))  # Rate limit semaphore / API 호출 제한

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
        """
        Check if OpenAI API key is set
        OpenAI API 키가 설정되어 있는지 확인합니다

        Returns:
            True if API key is set, False otherwise
            API 키가 설정되어 있으면 True, 그렇지 않으면 False
        """
        isTrue = True
        isTrue &= cls.EMBEDDING_OPENAI_API_KEY is not None and cls.EMBEDDING_OPENAI_API_KEY != ""
        isTrue &= cls.DEFAULT_LLM_OPENAI_API_KEY is not None and cls.DEFAULT_LLM_OPENAI_API_KEY != ""
        isTrue &= cls.STAGE_1_LLM_OPENAI_API_KEY is not None and cls.STAGE_1_LLM_OPENAI_API_KEY != ""
        isTrue &= cls.STAGE_2_LLM_OPENAI_API_KEY is not None and cls.STAGE_2_LLM_OPENAI_API_KEY != ""

        return isTrue

    @staticmethod
    def get_prompt_path(stage: str, prompt_type: str) -> str:
        """
        Get prompt file path based on PROMPT_MODE setting
        프롬프트 모드 설정에 따른 프롬프트 파일 경로 반환
        
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
        """
        Initialize configuration
        설정 초기화

        Creates necessary directories and validates settings.
        필요한 디렉토리를 생성하고 설정을 검증합니다.
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
    """
    Factory for creating OpenAI client instances
    OpenAI 클라이언트 인스턴스를 생성하는 팩토리
    """

    @staticmethod
    def create_client(model_type: str) -> Optional[OpenAI]:
        """
        Create an OpenAI client
        OpenAI 클라이언트를 생성합니다

        Args:
            model_type: Client type (EMBEDDING_CLIENT, DEFAULT_LLM_CLIENT, etc.)
                       클라이언트 유형

        Returns:
            OpenAI client instance or None if API key is not set
            OpenAI 클라이언트 인스턴스 또는 API 키가 없으면 None
        """
        # Create SSL context with disabled verification
        # SSL 검증을 비활성화한 컨텍스트 생성
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        if model_type == "EMBEDDING_CLIENT":
            if not Settings.EMBEDDING_OPENAI_API_KEY:
                print("[WARN] Cannot create OpenAI client - API key not set.")
                return None

            return OpenAI(
                api_key=Settings.EMBEDDING_OPENAI_API_KEY,
                http_client=httpx.Client(verify=ssl_context),
            )

        elif model_type == "DEFAULT_LLM_CLIENT":
            if not Settings.DEFAULT_LLM_OPENAI_API_KEY:
                print("[WARN] Cannot create OpenAI client - API key not set.")
                return None

            return OpenAI(
                api_key=Settings.DEFAULT_LLM_OPENAI_API_KEY,
                base_url=Settings.DEFAULT_LLM_OPENAI_BASE_URL if Settings.DEFAULT_LLM_OPENAI_BASE_URL else None,
                http_client=httpx.Client(verify=ssl_context),
            )

        elif model_type == "LLM_CLIENT_STAGE_1":
            if not Settings.STAGE_1_LLM_OPENAI_API_KEY:
                print("[WARN] Cannot create OpenAI client - API key not set.")
                return None

            return OpenAI(
                api_key=Settings.STAGE_1_LLM_OPENAI_API_KEY,
                base_url=Settings.STAGE_1_OPENAI_BASE_URL if Settings.STAGE_1_OPENAI_BASE_URL else None,
                http_client=httpx.Client(verify=ssl_context),
            )

        elif model_type == "LLM_CLIENT_STAGE_2":
            if not Settings.STAGE_2_LLM_OPENAI_API_KEY:
                print("[WARN] Cannot create OpenAI client - API key not set.")
                return None

            return OpenAI(
                api_key=Settings.STAGE_2_LLM_OPENAI_API_KEY,
                base_url=Settings.STAGE_2_OPENAI_BASE_URL if Settings.STAGE_2_OPENAI_BASE_URL else None,
                http_client=httpx.Client(verify=ssl_context),
            )

        return None


# Initialize settings and create client instances
# 설정 초기화 및 클라이언트 인스턴스 생성
Settings.initialize()
EMBEDDING_CLIENT = OpenAIClientFactory.create_client("EMBEDDING_CLIENT")
DEFAULT_LLM_CLIENT = OpenAIClientFactory.create_client("DEFAULT_LLM_CLIENT")
LLM_CLIENT_STAGE_1 = OpenAIClientFactory.create_client("LLM_CLIENT_STAGE_1")
LLM_CLIENT_STAGE_2 = OpenAIClientFactory.create_client("LLM_CLIENT_STAGE_2")
