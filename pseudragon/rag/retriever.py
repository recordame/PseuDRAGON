"""
RAG (Retrieval-Augmented Generation) System for PseuDRAGON Framework
PseuDRAGON 프레임워크를 위한 RAG (검색 증강 생성) 시스템

This module implements a RAG system that processes PDF documents,
generates embeddings, and retrieves relevant context for queries.
이 모듈은 PDF 문서를 처리하고, 임베딩을 생성하며,
쿼리에 대한 관련 컨텍스트를 검색하는 RAG 시스템을 구현합니다.
"""

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

from config.config import DEFAULT_LLM_CLIENT, Settings

# Constants for document retrieval
MAX_DOC_SNIPPET_LENGTH = 200  # Maximum length of document text snippet in logs


class RAGError(Exception):
    """
    Custom exception for RAG system operations
    RAG 시스템 작업을 위한 커스텀 예외
    """


class DocumentProcessor:
    """
    Document Processor for PDF files
    PDF 파일을 위한 문서 프로세서

    Handles PDF reading, text extraction, and chunking operations.
    PDF 읽기, 텍스트 추출 및 청킹 작업을 처리합니다.
    """

    CACHE_VERSION = "v3_legalbert"  # Legal-BERT for legal document RAG (unchanged)
    PAGE_MARKER_TEMPLATE = "\n[Page {}]\n"

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """
        Extract text from PDF with page markers
        페이지 마커와 함께 PDF에서 텍스트 추출

        Args:
            pdf_path: Path to PDF file
                     PDF 파일 경로

        Returns:
            Extracted text with page markers
            페이지 마커가 포함된 추출된 텍스트

        Raises:
            RAGError: If PDF reading fails
                     PDF 읽기 실패 시
        """
        try:
            reader = PdfReader(pdf_path)
            text_parts = []

            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    page_marker = DocumentProcessor.PAGE_MARKER_TEMPLATE.format(page_num)
                    text_parts.append(f"{page_marker}{page_text}\n")

            return "".join(text_parts)
        except Exception as e:
            raise RAGError(f"Failed to extract text from PDF '{pdf_path}': {e}")

    @staticmethod
    def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Split text into overlapping chunks
        텍스트를 겹치는 청크로 분할

        Args:
            text: Text to chunk
                 청크로 나눌 텍스트
            chunk_size: Number of words per chunk
                       청크당 단어 수
            overlap: Number of overlapping words between chunks
                    청크 간 겹치는 단어 수

        Returns:
            List of text chunks
            텍스트 청크 목록
        """
        words = text.split()
        chunks = []

        step_size = max(1, chunk_size - overlap)

        for i in range(0, len(words), step_size):
            chunk = " ".join(words[i: i + chunk_size])
            chunks.append(chunk)

            if i + chunk_size >= len(words):
                break

        return chunks

    @staticmethod
    def create_document_metadata(chunks: List[str], source: str) -> List[Dict[str, str]]:
        """
        Create document metadata for chunks
        청크에 대한 문서 메타데이터 생성

        Args:
            chunks: List of text chunks
                   텍스트 청크 목록
            source: Source file name
                   소스 파일 이름

        Returns:
            List of documents with metadata
            메타데이터가 포함된 문서 목록
        """
        return [{"text": chunk, "source": source} for chunk in chunks]


class EmbeddingGenerator:
    """
    EmbeddingGenerator
    임베딩 생성기

    Supports both OpenAI API and local models (via sentence-transformers).
    OpenAI API와 로컬 모델(sentence-transformers)을 모두 지원합니다.
    """

    DEFAULT_BATCH_SIZE = 20

    def __init__(self, client, model: str):
        """
        Initialize EmbeddingGenerator
        EmbeddingGenerator 초기화

        Args:
            client: OpenAI client instance (used if provider is 'openai')
                   OpenAI 클라이언트 인스턴스 (provider가 'openai'일 때 사용)
            model: Embedding model name
                  임베딩 모델 이름
        """
        self.client = client
        self.model = model
        self.provider = Settings.EMBEDDING_PROVIDER
        self.local_model = None

        if self.provider == "local":
            try:
                from sentence_transformers import SentenceTransformer
                import ssl
                import warnings

                warnings.filterwarnings('ignore', message='Unverified HTTPS request')

                ssl._create_default_https_context = ssl._create_unverified_context

                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                os.environ['CURL_CA_BUNDLE'] = ''
                os.environ['REQUESTS_CA_BUNDLE'] = ''
                os.environ['SSL_CERT_FILE'] = ''
                os.environ['PYTHONHTTPSVERIFY'] = '0'

                try:
                    import urllib3
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                except ImportError:
                    pass

                try:
                    import requests
                    from requests.adapters import HTTPAdapter
                    from urllib3.util.retry import Retry

                    session = requests.Session()
                    session.verify = False

                    retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
                    adapter = HTTPAdapter(max_retries=retry_strategy)
                    session.mount("https://", adapter)
                    session.mount("http://", adapter)

                    import huggingface_hub
                    huggingface_hub.constants.HF_HUB_DISABLE_SSL_VERIFY = True

                    original_request = requests.Session.request

                    def patched_request(self, *args, **kwargs):
                        kwargs['verify'] = False
                        return original_request(self, *args, **kwargs)

                    requests.Session.request = patched_request
                except ImportError:
                    pass

                print(f"Loading local embedding model: {self.model}...")
                self.local_model = SentenceTransformer(self.model, cache_folder=os.path.join(os.getcwd(), ".cache", "sentence_transformers"), trust_remote_code=True, use_auth_token=False)
                print("Local model loaded successfully.")
            except ImportError:
                raise RAGError(
                    "sentence-transformers not installed. "
                    "Please run: pip install sentence-transformers"
                )
            except Exception as e:
                raise RAGError(f"Failed to load local model '{self.model}': {e}")

    def generate_embeddings(self, documents: List[Dict[str, str]], batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
        """
        Generate embeddings for documents
        문서에 대한 임베딩 생성

        Args:
            documents: List of documents with text
                      텍스트가 포함된 문서 목록
            batch_size: Number of documents per API call (for OpenAI)
                       API 호출당 문서 수 (OpenAI용)

        Returns:
            NumPy array of embeddings
            임베딩의 NumPy 배열
        """
        texts = [doc["text"] for doc in documents]
        if not texts:
            return np.array([])

        if self.provider == "local":
            return self._generate_local_embeddings(texts)
        else:
            return self._generate_openai_embeddings(texts, batch_size)

    def _generate_local_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using local model"""
        try:
            # SentenceTransformer handles batching internally, but we can pass it directly
            embeddings = self.local_model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            raise RAGError(f"Failed to generate local embeddings: {e}")

    def _generate_openai_embeddings(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Generate embeddings using OpenAI API"""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            try:
                response = self.client.embeddings.create(input=batch, model=self.model)
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                raise RAGError(f"Failed to generate embeddings for batch {i // batch_size + 1}: {e}")

        return np.array(all_embeddings)


class CacheManager:
    """
    Cache Manager for embeddings
    임베딩을 위한 캐시 매니저

    Handles caching and loading of processed documents and embeddings.
    처리된 문서와 임베딩의 캐싱 및 로딩을 처리합니다.
    """

    @staticmethod
    def get_cache_path(pdf_path: str, version: str = DocumentProcessor.CACHE_VERSION) -> str:
        """
        Get cache file path for a PDF
        PDF에 대한 캐시 파일 경로 가져오기

        Args:
            pdf_path: Path to PDF file
                     PDF 파일 경로
            version: Cache version identifier
                    캐시 버전 식별자

        Returns:
            Cache file path
            캐시 파일 경로
        """
        return f"{pdf_path}.{version}.pkl"

    @staticmethod
    def load_from_cache(cache_path: str, ) -> Optional[Tuple[List[Dict[str, str]], np.ndarray]]:
        """
        Load documents and embeddings from cache
        캐시에서 문서와 임베딩 로드

        Args:
            cache_path: Path to cache file
                       캐시 파일 경로

        Returns:
            Tuple of (documents, embeddings) or None if cache doesn't exist
            (문서, 임베딩) 튜플 또는 캐시가 없으면 None
        """
        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
                return data["documents"], data["embeddings"]
        except Exception as e:
            print(f"Warning: Failed to load cache from '{cache_path}': {e}")
            return None

    @staticmethod
    def save_to_cache(cache_path: str, documents: List[Dict[str, str]], embeddings: np.ndarray) -> None:
        """
        Save documents and embeddings to cache
        문서와 임베딩을 캐시에 저장

        Args:
            cache_path: Path to cache file
                       캐시 파일 경로
            documents: List of documents
                      문서 목록
            embeddings: NumPy array of embeddings
                       임베딩의 NumPy 배열

        Raises:
            RAGError: If caching fails
                     캐싱 실패 시
        """
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({"documents": documents, "embeddings": embeddings}, f)
        except Exception as e:
            raise RAGError(f"Failed to save cache to '{cache_path}': {e}")


class RAGSystem:
    """
    RAG (Retrieval-Augmented Generation) System
    RAG (검색 증강 생성) 시스템

    Main class for document processing, embedding generation, and retrieval.
    문서 처리, 임베딩 생성 및 검색을 위한 메인 클래스.
    """

    # Feedback cache version - increment when feedback document structure changes
    # 피드백 캐시 버전 - 피드백 문서 구조 변경 시 증가
    FEEDBACK_CACHE_VERSION = "v1"

    def __init__(self, client=None):
        """
        Initialize RAG System
        RAG 시스템 초기화

        Args:
            client: OpenAI client instance (uses global client if None)
                   OpenAI 클라이언트 인스턴스 (None이면 전역 클라이언트 사용)
        """
        self.client = client or DEFAULT_LLM_CLIENT

        # Check if RAG is enabled
        # RAG 활성화 여부 확인
        self.rag_enabled = getattr(Settings, 'RAG_ENABLED', True)

        if self.rag_enabled:
            # Legal-BERT for legal documents (법률 문서용)
            self.embedding_generator = EmbeddingGenerator(self.client, Settings.LEGAL_EMBEDDING_MODEL)

            # SBERT for feedback documents (전문가 피드백용 - 의미적 유사도 검색에 최적화)
            self.feedback_embedding_generator = EmbeddingGenerator(self.client, Settings.FEEDBACK_EMBEDDING_MODEL)
        else:
            # RAG disabled: skip embedding model loading
            # RAG 비활성화: 임베딩 모델 로드 건너뛰기
            print("[INFO] RAG disabled - skipping embedding model initialization")
            self.embedding_generator = None
            self.feedback_embedding_generator = None

        self.documents: List[Dict[str, str]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.feedback_documents: List[Dict[str, str]] = []
        self.feedback_embeddings: Optional[np.ndarray] = None
        self.is_initialized = False
        self._feedback_cache_path: Optional[str] = None

    def load_documents(self, dir_path: str) -> None:
        """
        Load and process all PDF documents in the directory with 3-tier priority system
        3단계 우선순위 시스템으로 디렉토리 내의 모든 PDF 문서 로드 및 처리

        Priority levels:
        1. institutional_policy (사내 법률) - HIGHEST PRIORITY
        2. national_law (국내 법률) - MEDIUM PRIORITY
        3. international_regulation (국제 법률) - LOW PRIORITY

        Args:
            dir_path: Path to directory containing PDF files
                     PDF 파일이 포함된 디렉토리 경로

        Raises:
            FileNotFoundError: If directory doesn't exist
                              디렉토리가 존재하지 않을 때
            RAGError: If document processing fails
                     문서 처리 실패 시
        """
        # Skip document loading if RAG is disabled
        # RAG 비활성화 시 문서 로드 건너뛰기
        if not self.rag_enabled:
            print("[INFO] RAG disabled - skipping document loading")
            self.is_initialized = True  # Mark as initialized to avoid errors
            return

        print(f"Scanning directory: {dir_path}")

        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        priority_dirs = [
            ("institutional_policy", 1, "INTERNAL STANDARD"),
            ("national_law", 2, "NATIONAL STANDARD"), 
            ("international_regulation", 3, "INTERNATIONAL STANDARD"),
        ]

        all_embeddings_list = []
        total_files = 0

        for subdir_name, priority_level, priority_label in priority_dirs:
            subdir_path = os.path.join(dir_path, subdir_name)

            if not os.path.exists(subdir_path):
                print(f"[WARNING] {subdir_name} directory not found, skipping...")
                continue

            pdf_files = self._get_pdf_files(subdir_path)

            if pdf_files:
                print(f"Found {len(pdf_files)} PDF files in {subdir_name} ({priority_label}): {pdf_files}")
                total_files += len(pdf_files)

                for pdf_file in pdf_files:
                    file_path = os.path.join(subdir_path, pdf_file)
                    docs, embs = self._process_single_pdf(file_path, priority_level=priority_level, priority_label=priority_label, )

                    self.documents.extend(docs)
                    if len(embs) > 0:
                        all_embeddings_list.append(embs)

        if all_embeddings_list:
            self.embeddings = np.vstack(all_embeddings_list)
            self.is_initialized = True
            print(f"[OK] All documents processed. Total files: {total_files}, Total chunks: {len(self.documents)}")
        else:
            print("[WARNING] No embeddings generated.")

    def load_feedback_knowledge(self, audit_log_dir: str) -> None:
        """
        Load feedback knowledge from audit logs and integrate into RAG system
        감사 로그에서 피드백 지식을 로드하고 RAG 시스템에 통합

        This method processes audit logs containing expert-approved policies and
        user modifications, extracting valuable patterns including:
        - Column name -> PII type mappings
        - Action selection rationale with legal evidence
        - Code snippets for specific transformations
        - Parameter configurations

        이 메서드는 전문가가 승인한 정책과 사용자 수정 사항을 포함하는 감사 로그를 처리하여
        다음과 같은 패턴을 추출합니다:
        - 컬럼 이름 -> PII 유형 매핑
        - 법적 근거가 포함된 액션 선택 근거
        - 특정 변환을 위한 코드 스니펫
        - 매개변수 구성

        Skip loading if RAG is disabled.
        RAG 비활성화 시 로드 건너뛰기.

        Caching mechanism:
        - Cache is stored in resources/feedback_cache_{version}.pkl
        - Cache is invalidated when audit log files are newer than cache
        - Cache stores both documents and their embeddings

        캐싱 메커니즘:
        - 캐시는 resources/feedback_cache_{version}.pkl에 저장
        - 감사 로그 파일이 캐시보다 새로우면 캐시 무효화
        - 캐시는 문서와 임베딩을 모두 저장

        Args:
            audit_log_dir: Directory containing audit log JSON files
                          감사 로그 JSON 파일이 포함된 디렉토리
        """
        # Skip feedback loading if RAG is disabled
        # RAG 비활성화 시 피드백 로드 건너뛰기
        if not self.rag_enabled:
            print("[INFO] RAG disabled - skipping feedback knowledge loading")
            return

        import json
        import glob

        if not os.path.exists(audit_log_dir):
            print(f"[INFO] Audit log directory not found: {audit_log_dir}")
            return

        # Set feedback cache path
        # 피드백 캐시 경로 설정
        RESOURCES_DIR = os.path.join(BASE_DIR, "resources")

        self._feedback_cache_path = os.path.join(
            RESOURCES_DIR, f"feedback_cache_{self.FEEDBACK_CACHE_VERSION}.pkl"
        )

        # Find all audit log files (support both .json and .jsonl)
        # 모든 감사 로그 파일 찾기 (.json 및 .jsonl 모두 지원)
        audit_files = glob.glob(os.path.join(audit_log_dir, "**", "audit_*.jsonl"), recursive=True)
        audit_files.extend(glob.glob(os.path.join(audit_log_dir, "**", "audit_log.json"), recursive=True))

        if not audit_files:
            print(f"[INFO] No audit logs found in {audit_log_dir}")
            return

        # Calculate hash of audit files to detect changes
        # 변경 감지를 위한 감사 파일 해시 계산
        audit_files_hash = self._calculate_audit_files_hash(audit_files)

        # Try to load from cache
        # 캐시에서 로드 시도
        cached_data = self._load_feedback_cache(audit_files_hash)
        if cached_data is not None:
            feedback_documents, feedback_embeddings = cached_data
            print(f"[INFO] Loaded {len(feedback_documents)} feedback documents from cache")

            # Integrate cached feedback into RAG
            self._integrate_feedback_into_rag(feedback_documents, feedback_embeddings)
            return

        print(f"[INFO] Loading feedback from {len(audit_files)} audit log(s)...")

        feedback_documents = []

        for audit_file in audit_files:
            try:
                with open(audit_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            log_entry = json.loads(line.strip())
                            event_type = log_entry.get('event_type', '')
                            data = log_entry.get('data', {})

                            # Process POLICY_APPROVED events (expert-approved policies)
                            if event_type == 'POLICY_APPROVED':
                                table_name = data.get('table_name', 'unknown')
                                columns_detail = data.get('columns_detail', [])

                                for col_data in columns_detail:
                                    feedback_text = self._create_feedback_document_from_approval(table_name, col_data)
                                    if feedback_text:
                                        feedback_documents.append(
                                            {
                                                'text': feedback_text,
                                                'source': f'feedback:{os.path.basename(os.path.dirname(audit_file))}',
                                                'type': 'expert_approved',
                                                'table': table_name,
                                                'column': col_data.get('column_name', 'unknown'),
                                                'priority_level': 0
                                            }
                                        )

                            # Process USER_EDIT events (expert modifications)
                            elif event_type == 'USER_EDIT':
                                feedback_text = self._create_feedback_document_from_edit(data)
                                if feedback_text:
                                    feedback_documents.append(
                                        {
                                            'text': feedback_text,
                                            'source': f'feedback:{os.path.basename(os.path.dirname(audit_file))}',
                                            'type': 'expert_modified',
                                            'table': data.get('table', 'unknown'),
                                            'column': data.get('column', 'unknown'),
                                            'priority_level': 0
                                        }
                                    )

                            # Process PII_CLASSIFICATION_CHANGE events
                            elif event_type == 'PII_CLASSIFICATION_CHANGE':
                                feedback_text = self._create_feedback_document_from_status_change(data)
                                if feedback_text:
                                    feedback_documents.append(
                                        {
                                            'text': feedback_text,
                                            'source': f'feedback:{os.path.basename(os.path.dirname(audit_file))}',
                                            'type': 'status_change',
                                            'table': data.get('table', 'unknown'),
                                            'column': data.get('column', 'unknown'),
                                            'priority_level': 0
                                        }
                                    )

                        except json.JSONDecodeError:
                            continue

            except Exception as e:
                print(f"[WARNING] Failed to process audit log {audit_file}: {e}")
                continue

        if feedback_documents:
            print(f"[INFO] Extracted {len(feedback_documents)} feedback documents")

            # Generate embeddings for feedback documents using SBERT
            # SBERT를 사용하여 피드백 문서에 대한 임베딩 생성 (의미적 유사도 검색에 최적화)
            try:
                feedback_embeddings = self.feedback_embedding_generator.generate_embeddings(feedback_documents)

                # Save to cache before integrating
                # 통합 전에 캐시에 저장
                self._save_feedback_cache(feedback_documents, feedback_embeddings, audit_files_hash)

                # Integrate into RAG
                # RAG에 통합
                self._integrate_feedback_into_rag(feedback_documents, feedback_embeddings)

            except Exception as e:
                print(f"[WARNING] Failed to integrate feedback documents: {e}")
        else:
            print("[INFO] No feedback documents extracted from audit logs")

    def _calculate_audit_files_hash(self, audit_files: List[str]) -> str:
        """
        Calculate a hash based on audit file paths and modification times
        감사 파일 경로 및 수정 시간 기반 해시 계산

        Args:
            audit_files: List of audit file paths
                        감사 파일 경로 목록

        Returns:
            MD5 hash string representing the current state of audit files
            감사 파일의 현재 상태를 나타내는 MD5 해시 문자열
        """
        import hashlib

        hash_input = ""
        for audit_file in sorted(audit_files):
            try:
                mtime = os.path.getmtime(audit_file)
                size = os.path.getsize(audit_file)
                hash_input += f"{audit_file}:{mtime}:{size};"
            except OSError:
                continue

        return hashlib.md5(hash_input.encode()).hexdigest()

    def _load_feedback_cache(self, expected_hash: str) -> Optional[Tuple[List[Dict[str, Any]], np.ndarray]]:
        """
        Load feedback documents and embeddings from cache if valid
        유효한 경우 캐시에서 피드백 문서 및 임베딩 로드

        Args:
            expected_hash: Expected hash of audit files
                          예상되는 감사 파일 해시

        Returns:
            Tuple of (documents, embeddings) if cache is valid, None otherwise
            캐시가 유효하면 (문서, 임베딩) 튜플, 그렇지 않으면 None
        """
        if self._feedback_cache_path is None or not os.path.exists(self._feedback_cache_path):
            return None

        try:
            with open(self._feedback_cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            # Verify cache integrity
            # 캐시 무결성 검증
            if cache_data.get('version') != self.FEEDBACK_CACHE_VERSION:
                print(f"[INFO] Feedback cache version mismatch, regenerating...")
                return None

            if cache_data.get('audit_files_hash') != expected_hash:
                print(f"[INFO] Audit files changed, regenerating feedback cache...")
                return None

            documents = cache_data.get('documents', [])
            embeddings = cache_data.get('embeddings')

            if not documents or embeddings is None or len(embeddings) == 0:
                return None

            print(f"[INFO] Feedback cache is valid (hash: {expected_hash[:8]}...)")
            return documents, embeddings

        except Exception as e:
            print(f"[WARNING] Failed to load feedback cache: {e}")
            return None

    def _save_feedback_cache(self, documents: List[Dict[str, Any]], embeddings: np.ndarray, audit_files_hash: str) -> None:
        """
        Save feedback documents and embeddings to cache
        피드백 문서 및 임베딩을 캐시에 저장

        Args:
            documents: List of feedback documents
                      피드백 문서 목록
            embeddings: Numpy array of embeddings
                       임베딩 numpy 배열
            audit_files_hash: Hash of source audit files
                             소스 감사 파일의 해시
        """
        if self._feedback_cache_path is None:
            return

        try:
            cache_data = {
                'version': self.FEEDBACK_CACHE_VERSION,
                'audit_files_hash': audit_files_hash,
                'documents': documents,
                'embeddings': embeddings,
                'created_at': os.path.getmtime(self._feedback_cache_path) if os.path.exists(self._feedback_cache_path) else None
            }

            with open(self._feedback_cache_path, 'wb') as f:
                pickle.dump(cache_data, f)

            print(f"[INFO] Saved feedback cache to {self._feedback_cache_path}")

        except Exception as e:
            print(f"[WARNING] Failed to save feedback cache: {e}")

    def _integrate_feedback_into_rag(self, feedback_documents: List[Dict[str, Any]], feedback_embeddings: np.ndarray) -> None:
        """
        Integrate feedback documents and embeddings into the RAG system
        피드백 문서 및 임베딩을 RAG 시스템에 통합

        NOTE: Feedback documents are stored SEPARATELY from legal documents because
        they use different embedding models (SBERT vs Legal-BERT).
        피드백 문서는 법률 문서와 별도로 저장됩니다 (서로 다른 임베딩 모델 사용).

        Args:
            feedback_documents: List of feedback documents
                               피드백 문서 목록
            feedback_embeddings: Numpy array of feedback embeddings (SBERT)
                                피드백 임베딩 numpy 배열 (SBERT)
        """
        # Store feedback separately (different embedding space from legal documents)
        # 피드백을 별도로 저장 (법률 문서와 다른 임베딩 공간)
        if self.feedback_embeddings is not None and len(self.feedback_embeddings) > 0:
            self.feedback_embeddings = np.vstack([self.feedback_embeddings, feedback_embeddings])
            self.feedback_documents.extend(feedback_documents)
        else:
            self.feedback_embeddings = feedback_embeddings
            self.feedback_documents = feedback_documents

        print(f"[INFO] Successfully integrated {len(feedback_documents)} feedback documents into RAG (separate from legal docs)")

    def _truncate_feedback_text(self, text: str) -> str:
        """
        Truncate feedback text to max length for SBERT embedding
        SBERT 임베딩을 위해 피드백 텍스트를 최대 길이로 자름

        Args:
            text: Original feedback text

        Returns:
            Truncated text if exceeds FEEDBACK_MAX_LENGTH
        """
        max_length = Settings.FEEDBACK_MAX_LENGTH
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text

    def _create_feedback_document_from_approval(self, table_name: str, col_data: Dict[str, Any]) -> str:
        """
        Create a structured feedback document from approved policy
        승인된 정책에서 구조화된 피드백 문서 생성

        Args:
            table_name: Table name
            col_data: Column policy data with code snippet

        Returns:
            Formatted feedback text for RAG (truncated to FEEDBACK_MAX_LENGTH)
        """
        column_name = col_data.get('column_name', 'unknown')
        pii_type = col_data.get('pii_type', 'unknown')
        action = col_data.get('action', 'KEEP')
        parameters = col_data.get('parameters', {})
        rationale = col_data.get('rationale', '')
        legal_evidence = col_data.get('legal_evidence', '')
        code_snippet = col_data.get('code_snippet', '')

        # Create structured feedback document
        parts = [f"Expert-Approved Policy for {table_name}.{column_name}", f"PII Type: {pii_type}", f"Action: {action}", ]

        if parameters:
            params_str = ', '.join([f"{k}={v}" for k, v in parameters.items()])
            parts.append(f"Parameters: {params_str}")

        if rationale:
            parts.append(f"Rationale: {rationale}")

        if legal_evidence:
            parts.append(f"Legal Basis: {legal_evidence}")

        if code_snippet:
            parts.append(f"Implementation:\n{code_snippet}")

        return self._truncate_feedback_text('\n'.join(parts))

    def _create_feedback_document_from_edit(self, data: Dict[str, Any]) -> str:
        """
        Create a structured feedback document from user edit
        사용자 수정에서 구조화된 피드백 문서 생성

        Args:
            data: User edit data with old/new actions and code snippets

        Returns:
            Formatted feedback text for RAG
        """
        table = data.get('table', 'unknown')
        column = data.get('column', 'unknown')
        old_action = data.get('old_action', '')
        new_action = data.get('new_action', '')
        rationale = data.get('rationale', '')
        old_parameters = data.get('old_parameters', {})
        new_parameters = data.get('new_parameters', {})
        old_code_snippet = data.get('old_code_snippet', '')
        new_code_snippet = data.get('new_code_snippet', '')
        legal_evidence = data.get('legal_evidence', '')

        # Create structured feedback document showing the correction
        parts = [f"Expert Correction for {table}.{column}", f"Changed Action: {old_action} -> {new_action}", ]

        if old_parameters or new_parameters:
            old_params_str = ', '.join([f"{k}={v}" for k, v in old_parameters.items()]) if old_parameters else 'none'
            new_params_str = ', '.join([f"{k}={v}" for k, v in new_parameters.items()]) if new_parameters else 'none'
            parts.append(f"Parameters: {old_params_str} -> {new_params_str}")

        if rationale:
            parts.append(f"Expert Rationale: {rationale}")

        if legal_evidence:
            parts.append(f"Legal Basis: {legal_evidence}")

        if new_code_snippet:
            parts.append(f"Corrected Implementation:\n{new_code_snippet}")
        elif old_code_snippet:
            parts.append(f"Previous Implementation (rejected):\n{old_code_snippet}")

        return self._truncate_feedback_text('\n'.join(parts))

    def _create_feedback_document_from_status_change(self, data: Dict[str, Any]) -> str:
        """
        Create a structured feedback document from PII status change
        PII 상태 변경에서 구조화된 피드백 문서 생성

        Args:
            data: Status change data

        Returns:
            Formatted feedback text for RAG (truncated to FEEDBACK_MAX_LENGTH)
        """
        table = data.get('table', 'unknown')
        column = data.get('column', 'unknown')
        old_status = data.get('old_status', '')
        new_status = data.get('new_status', '')
        rationale = data.get('rationale', '')

        parts = [
            f"Expert PII Status Change for {table}.{column}",
            f"Changed Status: {old_status} -> {new_status}",
        ]

        if rationale:
            parts.append(f"Rationale: {rationale}")

        return self._truncate_feedback_text('\n'.join(parts))

    def _get_pdf_files(self, dir_path: str) -> List[str]:
        """
        Get list of PDF files in directory
        디렉토리 내 PDF 파일 목록 가져오기

        Args:
            dir_path: Path to directory
                     디렉토리 경로

        Returns:
            List of PDF file names
            PDF 파일 이름 목록
        """
        return [f for f in os.listdir(dir_path) if f.lower().endswith(".pdf")]

    def _process_single_pdf(self, pdf_path: str, priority_level: int = 3, priority_label: str = "GENERAL") -> Tuple[List[Dict[str, str]], np.ndarray]:
        """
        Process a single PDF file: read, chunk, embed, and cache
        단일 PDF 파일 처리: 읽기, 청킹, 임베딩 및 캐싱

        Args:
            pdf_path: Path to PDF file
                     PDF 파일 경로
            priority_level: Priority level (1=high, 2=medium, 3=lowest)
                           우선순위 레벨 (1=최고, 2=중간, 3=최저)
            priority_label: Priority label for display
                           표시용 우선순위 레이블

        Returns:
            Tuple of (documents, embeddings)
            (문서, 임베딩) 튜플
        """
        print(f"Processing {pdf_path}...")

        cache_path = CacheManager.get_cache_path(pdf_path)
        cached_data = CacheManager.load_from_cache(cache_path)

        if cached_data:
            print(f"  - Loading from cache: {os.path.basename(cache_path)}")
            documents, embeddings = cached_data
            for doc in documents:
                doc["priority_level"] = priority_level
                doc["priority_label"] = priority_label
            return documents, embeddings

        try:
            text = DocumentProcessor.extract_text_from_pdf(pdf_path)
            chunks = DocumentProcessor.chunk_text(text, Settings.CHUNK_SIZE, Settings.CHUNK_OVERLAP)
            documents = DocumentProcessor.create_document_metadata(chunks, os.path.basename(pdf_path))

            for doc in documents:
                doc["priority_level"] = priority_level
                doc["priority_label"] = priority_label

            embeddings = self.embedding_generator.generate_embeddings(documents)

            CacheManager.save_to_cache(cache_path, documents, embeddings)

            return documents, embeddings
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return [], np.array([])

    def retrieve(self, query: str, top_k: int = Settings.TOP_K_RESULTS, exclude_feedback: bool = False, allowed_feedback_types: List[str] = None, metadata_filter: Dict[str, Any] = None) -> tuple[
        str, list[str]]:
        """
        Retrieve relevant context from documents using dual-model search
        이중 모델 검색을 사용하여 문서에서 관련 컨텍스트 검색

        This method performs:
        이 메서드는 다음을 수행합니다:
        1. Legal document search using Legal-BERT embeddings
           Legal-BERT 임베딩을 사용한 법률 문서 검색
        2. Feedback document search using SBERT embeddings (for semantic similarity)
           SBERT 임베딩을 사용한 피드백 문서 검색 (의미적 유사도)

        Args:
            query: Search query
                  검색 쿼리
            top_k: Number of top results to return
                  반환할 상위 결과 수
            exclude_feedback: If True, exclude feedback documents from search
                             True면 피드백 문서를 검색에서 제외
            allowed_feedback_types: Filter feedback by type (e.g., ['status_change'])
                                   피드백 유형 필터
            metadata_filter: NOT used for strict filtering (allows similar column discovery)
                            엄격한 필터링에 사용되지 않음 (유사 컬럼 발견 허용)

        Returns:
            Tuple of (formatted context string with sources, list of source document names)
            (소스가 포함된 포맷팅된 컨텍스트 문자열, 소스 문서명 목록) 튜플

        Raises:
            RAGError: If system is not initialized or retrieval fails
                     시스템이 초기화되지 않았거나 검색 실패 시
        """
        if not self.is_initialized:
            raise RAGError("RAG System not initialized. Call load_documents first.")

        # Check if we have any documents to search
        has_legal_docs = self.embeddings is not None and len(self.embeddings) > 0
        has_feedback_docs = self.feedback_embeddings is not None and len(self.feedback_embeddings) > 0

        if not has_legal_docs and not has_feedback_docs:
            print("[WARNING] No documents available. Returning empty context.")
            return "", []

        try:
            query_embedding = self._generate_query_embedding(query)
            _, doc_info_list = self._find_similar_documents(
                query_embedding, top_k, exclude_feedback, allowed_feedback_types, metadata_filter,
                query_text=query  # Pass original query for SBERT embedding
            )
            context_str = self._format_results_from_doc_info(doc_info_list)

            estimated_tokens = len(context_str.split()) * 1.3
            if estimated_tokens > Settings.MAX_CONTEXT_TOKENS:
                print(f"[WARNING] Context too large ({estimated_tokens:.0f} tokens), reducing to top {top_k - 1} results")
                if top_k > 1:
                    return self.retrieve(query, top_k - 1, exclude_feedback, allowed_feedback_types, metadata_filter)
                else:
                    context_str = context_str[:int(Settings.MAX_CONTEXT_TOKENS * 4)]

            source_docs = self._extract_source_names_from_doc_info(doc_info_list)
            return context_str, source_docs
        except Exception as e:
            raise RAGError(f"Failed to retrieve context for query: {e}")

    def retrieve_with_details(
            self,
            query: str,
            top_k: int = Settings.TOP_K_RESULTS,
            exclude_feedback: bool = False,
            allowed_feedback_types: List[str] = None,
            metadata_filter: Dict[str, Any] = None
    ) -> Tuple[str, List[str], List[Dict[str, Any]]]:
        """
        Retrieve relevant context from documents with detailed information using dual-model search
        이중 모델 검색을 사용하여 문서에서 관련 컨텍스트를 상세 정보와 함께 검색

        This method performs:
        이 메서드는 다음을 수행합니다:
        1. Legal document search using Legal-BERT embeddings
           Legal-BERT 임베딩을 사용한 법률 문서 검색
        2. Feedback document search using SBERT embeddings (for semantic similarity)
           SBERT 임베딩을 사용한 피드백 문서 검색 (의미적 유사도)

        Args:
            query: Search query
                  검색 쿼리
            top_k: Number of top results to return
                  반환할 상위 결과 수
            exclude_feedback: If True, exclude feedback documents from search
                             True면 피드백 문서를 검색에서 제외
            allowed_feedback_types: Filter feedback by type (e.g., ['status_change'])
                                   피드백 유형 필터
            metadata_filter: NOT used for strict filtering (allows similar column discovery)
                            엄격한 필터링에 사용되지 않음 (유사 컬럼 발견 허용)

        Returns:
            Tuple of (formatted context string, list of source names, list of document details)
            (포맷팅된 컨텍스트 문자열, 소스명 목록, 문서 상세 정보 목록) 튜플

            Document details is a list of dicts with keys:
            - 'source' (str): Source document name
            - 'priority_level' (int): Priority level (0-3, lower is higher priority)
            - 'priority_label' (str): Human-readable priority label
            - 'text_snippet' (str): Text snippet truncated to MAX_DOC_SNIPPET_LENGTH
            - 'similarity' (float): Similarity score
            - 'doc_source' (str): 'legal' or 'feedback'

        Raises:
            RAGError: If system is not initialized or retrieval fails
                     시스템이 초기화되지 않았거나 검색 실패 시
        """
        if not self.is_initialized:
            raise RAGError("RAG System not initialized. Call load_documents first.")

        # Check if we have any documents to search
        has_legal_docs = self.embeddings is not None and len(self.embeddings) > 0
        has_feedback_docs = self.feedback_embeddings is not None and len(self.feedback_embeddings) > 0

        if not has_legal_docs and not has_feedback_docs:
            print("[WARNING] No documents available. Returning empty context.")
            return "", [], []

        try:
            query_embedding = self._generate_query_embedding(query)
            _, doc_info_list = self._find_similar_documents(
                query_embedding, top_k, exclude_feedback, allowed_feedback_types, metadata_filter,
                query_text=query  # Pass original query for SBERT embedding
            )
            context_str = self._format_results_from_doc_info(doc_info_list)

            estimated_tokens = len(context_str.split()) * 1.3
            if estimated_tokens > Settings.MAX_CONTEXT_TOKENS:
                print(f"[WARNING] Context too large ({estimated_tokens:.0f} tokens), reducing to top {top_k - 1} results")
                if top_k > 1:
                    return self.retrieve_with_details(query, top_k - 1, exclude_feedback, allowed_feedback_types, metadata_filter)
                else:
                    context_str = context_str[:int(Settings.MAX_CONTEXT_TOKENS * 4)]

            source_docs = self._extract_source_names_from_doc_info(doc_info_list)

            # Extract detailed document information from doc_info_list
            doc_details = []
            for doc_info in doc_info_list:
                doc = doc_info.get("doc", {})
                priority_level = doc_info.get("priority_level", 3)
                similarity = doc_info.get("similarity", 0.0)
                doc_source = doc_info.get("source", "legal")

                priority_tag = ""
                if priority_level == 0:
                    priority_tag = "EXPERT FEEDBACK - HIGHEST PRIORITY"
                elif priority_level == 1:
                    priority_tag = "INTERNAL STANDARD - HIGH PRIORITY"
                elif priority_level == 2:
                    priority_tag = "NATIONAL STANDARD - MEDIUM PRIORITY"
                elif priority_level == 3:
                    priority_tag = "INTERNATIONAL STANDARD - LOW PRIORITY"

                text = doc.get("text", doc.get("content", ""))
                # Ensure text is a string before truncation
                if text is None:
                    text = ""
                # Truncate text for logging (configurable via MAX_DOC_SNIPPET_LENGTH)
                text_snippet = text[:MAX_DOC_SNIPPET_LENGTH] + "..." if len(text) > MAX_DOC_SNIPPET_LENGTH else text

                doc_details.append({
                    "source": doc.get("source", "Unknown"),
                    "priority_level": priority_level,
                    "priority_label": priority_tag,
                    "text_snippet": text_snippet,
                    "similarity": similarity,
                    "doc_source": doc_source
                })

            return context_str, source_docs, doc_details
        except Exception as e:
            raise RAGError(f"Failed to retrieve context for query: {e}")

    def _generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for query
        쿼리에 대한 임베딩 생성

        Args:
            query: Search query
                  검색 쿼리

        Returns:
            Query embedding as NumPy array
            NumPy 배열로 된 쿼리 임베딩
        """
        if Settings.EMBEDDING_PROVIDER == "local":
            return self.embedding_generator._generate_local_embeddings([query])

        response = self.client.embeddings.create(input=[query], model=Settings.LEGAL_EMBEDDING_MODEL)
        return np.array(response.data[0].embedding).reshape(1, -1)

    def _generate_feedback_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for query using SBERT (for feedback search)
        SBERT를 사용하여 쿼리 임베딩 생성 (피드백 검색용)

        Args:
            query: Search query / 검색 쿼리

        Returns:
            Query embedding as NumPy array / NumPy 배열로 된 쿼리 임베딩
        """
        if Settings.EMBEDDING_PROVIDER == "local":
            return self.feedback_embedding_generator._generate_local_embeddings([query])

        # For API-based embeddings, use the feedback model
        response = self.client.embeddings.create(input=[query], model=Settings.FEEDBACK_EMBEDDING_MODEL)
        return np.array(response.data[0].embedding).reshape(1, -1)

    def _find_similar_documents(
            self,
            query_embedding: np.ndarray,
            top_k: int,
            exclude_feedback: bool = False,
            allowed_feedback_types: List[str] = None,
            metadata_filter: Dict[str, Any] = None,
            query_text: str = None
    ) -> np.ndarray:
        """
        Find most similar documents using dual-model search with 4-tier priority
        이중 모델 검색을 사용하여 4단계 우선순위 기반 유사 문서 검색

        This function performs TWO separate searches:
        이 함수는 두 개의 별도 검색을 수행합니다:
        1. Legal documents: Legal-BERT embeddings (법률 문서: Legal-BERT 임베딩)
        2. Feedback documents: SBERT embeddings for semantic similarity (피드백: SBERT 의미적 유사도)

        For feedback search, metadata_filter is NOT applied strictly - instead,
        similar columns from past feedback are included to enable learning from
        semantically similar cases.
        피드백 검색에서는 metadata_filter가 엄격하게 적용되지 않습니다.
        대신 의미적으로 유사한 과거 피드백에서 학습할 수 있도록 유사 컬럼을 포함합니다.

        Priority order and weights:
        우선순위 및 가중치:
        0. EXPERT_FEEDBACK (priority_level=0) - w=1.0 (과거 전문가 피드백)
        1. institutional_policy (priority_level=1) - w=0.8 (사내 법률)
        2. national_law (priority_level=2) - w=0.6 (국내 법률)
        3. international_regulation (priority_level=3) - w=0.4 (국제 법률)

        Final score: score(d_i) = α * s_i + (1-α) * w_{p_i}

        Args:
            query_embedding: Query embedding (Legal-BERT) for legal documents
                           쿼리 임베딩 (Legal-BERT) - 법률 문서용
            top_k: Number of results to return
                  반환할 결과 수
            exclude_feedback: If True, exclude feedback documents
                             True면 피드백 문서 제외
            allowed_feedback_types: Filter feedback by type (e.g., ['status_change'])
                                   피드백 유형 필터
            metadata_filter: NOT used for strict filtering, only for logging
                            엄격한 필터링에 사용되지 않음, 로깅용
            query_text: Original query text for SBERT embedding (optional)
                       SBERT 임베딩용 원본 쿼리 텍스트 (선택적)

        Returns:
            Indices of most similar documents (combined from both searches)
            가장 유사한 문서의 인덱스 (두 검색 결과 병합)
        """
        # Priority weights from paper
        # 논문에 따른 우선순위 가중치
        PRIORITY_WEIGHTS = {
            0: float(os.environ.get("EXPERT_FEEDBACK_WEIGHT", 1.0)),  # EXPERT_FEEDBACK
            1: float(os.environ.get("INTERNAL_POLICY_WEIGHT", 0.8)),  # ORGANIZATION / INTERNAL
            2: float(os.environ.get("NATIONAL_LAW_WEIGHT", 0.6)),  # NATIONAL
            3: float(os.environ.get("INTERNATIONAL_REGULATIONS_WEIGHT", 0.4))  # INTERNATIONAL
        }

        # Weighting factor for semantic similarity vs priority
        # α: 의미적 유사도 vs 우선순위에 대한 가중치 계수
        ALPHA = float(os.environ.get("LEGAL_SOURCE_PRIORITY_WEIGHT", 0.7))

        doc_info = []

        # ========== PART 1: Search Legal Documents (Legal-BERT) ==========
        # 법률 문서 검색 (Legal-BERT 임베딩 사용)
        if self.embeddings is not None and len(self.embeddings) > 0:
            legal_similarities = cosine_similarity(query_embedding.reshape(1, -1), self.embeddings)[0]

            for idx, sim in enumerate(legal_similarities):
                priority_level = self.documents[idx].get("priority_level", 3)

                # Skip feedback documents in legal search (they are searched separately)
                if priority_level == 0:
                    continue

                # Get priority weight
                w_p = PRIORITY_WEIGHTS.get(priority_level, 0.4)

                # Calculate weighted score: score(d_i) = alpha * s_i + (1-alpha) * w_{p_i}
                score = ALPHA * sim + (1 - ALPHA) * w_p

                doc_info.append({
                    "idx": idx,
                    "similarity": float(sim),
                    "priority_level": priority_level,
                    "score": float(score),
                    "source": "legal",
                    "doc": self.documents[idx]
                })

        # ========== PART 2: Search Feedback Documents (SBERT) ==========
        # 피드백 문서 검색 (SBERT 임베딩 사용 - 유사 컬럼 포함)
        if not exclude_feedback and self.feedback_embeddings is not None and len(self.feedback_embeddings) > 0:
            # Generate SBERT embedding for feedback search
            if query_text:
                feedback_query_embedding = self._generate_feedback_query_embedding(query_text)
            else:
                # Fallback: use legal embedding (less accurate for semantic search)
                feedback_query_embedding = query_embedding

            feedback_similarities = cosine_similarity(
                feedback_query_embedding.reshape(1, -1),
                self.feedback_embeddings
            )[0]

            for idx, sim in enumerate(feedback_similarities):
                feedback_doc = self.feedback_documents[idx]

                # Filter by feedback type if specified
                if allowed_feedback_types is not None:
                    doc_type = feedback_doc.get("type", "")
                    if doc_type not in allowed_feedback_types:
                        continue

                # NOTE: metadata_filter is NOT applied strictly for feedback
                # This allows finding SIMILAR columns, not just exact matches
                # 메타데이터 필터는 피드백에 엄격하게 적용되지 않음
                # 정확한 매칭이 아닌 유사한 컬럼을 찾을 수 있도록 함

                # Apply minimum similarity threshold for feedback (avoid noise)
                # 피드백에 대한 최소 유사도 임계값 적용 (노이즈 방지)
                MIN_FEEDBACK_SIMILARITY = float(os.environ.get("MIN_FEEDBACK_SIMILARITY", 0.3))
                if sim < MIN_FEEDBACK_SIMILARITY:
                    continue

                # Priority level 0 for expert feedback
                priority_level = 0
                w_p = PRIORITY_WEIGHTS[0]

                # Calculate weighted score
                score = ALPHA * sim + (1 - ALPHA) * w_p

                # Use negative index to distinguish from legal documents
                # 법률 문서와 구분하기 위해 음수 인덱스 사용
                doc_info.append({
                    "idx": -(idx + 1),  # Negative index for feedback docs
                    "similarity": float(sim),
                    "priority_level": priority_level,
                    "score": float(score),
                    "source": "feedback",
                    "doc": feedback_doc
                })

        # Sort by score descending
        # 점수 내림차순 정렬
        doc_info.sort(key=lambda x: -x["score"])

        # Return top_k results
        top_indices = [doc["idx"] for doc in doc_info[:top_k]]

        return np.array(top_indices), doc_info[:top_k]

    def _format_results_from_doc_info(self, doc_info_list: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into a string with priority labels
        검색된 문서를 우선순위 레이블과 함께 문자열로 포맷

        Args:
            doc_info_list: List of document info dictionaries from _find_similar_documents
                          _find_similar_documents에서 반환된 문서 정보 딕셔너리 목록

        Returns:
            Formatted string of documents
            포맷된 문서 문자열
        """
        results = []
        for doc_info in doc_info_list:
            doc = doc_info.get("doc", {})
            priority_level = doc_info.get("priority_level", 3)
            similarity = doc_info.get("similarity", 0.0)

            priority_tag = ""
            if priority_level == 0:
                priority_tag = "[EXPERT FEEDBACK - HIGHEST PRIORITY] "
            elif priority_level == 1:
                priority_tag = "[INTERNAL STANDARD - HIGH PRIORITY] "
            elif priority_level == 2:
                priority_tag = "[NATIONAL STANDARD - MEDIUM PRIORITY] "
            elif priority_level == 3:
                priority_tag = "[INTERNATIONAL STANDARD - LOW PRIORITY] "

            # Use 'text' key instead of 'content' (documents are stored with 'text' key)
            content = doc.get("text", doc.get("content", ""))
            source = doc.get("source", "Unknown")

            # Add similarity score for transparency
            results.append(f"Source: {priority_tag}{source} (similarity: {similarity:.3f})\n{content}\n")

        return "\n---\n".join(results)

    def _extract_source_names_from_doc_info(self, doc_info_list: List[Dict[str, Any]]) -> list[str]:
        """
        Extract unique source document names from retrieved documents
        검색된 문서에서 고유한 소스 문서명 추출

        Args:
            doc_info_list: List of document info dictionaries from _find_similar_documents
                          _find_similar_documents에서 반환된 문서 정보 딕셔너리 목록

        Returns:
            List of unique source document names
            고유한 소스 문서명 목록
        """
        sources = []
        seen = set()
        for doc_info in doc_info_list:
            doc = doc_info.get("doc", {})
            source = doc.get("source", "Unknown")
            if source not in seen:
                sources.append(source)
                seen.add(source)
        return sources
