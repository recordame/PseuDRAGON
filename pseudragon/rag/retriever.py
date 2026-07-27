"""
RAG (Retrieval-Augmented Generation) System for PseuDRAGON Framework

This module implements a RAG system that processes PDF documents,
generates embeddings, and retrieves relevant context for queries.
"""

import hashlib
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

from config.config import DEFAULT_LLM_CLIENT, RAGConfig, Settings

# Constants for document retrieval
MAX_DOC_SNIPPET_LENGTH = 200  # Maximum length of document text snippet in logs


class RAGError(Exception):
    """
    Custom exception for RAG system operations
    """


class DocumentProcessor:
    """
    Document Processor for PDF files

    Handles PDF reading, text extraction, and chunking operations.
    """

    CACHE_VERSION = "v5_token_model_scoped"
    PAGE_MARKER_TEMPLATE = "\n[Page {}]\n"

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """
        Extract text from PDF with page markers

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text with page markers

        Raises:
            RAGError: If PDF reading fails
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
    def chunk_text(
        text: str,
        chunk_size: int,
        overlap: int,
        tokenizer: Any = None,
    ) -> List[str]:
        """
        Split text into overlapping chunks

        Args:
            text: Text to chunk
            chunk_size: Number of tokens per chunk when a tokenizer is
                        available; otherwise number of whitespace-delimited
                        words
            overlap: Number of overlapping tokens or words between chunks
            tokenizer: Optional embedding-model tokenizer

        Returns:
            List of text chunks
        """
        if tokenizer is not None:
            try:
                token_ids = tokenizer.encode(
                    text,
                    add_special_tokens=False,
                    truncation=False,
                )
                chunks = []
                step_size = max(1, chunk_size - overlap)

                for i in range(0, len(token_ids), step_size):
                    chunk_ids = token_ids[i: i + chunk_size]
                    chunk = tokenizer.decode(
                        chunk_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ).strip()
                    if chunk:
                        chunks.append(chunk)

                    if i + chunk_size >= len(token_ids):
                        break

                return chunks
            except Exception as e:
                print(
                    "[WARNING] Token-aware chunking failed; "
                    f"falling back to whitespace chunking: {e}"
                )

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

        Args:
            chunks: List of text chunks
            source: Source file name

        Returns:
            List of documents with metadata
        """
        return [{"text": chunk, "source": source} for chunk in chunks]


class EmbeddingGenerator:
    """
    EmbeddingGenerator

    Supports both OpenAI API and local models (via sentence-transformers).
    """

    DEFAULT_BATCH_SIZE = 20

    def __init__(self, client, model: str):
        """
        Initialize EmbeddingGenerator

        Args:
            client: OpenAI client instance (used if provider is 'openai')
            model: Embedding model name
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

                cache_folder = os.path.join(os.getcwd(), ".cache", "sentence_transformers")
                local_candidate = os.path.join(cache_folder, self.model)
                model_ref = local_candidate if os.path.isdir(local_candidate) else self.model
                print(f"Loading local embedding model: {self.model}...")
                self.local_model = SentenceTransformer(model_ref, cache_folder=cache_folder, trust_remote_code=True, use_auth_token=False)
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

        Args:
            documents: List of documents with text
            batch_size: Number of documents per API call (for OpenAI)

        Returns:
            NumPy array of embeddings
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

    Handles caching and loading of processed documents and embeddings.
    """

    @staticmethod
    def get_cache_path(pdf_path: str, version: Optional[str] = None) -> str:
        """
        Get cache file path for a PDF

        Args:
            pdf_path: Path to PDF file
            version: Cache version identifier

        Returns:
            Cache file path
        """
        if version is None:
            cache_identity = "|".join(
                (
                    Settings.LEGAL_EMBEDDING_MODEL,
                    str(Settings.CHUNK_SIZE),
                    str(Settings.CHUNK_OVERLAP),
                )
            )
            fingerprint = hashlib.sha256(
                cache_identity.encode("utf-8")
            ).hexdigest()[:12]
            version = f"{DocumentProcessor.CACHE_VERSION}_{fingerprint}"

        return f"{pdf_path}.{version}.pkl"

    @staticmethod
    def load_from_cache(cache_path: str, ) -> Optional[Tuple[List[Dict[str, str]], np.ndarray]]:
        """
        Load documents and embeddings from cache

        Args:
            cache_path: Path to cache file

        Returns:
            Tuple of (documents, embeddings) or None if cache doesn't exist
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

        Args:
            cache_path: Path to cache file
            documents: List of documents
            embeddings: NumPy array of embeddings

        Raises:
            RAGError: If caching fails
        """
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({"documents": documents, "embeddings": embeddings}, f)
        except Exception as e:
            raise RAGError(f"Failed to save cache to '{cache_path}': {e}")


class RAGSystem:
    """
    RAG (Retrieval-Augmented Generation) System

    Main class for document processing, embedding generation, and retrieval.
    """

    # Feedback cache version - increment when feedback document structure changes
    FEEDBACK_CACHE_VERSION = "v1"

    def __init__(self, client=None):
        """
        Initialize RAG System

        Args:
            client: OpenAI client instance (uses global client if None)
        """
        self.client = client or DEFAULT_LLM_CLIENT

        # Check if RAG is enabled
        self.rag_enabled = getattr(Settings, 'RAG_ENABLED', True)

        if self.rag_enabled:
            # Configured legal-domain embedding for legal documents
            self.embedding_generator = EmbeddingGenerator(self.client, Settings.LEGAL_EMBEDDING_MODEL)

            self.feedback_embedding_generator = EmbeddingGenerator(self.client, Settings.FEEDBACK_EMBEDDING_MODEL)
        else:
            # RAG disabled: skip embedding model loading
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

        Priority levels:

        Args:
            dir_path: Path to directory containing PDF files

        Raises:
            FileNotFoundError: If directory doesn't exist
            RAGError: If document processing fails
        """
        # Skip document loading if RAG is disabled
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

        This method processes audit logs containing expert-approved policies and
        user modifications, extracting valuable patterns including:
        - Column name -> PII type mappings
        - Action selection rationale with legal evidence
        - Code snippets for specific transformations
        - Parameter configurations


        Skip loading if RAG is disabled.

        Caching mechanism:
        - Cache is stored in resources/feedback_cache_{version}.pkl
        - Cache is invalidated when audit log files are newer than cache
        - Cache stores both documents and their embeddings


        Args:
            audit_log_dir: Directory containing audit log JSON files
        """
        # Skip feedback loading if RAG is disabled
        if not self.rag_enabled:
            print("[INFO] RAG disabled - skipping feedback knowledge loading")
            return

        import json
        import glob

        if not os.path.exists(audit_log_dir):
            print(f"[INFO] Audit log directory not found: {audit_log_dir}")
            return

        # Set feedback cache path
        RESOURCES_DIR = os.path.join(BASE_DIR, "resources")

        self._feedback_cache_path = os.path.join(
            RESOURCES_DIR, f"feedback_cache_{self.FEEDBACK_CACHE_VERSION}.pkl"
        )

        # Find all audit log files (support both .json and .jsonl)
        audit_files = glob.glob(os.path.join(audit_log_dir, "**", "audit_*.jsonl"), recursive=True)
        audit_files.extend(glob.glob(os.path.join(audit_log_dir, "**", "audit_log.json"), recursive=True))

        if not audit_files:
            print(f"[INFO] No audit logs found in {audit_log_dir}")
            return

        # Calculate hash of audit files to detect changes
        audit_files_hash = self._calculate_audit_files_hash(audit_files)

        # Try to load from cache
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
            try:
                feedback_embeddings = self.feedback_embedding_generator.generate_embeddings(feedback_documents)

                # Save to cache before integrating
                self._save_feedback_cache(feedback_documents, feedback_embeddings, audit_files_hash)

                # Integrate into RAG
                self._integrate_feedback_into_rag(feedback_documents, feedback_embeddings)

            except Exception as e:
                print(f"[WARNING] Failed to integrate feedback documents: {e}")
        else:
            print("[INFO] No feedback documents extracted from audit logs")

    def _calculate_audit_files_hash(self, audit_files: List[str]) -> str:
        """
        Calculate a hash based on audit file paths and modification times

        Args:
            audit_files: List of audit file paths

        Returns:
            MD5 hash string representing the current state of audit files
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

        Args:
            expected_hash: Expected hash of audit files

        Returns:
            Tuple of (documents, embeddings) if cache is valid, None otherwise
        """
        if self._feedback_cache_path is None or not os.path.exists(self._feedback_cache_path):
            return None

        try:
            with open(self._feedback_cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            # Verify cache integrity
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

        Args:
            documents: List of feedback documents
            embeddings: Numpy array of embeddings
            audit_files_hash: Hash of source audit files
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

        NOTE: Feedback documents are stored SEPARATELY from legal documents because
        they use different embedding models (general SBERT vs the configured
        legal-domain embedding).

        Args:
            feedback_documents: List of feedback documents
            feedback_embeddings: Numpy array of feedback embeddings (SBERT)
        """
        # Store feedback separately (different embedding space from legal documents)
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

        Args:
            dir_path: Path to directory

        Returns:
            List of PDF file names
        """
        return [f for f in os.listdir(dir_path) if f.lower().endswith(".pdf")]

    def _process_single_pdf(self, pdf_path: str, priority_level: int = 3, priority_label: str = "GENERAL") -> Tuple[List[Dict[str, str]], np.ndarray]:
        """
        Process a single PDF file: read, chunk, embed, and cache

        Args:
            pdf_path: Path to PDF file
            priority_level: Priority level (1=high, 2=medium, 3=lowest)
            priority_label: Priority label for display

        Returns:
            Tuple of (documents, embeddings)
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
            tokenizer = getattr(
                self.embedding_generator.local_model,
                "tokenizer",
                None,
            )
            chunks = DocumentProcessor.chunk_text(
                text,
                Settings.CHUNK_SIZE,
                Settings.CHUNK_OVERLAP,
                tokenizer=tokenizer,
            )
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

    def retrieve(
            self,
            query: str,
            top_k: int = Settings.TOP_K_RESULTS,
            exclude_feedback: bool = False,
            allowed_feedback_types: List[str] = None,
            metadata_filter: Dict[str, Any] = None,
            semantic_similarity_weight: Optional[float] = None,
    ) -> tuple[str, list[str]]:
        """
        Retrieve relevant context from documents using dual-model search

        This method performs:
        1. Legal document search using the configured legal-domain embeddings
        2. Feedback document search using SBERT embeddings (for semantic similarity)

        Args:
            query: Search query
            top_k: Number of top results to return
            exclude_feedback: If True, exclude feedback documents from search
            allowed_feedback_types: Filter feedback by type (e.g., ['status_change'])
            metadata_filter: NOT used for strict filtering (allows similar column discovery)
            semantic_similarity_weight: Optional per-run alpha override

        Returns:
            Tuple of (formatted context string with sources, list of source document names)

        Raises:
            RAGError: If system is not initialized or retrieval fails
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
                query_text=query,  # Pass original query for SBERT embedding
                semantic_similarity_weight=semantic_similarity_weight,
            )
            context_str = self._format_results_from_doc_info(doc_info_list)

            estimated_tokens = len(context_str.split()) * 1.3
            if estimated_tokens > Settings.MAX_CONTEXT_TOKENS:
                print(f"[WARNING] Context too large ({estimated_tokens:.0f} tokens), reducing to top {top_k - 1} results")
                if top_k > 1:
                    return self.retrieve(
                        query,
                        top_k - 1,
                        exclude_feedback,
                        allowed_feedback_types,
                        metadata_filter,
                        semantic_similarity_weight,
                    )
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
            metadata_filter: Dict[str, Any] = None,
            semantic_similarity_weight: Optional[float] = None,
    ) -> Tuple[str, List[str], List[Dict[str, Any]]]:
        """
        Retrieve relevant context from documents with detailed information using dual-model search

        This method performs:
        1. Legal document search using the configured legal-domain embeddings
        2. Feedback document search using SBERT embeddings (for semantic similarity)

        Args:
            query: Search query
            top_k: Number of top results to return
            exclude_feedback: If True, exclude feedback documents from search
            allowed_feedback_types: Filter feedback by type (e.g., ['status_change'])
            metadata_filter: NOT used for strict filtering (allows similar column discovery)
            semantic_similarity_weight: Optional per-run alpha override

        Returns:
            Tuple of (formatted context string, list of source names, list of document details)

            Document details is a list of dicts with keys:
            - 'source' (str): Source document name
            - 'priority_level' (int): Priority level (0-3, lower is higher priority)
            - 'priority_label' (str): Human-readable priority label
            - 'text_snippet' (str): Text snippet truncated to MAX_DOC_SNIPPET_LENGTH
            - 'similarity' (float): Similarity score
            - 'doc_source' (str): 'legal' or 'feedback'

        Raises:
            RAGError: If system is not initialized or retrieval fails
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
                query_text=query,  # Pass original query for SBERT embedding
                semantic_similarity_weight=semantic_similarity_weight,
            )
            context_str = self._format_results_from_doc_info(doc_info_list)

            estimated_tokens = len(context_str.split()) * 1.3
            if estimated_tokens > Settings.MAX_CONTEXT_TOKENS:
                print(f"[WARNING] Context too large ({estimated_tokens:.0f} tokens), reducing to top {top_k - 1} results")
                if top_k > 1:
                    return self.retrieve_with_details(
                        query,
                        top_k - 1,
                        exclude_feedback,
                        allowed_feedback_types,
                        metadata_filter,
                        semantic_similarity_weight,
                    )
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

        Args:
            query: Search query

        Returns:
            Query embedding as NumPy array
        """
        if Settings.EMBEDDING_PROVIDER == "local":
            return self.embedding_generator._generate_local_embeddings([query])

        response = self.client.embeddings.create(input=[query], model=Settings.LEGAL_EMBEDDING_MODEL)
        return np.array(response.data[0].embedding).reshape(1, -1)

    def _generate_feedback_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for query using SBERT (for feedback search)

        Args:
            query: Search query

        Returns:
            Query embedding as NumPy array
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
            query_text: str = None,
            semantic_similarity_weight: Optional[float] = None,
    ) -> np.ndarray:
        """
        Find most similar documents using dual-model search with 4-tier priority

        This function performs TWO separate searches:
        1. Legal documents: configured legal-domain embeddings

        For feedback search, metadata_filter is NOT applied strictly - instead,
        similar columns from past feedback are included to enable learning from
        semantically similar cases.

        Priority order and source-authority weights:

        Final score: score(d_i) = α * s_i + (1-α) * w_{p_i}

        Args:
            query_embedding: Query embedding for legal documents
            top_k: Number of results to return
            exclude_feedback: If True, exclude feedback documents
            allowed_feedback_types: Filter feedback by type (e.g., ['status_change'])
            metadata_filter: NOT used for strict filtering, only for logging
            query_text: Original query text for SBERT embedding (optional)
            semantic_similarity_weight: Optional per-run alpha override

        Returns:
            Indices of most similar documents (combined from both searches)
        """
        # Use the same validated configuration that is exposed by the web UI.
        ranking_config = RAGConfig.get_ranking_config(
            semantic_similarity_weight
        )
        priority_weights = ranking_config["authority_source_weights"]
        alpha = ranking_config["semantic_similarity_weight"]

        doc_info = []

        # ========== PART 1: Search Legal Documents ==========
        if self.embeddings is not None and len(self.embeddings) > 0:
            legal_similarities = cosine_similarity(query_embedding.reshape(1, -1), self.embeddings)[0]

            for idx, sim in enumerate(legal_similarities):
                priority_level = self.documents[idx].get("priority_level", 3)

                # Skip feedback documents in legal search (they are searched separately)
                if priority_level == 0:
                    continue

                # Get priority weight
                w_p = priority_weights.get(
                    priority_level,
                    RAGConfig.DEFAULT_AUTHORITY_WEIGHTS[3],
                )

                # Calculate weighted score: score(d_i) = alpha * s_i + (1-alpha) * w_{p_i}
                score = alpha * sim + (1 - alpha) * w_p

                doc_info.append({
                    "idx": idx,
                    "similarity": float(sim),
                    "priority_level": priority_level,
                    "score": float(score),
                    "source": "legal",
                    "doc": self.documents[idx]
                })

        # ========== PART 2: Search Feedback Documents (SBERT) ==========
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

                # Apply minimum similarity threshold for feedback (avoid noise)
                MIN_FEEDBACK_SIMILARITY = float(os.environ.get("MIN_FEEDBACK_SIMILARITY", 0.3))
                if sim < MIN_FEEDBACK_SIMILARITY:
                    continue

                # Priority level 0 for expert feedback
                priority_level = 0
                w_p = priority_weights[0]

                # Calculate weighted score
                score = alpha * sim + (1 - alpha) * w_p

                # Use negative index to distinguish from legal documents
                doc_info.append({
                    "idx": -(idx + 1),  # Negative index for feedback docs
                    "similarity": float(sim),
                    "priority_level": priority_level,
                    "score": float(score),
                    "source": "feedback",
                    "doc": feedback_doc
                })

        # Sort by score descending
        doc_info.sort(key=lambda x: -x["score"])

        # Return top_k results
        top_indices = [doc["idx"] for doc in doc_info[:top_k]]

        return np.array(top_indices), doc_info[:top_k]

    def _format_results_from_doc_info(self, doc_info_list: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into a string with priority labels

        Args:
            doc_info_list: List of document info dictionaries from _find_similar_documents

        Returns:
            Formatted string of documents
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

        Args:
            doc_info_list: List of document info dictionaries from _find_similar_documents

        Returns:
            List of unique source document names
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
