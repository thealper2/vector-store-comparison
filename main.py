from __future__ import annotations

import json
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import (
    FAISS,
    Chroma,
    ElasticsearchStore,
    Milvus,
    OpenSearchVectorSearch,
)
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

#  Step 1 parameters
BEST_CHUNK_SIZE: int = 512
BEST_CHUNK_OVERLAP: int = 0
BEST_EMBEDDING: str = "BAAI/bge-m3"

# BM25 parameters (Step 1)
BEST_BM25_K1: float = 0.5
BEST_BM25_B: float = 0.75

# RRF parameters (Step 1)
BEST_RRF_K: int = 10
BEST_FETCH_TOP_K: int = 10

#  Step 2 parameters
BEST_RERANKER: str = "seroe/bge-reranker-v2-m3-turkish-triplet"
BEST_INITIAL_K: int = 20  # Candidate count pulled from hybrid RRF
BEST_FINAL_K: int = 5  # Results returned after reranking

#  LLM
OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_MODEL: str = "llama3.1:8b"

#  Vector store connections (Docker default ports) ─
QDRANT_URL: str = "http://localhost:6333"
QDRANT_COLLECTION: str = "vs_comparison"

MILVUS_HOST: str = "localhost"
MILVUS_PORT: int = 19530
MILVUS_COLLECTION: str = "vs_comparison"

CHROMA_HOST: str = "localhost"
CHROMA_PORT: int = 8000
CHROMA_COLLECTION: str = "vs_comparison"

ELASTICSEARCH_URL: str = "http://localhost:9200"
ELASTICSEARCH_INDEX: str = "vs_comparison"

OPENSEARCH_URL: str = "http://localhost:9201"
OPENSEARCH_INDEX: str = "vs_comparison"

# CockroachDB with pgvector extension
COCKROACHDB_URL: str = "postgresql://root@localhost:26257/defaultdb?sslmode=disable"
COCKROACHDB_TABLE: str = "vs_comparison"

FAISS_INDEX_PATH: str = "./faiss_index"

#  Evaluation ─
EVAL_K_VALUES: list[int] = [1, 3, 5]

#  Paths
DATA_CSV_PATH: str = "data.csv"
EVAL_CSV_PATH: str = "evaluation_dataset_2.csv"
RESULTS_JSON_PATH: str = "results.json"
RESULTS_CSV_PATH: str = "results.csv"

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("vector_store_comparison.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════


def extract_url_metadata(url: str) -> dict[str, Any]:
    """
    Parse a URL and return structured metadata for use as document metadata.

    Path segments are split on '/'. The first 3 non-empty segments (scheme,
    domain, top-level path) are skipped; remaining segments are stored as
    individual 'sub_path_N' keys and joined as 'sub_paths'.

    Example:
        'https://example.com/docs/api/v2/auth/login'
        → sub_paths = 'auth/login'
        → sub_path_1 = 'auth', sub_path_2 = 'login'

    Args:
        url: Full URL string.

    Returns:
        Metadata dict with keys: url, domain, full_path, sub_paths, sub_path_N...
    """
    metadata: dict[str, Any] = {"url": url}
    try:
        parsed = urlparse(url)
        metadata["domain"] = parsed.netloc
        metadata["full_path"] = parsed.path

        # Split path into non-empty segments, skip the first 3
        segments = [s for s in parsed.path.split("/") if s]
        remaining = segments[3:]

        metadata["sub_paths"] = "/".join(remaining) if remaining else ""
        for idx, seg in enumerate(remaining, start=1):
            metadata[f"sub_path_{idx}"] = seg

    except Exception as exc:
        logger.warning("URL parse failed for '%s': %s", url, exc)
        metadata.update({"domain": "", "full_path": "", "sub_paths": ""})

    return metadata


def load_and_validate_csv(csv_path: str) -> pd.DataFrame:
    """
    Load the data CSV, validate required columns, and clean empty rows.

    Args:
        csv_path: Path to the CSV with 'Content' and 'URL' columns.

    Returns:
        Cleaned DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are absent.
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Data CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, usecols=["Processed", "URL"])
    # Preserve semantic mapping: Processed -> Content, URL -> URL
    df.columns = ["URL", "Content"]
    logger.info("CSV loaded: %d rows from '%s'", len(df), csv_path)

    missing = {"Content", "URL"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    before = len(df)
    df = df.dropna(subset=["Content", "URL"]).reset_index(drop=True)
    df["Content"] = df["Content"].astype(str).str.strip()
    df["URL"] = df["URL"].astype(str).str.strip()
    df = df[df["Content"] != ""].reset_index(drop=True)

    logger.info("Rows after cleaning: %d (dropped %d)", len(df), before - len(df))
    return df


def load_evaluation_csv(csv_path: str) -> pd.DataFrame:
    """
    Load the evaluation CSV and validate required columns.

    Args:
        csv_path: Path to the CSV with 'Question' and 'Answer' columns.

    Returns:
        Cleaned DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are absent.
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Evaluation CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = {"Question", "Answer"} - set(df.columns)
    if missing:
        raise ValueError(f"Evaluation CSV missing columns: {missing}")

    df = df.dropna(subset=["Question", "Answer"]).reset_index(drop=True)
    df["Question"] = df["Question"].astype(str).str.strip()
    df["Answer"] = df["Answer"].astype(str).str.strip()
    logger.info("Evaluation CSV loaded: %d rows", len(df))
    return df


def build_documents(df: pd.DataFrame) -> list[Document]:
    """
    Convert DataFrame rows into LangChain Document objects.

    Metadata includes original URL plus all sub-path segments from
    :func:`extract_url_metadata`.

    Args:
        df: DataFrame with 'Content' and 'URL' columns.

    Returns:
        List of LangChain Document objects.
    """
    docs = [
        Document(
            page_content=row["Content"],
            metadata=extract_url_metadata(row["URL"]),
        )
        for _, row in df.iterrows()
    ]
    logger.info("Built %d documents", len(docs))
    return docs


def chunk_documents(
    documents: list[Document],
    chunk_size: int = BEST_CHUNK_SIZE,
    chunk_overlap: int = BEST_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split documents into smaller chunks using RecursiveCharacterTextSplitter.

    Parent document metadata is preserved on every child chunk.

    Args:
        documents:    Input documents to split.
        chunk_size:   Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(
        "Chunked: %d docs → %d chunks (size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING MODEL
# ══════════════════════════════════════════════════════════════════════════════


def get_embedding_model(model_name: str = BEST_EMBEDDING) -> HuggingFaceEmbeddings:
    """
    Load the HuggingFace sentence embedding model.

    Uses CPU inference with L2-normalised embeddings (required for cosine similarity).

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        HuggingFaceEmbeddings instance.
    """
    logger.info("Loading embedding model: %s", model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ══════════════════════════════════════════════════════════════════════════════
# VECTOR STORE BUILDERS  (one function per backend, all start fresh)
# ══════════════════════════════════════════════════════════════════════════════


def build_qdrant(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
) -> QdrantVectorStore:
    """
    Create a fresh Qdrant collection and ingest document chunks.

    Drops the collection if it already exists to guarantee a clean state.

    Args:
        chunks:     Chunked documents to ingest.
        embeddings: Embedding model.

    Returns:
        LangChain Qdrant VectorStore.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    logger.info("[Qdrant] Connecting to %s", QDRANT_URL)
    client = QdrantClient(url=QDRANT_URL)

    # Drop existing collection for a clean start
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION in existing:
        client.delete_collection(QDRANT_COLLECTION)
        logger.info("[Qdrant] Dropped existing collection '%s'", QDRANT_COLLECTION)

    dim = len(embeddings.embed_query("test"))
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    store = QdrantVectorStore(
        client=client, collection_name=QDRANT_COLLECTION, embedding=embeddings
    )
    store.add_documents(chunks)
    logger.info("[Qdrant] Ingested %d chunks", len(chunks))
    return store


def build_faiss(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
) -> FAISS:
    """
    Build a FAISS index from scratch and persist it to FAISS_INDEX_PATH.

    Removes existing index directory before building to ensure a fresh start.

    Args:
        chunks:     Chunked documents to ingest.
        embeddings: Embedding model.

    Returns:
        LangChain FAISS VectorStore.
    """
    logger.info("[FAISS] Building index at '%s'", FAISS_INDEX_PATH)

    if os.path.exists(FAISS_INDEX_PATH):
        shutil.rmtree(FAISS_INDEX_PATH)

    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(FAISS_INDEX_PATH)
    logger.info("[FAISS] Saved index (%d chunks)", len(chunks))
    return store


def build_milvus(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
) -> Milvus:
    """
    Create a fresh Milvus collection and ingest document chunks.

    Drops the collection if it already exists.

    Args:
        chunks:     Chunked documents to ingest.
        embeddings: Embedding model.

    Returns:
        LangChain Milvus VectorStore.
    """
    from pymilvus import connections, utility

    logger.info("[Milvus] Connecting to %s:%d", MILVUS_HOST, MILVUS_PORT)
    connections.connect(host=MILVUS_HOST, port=str(MILVUS_PORT))

    if utility.has_collection(MILVUS_COLLECTION):
        utility.drop_collection(MILVUS_COLLECTION)
        logger.info("[Milvus] Dropped collection '%s'", MILVUS_COLLECTION)

    store = Milvus.from_documents(
        chunks,
        embeddings,
        connection_args={"host": MILVUS_HOST, "port": str(MILVUS_PORT)},
        collection_name=MILVUS_COLLECTION,
    )
    logger.info("[Milvus] Ingested %d chunks", len(chunks))
    return store


def build_chroma(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
) -> Chroma:
    """
    Create a fresh Chroma collection via HTTP client and ingest document chunks.

    Deletes the collection if it already exists.

    Args:
        chunks:     Chunked documents to ingest.
        embeddings: Embedding model.

    Returns:
        LangChain Chroma VectorStore.
    """
    import chromadb

    logger.info("[Chroma] Connecting to %s:%d", CHROMA_HOST, CHROMA_PORT)
    http_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    existing = [c.name for c in http_client.list_collections()]
    if CHROMA_COLLECTION in existing:
        http_client.delete_collection(CHROMA_COLLECTION)
        logger.info("[Chroma] Deleted collection '%s'", CHROMA_COLLECTION)

    store = Chroma(
        client=http_client,
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
    )
    store.add_documents(chunks)
    logger.info("[Chroma] Ingested %d chunks", len(chunks))
    return store


def build_inmemory(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
) -> InMemoryVectorStore:
    """
    Build an in-memory vector store (no persistence).

    Useful as a fast baseline — no Docker service required.

    Args:
        chunks:     Chunked documents to ingest.
        embeddings: Embedding model.

    Returns:
        LangChain InMemoryVectorStore.
    """
    logger.info("[InMemory] Building in-memory store (%d chunks)", len(chunks))
    store = InMemoryVectorStore(embedding=embeddings)
    store.add_documents(chunks)
    logger.info("[InMemory] Done")
    return store


def build_elasticsearch(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
) -> ElasticsearchStore:
    """
    Create a fresh Elasticsearch index and ingest document chunks.

    Deletes the index if it already exists (security is disabled for local dev).

    Args:
        chunks:     Chunked documents to ingest.
        embeddings: Embedding model.

    Returns:
        LangChain ElasticsearchStore.
    """
    from elasticsearch import Elasticsearch

    logger.info("[Elasticsearch] Connecting to %s", ELASTICSEARCH_URL)
    es_client = Elasticsearch(ELASTICSEARCH_URL)

    if es_client.indices.exists(index=ELASTICSEARCH_INDEX):
        es_client.indices.delete(index=ELASTICSEARCH_INDEX)
        logger.info("[Elasticsearch] Deleted index '%s'", ELASTICSEARCH_INDEX)

    store = ElasticsearchStore.from_documents(
        chunks,
        embeddings,
        es_url=ELASTICSEARCH_URL,
        index_name=ELASTICSEARCH_INDEX,
    )
    logger.info("[Elasticsearch] Ingested %d chunks", len(chunks))
    return store


def build_cockroachdb(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
) -> Any:
    """
    Create a pgvector table in CockroachDB and ingest document chunks.

    Uses langchain_postgres PGVector which is compatible with CockroachDB's
    built-in pgvector extension. The 'pre_delete_collection=True' flag drops
    and recreates the table for a fresh start.

    Prerequisite: Run once after container start:
        docker exec -it cockroachdb ./cockroach sql --insecure \\
          --execute="CREATE EXTENSION IF NOT EXISTS vector;"

    Args:
        chunks:     Chunked documents to ingest.
        embeddings: Embedding model.

    Returns:
        LangChain PGVector VectorStore.
    """
    from langchain_postgres import PGVector

    logger.info("[CockroachDB] Connecting via pgvector: %s", COCKROACHDB_URL)
    store = PGVector.from_documents(
        chunks,
        embeddings,
        connection=COCKROACHDB_URL,
        collection_name=COCKROACHDB_TABLE,
        pre_delete_collection=True,  # drop & recreate for a fresh state
    )
    logger.info("[CockroachDB] Ingested %d chunks", len(chunks))
    return store


def build_opensearch(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
) -> OpenSearchVectorSearch:
    """
    Create a fresh OpenSearch index and ingest document chunks.

    Uses the built-in FAISS k-NN engine with cosine similarity.
    Security is disabled for local development.

    Args:
        chunks:     Chunked documents to ingest.
        embeddings: Embedding model.

    Returns:
        LangChain OpenSearchVectorSearch VectorStore.
    """
    from opensearchpy import OpenSearch

    logger.info("[OpenSearch] Connecting to %s", OPENSEARCH_URL)
    os_client = OpenSearch(OPENSEARCH_URL)

    if os_client.indices.exists(index=OPENSEARCH_INDEX):
        os_client.indices.delete(index=OPENSEARCH_INDEX)
        logger.info("[OpenSearch] Deleted index '%s'", OPENSEARCH_INDEX)

    store = OpenSearchVectorSearch.from_documents(
        chunks,
        embeddings,
        opensearch_url=OPENSEARCH_URL,
        index_name=OPENSEARCH_INDEX,
        engine="faiss",
        space_type="cosinesimil",
    )
    logger.info("[OpenSearch] Ingested %d chunks", len(chunks))
    return store


# Store name → builder function registry
STORE_BUILDERS: dict[str, Any] = {
    "qdrant": build_qdrant,
    "faiss": build_faiss,
    "milvus": build_milvus,
    "chroma": build_chroma,
    "inmemory": build_inmemory,
    "elasticsearch": build_elasticsearch,
    "cockroachdb": build_cockroachdb,
    "opensearch": build_opensearch,
}


def build_all_stores(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
) -> dict[str, Any]:
    """
    Build all 8 vector stores sequentially.

    Failed stores are skipped with an error log; remaining stores still run.

    Args:
        chunks:     Chunked documents to ingest into every store.
        embeddings: Shared embedding model instance.

    Returns:
        Dict mapping store name → VectorStore instance (only successful builds).
    """
    built: dict[str, Any] = {}
    for name, builder in STORE_BUILDERS.items():
        logger.info("━━ Building: %s ━━", name.upper())
        try:
            built[name] = builder(chunks, embeddings)
        except Exception as exc:
            logger.error("Store '%s' FAILED — skipping: %s", name, exc)
    logger.info(
        "Built %d / %d stores: %s", len(built), len(STORE_BUILDERS), list(built.keys())
    )
    return built


# ══════════════════════════════════════════════════════════════════════════════
# BM25 RETRIEVER
# ══════════════════════════════════════════════════════════════════════════════


def build_bm25_retriever(
    documents: list[Document],
    k: int = BEST_FETCH_TOP_K,
    k1: float = BEST_BM25_K1,
    b: float = BEST_BM25_B,
) -> BM25Retriever:
    """
    Build a BM25 retriever and apply custom k1/b parameters.

    LangChain's BM25Retriever does not expose k1/b as constructor kwargs, so
    they are patched directly onto the internal rank_bm25 object after creation.

    Args:
        documents: Chunked documents to index.
        k:         Number of top documents to retrieve.
        k1:        BM25 term-frequency saturation parameter.
        b:         BM25 document-length normalisation parameter.

    Returns:
        Configured BM25Retriever.
    """
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k
    try:
        # Patch internal BM25Okapi parameters
        retriever.vectorizer.k1 = k1  # type: ignore[attr-defined]
        retriever.vectorizer.b = b  # type: ignore[attr-defined]
    except AttributeError:
        logger.warning("Could not patch BM25 k1/b — library API may have changed")
    logger.info("BM25 retriever built (k=%d, k1=%.2f, b=%.2f)", k, k1, b)
    return retriever


# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL PIPELINE  (RRF + Reranker)
# ══════════════════════════════════════════════════════════════════════════════


def load_reranker(model_name: str = BEST_RERANKER) -> CrossEncoder:
    """
    Load a CrossEncoder reranker model from HuggingFace.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Loaded CrossEncoder instance.
    """
    logger.info("Loading reranker: %s", model_name)
    reranker = CrossEncoder(model_name)
    logger.info("Reranker loaded")
    return reranker


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    rrf_k: int = BEST_RRF_K,
) -> list[tuple[Document, float]]:
    """
    Merge multiple ranked document lists with Reciprocal Rank Fusion (RRF).

    RRF score:  score(d) = Σ  1 / (k + rank(d, list_i))

    Documents are deduplicated by page_content; the highest-scoring copy is kept.

    Args:
        ranked_lists: Each element is one retriever's ordered result list.
        rrf_k:        Smoothing constant.

    Returns:
        List of (Document, score) tuples sorted by descending score.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            doc_map[key] = doc

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[content], score) for content, score in merged]


def rerank_documents(
    query: str,
    documents: list[Document],
    reranker: CrossEncoder,
    final_k: int = BEST_FINAL_K,
) -> list[Document]:
    """
    Re-score candidate documents with a cross-encoder and return top-k.

    Args:
        query:     User query string.
        documents: Candidate documents from RRF.
        reranker:  Loaded CrossEncoder instance.
        final_k:   Number of top documents to return.

    Returns:
        Top-k documents by reranker score (descending).
    """
    if not documents:
        return []
    pairs = [(query, doc.page_content) for doc in documents]
    try:
        scores: list[float] = reranker.predict(pairs).tolist()
    except Exception as exc:
        logger.error("Reranker prediction failed: %s", exc)
        return documents[:final_k]

    scored = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:final_k]]


def hybrid_retrieve(
    query: str,
    vector_retriever: Any,
    bm25_retriever: BM25Retriever,
    reranker: CrossEncoder,
    initial_k: int = BEST_INITIAL_K,
    final_k: int = BEST_FINAL_K,
    rrf_k: int = BEST_RRF_K,
) -> tuple[list[Document], float]:
    """
    Full hybrid retrieval pipeline for a single query.

    Steps:
        1. Dense (vector) retrieval      → initial_k candidates
        2. Sparse BM25 retrieval         → initial_k candidates
        3. Reciprocal Rank Fusion (RRF)  → merged & re-ranked list
        4. Cross-encoder reranking       → final_k documents

    Args:
        query:            User query.
        vector_retriever: Dense retriever from a vector store.
        bm25_retriever:   Sparse BM25 retriever.
        reranker:         CrossEncoder reranker.
        initial_k:        Candidates to pull from each retriever before RRF.
        final_k:          Documents to return after reranking.
        rrf_k:            RRF smoothing constant.

    Returns:
        Tuple of (final_documents, elapsed_seconds).
    """
    t_start = time.perf_counter()

    # 1. Dense retrieval
    try:
        vector_retriever.search_kwargs = {"k": initial_k}  # type: ignore[attr-defined]
        dense_docs: list[Document] = vector_retriever.invoke(query)
    except Exception as exc:
        logger.warning("Dense retrieval failed: %s", exc)
        dense_docs = []

    # 2. BM25 retrieval
    try:
        bm25_retriever.k = initial_k
        sparse_docs: list[Document] = bm25_retriever.invoke(query)
    except Exception as exc:
        logger.warning("BM25 retrieval failed: %s", exc)
        sparse_docs = []

    # 3. RRF fusion — merge both ranked lists
    fused = reciprocal_rank_fusion([dense_docs, sparse_docs], rrf_k=rrf_k)
    candidates = [doc for doc, _ in fused[:initial_k]]

    # 4. Reranking — cross-encoder selects final top-k
    final_docs = rerank_documents(query, candidates, reranker, final_k=final_k)

    elapsed = time.perf_counter() - t_start
    return final_docs, elapsed


# ══════════════════════════════════════════════════════════════════════════════
# LLM GENERATION  (Ollama — answer in Turkish)
# ══════════════════════════════════════════════════════════════════════════════


def get_llm(temperature: float = 0.0) -> OllamaLLM:
    """
    Initialise and return an Ollama LLM instance.

    Args:
        temperature: Sampling temperature (0 = deterministic).

    Returns:
        OllamaLLM instance.
    """
    logger.info("Initialising Ollama LLM: %s @ %s", OLLAMA_MODEL, OLLAMA_BASE_URL)
    return OllamaLLM(
        model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=temperature
    )


def build_rag_prompt(question: str, context_docs: list[Document]) -> str:
    """
    Build the RAG prompt from the question and retrieved context documents.

    The prompt is written in English to maximise instruction-following quality,
    but the model is explicitly instructed to answer in Turkish.

    Args:
        question:     User question.
        context_docs: Retrieved documents to use as context.

    Returns:
        Formatted prompt string.
    """
    context_parts: list[str] = []
    for i, doc in enumerate(context_docs, start=1):
        source = doc.metadata.get("url", "unknown")
        context_parts.append(f"[Document {i} | Source: {source}]\n{doc.page_content}")

    context_text = "\n\n".join(context_parts)

    return f"""You are a helpful assistant. Use ONLY the provided context documents to answer the question.
If the answer is not found in the context, say "Bu sorunun cevabını verilen belgelerde bulamadım."
IMPORTANT: Your answer MUST be written entirely in Turkish.

--- CONTEXT START ---
{context_text}
--- CONTEXT END ---

Question: {question}

Answer (in Turkish):"""


def generate_answer(
    question: str,
    context_docs: list[Document],
    llm: OllamaLLM,
) -> str:
    """
    Generate a Turkish RAG answer using the Ollama LLM.

    Args:
        question:     User question.
        context_docs: Retrieved documents for context.
        llm:          OllamaLLM instance.

    Returns:
        Generated answer string in Turkish.
    """
    if not context_docs:
        return "Bu sorunun cevabını verilen belgelerde bulamadım."

    prompt = build_rag_prompt(question, context_docs)
    try:
        return llm.invoke(prompt).strip()
    except Exception as exc:
        logger.error("LLM generation failed: %s", exc)
        return f"[Hata] Yanıt oluşturulamadı: {exc}"


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════════════════


def is_relevant(doc: Document, ground_truth_answer: str) -> bool:
    """
    Judge whether a retrieved document is relevant to the ground-truth answer.

    Relevance is determined by token overlap: a document is considered relevant
    if at least 50% of the meaningful tokens (length > 3) from the answer appear
    (case-insensitive substring match) in the document content.

    Args:
        doc:                  Retrieved document.
        ground_truth_answer:  Reference answer string.

    Returns:
        True if the document is judged relevant.
    """
    content_lower = doc.page_content.lower()
    tokens = [t.lower() for t in ground_truth_answer.split() if len(t) > 3]
    if not tokens:
        return ground_truth_answer.lower() in content_lower
    matches = sum(1 for t in tokens if t in content_lower)
    return matches / len(tokens) >= 0.5


def get_relevance_flags(docs: list[Document], answer: str) -> list[int]:
    """
    Convert a ranked document list into binary relevance flags (1=relevant, 0=not).

    Args:
        docs:   Ranked retrieved documents.
        answer: Ground-truth answer string.

    Returns:
        List of 0/1 integers aligned with docs.
    """
    return [int(is_relevant(doc, answer)) for doc in docs]


def mrr_at_k(flags: list[int], k: int) -> float:
    """
    Compute Mean Reciprocal Rank at k for a single query.

    Returns 1/rank of the first relevant document within top-k, or 0 if none found.

    Args:
        flags: Binary relevance list.
        k:     Cut-off rank.

    Returns:
        MRR@k in [0, 1].
    """
    for rank, rel in enumerate(flags[:k], start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def recall_at_k(flags: list[int], k: int, total_relevant: int) -> float:
    """
    Compute Recall at k for a single query.

    Recall@k = (relevant in top-k) / total_relevant

    Args:
        flags:          Binary relevance list.
        k:              Cut-off rank.
        total_relevant: Total number of relevant docs in the corpus.

    Returns:
        Recall@k in [0, 1].
    """
    if total_relevant <= 0:
        return 0.0
    return min(sum(flags[:k]) / total_relevant, 1.0)


def precision_at_k(flags: list[int], k: int) -> float:
    """
    Compute Precision at k for a single query.

    Precision@k = (relevant in top-k) / k

    Args:
        flags: Binary relevance list.
        k:     Cut-off rank.

    Returns:
        Precision@k in [0, 1].
    """
    return sum(flags[:k]) / k if k > 0 else 0.0


def ndcg_at_k(flags: list[int], k: int, total_relevant: int) -> float:
    """
    Compute normalised Discounted Cumulative Gain at k for a single query.

    DCG@k  = Σ rel_i / log2(i+1)
    IDCG@k = DCG of the ideal ranking with min(total_relevant, k) relevant docs
    nDCG@k = DCG@k / IDCG@k

    Args:
        flags: Binary relevance list.
        k:              Cut-off rank.
        total_relevant: Total number of relevant docs in the corpus.

    Returns:
        nDCG@k in [0, 1].
    """
    dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(flags[:k], start=1))
    ideal_hits = min(max(total_relevant, 0), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_store(
    store_name: str,
    vector_retriever: Any,
    bm25_retriever: BM25Retriever,
    reranker: CrossEncoder,
    eval_df: pd.DataFrame,
    corpus_docs: list[Document],
    k_values: list[int] = EVAL_K_VALUES,
    final_k: int = BEST_FINAL_K,
) -> dict[str, Any]:
    """
    Run the full evaluation loop for one vector store backend.

    For each (Question, Answer) pair:
        1. hybrid_retrieve → documents + elapsed time
        2. get_relevance_flags → binary relevance per doc
        3. Accumulate MRR@k, Recall@k, Precision@k, nDCG@k

    Averages all per-query scores into a single results dictionary.

    Args:
        store_name:       Human-readable name for logging.
        vector_retriever: Dense vector retriever.
        bm25_retriever:   Shared BM25 retriever.
        reranker:         CrossEncoder reranker.
        eval_df:          DataFrame with 'Question' and 'Answer' columns.
        corpus_docs:      Full retrieval corpus used to estimate total relevant docs.
        k_values:         Cut-off values for metric computation.
        final_k:          Max documents returned by hybrid_retrieve.

    Returns:
        Dict with keys: store, num_queries, mean_retrieval_time_s, MRR@k, Recall@k, etc.
    """
    logger.info("[Eval] Store: %s | Queries: %d", store_name, len(eval_df))

    # Accumulate per-query scores for each metric+k combination
    acc: dict[str, list[float]] = {
        f"{m}@{k}": [] for m in ("MRR", "Recall", "Precision", "nDCG") for k in k_values
    }
    times: list[float] = []
    total_relevant_cache: dict[str, int] = {}

    for idx, row in eval_df.iterrows():
        question: str = row["Question"]
        answer: str = row["Answer"]

        try:
            docs, elapsed = hybrid_retrieve(
                query=question,
                vector_retriever=vector_retriever,
                bm25_retriever=bm25_retriever,
                reranker=reranker,
                final_k=final_k,
            )
            times.append(elapsed)
        except Exception as exc:
            logger.warning("[Eval][%s] Query %s failed: %s", store_name, idx, exc)
            # Record zeros to keep the denominator consistent
            for key in acc:
                acc[key].append(0.0)
            times.append(0.0)
            continue

        flags = get_relevance_flags(docs, answer)
        if answer not in total_relevant_cache:
            total_relevant_cache[answer] = sum(
                1 for doc in corpus_docs if is_relevant(doc, answer)
            )
        total_relevant = total_relevant_cache[answer]

        for k in k_values:
            acc[f"MRR@{k}"].append(mrr_at_k(flags, k))
            acc[f"Recall@{k}"].append(recall_at_k(flags, k, total_relevant))
            acc[f"Precision@{k}"].append(precision_at_k(flags, k))
            acc[f"nDCG@{k}"].append(ndcg_at_k(flags, k, total_relevant))

    results: dict[str, Any] = {
        "store": store_name,
        "num_queries": len(eval_df),
        "mean_retrieval_time_s": round(sum(times) / len(times), 4) if times else 0.0,
    }
    for key, vals in acc.items():
        results[key] = round(sum(vals) / len(vals), 4) if vals else 0.0

    logger.info(
        "[Eval][%s] mean_time=%.3fs | MRR@%d=%.4f | nDCG@%d=%.4f",
        store_name,
        results["mean_retrieval_time_s"],
        k_values[-1],
        results.get(f"MRR@{k_values[-1]}", 0.0),
        k_values[-1],
        results.get(f"nDCG@{k_values[-1]}", 0.0),
    )
    return results


def build_results_table(all_results: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert per-store result dicts into a DataFrame sorted by best MRR@k.

    Args:
        all_results: List of dicts from evaluate_store().

    Returns:
        Sorted DataFrame with one row per store.
    """
    df = pd.DataFrame(all_results)
    sort_col = f"MRR@{max(EVAL_K_VALUES)}"
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """
    Orchestrate the full vector store comparison pipeline.

    Steps:
        1.  Load & chunk data CSV
        2.  Load evaluation CSV
        3.  Load embedding model
        4.  Load reranker
        5.  Build all 8 vector stores (fresh from scratch)
        6.  Build shared BM25 retriever
        7.  For each store: evaluate hybrid retrieval
        8.  Save results to JSON and CSV, print summary table
    """
    logger.info("=" * 70)
    logger.info("Vector Store Comparison — START")
    logger.info("=" * 70)

    #  1. Load and chunk data
    logger.info("Step 1 — Loading and chunking data CSV: %s", DATA_CSV_PATH)
    try:
        df = load_and_validate_csv(DATA_CSV_PATH)
        raw_docs = build_documents(df)
        chunks = chunk_documents(raw_docs)
    except (FileNotFoundError, ValueError) as exc:
        logger.critical("Data loading failed: %s", exc)
        sys.exit(1)

    #  2. Load evaluation data ─
    logger.info("Step 2 — Loading evaluation CSV: %s", EVAL_CSV_PATH)
    try:
        eval_df = load_evaluation_csv(EVAL_CSV_PATH)
    except (FileNotFoundError, ValueError) as exc:
        logger.critical("Evaluation data loading failed: %s", exc)
        sys.exit(1)

    logger.info(
        "Ready: %d raw docs | %d chunks | %d eval queries",
        len(raw_docs),
        len(chunks),
        len(eval_df),
    )

    #  3. Embedding model
    logger.info("Step 3 — Loading embedding model: %s", BEST_EMBEDDING)
    try:
        embeddings = get_embedding_model()
    except Exception as exc:
        logger.critical("Embedding model load failed: %s", exc)
        sys.exit(1)

    #  4. Reranker
    logger.info("Step 4 — Loading reranker: %s", BEST_RERANKER)
    try:
        reranker = load_reranker()
    except Exception as exc:
        logger.critical("Reranker load failed: %s", exc)
        sys.exit(1)

    #  5. Build all vector stores
    logger.info("Step 5 — Building all vector stores (fresh from scratch)")
    stores = build_all_stores(chunks, embeddings)

    if not stores:
        logger.critical("No vector stores were built — aborting")
        sys.exit(1)

    #  6. BM25 retriever (built once; shared across all stores)
    logger.info("Step 6 — Building shared BM25 retriever")
    try:
        bm25_retriever = build_bm25_retriever(chunks)
    except Exception as exc:
        logger.critical("BM25 retriever build failed: %s", exc)
        sys.exit(1)

    #  7. Evaluate each store
    logger.info("Step 7 — Evaluating %d stores", len(stores))
    all_results: list[dict[str, Any]] = []

    for store_name, store in stores.items():
        logger.info("━━━━━━━━ Evaluating: %s ━━━━━━━━", store_name.upper())
        try:
            vector_retriever = store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": BEST_FINAL_K},
            )
            result = evaluate_store(
                store_name=store_name,
                vector_retriever=vector_retriever,
                bm25_retriever=bm25_retriever,
                reranker=reranker,
                eval_df=eval_df,
                corpus_docs=chunks,
            )
            all_results.append(result)
        except Exception as exc:
            logger.error("Evaluation failed for '%s': %s", store_name, exc)
            all_results.append({"store": store_name, "error": str(exc)})

    #  8. Save and display results ─
    results_df = build_results_table(all_results)

    logger.info("\n%s", "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("%s", "=" * 70)
    print(results_df.to_string(index=False))

    try:
        with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info("Results saved → %s", RESULTS_JSON_PATH)
    except OSError as exc:
        logger.error("Failed to save JSON: %s", exc)

    try:
        results_df.to_csv(RESULTS_CSV_PATH, index=False, encoding="utf-8")
        logger.info("Results saved → %s", RESULTS_CSV_PATH)
    except OSError as exc:
        logger.error("Failed to save CSV: %s", exc)

    logger.info("=" * 70)
    logger.info("Vector Store Comparison — DONE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
