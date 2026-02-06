"""
RAG Engine Module
Integrates LightRAG for knowledge graph creation and Supabase for vector storage.
Provides context retrieval for chat and handbook generation.
"""

import os
import asyncio
import hashlib
import json
import logging
from typing import List, Optional

import numpy as np
from openai import OpenAI
from supabase import create_client, Client

from config import (
    XAI_API_KEY, XAI_BASE_URL, XAI_MODEL,
    SUPABASE_URL, SUPABASE_KEY,
    LIGHTRAG_WORKING_DIR, EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Embedding Helpers
# ─────────────────────────────────────────────────────────────

def get_openai_client() -> OpenAI:
    """Create OpenAI-compatible client for xAI / Grok."""
    return OpenAI(api_key=XAI_API_KEY, base_url=XAI_BASE_URL)


def compute_embedding(text: str) -> List[float]:
    """Compute embedding — uses fast local TF-IDF-style hashing.
    Groq and many free providers don't support embedding endpoints,
    so we use a deterministic hash-based approach that's instant."""
    return _fallback_embedding(text)


def _fallback_embedding(text: str, dim: int = 384) -> List[float]:
    """Fast deterministic TF-IDF-style embedding using word hashing."""
    vec = np.zeros(dim, dtype=np.float32)
    words = text.lower().split()
    for i, word in enumerate(words):
        h = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
        idx = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        # Weight by inverse position (earlier words matter more)
        weight = 1.0 / (1.0 + 0.001 * i)
        vec[idx] += sign * weight
    # Normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


# ─────────────────────────────────────────────────────────────
#  Supabase Vector Store
# ─────────────────────────────────────────────────────────────

class SupabaseVectorStore:
    """Manages document chunks and embeddings in Supabase with pgvector."""

    TABLE_NAME = "document_chunks"

    def __init__(self):
        self.local_store: List[dict] = []  # Always initialize for fallback
        if SUPABASE_URL and SUPABASE_KEY:
            self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
            self.enabled = True
            self._ensure_table()
        else:
            self.client = None
            self.enabled = False
            logger.warning("Supabase not configured. Using local storage fallback.")

    def _ensure_table(self):
        """Create the document_chunks table if it doesn't exist via RPC or skip."""
        try:
            # Try to query the table to see if it exists
            self.client.table(self.TABLE_NAME).select("id").limit(1).execute()
        except Exception:
            logger.info("Table may not exist yet. It will be created via Supabase dashboard or migration.")

    def add_chunks(self, chunks: List[str], source: str = "uploaded_pdf"):
        """Store text chunks with their embeddings."""
        for i, chunk in enumerate(chunks):
            embedding = compute_embedding(chunk)
            record = {
                "content": chunk,
                "source": source,
                "chunk_index": i,
                "embedding": embedding,
                "metadata": json.dumps({"source": source, "index": i}),
            }
            if self.enabled:
                try:
                    self.client.table(self.TABLE_NAME).insert(record).execute()
                except Exception as e:
                    logger.error(f"Supabase insert failed: {e}")
                    self.local_store.append(record)
            else:
                self.local_store.append(record)

    def search(self, query: str, top_k: int = 10) -> List[str]:
        """Retrieve most relevant chunks for a query."""
        query_embedding = compute_embedding(query)

        if self.enabled:
            try:
                # Use Supabase RPC for vector similarity search
                response = self.client.rpc(
                    "match_documents",
                    {
                        "query_embedding": query_embedding,
                        "match_threshold": 0.3,
                        "match_count": top_k,
                    }
                ).execute()
                return [r["content"] for r in response.data]
            except Exception as e:
                logger.warning(f"Supabase search failed ({e}), falling back to local")

        # Local fallback: cosine similarity
        return self._local_search(query_embedding, top_k)

    def _local_search(self, query_embedding: List[float], top_k: int) -> List[str]:
        """Cosine similarity search on local store."""
        if not self.local_store:
            return []

        q = np.array(query_embedding)
        scores = []
        for record in self.local_store:
            emb = np.array(record["embedding"])
            # Cosine similarity
            sim = np.dot(q, emb) / (np.linalg.norm(q) * np.linalg.norm(emb) + 1e-10)
            scores.append((sim, record["content"]))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [content for _, content in scores[:top_k]]

    def get_all_content(self) -> str:
        """Retrieve all stored content (for handbook generation context)."""
        if self.enabled:
            try:
                response = self.client.table(self.TABLE_NAME).select("content").execute()
                return "\n\n".join([r["content"] for r in response.data])
            except Exception:
                pass

        return "\n\n".join([r["content"] for r in self.local_store])

    def clear(self):
        """Clear all stored documents."""
        if self.enabled:
            try:
                self.client.table(self.TABLE_NAME).delete().neq("id", 0).execute()
            except Exception:
                pass
        self.local_store = []


# ─────────────────────────────────────────────────────────────
#  LightRAG Knowledge Graph
# ─────────────────────────────────────────────────────────────

class KnowledgeGraph:
    """
    LightRAG-based knowledge graph for enhanced retrieval.
    Falls back to vector search if LightRAG is not available.
    """

    def __init__(self, vector_store: SupabaseVectorStore):
        self.vector_store = vector_store
        self.rag = None
        self.initialized = False
        self._try_init_lightrag()

    def _try_init_lightrag(self):
        """Attempt to initialize LightRAG."""
        try:
            from lightrag import LightRAG, QueryParam
            from lightrag.llm.openai import openai_complete_if_cache, openai_embed
            
            async def grok_model_func(prompt, **kwargs) -> str:
                return await openai_complete_if_cache(
                    XAI_MODEL,
                    prompt,
                    api_key=XAI_API_KEY,
                    base_url=XAI_BASE_URL,
                    **kwargs
                )

            async def embedding_func(texts: list[str]) -> np.ndarray:
                return await openai_embed(
                    texts,
                    model=EMBEDDING_MODEL,
                    api_key=XAI_API_KEY,
                    base_url=XAI_BASE_URL,
                )

            self.rag = LightRAG(
                working_dir=LIGHTRAG_WORKING_DIR,
                llm_model_func=grok_model_func,
                embedding_func=embedding_func,
            )
            self.initialized = True
            logger.info("LightRAG initialized successfully")
        except ImportError:
            logger.warning("LightRAG not installed. Using vector-only search.")
        except Exception as e:
            logger.warning(f"LightRAG init failed: {e}. Using vector-only search.")

    async def ingest(self, text: str):
        """Add document text to the knowledge graph."""
        if self.initialized and self.rag:
            try:
                await self.rag.ainsert(text)
                logger.info("Text ingested into LightRAG knowledge graph")
            except Exception as e:
                logger.error(f"LightRAG ingestion failed: {e}")

    def ingest_sync(self, text: str):
        """Synchronous wrapper for ingestion."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    pool.submit(asyncio.run, self.ingest(text)).result()
            else:
                loop.run_until_complete(self.ingest(text))
        except RuntimeError:
            asyncio.run(self.ingest(text))

    async def query(self, question: str, mode: str = "hybrid") -> str:
        """Query the knowledge graph."""
        if self.initialized and self.rag:
            try:
                from lightrag import QueryParam
                result = await self.rag.aquery(
                    question,
                    param=QueryParam(mode=mode)
                )
                return result
            except Exception as e:
                logger.warning(f"LightRAG query failed: {e}")

        # Fallback to vector search
        chunks = self.vector_store.search(question, top_k=8)
        if chunks:
            return "\n\n---\n\n".join(chunks)
        return "No relevant context found."

    def query_sync(self, question: str, mode: str = "hybrid") -> str:
        """Synchronous wrapper for querying."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, self.query(question, mode)).result()
            else:
                return loop.run_until_complete(self.query(question, mode))
        except RuntimeError:
            return asyncio.run(self.query(question, mode))


# ─────────────────────────────────────────────────────────────
#  RAG Engine (Main Interface)
# ─────────────────────────────────────────────────────────────

class RAGEngine:
    """High-level RAG interface combining vector store and knowledge graph."""

    def __init__(self):
        self.vector_store = SupabaseVectorStore()
        self.knowledge_graph = KnowledgeGraph(self.vector_store)
        self.documents: List[str] = []  # Track uploaded document names

    def ingest_document(self, text: str, chunks: List[str], source: str):
        """Ingest a processed document into both vector store and knowledge graph."""
        # Store chunks in vector store (fast, local embeddings)
        self.vector_store.add_chunks(chunks, source=source)

        # Ingest into knowledge graph (only if LightRAG is available & initialized)
        if self.knowledge_graph.initialized:
            try:
                self.knowledge_graph.ingest_sync(text)
            except Exception as e:
                logger.warning(f"KG ingestion skipped: {e}")

        self.documents.append(source)
        logger.info(f"Document '{source}' ingested: {len(chunks)} chunks")

    def get_context(self, query: str, top_k: int = 8) -> str:
        """Retrieve relevant context for a query."""
        context_parts = []

        # Try knowledge graph if available
        if self.knowledge_graph.initialized:
            try:
                kg_result = self.knowledge_graph.query_sync(query)
                if kg_result and kg_result != "No relevant context found.":
                    context_parts.append("### Knowledge Graph Context:\n" + kg_result)
            except Exception as e:
                logger.debug(f"KG query skipped: {e}")

        # Vector search (always fast)
        vector_results = self.vector_store.search(query, top_k=top_k)
        if vector_results:
            context_parts.append("### Document Excerpts:\n" + "\n\n---\n\n".join(vector_results))

        return "\n\n".join(context_parts) if context_parts else "No relevant context found in uploaded documents."

    def get_full_context(self) -> str:
        """Get all stored content for comprehensive handbook generation."""
        return self.vector_store.get_all_content()

    def get_document_list(self) -> List[str]:
        """Return list of uploaded documents."""
        return self.documents

    def clear(self):
        """Clear all stored data."""
        self.vector_store.clear()
        self.documents = []
