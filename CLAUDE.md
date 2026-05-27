# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG-based intelligent legal Q&A system (Chinese law). Users ask legal questions and get answers grounded in a knowledge base of 80+ Chinese law full-text documents. The system uses three-way fusion retrieval (FAISS vector + BM25 keyword + Neo4j knowledge graph), DashScope reranker, and MiMo LLM for streaming generation.

## Running the Server

```bash
# Activate conda environment
conda activate pytorch

# Install dependencies
pip install -r requirements.txt

# Start server (port 8080)
python -m uvicorn app:app --host 127.0.0.1 --port 8080

# Alternative: use the batch script
start.bat
```

First startup builds FAISS vector index and BM25 index from `knowledge_base/` (takes 5-10 min). Subsequent startups load cached indexes from `law_faiss/` and `bm25_index.pkl`.

## Environment Configuration

Requires a `.env` file with at minimum:
- `MIMO_API_KEY` (or `DEEPSEEK_API_KEY` as fallback) - LLM API key
- `RERANKER_API_KEY` - DashScope reranker key
- Optional: `EMBEDDING_PROVIDER=local|dashscope`, `VECTOR_DB_PATH`, `DATABASE_URL`, `RELEVANCE_THRESHOLD`, retrieval weights
- Optional Neo4j: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`, `GRAPH_RETRIEVAL_WEIGHT`

## Architecture

**Core RAG pipeline** (`law_assistant/rag.py` - `DeepSeekApiRag` class):

1. Query analysis (1 LLM call): conversational rewrite + terminology rewrite + query decomposition + HyDE doc generation — all in a single `analyze_query()` call using structured JSON output
2. Parallel three-way fusion retrieval: main query + sub-queries + HyDE doc run concurrently via `ThreadPoolExecutor`, each doing FAISS vector search (weight 0.4) + BM25 keyword search (weight 0.3) + Neo4j graph search (weight 0.3)
3. Candidate deduplication and fusion across all sub-query results
4. DashScope reranker (`gte-rerank`) on merged candidates
5. Relevance threshold filtering: results below `RELEVANCE_THRESHOLD` (default 0.15) are discarded
6. Optional knowledge base filtering by `knowledge_base_id` metadata
7. Prompt assembly: retrieved context with citation tags `[来源N]` + conversation history + system prompt from `law_assistant/prompts.yaml`
8. LLM streaming response via SSE (`/ask_stream` endpoint)

Total LLM calls per request: 2 (1 for query analysis, 1 for answer generation)

**Web layer** (`app.py`): FastAPI with Jinja2 templates, cookie-based session auth (bcrypt passwords, SQLite via SQLAlchemy). SSE streaming at `/ask_stream` supports both GET and POST.

**Document processing** pipeline:
- `DocumentProcessor` (`law_assistant/processor.py`) auto-detects legal vs general documents by filename keywords and content patterns
- Legal docs → `DocumentSplitter` (`law_assistant/splitter.py`, splits by article number "第X条")
- General docs → `GeneralDocumentSplitter` (recursive character splitter, 200-char chunks)
- Long legal articles (>500 chars) get sub-split into 400-char chunks

**Key module relationships:**

- `app.py` creates `DeepSeekApiRag` instance on startup, which owns `BM25Retriever`, `DocumentProcessor`, `ConversationMemory`, `LegalKnowledgeGraph`
- All source modules are in `law_assistant/` package: `rag.py`, `bm25.py`, `memory.py`, `processor.py`, `splitter.py`, `redis_utils.py`, `graph.py`
- `ConversationMemory` (`law_assistant/memory.py`) is in-memory only (dict), keyed by `chat_{id}` — not persisted to DB
- Chat messages (user + bot) are persisted to SQLite `message` table
- Knowledge base per-chat selection: experts/admins can bind a knowledge base to a chat; creates a temporary `DeepSeekApiRag` with that KB's documents

## Key Design Decisions

- Embedding model: defaults to local `BAAI/bge-small-zh-v1.5` on CUDA; can switch to DashScope API via `EMBEDDING_PROVIDER=dashscope`
- BM25 uses jieba for Chinese tokenization
- BM25 index uses lazy batch rebuild (threshold: 50 pending documents) rather than per-document updates; single file upload forces immediate rebuild
- Knowledge graph: Neo4j backend (`law_assistant/graph.py`), rule-based entity extraction (laws, articles, chapters, concepts, citations). Graph search via entity linking + 1-2 hop subgraph traversal. Neo4j is optional — system degrades to dual-path (vector + BM25) if unavailable. Requires `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` in `.env`
- Retrieval weights: vector 0.4 + BM25 0.3 + graph 0.3 (configurable via env vars)
- Session tokens are simple `user_id:random_hex` format (not JWT)
- Document uploads (PDF/DOCX/TXT) are expert/admin only
- Query analysis: single LLM call (`analyze_query()`) does conversational rewrite + terminology rewrite + query decomposition + HyDE doc generation via structured JSON output
- Parallel retrieval: sub-queries and HyDE doc are retrieved concurrently via `ThreadPoolExecutor`
- Citation: context uses `[来源N]` tags; prompt instructs LLM to cite sources in answers
- Relevance threshold: `RELEVANCE_THRESHOLD` (default 0.15) filters out low-scoring reranker results
- Document deletion: removes text chunks from BM25 index before deleting file (FAISS vectors not removed — full rebuild needed)
- Document upload: uses FastAPI `BackgroundTasks` for async index building
- Conversation memory: in-memory cache with DB fallback (`_load_from_db`)
- Multi-knowledge base: `knowledge_base_id` passed to `retrieve_documents()` for metadata filtering, no temp RAG instances
