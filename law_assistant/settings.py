"""Centralized configuration via Pydantic Settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── LLM ──
    mimo_api_key: str | None = None
    mimo_base_url: str = "https://token-plan-cn.xiaomimimo.com/v1"
    mimo_model: str = "mimo-v2.5-pro"
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    # ── Embedding ──
    embedding_provider: str = "dashscope"
    embedding_api_key: str | None = None
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model: str = "text-embedding-v2"
    embedding_fallback_model: str = "text-embedding-async-v2"

    # ── Reranker ──
    reranker_provider: str = "local"
    reranker_api_key: str | None = None
    reranker_base_url: str = "https://api.siliconflow.cn/v1/rerank"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_model_path: str = "BAAI/bge-reranker-v2-m3"

    # ── Database ──
    database_url: str

    # ── Redis ──
    redis_url: str = "redis://localhost:6379/0"

    # ── Neo4j ──
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"

    # ── Security ──
    session_secret: str | None = None
    session_expire_hours: int = 24
    admin_password: str | None = None

    # ── RAG ──
    vector_db_path: str = "law_faiss"
    vector_retrieval_weight: float = 0.4
    bm25_retrieval_weight: float = 0.3
    graph_retrieval_weight: float = 0.3
    relevance_threshold: float = 0.15
    max_context_tokens: int = 3000
    graph_retrieval_weight_fallback: float = 0.3

    # ── Server ──
    env: str = "development"
    trusted_proxy: str | None = None

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
