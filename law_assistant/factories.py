"""Factory pattern for Embedding, Reranker, and LLM providers."""

import logging
import os

logger = logging.getLogger(__name__)


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class EmbeddingFactory:
    """Create embedding models based on provider config."""

    @staticmethod
    def create(provider: str = None):
        provider = provider or os.getenv("EMBEDDING_PROVIDER", "dashscope").lower()

        if provider == "dashscope":
            from langchain_openai import OpenAIEmbeddings

            api_key = os.getenv("EMBEDDING_API_KEY")
            base_url = os.getenv("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            model = os.getenv("EMBEDDING_MODEL", "text-embedding-v2")
            fallback_model = os.getenv("EMBEDDING_FALLBACK_MODEL", "text-embedding-async-v2")
            logger.info(f"Creating DashScope embedding: {model}")

            primary = OpenAIEmbeddings(
                api_key=api_key, base_url=base_url, model=model,
                check_embedding_ctx_length=False, chunk_size=10,
            )
            fallback = OpenAIEmbeddings(
                api_key=api_key, base_url=base_url, model=fallback_model,
                check_embedding_ctx_length=False, chunk_size=10,
            )
            return primary, fallback
        else:
            from langchain_huggingface import HuggingFaceEmbeddings

            model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
            model_path = model_name
            if not os.path.isabs(model_path) and not os.path.isdir(model_path):
                local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_name)
                if os.path.isdir(local_path):
                    model_path = local_path

            device = "cuda" if _cuda_available() else "cpu"
            logger.info(f"Creating local embedding: {model_path} (device={device})")

            primary = HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
            return primary, None


class RerankerFactory:
    """Create reranker instances based on provider config."""

    @staticmethod
    def create(provider: str = None):
        provider = provider or os.getenv("RERANKER_PROVIDER", "local").lower()

        if provider == "local":
            from sentence_transformers import CrossEncoder

            model_path = os.getenv("RERANKER_MODEL_PATH", "BAAI/bge-reranker-v2-m3")
            device = "cuda" if _cuda_available() else "cpu"
            logger.info(f"Creating local reranker: {model_path} (device={device})")
            return CrossEncoder(model_path, max_length=512, device=device)
        else:
            api_key = os.getenv("RERANKER_API_KEY")
            base_url = os.getenv("RERANKER_BASE_URL", "https://api.siliconflow.cn/v1/rerank")
            model = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
            logger.info(f"Creating DashScope reranker: {model}")
            return {"api_key": api_key, "base_url": base_url, "model": model}


class LLMFactory:
    """Create LLM instances based on provider config."""

    @staticmethod
    def create():
        from langchain_openai import ChatOpenAI

        mimo_key = os.getenv("MIMO_API_KEY")
        if mimo_key:
            base_url = os.getenv("MIMO_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1")
            model = os.getenv("MIMO_MODEL", "mimo-v2.5-pro")
            logger.info(f"Creating MiMo LLM: {model}")
            return ChatOpenAI(api_key=mimo_key, base_url=base_url, model=model)

        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        logger.info(f"Creating DeepSeek LLM: {model}")
        return ChatOpenAI(api_key=deepseek_key, base_url=base_url, model=model)
