"""
RAGAS 评估数据收集器

直接调用 RAG 模型（不经过 HTTP），收集各阶段中间数据。
"""

import json
import logging
import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def init_rag():
    """初始化 RAG 模型（不触发 app.py 的模块级初始化）"""
    import os

    from dotenv import load_dotenv

    load_dotenv()

    from law_assistant.rag import DeepSeekApiRag

    api_key = os.getenv("MIMO_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    db_path = os.getenv("VECTOR_DB_PATH", "law_faiss")

    rag = DeepSeekApiRag(api_key, db_path)
    return rag


def collect_single(rag, question: str, conversation_id: str = None) -> dict:
    """
    单次调用 RAG，收集完整中间数据。

    返回:
    {
        "question": str,
        "answer": str,
        "contexts": [str],                    # 检索到的文档文本
        "contexts_with_scores": [(str, float)], # 带 reranker 分数的文档
        "analysis": {                          # 查询分析结果
            "rewritten_query": str,
            "sub_queries": [str],
            "hypothetical_doc": str,
        },
        "raw_context": str,                    # 带引用编号的完整上下文
        "elapsed_sec": float,
        "error": str | None,
    }
    """
    result = {
        "question": question,
        "answer": "",
        "contexts": [],
        "contexts_with_scores": [],
        "analysis": {},
        "raw_context": "",
        "elapsed_sec": 0,
        "error": None,
    }

    start = time.time()
    try:
        response = rag.generate_response_stream(
            question, conversation_id=conversation_id, top_k=10
        )

        # 收集中间数据
        result["contexts"] = response.get("retrieved_documents", [])
        result["contexts_with_scores"] = response.get("retrieved_documents_with_scores", [])
        result["analysis"] = response.get("analysis", {})
        result["raw_context"] = response.get("context", "")

        # 消费流获取完整回答
        full_answer = []
        for chunk in response["stream"]:
            if hasattr(chunk, "content"):
                full_answer.append(chunk.content)
        result["answer"] = "".join(full_answer)

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"收集失败: {e}")

    result["elapsed_sec"] = time.time() - start
    return result


def collect_dataset(rag, dataset: list[dict], save_path: str = None) -> list[dict]:
    """
    批量收集评估数据（增量保存，支持断点续跑）。

    参数:
        rag: RAG 模型实例
        dataset: 评估数据集 (来自 ragas_dataset.py)
        save_path: 增量保存路径

    返回:
        收集结果列表，每个元素包含原始用例数据 + RAG 输出
    """
    if save_path is None:
        save_path = str(Path(__file__).parent / "ragas_collected.json")

    # 加载已有结果（断点续跑）
    existing_ids = set()
    results = []
    if Path(save_path).exists():
        try:
            with open(save_path, encoding="utf-8") as f:
                results = json.load(f)
            existing_ids = {r.get("id") for r in results}
            print(f"  已有 {len(results)} 个结果，跳过已收集的用例")
        except Exception:
            pass

    total = len(dataset)

    for i, case in enumerate(dataset, 1):
        case_id = case.get("id", f"CASE-{i}")
        if case_id in existing_ids:
            print(f"  [{i}/{total}] 跳过 {case_id} (已收集)")
            continue

        print(f"  [{i}/{total}] 收集 {case_id}: {case['question'][:30]}...", end=" ", flush=True)

        # 多轮对话用例需要模拟上下文
        conv_id = f"eval_{case_id}_{int(time.time())}"
        if case.get("context"):
            rag.memory.add_message(conv_id, "user", case["context"])
            rag.memory.add_message(conv_id, "assistant", "好的，我了解了您的问题背景。")

        collected = collect_single(rag, case["question"], conversation_id=conv_id)

        # 合并原始用例数据和收集结果
        merged = {**case, **collected}
        results.append(merged)

        status = "OK" if not collected["error"] else f"ERR: {collected['error']}"
        print(f"{status} ({collected['elapsed_sec']:.1f}s)")

        # 清理对话记忆，避免污染下一个用例
        rag.clear_conversation_memory(conv_id)

        # 增量保存
        save_collected(results, save_path)

        time.sleep(0.3)

    return results


def save_collected(results: list[dict], output_path: str = None):
    """保存收集结果到 JSON 文件"""
    if output_path is None:
        output_path = str(Path(__file__).parent / "ragas_collected.json")

    # 序列化时需要处理不可 JSON 化的对象
    serializable = []
    for r in results:
        item = {}
        for k, v in r.items():
            if k == "contexts_with_scores":
                item[k] = [{"text": t, "score": s} for t, s in v] if v else []
            elif k == "analysis":
                item[k] = v if v else {}
            else:
                item[k] = v
        serializable.append(item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"收集结果已保存到: {output_path}")


if __name__ == "__main__":
    from tests.eval.ragas_dataset import EVAL_DATASET

    print("初始化 RAG 模型...")
    rag = init_rag()
    print(f"数据集: {len(EVAL_DATASET)} 个用例")
    print("开始收集...")
    results = collect_dataset(rag, EVAL_DATASET)
    save_collected(results)
    print("收集完成!")
