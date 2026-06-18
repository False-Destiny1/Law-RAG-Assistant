"""三路融合检索 baseline 对比实验

通过 RETRIEVAL_MODE 环境变量控制检索模式，对比不同检索路的贡献。

用法:
    # 跑 full 模式（三路融合）
    python -m tests.eval.baseline_eval

    # 跑单路模式
    RETRIEVAL_MODE=vector_only python -m tests.eval.baseline_eval

    # 一次性跑全部 4 组
    python -m tests.eval.baseline_eval --all

输出: baseline_{mode}.json, baseline_comparison.md
"""
import os
import sys
import json
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tests.eval.ragas_dataset import EVAL_DATASET

MODES = [
    ("full", "三路融合 (FAISS+BM25+Neo4j)"),
    ("vector_only", "FAISS Only"),
    ("bm25_only", "BM25 Only"),
    ("graph_only", "Neo4j Only"),
]


def _compute_precision(retrieved, reference):
    """检索到的文档中有多少是相关的"""
    if not retrieved or not reference:
        return 0.0
    ref_set = set(r[:50] for r in reference)
    hits = sum(1 for doc in retrieved if any(ref in doc[:100] for ref in ref_set))
    return hits / len(retrieved)


def _compute_recall(retrieved, reference):
    """相关文档中有多少被检索到了"""
    if not reference:
        return 0.0
    ref_set = set(r[:50] for r in reference)
    hits = sum(1 for ref in ref_set if any(ref in doc[:100] for doc in retrieved))
    return hits / len(ref_set)


def _compute_hit_rate(retrieved, reference):
    """是否命中至少一条相关文档"""
    if not reference:
        return 0.0
    ref_set = set(r[:50] for r in reference)
    for doc in retrieved:
        if any(ref in doc[:100] for ref in ref_set):
            return 1.0
    return 0.0


async def evaluate_single(rag, case):
    """对单条 case 跑检索，返回基础检索指标"""
    query = case["question"]
    reference_contexts = case.get("reference_contexts", [])

    try:
        analysis = await rag.analyze_query(query)
        docs = await rag.retrieve_documents(
            query=query,
            rewritten_query=analysis.get("rewritten_query", query),
            sub_queries=analysis.get("sub_queries", []),
            hypothetical_doc=analysis.get("hypothetical_doc", ""),
        )
    except Exception as e:
        print(f"  检索失败: {e}")
        docs = []

    retrieved_texts = [doc for doc, _ in docs] if docs else []
    return {
        "context_precision": _compute_precision(retrieved_texts, reference_contexts),
        "context_recall": _compute_recall(retrieved_texts, reference_contexts),
        "retrieval_hit_rate": _compute_hit_rate(retrieved_texts, reference_contexts),
    }


async def run_mode(mode):
    """运行单个检索模式"""
    from law_assistant.rag import DeepSeekApiRag

    label = dict(MODES).get(mode, mode)
    print(f"\n{'='*60}")
    print(f"  检索模式: {label}")
    print(f"{'='*60}\n")

    os.environ["RETRIEVAL_MODE"] = mode
    rag = DeepSeekApiRag()

    results = []
    for i, case in enumerate(EVAL_DATASET):
        print(f"[{i+1}/15] {case['id']}: {case['question'][:40]}...")
        metrics = await evaluate_single(rag, case)
        results.append({"id": case["id"], "question": case["question"], **metrics})
        print(f"  P={metrics['context_precision']:.3f}  R={metrics['context_recall']:.3f}  H={metrics['retrieval_hit_rate']:.3f}")

    # 计算均值
    avg = {}
    for key in ["context_precision", "context_recall", "retrieval_hit_rate"]:
        values = [r[key] for r in results]
        avg[key] = sum(values) / len(values) if values else 0

    print(f"\n--- {label} 均值 ---")
    for k, v in avg.items():
        print(f"  {k}: {v:.4f}")

    # 保存
    output = {"mode": mode, "label": label, "avg": avg, "per_case": results}
    out_path = os.path.join(os.path.dirname(__file__), f"baseline_{mode}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {out_path}")

    return mode, avg


def generate_comparison_md(all_avgs):
    """生成对比报告"""
    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "baseline_comparison.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# 三路融合检索 Baseline 对比实验\n\n")
        f.write("## 实验设计\n\n")
        f.write("通过 `RETRIEVAL_MODE` 环境变量控制检索模式，使用现有 RAGAS 评估数据集（15 条）进行对比。\n\n")
        f.write("| 对比组 | 检索方式 |\n|---|---|\n")
        for mode, label in MODES:
            f.write(f"| {label} | `RETRIEVAL_MODE={mode}` |\n")

        f.write("\n## 对比结果\n\n")
        f.write("| 指标 | 三路融合 | FAISS Only | BM25 Only | Neo4j Only |\n")
        f.write("|---|---|---|---|---|\n")
        for metric in ["context_precision", "context_recall", "retrieval_hit_rate"]:
            row = f"| {metric} |"
            for mode, _ in MODES:
                val = all_avgs.get(mode, {}).get(metric, 0)
                row += f" {val:.4f} |"
            f.write(row + "\n")

        # 分析
        full_r = all_avgs.get("full", {}).get("context_recall", 0)
        vec_r = all_avgs.get("vector_only", {}).get("context_recall", 0)
        bm_r = all_avgs.get("bm25_only", {}).get("context_recall", 0)
        gph_r = all_avgs.get("graph_only", {}).get("context_recall", 0)

        f.write("\n## 分析\n\n")
        best_single = max(vec_r, bm_r, gph_r)
        if full_r > best_single:
            f.write(f"- 三路融合 context_recall ({full_r:.4f}) 优于最佳单路 ({best_single:.4f})，融合有效\n")
        else:
            f.write(f"- 三路融合 context_recall ({full_r:.4f}) 未超过最佳单路 ({best_single:.4f})，需分析原因\n")

        f.write(f"- FAISS Recall: {vec_r:.4f} — BM25 Recall: {bm_r:.4f} — Graph Recall: {gph_r:.4f}\n")

    print(f"\n对比报告已生成: {out_path}")


async def main():
    run_all = "--all" in sys.argv
    mode = os.getenv("RETRIEVAL_MODE", "full")

    if run_all:
        all_avgs = {}
        for m, _ in MODES:
            _, avg = await run_mode(m)
            all_avgs[m] = avg
        generate_comparison_md(all_avgs)
    else:
        _, avg = await run_mode(mode)
        if mode == "full":
            generate_comparison_md({"full": avg})


if __name__ == "__main__":
    asyncio.run(main())
