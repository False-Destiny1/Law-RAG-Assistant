"""消融实验运行器 — 单次运行，按场景分批

用法:
    # 跑 fusion 消融（4 配置 x 全部场景）
    python -m tests.eval.ablation_runner --ablation fusion

    # 只跑某个场景
    python -m tests.eval.ablation_runner --ablation fusion --scene A
    python -m tests.eval.ablation_runner --ablation fusion --scene B
    python -m tests.eval.ablation_runner --ablation fusion --scene C

    # 跑 hyde 消融
    python -m tests.eval.ablation_runner --ablation hyde --scene B
"""
import os
import sys
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv
load_dotenv()

from tests.eval.ragas_dataset import EVAL_DATASET

SCENES = {
    "A": {"name": "精确法律查询", "ids": ["R1", "R2", "R3", "R4", "R5"]},
    "B": {"name": "口语化查询", "ids": ["C1", "C2"]},
    "C": {"name": "复合问题", "ids": ["M1"]},
    "D": {"name": "冷门法律", "ids": ["D1", "D2"]},
    "E": {"name": "多轮对话", "ids": ["T1", "T2"]},
}

ABLATIONS = {
    "fusion": {
        "name": "三路融合检索",
        "attr": "retrieval_mode",
        "configs": ["full", "vector_only", "bm25_only", "graph_only"],
        "labels": {"full": "三路融合", "vector_only": "FAISS Only", "bm25_only": "BM25 Only", "graph_only": "Neo4j Only"},
    },
    "hyde": {
        "name": "HyDE 检索增强",
        "attr": "enable_hyde",
        "configs": [True, False],
        "labels": {True: "有 HyDE", False: "无 HyDE"},
    },
}


def compute_metrics(retrieved, reference_contexts):
    if not reference_contexts:
        return {"precision": 0, "recall": 0, "hit_rate": 0}
    ref_set = set(r[:50] for r in reference_contexts)
    if not retrieved:
        return {"precision": 0, "recall": 0, "hit_rate": 0}
    hits = sum(1 for doc in retrieved if any(ref in doc[:100] for ref in ref_set))
    precision = hits / len(retrieved)
    recall_hits = sum(1 for ref in ref_set if any(ref in doc[:100] for doc in retrieved))
    recall = recall_hits / len(ref_set)
    return {"precision": precision, "recall": recall, "hit_rate": 1.0 if hits > 0 else 0.0}


def run_ablation(ablation_name, scene_key=None):
    from law_assistant.rag import DeepSeekApiRag

    ablation = ABLATIONS[ablation_name]
    attr = ablation["attr"]
    configs = ablation["configs"]
    labels = ablation["labels"]

    # 筛选 case
    if scene_key and scene_key in SCENES:
        case_ids = SCENES[scene_key]["ids"]
        cases = [c for c in EVAL_DATASET if c["id"] in case_ids]
        scene_name = SCENES[scene_key]["name"]
    else:
        cases = EVAL_DATASET
        scene_name = "全部场景"

    print(f"\n{'='*50}")
    print(f"  消融: {ablation['name']}")
    print(f"  场景: {scene_name} ({len(cases)} cases)")
    print(f"  配置: {[labels[c] for c in configs]}")
    print(f"{'='*50}\n")

    # 初始化 RAG（只一次）
    print("Loading RAG model...")
    api_key = os.getenv("MIMO_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    db_path = os.getenv("VECTOR_DB_PATH", "law_faiss")
    rag = DeepSeekApiRag(api_key, db_path)
    print(f"FAISS: {rag.vector_db.index.ntotal}, BM25: {len(rag.bm25_retriever.documents)}\n")

    all_results = {}

    for config_val in configs:
        config_label = labels[config_val]
        setattr(rag, attr, config_val)
        print(f"--- {config_label} ({attr}={config_val}) ---")

        results = []
        for case in cases:
            try:
                analysis = rag.analyze_query(case["question"])
                docs = rag.retrieve_documents(
                    query=case["question"],
                    sub_queries=analysis.get("sub_queries", []),
                    hypothetical_doc=analysis.get("hypothetical_doc", ""),
                )
            except Exception as e:
                print(f"  Error {case['id']}: {e}")
                docs = []

            retrieved = [doc for doc, _ in docs] if docs else []
            metrics = compute_metrics(retrieved, case.get("reference_contexts", []))
            results.append({"id": case["id"], "category": case["category"], "question": case["question"][:50], **metrics})
            print(f"  {case['id']}: P={metrics['precision']:.3f} R={metrics['recall']:.3f}")

        all_results[config_val] = results
        avg_p = sum(r["precision"] for r in results) / len(results)
        avg_r = sum(r["recall"] for r in results) / len(results)
        print(f"  >> {config_label} avg: P={avg_p:.3f} R={avg_r:.3f}\n")

    return all_results


def save_report(ablation_name, scene_key, all_results):
    ablation = ABLATIONS[ablation_name]
    configs = ablation["configs"]
    labels = ablation["labels"]

    scene_label = f"_{scene_key}" if scene_key else ""
    scene_name = SCENES[scene_key]["name"] if scene_key else "全部"

    out_path = os.path.join(os.path.dirname(__file__), "results", "ablation", f"ablation_{ablation_name}{scene_label}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# 消融: {ablation['name']} — {scene_name}\n\n")
        f.write(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        # 数据表
        f.write("| Case | Category |")
        for c in configs:
            f.write(f" {labels[c]} P | {labels[c]} R |")
        f.write("\n|---|---|")
        for _ in configs:
            f.write("---|---|")
        f.write("\n")

        n_cases = len(list(all_results.values())[0])
        for i in range(n_cases):
            first = list(all_results.values())[0][i]
            f.write(f"| {first['id']} | {first['category']} |")
            for c in configs:
                r = all_results[c][i]
                f.write(f" {r['precision']:.3f} | {r['recall']:.3f} |")
            f.write("\n")

        # 均值
        f.write("| **avg** | |")
        for c in configs:
            avg_p = sum(r["precision"] for r in all_results[c]) / n_cases
            avg_r = sum(r["recall"] for r in all_results[c]) / n_cases
            f.write(f" **{avg_p:.3f}** | **{avg_r:.3f}** |")
        f.write("\n")

        # 效果量
        if len(configs) >= 2:
            baseline = configs[0]
            f.write(f"\n## 效果量 (baseline: {labels[baseline]})\n\n")
            f.write("| 配置 | precision 变化 | recall 变化 |\n|---|---|---|\n")
            base_avg_p = sum(r["precision"] for r in all_results[baseline]) / n_cases
            base_avg_r = sum(r["recall"] for r in all_results[baseline]) / n_cases
            for c in configs[1:]:
                avg_p = sum(r["precision"] for r in all_results[c]) / n_cases
                avg_r = sum(r["recall"] for r in all_results[c]) / n_cases
                p_pct = ((avg_p - base_avg_p) / base_avg_p * 100) if base_avg_p > 0 else 0
                r_pct = ((avg_r - base_avg_r) / base_avg_r * 100) if base_avg_r > 0 else 0
                f.write(f"| {labels[c]} | {p_pct:+.1f}% ({avg_p - base_avg_p:+.3f}) | {r_pct:+.1f}% ({avg_r - base_avg_r:+.3f}) |\n")

    print(f"报告: {out_path}")

    # JSON
    json_path = out_path.replace(".md", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"数据: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", choices=list(ABLATIONS.keys()), required=True)
    parser.add_argument("--scene", choices=list(SCENES.keys()), default=None)
    args = parser.parse_args()

    results = run_ablation(args.ablation, args.scene)
    save_report(args.ablation, args.scene, results)
