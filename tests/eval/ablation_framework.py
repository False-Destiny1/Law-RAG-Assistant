"""消融实验通用框架

支持多配置 x 多场景 x 多次运行，输出 mean ± std。

用法:
    python -m tests.eval.ablation_framework --ablation fusion
    python -m tests.eval.ablation_framework --ablation hyde
    python -m tests.eval.ablation_framework --ablation reranker
    python -m tests.eval.ablation_framework --ablation citation
"""
import os
import sys
import json
import asyncio
import statistics
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tests.eval.ragas_dataset import EVAL_DATASET

# ── 场景分组 ──────────────────────────────────────────────────────────

SCENES = {
    "A_精确法律查询": ["R1", "R2", "R3", "R4", "R5"],
    "B_口语化查询": ["C1", "C2"],
    "C_复合问题": ["M1"],
    "D_冷门法律": ["D1", "D2"],
    "E_多轮对话": ["T1", "T2"],
}

# ── 消融配置定义 ──────────────────────────────────────────────────────

ABLATIONS = {
    "fusion": {
        "name": "三路融合检索",
        "env_key": "RETRIEVAL_MODE",
        "configs": ["full", "vector_only", "bm25_only", "graph_only"],
        "config_labels": {
            "full": "三路融合",
            "vector_only": "FAISS Only",
            "bm25_only": "BM25 Only",
            "graph_only": "Neo4j Only",
        },
    },
    "hyde": {
        "name": "HyDE 检索增强",
        "env_key": "ENABLE_HYDE",
        "configs": ["true", "false"],
        "config_labels": {"true": "有 HyDE", "false": "无 HyDE"},
    },
    "reranker": {
        "name": "CrossEncoder Reranker",
        "env_key": "ENABLE_RERANKER",
        "configs": ["true", "false"],
        "config_labels": {"true": "有 Reranker", "false": "无 Reranker"},
    },
    "citation": {
        "name": "引用后处理",
        "env_key": "ENABLE_CITATION_POSTPROCESS",
        "configs": ["true", "false"],
        "config_labels": {"true": "有后处理", "false": "无后处理"},
    },
}

RUNS_PER_CONFIG = 3


# ── 评估函数 ──────────────────────────────────────────────────────────

def compute_metrics(retrieved_texts, reference_contexts):
    """计算检索指标"""
    if not reference_contexts:
        return {"precision": 0, "recall": 0, "hit_rate": 0}

    ref_set = set(r[:50] for r in reference_contexts)

    if not retrieved_texts:
        return {"precision": 0, "recall": 0, "hit_rate": 0}

    hits = sum(1 for doc in retrieved_texts if any(ref in doc[:100] for ref in ref_set))
    precision = hits / len(retrieved_texts)

    recall_hits = sum(1 for ref in ref_set if any(ref in doc[:100] for doc in retrieved_texts))
    recall = recall_hits / len(ref_set)

    hit_rate = 1.0 if hits > 0 else 0.0

    return {"precision": precision, "recall": recall, "hit_rate": hit_rate}


async def evaluate_case(rag, case):
    """评估单条 case"""
    try:
        analysis = await rag.analyze_query(case["question"])
        docs = await rag.retrieve_documents(
            query=case["question"],
            rewritten_query=analysis.get("rewritten_query", case["question"]),
            sub_queries=analysis.get("sub_queries", []),
            hypothetical_doc=analysis.get("hypothetical_doc", ""),
        )
    except Exception as e:
        print(f"  Error on {case['id']}: {e}")
        docs = []

    retrieved = [doc for doc, _ in docs] if docs else []
    return compute_metrics(retrieved, case.get("reference_contexts", []))


# ── 主运行逻辑 ────────────────────────────────────────────────────────

async def run_ablation(ablation_name):
    from law_assistant.rag import DeepSeekApiRag

    ablation = ABLATIONS[ablation_name]
    print(f"\n{'='*60}")
    print(f"  消融实验: {ablation['name']}")
    print(f"  配置: {ablation['configs']}")
    print(f"  每配置运行: {RUNS_PER_CONFIG} 次")
    print(f"  评估数据: {len(EVAL_DATASET)} 条")
    print(f"{'='*60}\n")

    all_results = {}

    for config_val in ablation["configs"]:
        config_label = ablation["config_labels"][config_val]
        os.environ[ablation["env_key"]] = config_val

        mode_results = []
        for run_idx in range(RUNS_PER_CONFIG):
            print(f"\n--- {config_label} | Run {run_idx + 1}/{RUNS_PER_CONFIG} ---")
            rag = DeepSeekApiRag()
            run_data = []

            for i, case in enumerate(EVAL_DATASET):
                metrics = await evaluate_case(rag, case)
                run_data.append({
                    "id": case["id"],
                    "category": case["category"],
                    **metrics,
                })
                print(f"  [{i+1}/15] {case['id']}: P={metrics['precision']:.3f} R={metrics['recall']:.3f}")

            mode_results.append(run_data)

        # 计算每个 case 的 mean ± std
        averaged = []
        for i, case in enumerate(EVAL_DATASET):
            precisions = [run[i]["precision"] for run in mode_results]
            recalls = [run[i]["recall"] for run in mode_results]
            hits = [run[i]["hit_rate"] for run in mode_results]
            averaged.append({
                "id": case["id"],
                "category": case["category"],
                "question": case["question"][:50],
                "precision_mean": statistics.mean(precisions),
                "precision_std": statistics.stdev(precisions) if len(precisions) > 1 else 0,
                "recall_mean": statistics.mean(recalls),
                "recall_std": statistics.stdev(recalls) if len(recalls) > 1 else 0,
                "hit_rate_mean": statistics.mean(hits),
            })

        all_results[config_val] = averaged

    return all_results


def generate_report(ablation_name, all_results):
    """生成消融实验报告"""
    ablation = ABLATIONS[ablation_name]
    configs = ablation["configs"]
    labels = ablation["config_labels"]

    # 按场景分组
    scene_data = {}
    for scene_name, case_ids in SCENES.items():
        scene_data[scene_name] = {}
        for config in configs:
            cases = [r for r in all_results[config] if r["id"] in case_ids]
            if cases:
                precisions = [c["precision_mean"] for c in cases]
                recalls = [c["recall_mean"] for c in cases]
                scene_data[scene_name][config] = {
                    "precision": statistics.mean(precisions),
                    "precision_std": statistics.stdev(precisions) if len(precisions) > 1 else 0,
                    "recall": statistics.mean(recalls),
                    "recall_std": statistics.stdev(recalls) if len(recalls) > 1 else 0,
                }

    # Overall 均值
    overall = {}
    for config in configs:
        precisions = [r["precision_mean"] for r in all_results[config]]
        recalls = [r["recall_mean"] for r in all_results[config]]
        overall[config] = {
            "precision": statistics.mean(precisions),
            "precision_std": statistics.stdev(precisions) if len(precisions) > 1 else 0,
            "recall": statistics.mean(recalls),
            "recall_std": statistics.stdev(recalls) if len(recalls) > 1 else 0,
        }

    # 写报告
    out_path = os.path.join(os.path.dirname(__file__), "results", "ablation", f"ablation_{ablation_name}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# 消融实验: {ablation['name']}\n\n")
        f.write(f"> {len(configs)} 配置 x {len(SCENES)} 场景 x {RUNS_PER_CONFIG} 次运行\n")
        f.write(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        # 实验设计
        f.write("## 实验设计\n\n")
        f.write(f"| 配置 | 环境变量 `{ablation['env_key']}` | 说明 |\n")
        f.write("|---|---|---|\n")
        for config in configs:
            f.write(f"| {labels[config]} | `{config}` | |\n")

        # 场景分组对比
        f.write("\n## 场景分组对比\n\n")
        f.write("| 场景 | 指标 |")
        for config in configs:
            f.write(f" {labels[config]} |")
        f.write("\n|---|---|")
        for _ in configs:
            f.write("---|")
        f.write("\n")

        for scene_name in SCENES:
            for metric, metric_label in [("precision", "precision"), ("recall", "recall")]:
                f.write(f"| {scene_name if metric == 'precision' else ''} | {metric_label} |")
                for config in configs:
                    if config in scene_data.get(scene_name, {}):
                        d = scene_data[scene_name][config]
                        val = d[metric]
                        std = d[f"{metric}_std"]
                        f.write(f" {val:.3f} ± {std:.3f} |")
                    else:
                        f.write(" - |")
                f.write("\n")

        # Overall
        f.write("\n## Overall 均值\n\n")
        f.write("| 指标 |")
        for config in configs:
            f.write(f" {labels[config]} |")
        f.write("\n|---|")
        for _ in configs:
            f.write("---|")
        f.write("\n")

        for metric in ["precision", "recall"]:
            f.write(f"| {metric} |")
            for config in configs:
                d = overall[config]
                f.write(f" {d[metric]:.3f} ± {d[f'{metric}_std']:.3f} |")
            f.write("\n")

        # 效果量分析
        if len(configs) >= 2:
            baseline = configs[0]
            f.write("\n## 效果量分析\n\n")
            f.write(f"以 {labels[baseline]} 为 baseline:\n\n")
            f.write("| 配置 | precision 变化 | recall 变化 |\n")
            f.write("|---|---|---|\n")
            for config in configs[1:]:
                p_change = overall[config]["precision"] - overall[baseline]["precision"]
                r_change = overall[config]["recall"] - overall[baseline]["recall"]
                p_pct = (p_change / overall[baseline]["precision"] * 100) if overall[baseline]["precision"] > 0 else 0
                r_pct = (r_change / overall[baseline]["recall"] * 100) if overall[baseline]["recall"] > 0 else 0
                f.write(f"| {labels[config]} | {p_pct:+.1f}% ({p_change:+.3f}) | {r_pct:+.1f}% ({r_change:+.3f}) |\n")

        # 逐 case 结果
        f.write("\n## 逐 Case 结果\n\n")
        f.write("| Case | Category |")
        for config in configs:
            f.write(f" {labels[config]} P | {labels[config]} R |")
        f.write("\n|---|---|")
        for _ in configs:
            f.write("---|---|")
        f.write("\n")

        for i, case in enumerate(EVAL_DATASET):
            f.write(f"| {case['id']} | {case['category']} |")
            for config in configs:
                r = all_results[config][i]
                f.write(f" {r['precision_mean']:.3f} | {r['recall_mean']:.3f} |")
            f.write("\n")

    print(f"\n报告已生成: {out_path}")
    return out_path


async def main():
    parser = argparse.ArgumentParser(description="消融实验框架")
    parser.add_argument("--ablation", choices=list(ABLATIONS.keys()), required=True)
    parser.add_argument("--runs", type=int, default=RUNS_PER_CONFIG)
    args = parser.parse_args()

    global RUNS_PER_CONFIG
    RUNS_PER_CONFIG = args.runs

    results = await run_ablation(args.ablation)
    report_path = generate_report(args.ablation, results)

    # 保存原始数据
    json_path = report_path.replace(".md", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"原始数据已保存: {json_path}")


if __name__ == "__main__":
    asyncio.run(main())
