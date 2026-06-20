"""评估报告格式化器

将 ragas_report.json 整理为面试展示用的 Markdown 报告。

用法:
    python -m tests.eval.report_formatter

输出: ragas_report.md
"""

import json
import os
import sys

sys_path = os.path.join(os.path.dirname(__file__), "..", "..")


def generate():
    report_path = os.path.join(sys_path, "tests", "eval", "ragas_report.json")
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    cases = report.get("per_case", [])
    overall = report.get("overall", {})

    out_path = os.path.join(os.path.dirname(__file__), "results", "ragas_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# RAGAS 评估报告\n\n")
        f.write(f"> 基于 {len(cases)} 条评估数据，8 项自定义指标\n\n")

        # 总览表格
        f.write("## 总体指标\n\n")
        f.write("| 指标 | 均值 | 最小值 | 最大值 | 评价 |\n")
        f.write("|---|---|---|---|---|\n")
        for metric, stats in overall.items():
            mean_val = stats["mean"]
            if mean_val >= 0.9:
                tag = "优秀"
            elif mean_val >= 0.7:
                tag = "良好"
            elif mean_val >= 0.5:
                tag = "一般"
            else:
                tag = "需改进"
            f.write(f"| {metric} | {mean_val:.4f} | {stats['min']:.4f} | {stats['max']:.4f} | {tag} |\n")

        # 强项分析
        f.write("\n## 强项\n\n")
        strong = [(k, v) for k, v in overall.items() if v["mean"] >= 0.9]
        for metric, stats in strong:
            descs = {
                "faithfulness": "回答忠实度高，LLM 几乎不会编造未在检索文档中出现的内容",
                "answer_relevancy": "回答与问题高度相关，没有跑题",
                "citation_accuracy": "引用的法条准确无误，没有张冠李戴",
            }
            f.write(f"- **{metric}** ({stats['mean']:.3f}): {descs.get(metric, '')}\n")

        # 弱项分析
        f.write("\n## 弱项\n\n")
        weak = [(k, v) for k, v in overall.items() if v["mean"] < 0.7]
        for metric, stats in weak:
            descs = {
                "context_precision": "检索结果中相关文档占比偏低，夹杂了较多无关文档",
                "retrieval_hit_rate": "部分查询未能命中任何相关文档",
                "reranker_score_avg": "重排序分数整体偏低，reranker 对相关性区分度不足",
                "citation_coverage": "回答中引用标签覆盖率低，大部分回答缺少 [来源N] 标注",
            }
            f.write(f"- **{metric}** ({stats['mean']:.3f}): {descs.get(metric, '')}\n")

        # 分类表现
        f.write("\n## 分类表现\n\n")
        categories = {}
        for case in cases:
            cat = case.get("category", "unknown")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(case)

        f.write("| 类别 | 数量 | avg_precision | avg_recall | avg_faithfulness | avg_citation_coverage |\n")
        f.write("|---|---|---|---|---|---|\n")
        for cat, cat_cases in sorted(categories.items()):
            n = len(cat_cases)
            avg_p = sum(c["metrics"]["context_precision"] for c in cat_cases) / n
            avg_r = sum(c["metrics"]["context_recall"] for c in cat_cases) / n
            avg_f = sum(c["metrics"]["faithfulness"] for c in cat_cases) / n
            avg_c = sum(c["metrics"]["citation_coverage"] for c in cat_cases) / n
            f.write(f"| {cat} | {n} | {avg_p:.3f} | {avg_r:.3f} | {avg_f:.3f} | {avg_c:.3f} |\n")

        # 改进方向
        f.write("\n## 改进方向\n\n")
        f.write(
            "1. **优化 citation_coverage**: 在 chunk 切割时保留引用元数据（法条编号、章节标题），扩展引用提取正则规则\n"
        )
        f.write("2. **改善冷门法律检索**: 扩展 BM25 法律词典覆盖更多领域，或引入多知识库检索\n")
        f.write("3. **提升 reranker 分数**: 调优 reranker 阈值（当前 0.15），或尝试更大的 reranker 模型\n")
        f.write("4. **消融实验验证**: 补充 baseline 对比（FAISS-only / BM25-only / Graph-only）和 HyDE 消融实验\n")

    print(f"评估报告已生成: {out_path}")


if __name__ == "__main__":
    sys.path.insert(0, sys_path)
    generate()
