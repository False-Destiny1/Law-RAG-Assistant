"""Bad Case 分析器

从 ragas_report.json 中提取低分 case，分析失败原因。

用法:
    python -m tests.eval.bad_case_analyzer

输出: bad_case_list.md
"""
import os
import json

sys_path = os.path.join(os.path.dirname(__file__), "..", "..")


def classify_failure(metrics):
    """分类失败类型"""
    ctx_prec = metrics.get("context_precision", 1)
    ctx_rec = metrics.get("context_recall", 1)
    faith = metrics.get("faithfulness", 1)
    cit_cov = metrics.get("citation_coverage", 1)

    if ctx_prec < 0.3 and ctx_rec < 0.3:
        return "检索失败", "检索结果与问题完全不相关，可能缺少对应法律文档或分词/向量化有偏差"
    if ctx_prec < 0.5:
        return "检索噪音多", "检索到了一些相关文档但也夹杂了大量无关文档"
    if ctx_rec < 0.5:
        return "检索遗漏", "部分相关文档未被检索到，可能是分块策略或检索权重问题"
    if faith < 0.8:
        return "生成幻觉", "LLM 生成了与检索文档不一致的内容"
    if cit_cov < 0.2:
        return "引用缺失", "回答正确但缺少引用标签 [来源N]，引用覆盖率低"
    return "轻度问题", "指标略低但整体可接受"


def analyze():
    report_path = os.path.join(sys_path, "tests", "eval", "ragas_report.json")
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    cases = report.get("per_case", [])

    bad_cases = []
    for case in cases:
        m = case.get("metrics", {})
        # 标记 bad case 的条件
        is_bad = (
            m.get("context_precision", 1) < 0.5
            or m.get("context_recall", 1) < 0.5
            or m.get("faithfulness", 1) < 0.8
            or m.get("citation_coverage", 1) < 0.15
        )
        if is_bad:
            failure_type, reason = classify_failure(m)
            bad_cases.append({
                "id": case["id"],
                "category": case.get("category", ""),
                "question": case.get("question", ""),
                "context_precision": m.get("context_precision", 0),
                "context_recall": m.get("context_recall", 0),
                "faithfulness": m.get("faithfulness", 0),
                "citation_coverage": m.get("citation_coverage", 0),
                "retrieval_hit_rate": m.get("retrieval_hit_rate", 0),
                "failure_type": failure_type,
                "reason": reason,
            })

    # 按严重程度排序（context_precision 最低的排前面）
    bad_cases.sort(key=lambda x: x["context_precision"])

    # 生成 Markdown
    out_path = os.path.join(os.path.dirname(__file__), "results", "bad_case_list.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Bad Case 分析报告\n\n")
        f.write(f"共 **{len(bad_cases)}** 条 bad case（共 {len(cases)} 条评估数据）\n\n")

        # 汇总
        f.write("## 失败类型分布\n\n")
        type_counts = {}
        for bc in bad_cases:
            t = bc["failure_type"]
            type_counts[t] = type_counts.get(t, 0) + 1
        f.write("| 失败类型 | 数量 | 说明 |\n")
        f.write("|---|---|---|\n")
        type_descs = {
            "检索失败": "检索结果与问题完全不相关",
            "检索噪音多": "检索结果夹杂大量无关文档",
            "检索遗漏": "相关文档未被检索到",
            "生成幻觉": "LLM 生成与文档不一致的内容",
            "引用缺失": "回答正确但缺少引用标签",
            "轻度问题": "指标略低但整体可接受",
        }
        for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            f.write(f"| {t} | {count} | {type_descs.get(t, '')} |\n")

        # 详细列表
        f.write("\n## Bad Case 详情\n\n")
        for bc in bad_cases:
            f.write(f"### {bc['id']} [{bc['category']}]\n\n")
            f.write(f"**问题**: {bc['question']}\n\n")
            f.write(f"| 指标 | 值 |\n|---|---|\n")
            f.write(f"| context_precision | {bc['context_precision']:.3f} |\n")
            f.write(f"| context_recall | {bc['context_recall']:.3f} |\n")
            f.write(f"| faithfulness | {bc['faithfulness']:.3f} |\n")
            f.write(f"| citation_coverage | {bc['citation_coverage']:.3f} |\n")
            f.write(f"| retrieval_hit_rate | {bc['retrieval_hit_rate']:.3f} |\n\n")
            f.write(f"**失败类型**: {bc['failure_type']}\n\n")
            f.write(f"**原因分析**: {bc['reason']}\n\n")
            f.write("---\n\n")

        # 改进建议
        f.write("## 改进建议\n\n")
        if type_counts.get("检索失败", 0) > 0 or type_counts.get("检索遗漏", 0) > 0:
            f.write("1. **检索问题**: 扩展 BM25 法律词典、优化 chunk 策略、调整 RRF 融合参数\n")
        if type_counts.get("检索噪音多", 0) > 0:
            f.write("2. **噪音问题**: 提高 reranker 阈值（当前 0.15）、优化 reranker 模型\n")
        if type_counts.get("引用缺失", 0) > 0:
            f.write("3. **引用问题**: 在 chunk 切割时保留引用元数据、扩展引用提取正则规则\n")
        if type_counts.get("生成幻觉", 0) > 0:
            f.write("4. **幻觉问题**: 强化 system prompt 的引用要求、添加引用验证后处理\n")

    print(f"Bad case 分析完成: {len(bad_cases)} 条")
    print(f"报告已保存到: {out_path}")
    return bad_cases


if __name__ == "__main__":
    import sys
    sys.path.insert(0, sys_path)
    analyze()
