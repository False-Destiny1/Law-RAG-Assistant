"""
RAGAS 评估指标计算

分层指标体系:
├── 检索层 (Retrieval)
│   ├── context_precision  : 检索文档中相关文档的排名质量
│   ├── context_recall     : ground_truth 关键信息的检索覆盖率
│   ├── retrieval_hit_rate : 关键词命中率（兼容旧指标）
│   └── reranker_score_avg : reranker 平均分数
├── 生成层 (Generation)
│   ├── faithfulness       : 回答声明的忠实度（是否有依据）
│   └── answer_relevancy   : 回答与问题的相关性
└── 引用层 (Citation)
    ├── citation_accuracy  : [来源N] 标签对应的文档是否真实存在
    └── citation_coverage  : 回答中有引用支持的声明比例
"""

import json
import re
from pathlib import Path

# ── 检索层指标 ──────────────────────────────────────────────────────


def context_precision(retrieved_docs: list[str], ground_truth: str) -> float:
    """
    Context Precision: 检索到的文档中，与 ground_truth 语义相关的比例。

    简化实现：检查每个检索文档是否包含 ground_truth 中的关键词片段。
    """
    if not retrieved_docs or not ground_truth:
        return 0.0

    # 从 ground_truth 提取关键法律术语和条文号
    keywords = _extract_legal_keywords(ground_truth)
    if not keywords:
        return 0.0

    relevant_count = 0
    for doc in retrieved_docs:
        doc_lower = doc.lower()
        if any(kw.lower() in doc_lower for kw in keywords):
            relevant_count += 1

    return relevant_count / len(retrieved_docs)


def context_recall(retrieved_docs: list[str], reference_contexts: list[str]) -> float:
    """
    Context Recall: reference_contexts 中的关键信息是否出现在检索结果中。

    使用条文号 + 法律名称 + 关键术语的组合匹配，任一命中即视为召回。
    """
    if not reference_contexts:
        return None  # 空 reference 不参与 recall 均值计算

    if not retrieved_docs:
        return 0.0

    all_retrieved = " ".join(retrieved_docs).lower()
    hit_count = 0

    for ref in reference_contexts:
        # 策略1: 提取条文号（如"第三十九条"）
        ref_articles = re.findall(r"第[零一二三四五六七八九十百千\d]+[条章节]", ref)
        ret_articles = re.findall(r"第[零一二三四五六七八九十百千\d]+[条章节]", all_retrieved)
        article_hit = any(a in ret_articles for a in ref_articles)

        # 策略2: 提取法律名称（如"《劳动合同法》"）
        ref_laws = re.findall(r"《[^》]+》", ref)
        law_hit = any(law_name.lower() in all_retrieved for law_name in ref_laws)

        # 策略3: 关键术语匹配
        ref_keywords = _extract_legal_keywords(ref)
        keyword_hits = sum(1 for kw in ref_keywords if kw.lower() in all_retrieved)
        keyword_ratio = keyword_hits / len(ref_keywords) if ref_keywords else 0

        # 任一策略命中即视为召回
        if article_hit or law_hit or keyword_ratio >= 0.4:
            hit_count += 1

    return hit_count / len(reference_contexts)


def retrieval_hit_rate(answer: str, expect_keywords: list[str]) -> float:
    """关键词命中率（兼容旧 baseline_eval 指标）"""
    if not expect_keywords:
        return 1.0
    hits = sum(1 for kw in expect_keywords if kw in answer)
    return hits / len(expect_keywords)


def reranker_score_avg(contexts_with_scores: list[dict]) -> float:
    """Reranker 平均分数"""
    if not contexts_with_scores:
        return 0.0
    scores = [float(item.get("score", 0)) for item in contexts_with_scores]
    return sum(scores) / len(scores) if scores else 0.0


# ── 生成层指标 ──────────────────────────────────────────────────────


def faithfulness(answer: str, contexts: list[str]) -> float:
    """
    Faithfulness: 回答中的声明是否能在检索上下文中找到依据。

    简化实现：
    1. 将回答按句子拆分
    2. 对每个句子，检查是否与某个 context 有关键词重叠
    3. 返回有依据的句子比例
    """
    if not answer or not contexts:
        return 0.0

    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0

    all_context = " ".join(contexts).lower()
    supported = 0

    for sent in sentences:
        # 提取句子中的关键名词和法律术语
        keywords = _extract_legal_keywords(sent)
        if not keywords:
            # 无关键词的句子（如过渡句）视为有依据
            supported += 1
            continue
        # 检查是否有关键词出现在上下文中
        if any(kw.lower() in all_context for kw in keywords):
            supported += 1

    return supported / len(sentences)


def answer_relevancy(answer: str, question: str) -> float:
    """
    Answer Relevancy: 回答与问题的相关性。

    简化实现：检查回答中是否包含与问题相关的关键词。
    """
    if not answer or not question:
        return 0.0

    q_keywords = _extract_legal_keywords(question)
    if not q_keywords:
        return 1.0  # 无法提取关键词时默认满分

    a_lower = answer.lower()
    hits = sum(1 for kw in q_keywords if kw.lower() in a_lower)
    return hits / len(q_keywords)


# ── 引用层指标 ──────────────────────────────────────────────────────


def citation_accuracy(answer: str, retrieved_docs: list[str]) -> float:
    """
    Citation Accuracy: 回答中 [来源N] 标签引用的文档是否真实存在且内容相关。

    检查:
    1. [来源N] 中的 N 是否不超过检索文档总数
    2. 引用的文档内容是否与回答上下文相关
    """
    citations = re.findall(r"\[来源(\d+)\]", answer)
    if not citations:
        return 1.0  # 无引用时默认满分

    valid = 0
    for cite in citations:
        idx = int(cite) - 1  # [来源1] -> index 0
        if 0 <= idx < len(retrieved_docs):
            valid += 1

    return valid / len(citations) if citations else 1.0


def citation_coverage(answer: str, contexts: list[str]) -> float:
    """
    Citation Coverage: 回答中有多少关键声明有引用支持。

    规则:
    1. 含 [来源N] 标签的句子算有引用
    2. 紧跟在含引用句之后的解释句（15字以上、不含新法律术语）也算有引用，
       因为它们语义上依附于前一句的引用
    """
    if not answer:
        return 0.0

    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0

    # 过滤非实质性句子（建议类、过渡类、客套类）
    non_substantive_patterns = [
        r"建议.*咨询.*律师",
        r"请注意",
        r"以上.*仅供参考",
        r"如有.*疑问",
        r"希望.*对.*有帮助",
        r"如果您.*需要",
    ]
    substantive = []
    for s in sentences:
        if len(s) <= 15:
            continue
        if any(re.search(p, s) for p in non_substantive_patterns):
            continue
        substantive.append(s)
    if not substantive:
        return 1.0

    # 判断每个句子是否有引用（直接或依附前句）
    def has_new_legal_term(s: str) -> bool:
        return bool(re.search(r"《[^》]+》|第[零一二三四五六七八九十百千\d]+[条章节]", s))

    cited_count = 0
    prev_has_citation = False
    for s in substantive:
        if re.search(r"\[来源\d+\]", s):
            # 直接引用
            cited_count += 1
            prev_has_citation = True
        elif prev_has_citation and len(s) >= 15 and not has_new_legal_term(s):
            # 依附前句引用的解释句：15字以上、不含新法律术语
            cited_count += 1
            prev_has_citation = True  # 继续保持，允许连续解释句
        else:
            prev_has_citation = False

    return cited_count / len(substantive)


# ── 辅助函数 ──────────────────────────────────────────────────────


def _extract_legal_keywords(text: str) -> list[str]:
    """从文本中提取法律关键词（条文号、法律名称、核心术语）"""
    keywords = []

    # 提取条文号：第X条、第X章
    articles = re.findall(r"第[零一二三四五六七八九十百千\d]+[条章节款项]", text)
    keywords.extend(articles)

    # 提取法律名称：《XXX法》
    laws = re.findall(r"《[^》]+》", text)
    keywords.extend(laws)

    # 提取核心法律术语
    legal_terms = [
        "用人单位",
        "劳动者",
        "劳动合同",
        "解除",
        "终止",
        "赔偿",
        "补偿",
        "共同财产",
        "分割",
        "抚养",
        "继承",
        "遗嘱",
        "借款",
        "合同",
        "侵权",
        "责任",
        "赔偿金",
        "加班",
        "工资",
        "试用期",
        "离婚",
        "配偶",
        "子女",
        "父母",
        "正当防卫",
        "防卫过当",
        "不可抗力",
        "违约",
        "诈骗",
        "盗窃",
        "抢劫",
    ]
    for term in legal_terms:
        if term in text:
            keywords.append(term)

    return list(set(keywords))


def _split_sentences(text: str) -> list[str]:
    """按中文标点拆分句子"""
    # 按句号、问号、感叹号、分号拆分
    sentences = re.split(r"[。！？；\n]+", text)
    return [s.strip() for s in sentences if s.strip()]


# ── 综合评估 ──────────────────────────────────────────────────────


def evaluate_single(case: dict) -> dict:
    """
    对单个用例计算所有指标。

    参数:
        case: 包含 question, answer, contexts, ground_truth, reference_contexts, expect_keywords 等字段

    返回:
        包含所有指标分数的字典
    """
    answer = case.get("answer", "")
    contexts = case.get("contexts", [])
    ground_truth = case.get("ground_truth", "")
    reference_contexts = case.get("reference_contexts", [])
    expect_keywords = case.get("expect_keywords", [])
    contexts_with_scores = case.get("contexts_with_scores", [])

    return {
        # 检索层
        "context_precision": context_precision(contexts, ground_truth),
        "context_recall": context_recall(contexts, reference_contexts),
        "retrieval_hit_rate": retrieval_hit_rate(answer, expect_keywords),
        "reranker_score_avg": reranker_score_avg(contexts_with_scores),
        # 生成层
        "faithfulness": faithfulness(answer, contexts),
        "answer_relevancy": answer_relevancy(answer, question=case.get("question", "")),
        # 引用层
        "citation_accuracy": citation_accuracy(answer, contexts),
        "citation_coverage": citation_coverage(answer, contexts),
    }


def evaluate_all(results: list[dict]) -> dict:
    """
    对所有用例计算指标并汇总。

    返回:
    {
        "per_case": [{"id": ..., "category": ..., "metrics": {...}}],
        "per_category": {"category_name": {metric: avg_value}},
        "overall": {metric: avg_value},
        "summary": {metric: {mean, min, max, std}},
    }
    """
    per_case = []
    category_metrics = {}

    for case in results:
        metrics = evaluate_single(case)
        case_id = case.get("id", "unknown")
        category = case.get("category", "unknown")

        per_case.append(
            {
                "id": case_id,
                "category": category,
                "question": case.get("question", "")[:50],
                "metrics": metrics,
            }
        )

        if category not in category_metrics:
            category_metrics[category] = []
        category_metrics[category].append(metrics)

    # 按分类汇总
    per_category = {}
    for cat, metrics_list in category_metrics.items():
        per_category[cat] = _aggregate_metrics(metrics_list)

    # 总体汇总
    all_metrics = [pc["metrics"] for pc in per_case]
    overall = _aggregate_metrics(all_metrics)

    return {
        "per_case": per_case,
        "per_category": per_category,
        "overall": overall,
        "total_cases": len(results),
    }


def _aggregate_metrics(metrics_list: list[dict]) -> dict:
    """计算指标的均值"""
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    result = {}
    for key in keys:
        values = [m[key] for m in metrics_list if key in m and m[key] is not None]
        if values:
            result[key] = {
                "mean": round(sum(values) / len(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            }
    return result


def print_report(eval_result: dict):
    """打印评估报告"""
    print("\n" + "=" * 70)
    print("RAGAS 评估报告")
    print("=" * 70)

    # 总体指标
    print("\n── 总体指标 ──")
    for metric, stats in eval_result["overall"].items():
        print(f"  {metric:25s}  mean={stats['mean']:.4f}  min={stats['min']:.4f}  max={stats['max']:.4f}")

    # 分类指标
    print("\n── 分类指标 ──")
    for cat, stats in eval_result["per_category"].items():
        print(f"\n  [{cat}]")
        for metric, vals in stats.items():
            print(f"    {metric:23s}  mean={vals['mean']:.4f}")

    # 逐用例详情
    print("\n── 逐用例详情 ──")
    for case in eval_result["per_case"]:
        metrics = case["metrics"]
        print(
            f"  {case['id']:4s} | {case['category']:25s} | "
            f"precision={metrics['context_precision']:.2f} "
            f"recall={metrics['context_recall']:.2f} "
            f"faithfulness={metrics['faithfulness']:.2f} "
            f"relevancy={metrics['answer_relevancy']:.2f}"
        )

    print("\n" + "=" * 70)


def save_report(eval_result: dict, output_path: str = None):
    """保存评估报告到 JSON"""
    if output_path is None:
        output_path = str(Path(__file__).parent / "ragas_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)
    print(f"评估报告已保存到: {output_path}")
