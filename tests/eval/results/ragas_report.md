# RAGAS 评估报告

> 基于 15 条评估数据，8 项自定义指标

## 总体指标

| 指标 | 均值 | 最小值 | 最大值 | 评价 |
|---|---|---|---|---|
| context_precision | 0.7667 | 0.0000 | 1.0000 | 良好 |
| context_recall | 0.8444 | 0.0000 | 1.0000 | 良好 |
| retrieval_hit_rate | 0.6322 | 0.2000 | 1.0000 | 一般 |
| reranker_score_avg | 0.2776 | 0.0000 | 0.6126 | 需改进 |
| faithfulness | 0.9840 | 0.9310 | 1.0000 | 优秀 |
| answer_relevancy | 1.0000 | 1.0000 | 1.0000 | 优秀 |
| citation_accuracy | 1.0000 | 1.0000 | 1.0000 | 优秀 |
| citation_coverage | 0.2761 | 0.0000 | 0.6471 | 需改进 |

## 强项

- **faithfulness** (0.984): 回答忠实度高，LLM 几乎不会编造未在检索文档中出现的内容
- **answer_relevancy** (1.000): 回答与问题高度相关，没有跑题
- **citation_accuracy** (1.000): 引用的法条准确无误，没有张冠李戴

## 弱项

- **retrieval_hit_rate** (0.632): 部分查询未能命中任何相关文档
- **reranker_score_avg** (0.278): 重排序分数整体偏低，reranker 对相关性区分度不足
- **citation_coverage** (0.276): 回答中引用标签覆盖率低，大部分回答缺少 [来源N] 标注

## 分类表现

| 类别 | 数量 | avg_precision | avg_recall | avg_faithfulness | avg_citation_coverage |
|---|---|---|---|---|---|
| boundary | 3 | 0.667 | 0.889 | 0.988 | 0.225 |
| multi_turn | 2 | 1.000 | 1.000 | 1.000 | 0.312 |
| retrieval_cold | 2 | 0.000 | 0.000 | 0.939 | 0.250 |
| retrieval_colloquial | 2 | 1.000 | 1.000 | 1.000 | 0.250 |
| retrieval_direct | 5 | 0.920 | 1.000 | 0.993 | 0.239 |
| retrieval_multi | 1 | 0.900 | 1.000 | 0.952 | 0.647 |

## 改进方向

1. **优化 citation_coverage**: 在 chunk 切割时保留引用元数据（法条编号、章节标题），扩展引用提取正则规则
2. **改善冷门法律检索**: 扩展 BM25 法律词典覆盖更多领域，或引入多知识库检索
3. **提升 reranker 分数**: 调优 reranker 阈值（当前 0.15），或尝试更大的 reranker 模型
4. **消融实验验证**: 补充 baseline 对比（FAISS-only / BM25-only / Graph-only）和 HyDE 消融实验
