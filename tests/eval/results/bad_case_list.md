# Bad Case 分析报告

共 **3** 条 bad case（共 15 条评估数据）

## 失败类型分布

| 失败类型 | 数量 | 说明 |
|---|---|---|
| 检索失败 | 2 | 检索结果与问题完全不相关 |
| 检索噪音多 | 1 | 检索结果夹杂大量无关文档 |

## Bad Case 详情

### D1 [retrieval_cold]

**问题**: 深海海底资源勘探有什么法律规定

| 指标 | 值 |
|---|---|
| context_precision | 0.000 |
| context_recall | 0.000 |
| faithfulness | 0.947 |
| citation_coverage | 0.500 |
| retrieval_hit_rate | 1.000 |

**失败类型**: 检索失败

**原因分析**: 检索结果与问题完全不相关，可能缺少对应法律文档或分词/向量化有偏差

---

### D2 [retrieval_cold]

**问题**: 个人信息保护法对企业处理用户数据有什么要求

| 指标 | 值 |
|---|---|
| context_precision | 0.000 |
| context_recall | 0.000 |
| faithfulness | 0.931 |
| citation_coverage | 0.000 |
| retrieval_hit_rate | 0.333 |

**失败类型**: 检索失败

**原因分析**: 检索结果与问题完全不相关，可能缺少对应法律文档或分词/向量化有偏差

---

### B1 [boundary]

**问题**: 今天天气怎么样？

| 指标 | 值 |
|---|---|
| context_precision | 0.000 |
| context_recall | 1.000 |
| faithfulness | 1.000 |
| citation_coverage | 0.000 |
| retrieval_hit_rate | 1.000 |

**失败类型**: 检索噪音多

**原因分析**: 检索到了一些相关文档但也夹杂了大量无关文档

---

## 改进建议

1. **检索问题**: 扩展 BM25 法律词典、优化 chunk 策略、调整 RRF 融合参数
2. **噪音问题**: 提高 reranker 阈值（当前 0.15）、优化 reranker 模型
