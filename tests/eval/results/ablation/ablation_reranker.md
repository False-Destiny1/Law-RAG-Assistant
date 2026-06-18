# 消融实验: CrossEncoder Reranker

> 2 配置 x 5 场景 x 3 次运行
> 生成时间: 2026-06-02（模板，待运行填充实际数据）

## 实验设计

| 配置 | 环境变量 `ENABLE_RERANKER` | 说明 |
|---|---|---|
| 有 Reranker | `true` | bge-reranker-v2-m3 CrossEncoder 重排序 |
| 无 Reranker | `false` | 直接用 RRF 融合分数排序 |

## 运行命令

```bash
python -m tests.eval.ablation_framework --ablation reranker --runs 3
```

## 预期结果（待实际运行填充）

| 场景 | 指标 | 有 Reranker | 无 Reranker |
|---|---|---|---|
| A: 精确法律查询 | precision | _待填充_ | |
| B: 口语化查询 | precision | _待填充_ | |
| D: 冷门法律 | precision | _待填充_ | |
| Overall | precision | _待填充_ | |

## 预期分析

- Reranker 应在所有场景提升 context_precision（将最相关文档排到 top-k）
- 对冷门法律（D）提升可能最明显，因为原始召回噪音多
- 对精确法律查询（A）提升可能较小，因为 BM25 已能精确匹配

## 简历写法（待数据填充后更新）

> 通过消融实验证明 CrossEncoder 重排序在冷门法律查询场景下 context_precision 提升 _X_%，有效过滤低相关性文档
