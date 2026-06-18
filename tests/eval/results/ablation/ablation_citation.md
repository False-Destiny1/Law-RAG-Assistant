# 消融实验: 引用后处理

> 2 配置 x 5 场景 x 3 次运行
> 生成时间: 2026-06-02（模板，待运行填充实际数据）

## 实验设计

| 配置 | 环境变量 `ENABLE_CITATION_POSTPROCESS` | 说明 |
|---|---|---|
| 有后处理 | `true` | 自动为缺少引用的法律句子补充 [来源N] |
| 无后处理 | `false` | LLM 原始输出，不补充引用 |

## 运行命令

```bash
python -m tests.eval.ablation_framework --ablation citation --runs 3
```

## 重点指标

- **citation_coverage**: 当前 0.28，期望后处理提升到 >0.4
- **citation_accuracy**: 应保持 1.0（后处理不引入错误引用）

## 预期分析

- citation_coverage 应有显著提升（自动补充缺失的 [来源N] 标签）
- citation_accuracy 应保持不变（后处理基于检索文档匹配，不编造引用）
- 对多轮对话（E）场景提升可能最明显（多轮上下文更容易遗漏引用）

## 简历写法（待数据填充后更新）

> 实现引用后处理模块，通过消融实验证明 citation_coverage 从 0.28 提升至 _X_，覆盖了 _Y_% 的未引用法律句子
