# 消融: HyDE 检索增强 — 口语化查询

> 生成时间: 2026-06-02 16:46

| Case | Category | 有 HyDE P | 有 HyDE R | 无 HyDE P | 无 HyDE R |
|---|---|---|---|---|---|
| C1 | retrieval_colloquial | 0.167 | 0.500 | 0.200 | 0.500 |
| C2 | retrieval_colloquial | 0.500 | 0.500 | 0.000 | 0.000 |
| **avg** | | **0.333** | **0.500** | **0.100** | **0.250** |

## 效果量 (baseline: 有 HyDE)

| 配置 | precision 变化 | recall 变化 |
|---|---|---|
| 无 HyDE | -70.0% (-0.233) | -50.0% (-0.250) |
