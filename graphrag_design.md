# GraphRAG 增强方案 — 法律领域知识图谱 + 混合检索

## 1. 当前系统局限性分析

### 1.1 纯向量 + BM25 检索的瓶颈

当前系统使用 FAISS 向量检索 + BM25 关键词检索的混合方案，关键词命中率已达 90%，但存在以下结构性局限：

| 局限 | 具体表现 | 示例查询 |
|------|---------|---------|
| **跨文档关系缺失** | 无法关联不同法律之间的条文 | "《民法典》和《合同法》在违约责任方面有什么关系？" |
| **层次结构丢失** | 扁平 chunk 丢失了 章→节→条 的结构 | "《民法典》合同编有哪些章节？" |
| **引用链断裂** | 法条之间的引用关系无法利用 | "《民法典》第577条引用了哪些其他条文？" |
| **实体关联缺失** | 同一法律概念在不同法律中的表述无法关联 | "什么是'善意取得'？各法律是怎么规定的？" |
| **多跳推理不足** | 需要经过中间步骤才能回答的查询 | "甲公司违约→适用什么法律→该法律的赔偿标准是什么？" |

### 1.2 当前检索流程

```
用户查询 → 查询分析(改写+分解+HyDE) → 并行检索(向量+BM25) → Reranker → LLM生成
```

问题：检索完全依赖文本相似度，无法利用法律体系的结构化知识。

---

## 2. 技术选型对比

### 2.1 候选方案

| 方案 | 核心思路 | 优势 | 劣势 | 成本 |
|------|---------|------|------|------|
| **Microsoft GraphRAG** | LLM 自动抽取实体/关系，社区检测生成摘要 | 全自动，质量高，支持全局摘要 | LLM 调用成本高（148 篇文档约 ¥500-1000），构建慢（数小时），不可控 | 高 |
| **LightRAG** | 轻量级图谱，LLM 抽取 + 双层检索（低层/高层） | 成本低，支持增量更新，开源活跃 | 法律领域未经验证，社区较小 | 中 |
| **自建知识图谱 (Neo4j + 规则)** | 规则抽取法律实体/关系，Neo4j 存储，Cypher 查询 | 精确可控，无 LLM 成本，可增量更新 | 开发工作量大，需维护抽取规则 | 低 |
| **混合方案 (推荐)** | 规则抽取 + Neo4j + 向量检索 + BM25 三路融合 | 平衡成本和精度，渐进式集成 | 架构复杂度增加 | 中低 |

### 2.2 推荐方案：混合知识图谱

**选择理由：**

1. **法律领域高度结构化** — 法条编号、章节结构、引用关系都是固定格式，规则抽取比 LLM 抽取更精确
2. **成本敏感** — 148 篇法律全文用 LLM 抽取实体/关系需要大量 API 调用，规则抽取零成本
3. **可增量更新** — 新增法律文档时，规则抽取可以即时处理，无需等待 LLM
4. **精确可控** — 法律领域的实体类型和关系类型是明确的，规则可以精确定义

---

## 3. 法律领域知识图谱 Schema 设计

### 3.1 节点类型

```cypher
// 法律文档节点
(:Law {
    name: String,          // 简称，如 "民法典"
    full_name: String,     // 全称，如 "中华人民共和国民法典"
    category: String,      // 分类：民事/刑事/行政/经济/社会
    effective_date: String, // 施行日期
    status: String         // 状态：现行/已废止/已修订
})

// 章节节点
(:Chapter {
    number: String,        // 章节编号，如 "第一编"
    title: String,         // 章节标题
    law_name: String       // 所属法律
})

// 条文节点
(:Article {
    number: String,        // 条文编号，如 "第一百四十三条"
    text: String,          // 条文全文
    law_name: String,      // 所属法律
    chapter: String        // 所属章节
})

// 法律概念节点
(:Concept {
    name: String,          // 概念名称，如 "善意取得"
    definition: String     // 概念定义（来自条文）
})

// 立法机构节点
(:Institution {
    name: String           // 机构名称，如 "全国人民代表大会"
})
```

### 3.2 关系类型

```cypher
// 结构关系
(:Law)-[:HAS_CHAPTER]->(:Chapter)
(:Chapter)-[:CONTAINS]->(:Article)

// 引用关系
(:Article)-[:CITES {context: String}]->(:Article)
// 例：第577条 "依照本法第五百八十四条的规定" → CITES → 第584条

// 概念关系
(:Article)-[:DEFINES]->(:Concept)
(:Article)-[:USES]->(:Concept)

// 主题关系
(:Law)-[:BELONGS_TO]->(:Category {name: "民事法律"})

// 机构关系
(:Law)-[:ENACTED_BY]->(:Institution)
```

### 3.3 图谱规模估算

| 节点类型 | 预估数量 | 说明 |
|---------|---------|------|
| Law | ~150 | 148 篇法律 + 可能的补充 |
| Chapter | ~2,000 | 平均每篇 13 章 |
| Article | ~30,000 | 平均每篇 200 条 |
| Concept | ~5,000 | 法律术语词典 |
| **总计** | **~37,000 节点** | |

| 关系类型 | 预估数量 |
|---------|---------|
| HAS_CHAPTER | ~2,000 |
| CONTAINS | ~30,000 |
| CITES | ~5,000 |
| DEFINES | ~5,000 |
| **总计** | **~42,000 关系** |

规模适中，Neo4j 单机即可轻松处理。

---

## 4. 实体/关系抽取规则设计

### 4.1 法律文档识别

复用现有 `processor.py` 的 `_is_legal_content()` 方法。

### 4.2 条文抽取

```python
# 已有：processor.py 的 _extract_structured_articles()
# 使用正则：第[零一二三四五六七八九十百千万\d]+条
```

### 4.3 章节抽取

```python
CHAPTER_PATTERN = r'第[零一二三四五六七八九十百千\d]+[编篇章节]\s*(.*)'
```

### 4.4 引用关系抽取

```python
# 条文内引用模式
CITE_PATTERNS = [
    r'依照本法第([零一二三四五六七八九十百千万\d]+)条',  # 自引用
    r'依据《([^》]+)》第([零一二三四五六七八九十百千万\d]+)条',  # 跨法律引用
    r'本法第([零一二三四五六七八九十百千万\d]+)条',  # 简写引用
    r'第([零一二三四五六七八九十百千万\d]+)条第([零一二三四五六七八九十百千万\d]+)款',  # 条款引用
]
```

### 4.5 概念抽取

```python
# 法律术语词典（可从法律文本中自动提取高频术语）
LEGAL_CONCEPTS = [
    "善意取得", "违约责任", "侵权责任", "不当得利",
    "无因管理", "物权", "债权", "知识产权",
    "正当防卫", "紧急避险", "代理", "时效",
    # ... 可扩展到数千个
]
```

### 4.6 法律分类

```python
LAW_CATEGORIES = {
    "民事法律": ["民法", "合同", "婚姻", "继承", "物权", "侵权"],
    "刑事法律": ["刑法", "刑事诉讼"],
    "行政法律": ["行政", "行政处罚", "行政许可"],
    "经济法律": ["公司", "证券", "保险", "银行", "税"],
    "社会法律": ["劳动", "社会保", "环境", "教育"],
}
```

---

## 5. 检索架构设计

### 5.1 三路融合检索

```
                         ┌─────────────────────────────────────┐
                         │         用户查询                      │
                         └──────────────┬──────────────────────┘
                                        │
                         ┌──────────────▼──────────────────────┐
                         │       查询分析 (analyze_query)        │
                         │  - 改写 + 分解 + HyDE                │
                         │  - 实体识别（法律名/条文号/概念）       │
                         └──────────────┬──────────────────────┘
                                        │
                   ┌────────────────────┼────────────────────┐
                   ▼                    ▼                    ▼
         ┌──────────────────┐  ┌──────────────┐  ┌──────────────────┐
         │  向量检索 (FAISS) │  │ BM25 检索    │  │  图谱检索 (Neo4j) │
         │  语义相似度       │  │ 关键词匹配   │  │  实体关系遍历     │
         │  weight: α       │  │ weight: β    │  │  weight: γ       │
         └────────┬─────────┘  └──────┬───────┘  └────────┬─────────┘
                  │                   │                   │
                  └───────────────────┼───────────────────┘
                                      ▼
                         ┌──────────────────────────────┐
                         │    结果融合 + Reranker         │
                         │  - 去重 + 加权求和             │
                         │  - DashScope Reranker         │
                         │  - 相关性阈值过滤              │
                         └──────────────┬───────────────┘
                                        ▼
                         ┌──────────────────────────────┐
                         │    LLM 生成 + 引用溯源        │
                         └──────────────────────────────┘
```

### 5.2 图谱检索策略

**实体链接：** 从查询中识别法律实体并映射到图谱节点

```python
def extract_legal_entities(query: str) -> dict:
    """从查询中提取法律实体"""
    entities = {"laws": [], "articles": [], "concepts": []}

    # 法律名称：《民法典》《刑法》等
    law_matches = re.findall(r'《([^》]+)》', query)
    entities["laws"] = law_matches

    # 条文引用：第X条
    article_matches = re.findall(r'第([零一二三四五六七八九十百千万\d]+)条', query)
    entities["articles"] = article_matches

    # 概念匹配：与概念词典做交集
    for concept in LEGAL_CONCEPTS:
        if concept in query:
            entities["concepts"].append(concept)

    return entities
```

**子图检索：** 1-2 跳遍历获取相关条文

```cypher
// 1跳：获取某条文的直接上下文
MATCH (a:Article {number: $article_num, law_name: $law_name})
OPTIONAL MATCH (a)-[:CITES]->(cited:Article)
OPTIONAL MATCH (citer:Article)-[:CITES]->(a)
OPTIONAL MATCH (a)-[:DEFINES]->(concept:Concept)
RETURN a, collect(DISTINCT cited) AS cited_articles,
       collect(DISTINCT citer) AS citer_articles,
       collect(DISTINCT concept) AS concepts

// 2跳：获取某概念的所有相关条文
MATCH (c:Concept {name: $concept_name})
MATCH (a:Article)-[:DEFINES|USES]->(c)
OPTIONAL MATCH (a)-[:CITES]->(cited:Article)
RETURN a, collect(DISTINCT cited) AS related_articles
```

**路径检索：** 找到两个条文之间的引用链

```cypher
MATCH path = shortestPath(
    (a1:Article)-[:CITES*..5]->(a2:Article)
)
WHERE a1.number = $from_article AND a2.number = $to_article
RETURN path
```

### 5.3 融合评分

```python
def fuse_scores(vector_results, bm25_results, graph_results,
                alpha=0.4, beta=0.3, gamma=0.3):
    """三路融合评分"""
    scores = {}

    for doc, score in vector_results:
        scores[doc] = scores.get(doc, 0) + alpha * score

    for doc, score in bm25_results:
        scores[doc] = scores.get(doc, 0) + beta * score

    for doc, score in graph_results:
        # 图谱分数基于 PageRank 中心度和跳数距离
        scores[doc] = scores.get(doc, 0) + gamma * score

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

---

## 6. 实现路线图

### 阶段 A：图谱构建（1-2 周）

1. 安装 Neo4j（Docker 或本地）
2. 实现 `law_assistant/graph.py`：
   - `LegalKnowledgeGraph` 类
   - 节点/关系 Schema 定义
   - 批量导入 `knowledge_base/` 中的 148 篇法律
3. 实现抽取规则：
   - 章节抽取
   - 条文抽取（复用现有 `_extract_structured_articles`）
   - 引用关系抽取
   - 概念词典匹配

### 阶段 B：图谱检索集成（1 周）

1. 实现 `graph_search()` 方法：
   - 实体链接
   - 子图检索（1-2 跳）
   - 结果评分（PageRank + 距离）
2. 修改 `retrieve_documents()`：
   - 添加图谱检索分支
   - 三路融合评分
3. 测试权重调优（α, β, γ）

### 阶段 C：查询增强（1 周）

1. 在 `analyze_query()` 中添加实体识别
2. 利用图谱上下文增强 HyDE 文档生成
3. 在 prompt 中添加图谱结构化信息（如条文关系）

### 阶段 D：评估与优化（1 周）

1. 扩展 `baseline_eval.py` 测试用例：
   - 跨文档关系查询
   - 条文引用链查询
   - 概念关联查询
2. 对比 GraphRAG 前后的关键词命中率
3. 权重调优和性能优化

---

## 7. 成本估算

| 项目 | 成本 | 说明 |
|------|------|------|
| Neo4j | 免费 | Community Edition，单机足够 |
| 图谱构建 | 0 元 LLM 成本 | 纯规则抽取 |
| 开发时间 | 4 周 | 1 人全职 |
| 维护成本 | 低 | 新文档自动增量更新 |

对比 Microsoft GraphRAG：
- LLM 抽取成本：约 ¥500-1000（148 篇文档）
- 构建时间：数小时
- 无法精确控制抽取质量

**混合方案总成本约为 Microsoft GraphRAG 的 1/10，且精度更高。**

---

## 8. 依赖项

```txt
# 新增依赖
neo4j>=5.0.0          # Neo4j Python 驱动
py2neo>=2021.0        # 可选：更友好的 Neo4j ORM
```

```yaml
# docker-compose.yml (可选，用于本地开发)
services:
  neo4j:
    image: neo4j:5-community
    ports:
      - "7474:7474"  # Web UI
      - "7687:7687"  # Bolt 协议
    environment:
      NEO4J_AUTH: neo4j/password
    volumes:
      - neo4j_data:/data
```

---

## 9. 风险与缓解

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|---------|
| 引用关系抽取不完整 | 中 | 低 | 先覆盖主要法律，逐步完善规则 |
| Neo4j 运维复杂度 | 低 | 中 | Docker 部署，定期备份 |
| 图谱检索引入延迟 | 中 | 中 | 缓存热点查询，设置超时保护 |
| 权重调优困难 | 中 | 低 | 基于 baseline_eval 自动化调优 |
