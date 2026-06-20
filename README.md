# 基于 RAG 架构的智能法律问答系统

## 项目概述

基于 RAG（Retrieval-Augmented Generation）架构的智能法律问答系统，集成了用户权限管理、多格式文档解析（含 OCR）、智能问答交互、知识库动态管理等核心模块。

**技术栈：** FastAPI + PostgreSQL + Redis + FAISS + BM25 + Neo4j 知识图谱 + BAAI/bge-reranker-v2-m3 (本地) + PaddleOCR + BAAI/bge-small-zh-v1.5 + MiMo LLM

---

## 完整部署流程

### 第一步：环境准备

**系统要求：**
- Python 3.10+（推荐 3.13）
- 内存 ≥ 8GB
- 显存 ≥ 8GB（本地嵌入模型需要 CUDA）
- 存储 ≥ 10GB
- PostgreSQL 16+
- Redis（可选，用于缓存和限流，不安装时自动降级）
- Neo4j（可选，用于知识图谱检索，不安装时自动降级为双路检索）

**创建 Conda 环境：**
```bash
conda create -n pytorch python=3.13
conda activate pytorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 第二步：安装依赖

```bash
conda activate pytorch
cd "e:\python code\law_assistant-main"
pip install -r requirements.txt
```

**核心依赖清单：**

| 包名 | 版本 | 用途 |
|------|------|------|
| fastapi | ≥0.100 | Web 框架 |
| uvicorn | ≥0.20 | ASGI 服务器 |
| sqlalchemy | ≥2.0 | 数据库 ORM |
| psycopg2-binary | ≥2.9 | PostgreSQL 驱动 |
| langchain | ≥0.3 | LLM 编排框架 |
| faiss-cpu | ≥1.7 | 向量数据库 |
| sentence-transformers | ≥3.0 | 嵌入模型 + 本地 Reranker |
| rank-bm25 | ≥0.2 | BM25 检索 |
| paddleocr | ≥3.5 | 扫描文档 OCR |
| jieba | ≥0.42 | 中文分词 |
| bcrypt | ≥4.0 | 密码加密 |
| redis | ≥5.0 | 缓存、限流、会话管理 |
| neo4j | ≥5.0 | 知识图谱存储与检索（可选） |

**OCR 相关系统依赖：**
```bash
# pdf2image 需要 poppler
conda install -c conda-forge poppler
```

### 第三步：配置环境变量

编辑 `.env` 文件：

```env
# ── LLM 配置（必填）──
MIMO_API_KEY=你的API密钥
MIMO_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
MIMO_MODEL=mimo-v2.5-pro

# ── 嵌入模型（本地）──
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5

# ── 向量数据库 ──
VECTOR_DB_PATH=law_faiss

# ── Reranker（本地模型，默认）──
RERANKER_PROVIDER=local
RERANKER_MODEL_PATH=BAAI/bge-reranker-v2-m3

# ── Reranker（DashScope API，可选）──
# RERANKER_PROVIDER=dashscope
# RERANKER_API_KEY=你的DashScope密钥
# RERANKER_MODEL=gte-rerank

# ── 检索权重（三路融合）──
VECTOR_RETRIEVAL_WEIGHT=0.4
BM25_RETRIEVAL_WEIGHT=0.3
GRAPH_RETRIEVAL_WEIGHT=0.3
RELEVANCE_THRESHOLD=0.15

# ── Neo4j 知识图谱（可选）──
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=你的Neo4j密码
NEO4J_DATABASE=neo4j

# ── 数据库 ──
DATABASE_URL=postgresql://postgres:密码@localhost:5432/law_assistant

# ── Redis（可选）──
REDIS_URL=redis://localhost:6379/0
```

### 第四步：准备数据库

**PostgreSQL（推荐）：**
```bash
# 安装 PostgreSQL 后
psql -U postgres
CREATE DATABASE law_assistant;
\q
```

### 第五步：准备知识库

将法律文本文件（`.txt` 格式）放入 `knowledge_base/` 目录。项目自带 80+ 部中国法律全文。

支持的文档格式：`.txt`、`.pdf`、`.docx`、`.jpg`、`.png`、`.bmp`、`.tiff`

扫描版 PDF 和图片文件会自动通过 PaddleOCR 进行文字识别。

### 第六步：首次启动

```bash
# 方式一：双击 start.bat（推荐）

# 方式二：命令行
conda activate pytorch
python -m uvicorn app:app --host 127.0.0.1 --port 8080
```

**首次启动会自动执行：**
1. 创建数据库表和索引
2. 创建默认管理员账号（`admin` / `admin123`）
3. 加载 `BAAI/bge-small-zh-v1.5` 嵌入模型到 GPU
4. 扫描 `knowledge_base/` 目录，构建 FAISS 向量索引和 BM25 关键词索引
5. 保存索引到 `law_faiss/` 和 `bm25_index.pkl`
6. 连接 Neo4j 并创建知识图谱 Schema（如配置了 Neo4j）

> 首次构建索引需要 5-10 分钟，后续启动直接加载已有索引。

**构建知识图谱（首次使用 Neo4j 时需要）：**

```python
# 在 Python 交互环境或脚本中执行
from law_assistant.rag import DeepSeekApiRag
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("MIMO_API_KEY")
rag = DeepSeekApiRag(api_key)
rag.build_knowledge_graph("knowledge_base/")
```

> 知识图谱构建需要 5-15 分钟（取决于 knowledge_base/ 中的文件数量），构建完成后会持久化存储在 Neo4j 中，后续启动无需重建。

### 第七步：访问系统

浏览器打开：**http://127.0.0.1:8080**

- 管理员登录：`admin` / `admin123`
- 注册新账号（默认为普通用户角色）
- 专家/管理员可上传文档到知识库

---

## 项目结构

```
law_assistant-main/
├── app.py                    # FastAPI 主应用（路由、认证、API、数据库模型）
├── law_assistant/            # 核心 Python 包
│   ├── __init__.py           # 包导出
│   ├── rag.py                # RAG 引擎（检索、重排序、生成、记忆）
│   ├── bm25.py               # BM25 关键词检索器（jieba 分词）
│   ├── graph.py              # 知识图谱（规则抽取 + Neo4j 存储 + 图谱检索）
│   ├── processor.py          # 文档处理器（法律/通用识别 + OCR 回退）
│   ├── splitter.py           # 文本分块器（按条款/按字符）
│   ├── memory.py             # 对话记忆管理（L1 内存 + L2 Redis + L3 DB，LRU 淘汰）
│   ├── redis_utils.py        # Redis 客户端（连接池、缓存、限流、健康检查）
│   ├── security.py           # 安全模块（提示词注入检测、上下文过滤）
│   └── prompts.yaml          # LLM 提示词模板
│
├── templates/                # Jinja2 HTML 模板
│   ├── base.html             # 基础布局
│   ├── login.html            # 登录页
│   ├── register.html         # 注册页
│   ├── index.html            # 聊天主页
│   ├── upload.html           # 文档上传（专家/管理员）
│   ├── knowledge_base.html   # 知识库列表
│   ├── create_knowledge_base.html
│   └── edit_knowledge_base.html
│
├── static/
│   ├── css/
│   │   ├── shared.css        # 设计系统（CSS 变量、组件）
│   │   └── index.css         # 聊天界面样式
│   └── js/
│       └── index.js          # 聊天交互逻辑（SSE 流式）
│
├── tests/                    # 测试文件
│   ├── unit/                 # 单元测试（pytest）
│   ├── eval/                 # RAG 基准测试（需服务器运行）
│   │   ├── baseline_eval.py
│   │   └── run_all.py
│   └── manual/               # 手动测试脚本
│       ├── test_ocr.py
│       └── test_weights.py
│
├── knowledge_base/           # 法律文本知识库（80+ 部法律全文）
├── uploads/                  # 用户上传文档
│
├── .env                      # 环境变量配置（含 API 密钥，勿提交）
├── requirements.txt          # 依赖清单
├── start.bat                 # 一键启动脚本（Windows）
└── README.md                 # 本文件
```

**自动生成的文件（.gitignore 排除）：**
- `law_faiss/` — FAISS 向量索引
- `bm25_index.pkl` — BM25 关键词索引
- `BAAI/bge-small-zh-v1.5/` — 本地嵌入模型
- `BAAI/bge-reranker-v2-m3/` — 本地 Reranker 模型

---

## 核心架构

```
用户提问
    │
    ▼
FastAPI /ask_stream (SSE 流式)
    │
    ├─── 安全检查（提示词注入检测，命中则直接拒绝）
    │
    ├─── 查询分析（1 次 LLM 调用）
    │       ├── 多轮对话融合
    │       ├── 口语 → 法律术语改写
    │       ├── 查询分解（最多 3 个子查询）
    │       └── HyDE 假设性文档生成
    │
    ├─── 并行三路融合检索
    │       ├── 主查询 + 子查询 + HyDE 文档
    │       ├── 每个查询：FAISS 向量检索（0.4）‖ BM25 关键词检索（0.3）‖ Neo4j 图谱检索（0.3）
    │       └── 加权融合 → 候选文档
    │
    ├─── 本地 CrossEncoder Reranker（bge-reranker-v2-m3）
    │       └── 精排 → 相关性阈值过滤（0.15）
    │
    ├─── 知识库过滤（可选，按 knowledge_base_id）
    │
    ├─── 构建 Prompt
    │       ├── 检索上下文（带 [来源N] 引用编号，含注入过滤）
    │       ├── 对话历史（最近 5 轮）
    │       └── prompts.yaml 系统提示词（含安全规则）
    │
    ├─── MiMo LLM 流式生成（1 次 LLM 调用）
    │       └── SSE 逐 token 推送
    │
    └─── 保存对话记忆 + 写入 PostgreSQL
```

**每次请求共 2 次 LLM 调用**（查询分析 + 回答生成）+ 1 次本地 Reranker 推理。

---

## 功能说明

| 功能 | 说明 | 权限 |
|------|------|------|
| 用户注册/登录 | 手机号 + 密码，bcrypt 加密，HMAC 签名 session | 所有人 |
| 智能问答 | 基于 RAG 的法律咨询，流式输出，引用溯源 | 所有人 |
| 多轮对话 | 保留最近 5 轮对话记忆（内存 + Redis + DB 三级缓存） | 所有人 |
| 对话管理 | 创建/编辑/删除对话 | 所有人 |
| 知识库管理 | 创建/编辑/删除知识库 | 所有人 |
| 文档上传 | PDF/DOCX/TXT/JPG/PNG，自动向量化 | 专家/管理员 |
| OCR 识别 | 扫描版 PDF 和图片自动 PaddleOCR 文字识别 | 专家/管理员 |
| 三路融合检索 | 向量语义 + BM25 关键词 + 知识图谱，并行执行 | 系统自动 |
| 重排序 | 本地 CrossEncoder Reranker 精排（可选 DashScope API） | 系统自动 |
| 安全防护 | 提示词注入检测（输入过滤 + 上下文清洗 + 提示词加固） | 系统自动 |

---

## 测试

```bash
# 运行单元测试
pytest tests/unit/ -v

# 运行离线测试（OCR 文档处理）
python tests/eval/run_all.py

# 运行在线测试（需先启动服务器）
python tests/eval/run_all.py --online
```

---

## 常见问题

**Q: 启动报错 "sentence_transformers not found"**
```bash
pip install sentence-transformers
```

**Q: 启动报错 "CUDA not available"**
确保安装了 CUDA 版 PyTorch：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

**Q: Reranker 返回 400/404**
默认使用本地模型，无需 API 密钥。如果切换到 DashScope 模式（`RERANKER_PROVIDER=dashscope`），检查 `.env` 中 `RERANKER_API_KEY` 是否正确。

**Q: 扫描 PDF 无法识别文字**
确保安装了 PaddleOCR 和 poppler：
```bash
pip install paddleocr paddlepaddle
conda install -c conda-forge poppler
```

**Q: 端口被占用**
```bash
netstat -ano | findstr :8080
taskkill /F /PID <进程ID>
```

**Q: 知识库索引损坏，检索结果不对**
删除 `law_faiss/` 目录和 `bm25_index.pkl`，重启服务会自动重建。

**Q: PostgreSQL 连接失败**
确认 PostgreSQL 服务已启动，数据库已创建：
```bash
psql -U postgres -d law_assistant -c "SELECT 1;"
```

**Q: Redis 连接失败 / Redis 不可用**
Redis 为可选组件，不安装时系统自动降级到本地缓存和数据库回退，功能不受影响。如需使用：

- Windows：从 `e:\Redis\redis-server.exe` 启动，或使用 `start.bat` 自动启动
- Docker：`docker run -d -p 6379:6379 redis:latest`
- 安装后在 `.env` 中配置 `REDIS_URL=redis://localhost:6379/0`

**Q: Neo4j 连接失败 / 知识图谱不可用**
Neo4j 为可选组件，不安装时系统自动降级为双路检索（向量 + BM25）。如需使用知识图谱：

1. 安装 Neo4j Community Edition（本地或 Docker）
2. 在 `.env` 中配置 `NEO4J_URI`、`NEO4J_USER`、`NEO4J_PASSWORD`
3. 安装 Python 驱动：`pip install neo4j>=5.0.0`
4. 首次使用需构建图谱：运行 `rag.build_knowledge_graph("knowledge_base/")`
5. 使用 `start.bat` 启动会自动发现并启动 Neo4j（支持 `C:\neo4j` 和 `E:\neo4j*` 路径）

```bash
# Docker 方式运行 Neo4j
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/你的密码 neo4j:5
```

**Q: 知识图谱构建后如何查看**

浏览器打开 `http://localhost:7474`，用 Neo4j 账号登录，输入 Cypher 查询：

```cypher
// 查看所有法律节点
MATCH (l:Law) RETURN l LIMIT 50
// 查看某部法律的条文
MATCH (l:Law {name: "民法典"})-[:CONTAINS]->(a:Article) RETURN l, a LIMIT 30
```

## 数据集

### CAIL 法律判决数据集

项目包含完整的 CAIL（Chinese AI and Law）法律竞赛数据集，位于 `data/knowledge_base/final_all_data/`：

| 数据集 | 文件 | 数据量 | 大小 |
|--------|------|--------|------|
| exercise_contest | data_train.json | 154,592 条 | 236 MB |

**数据格式（JSONL）：**

```json
{
  "fact": "案件事实描述...",
  "meta": {
    "relevant_articles": [234],        // 相关法条编号
    "accusation": ["故意伤害"],         // 罪名
    "punish_of_money": 0,              // 罚金（元）
    "criminals": ["被告人"],            // 被告人
    "term_of_imprisonment": {          // 刑期
      "death_penalty": false,          // 是否死刑
      "imprisonment": 12,              // 有期徒刑（月）
      "life_imprisonment": false       // 是否无期徒刑
    }
  }
}
```

**数据用途：**

- 法律问答系统训练和评估
- 罪名预测、法条推荐、刑期预测等 NLP 任务
- RAG 检索增强的知识库补充

---

## 项目亮点

- **三路融合检索**: FAISS 向量检索 + BM25 关键词检索 + Neo4j 知识图谱检索，使用 Reciprocal Rank Fusion 融合排序，兼顾语义匹配、精确关键词和结构化关系
- **HyDE 检索增强**: 通过 LLM 生成假设法律文档，弥补用户口语化提问与法律文档正式表述之间的语义差距
- **自定义 RAGAS 评估**: 8 项指标（context_precision/recall、faithfulness、citation_accuracy/coverage 等）、15 条评估数据集覆盖 6 个类别，faithfulness 0.98、citation_accuracy 1.0
- **3 层 Prompt 注入防御**: 输入过滤（正则匹配已知攻击模式）+ 上下文消毒（移除历史中的注入片段）+ Prompt 硬化（system prompt 安全规则），法律场景特殊处理避免误杀
- **三级缓存架构**: L1 内存 LRU（500 会话、1h 超时）+ L2 Redis + L3 PostgreSQL，支持优雅降级
- **引用后处理**: 自动检测缺少 `[来源N]` 标签的法律句子并补充引用，提升 citation_coverage
