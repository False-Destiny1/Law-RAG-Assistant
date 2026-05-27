# 基于 RAG 架构的智能法律问答系统

## 项目概述

基于 RAG（Retrieval-Augmented Generation）架构的智能法律问答系统，集成了用户权限管理、多格式文档解析（含 OCR）、智能问答交互、知识库动态管理等核心模块。

**技术栈：** FastAPI + PostgreSQL + Redis + FAISS + BM25 + DashScope Reranker + PaddleOCR + BAAI/bge-small-zh-v1.5 + MiMo LLM

---

## 完整部署流程

### 第一步：环境准备

**系统要求：**
- Python 3.10+（推荐 3.13）
- 内存 ≥ 8GB
- 显存 ≥ 8GB（本地嵌入模型需要 CUDA）
- 存储 ≥ 10GB
- PostgreSQL 16+（或使用 SQLite）
- Redis（可选，用于缓存和限流，不安装时自动降级）

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
| sentence-transformers | ≥3.0 | 嵌入模型 |
| rank-bm25 | ≥0.2 | BM25 检索 |
| dashscope | ≥1.17 | Reranker API |
| paddleocr | ≥3.5 | 扫描文档 OCR |
| jieba | ≥0.42 | 中文分词 |
| bcrypt | ≥4.0 | 密码加密 |
| redis | ≥5.0 | 缓存、限流、会话管理 |

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

# ── Reranker（DashScope）──
RERANKER_API_KEY=你的DashScope密钥
RERANKER_BASE_URL=https://dashscope.aliyuncs.com/api/v1/services/reranking/reranking
RERANKER_MODEL=gte-rerank

# ── 检索权重 ──
VECTOR_RETRIEVAL_WEIGHT=0.8
BM25_RETRIEVAL_WEIGHT=0.2
RELEVANCE_THRESHOLD=0.15

# ── 数据库 ──
# PostgreSQL（推荐）
DATABASE_URL=postgresql://postgres:密码@localhost:5432/law_assistant
# 或 SQLite（开发用）
# DATABASE_URL=sqlite:///user.db

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

**SQLite：** 无需额外操作，首次启动自动创建。

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

> 首次构建索引需要 5-10 分钟，后续启动直接加载已有索引。

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
    ├─── 并行混合检索
    │       ├── 主查询 + 子查询 + HyDE 文档
    │       ├── 每个查询：FAISS 向量检索（权重 0.8）‖ BM25 关键词检索（权重 0.2）
    │       └── 加权融合 → 候选文档
    │
    ├─── DashScope Reranker（gte-rerank）
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

**每次请求共 2 次 LLM 调用**（查询分析 + 回答生成）+ 1 次 Reranker 网络调用。

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
| 混合检索 | 向量语义 + BM25 关键词，并行执行 | 系统自动 |
| 重排序 | DashScope Reranker 精排，超时自动回退 | 系统自动 |
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
检查 `.env` 中 `RERANKER_API_KEY` 和 `RERANKER_BASE_URL` 是否正确。

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
