# 基于 RAG 架构的智能法律问答系统

## 项目概述

基于 RAG（Retrieval-Augmented Generation）架构的智能法律问答系统，集成了用户权限管理、多格式文档解析、智能问答交互、知识库动态管理等核心模块。

**技术栈：** FastAPI + FAISS + BM25 + DashScope Reranker + bge-small-zh-v1.5 + MiMo LLM

---

## 完整部署流程

### 第一步：环境准备

**系统要求：**
- Python 3.10+（推荐 3.13）
- 内存 ≥ 8GB
- 显存 ≥ 8GB（本地嵌入模型需要 CUDA）
- 存储 ≥ 10GB

**创建 Conda 环境（如果还没有）：**
```bash
conda create -n pytorch python=3.13
conda activate pytorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 第二步：安装依赖

```bash
# 激活环境
conda activate pytorch

# 进入项目目录
cd "e:\python code\law_assistant-main"

# 安装所有依赖
pip install -r requirements.txt
```

**核心依赖清单：**
| 包名 | 版本 | 用途 |
|------|------|------|
| fastapi | ≥0.100 | Web 框架 |
| uvicorn | ≥0.20 | ASGI 服务器 |
| sqlalchemy | ≥2.0 | 数据库 ORM |
| langchain | ≥0.3 | LLM 编排框架 |
| faiss-cpu | ≥1.7 | 向量数据库 |
| sentence-transformers | ≥3.0 | 嵌入模型 |
| rank-bm25 | ≥0.2 | BM25 检索 |
| dashscope | ≥1.20 | Reranker API |
| jieba | ≥0.42 | 中文分词 |
| bcrypt | ≥4.0 | 密码加密 |

### 第三步：配置环境变量

编辑 `.env` 文件：

```env
# ── LLM 配置（必填）──
MIMO_API_KEY=你的API密钥
MIMO_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
MIMO_MODEL=mimo-v2.5-pro

# ── 嵌入模型（本地）──
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=bge-small-zh-v1.5

# ── 向量数据库 ──
VECTOR_DB_PATH=law_faiss

# ── Reranker（DashScope）──
RERANKER_API_KEY=你的DashScope密钥
RERANKER_BASE_URL=https://dashscope.aliyuncs.com/api/v1/services/reranking/reranking
RERANKER_MODEL=gte-rerank

# ── 检索权重 ──
VECTOR_RETRIEVAL_WEIGHT=0.6
BM25_RETRIEVAL_WEIGHT=0.4

# ── 数据库 ──
DATABASE_URL=sqlite:///user.db
```

### 第四步：准备知识库数据

将法律文本文件（`.txt` 格式）放入 `knowledge_base/` 目录。项目自带 80+ 部中国法律全文。

文件命名格式：`中华人民共和国XXX法.txt`

### 第五步：首次启动

```bash
# 方式一：双击 start.bat（推荐）

# 方式二：命令行
conda activate pytorch
cd "e:\python code\law_assistant-main"
python -m uvicorn app:app --host 127.0.0.1 --port 8080
```

**首次启动会自动执行：**
1. 创建 SQLite 数据库（`instance/user.db`）
2. 加载 `bge-small-zh-v1.5` 嵌入模型到 GPU
3. 扫描 `knowledge_base/` 目录，构建 FAISS 向量索引
4. 构建 BM25 关键词索引
5. 保存索引到 `law_faiss/` 和 `bm25_index.pkl`

> 首次构建索引需要 5-10 分钟（取决于知识库大小），后续启动直接加载已有索引。

### 第六步：访问系统

浏览器打开：**http://127.0.0.1:8080**

1. 注册账户（选择角色：普通用户 / 法律专家 / 管理员）
2. 登录系统
3. 开始提问法律问题

---

## 项目结构

```
law_assistant-main/
├── app.py                    # FastAPI 主应用（路由、认证、API）
├── model_utils.py            # RAG 核心引擎（检索、生成、记忆）
├── BM25Retriever.py          # BM25 关键词检索器
├── DocumentProcessor.py      # 文档处理器（法律/通用文档识别）
├── DocumentSplitter.py       # 文本分块器（按条款/按字符）
├── ConversationMemory.py     # 对话记忆管理
├── prompts.yaml              # LLM 提示词模板
│
├── templates/                # Jinja2 HTML 模板
│   ├── base.html             # 基础布局
│   ├── login.html            # 登录页
│   ├── register.html         # 注册页
│   ├── index.html            # 聊天主页
│   ├── upload.html           # 文档上传
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
├── knowledge_base/           # 法律文本知识库（80+ 部法律）
├── law_faiss/                # FAISS 向量索引（自动生成）
├── bm25_index.pkl            # BM25 索引缓存（自动生成）
├── bge-small-zh-v1.5/        # 本地嵌入模型
├── uploads/                  # 用户上传文档
├── instance/
│   └── user.db               # SQLite 数据库
│
├── .env                      # 环境变量配置
├── .python-version           # Python 解释器路径
├── requirements.txt          # 依赖清单
├── start.bat                 # 一键启动脚本
└── README.md                 # 本文件
```

---

## 核心架构

```
用户提问
    │
    ▼
FastAPI /ask_stream (SSE)
    │
    ├─── 混合检索 (top_k=20)
    │       ├── FAISS 向量检索 (权重 0.6)
    │       └── BM25 关键词检索 (权重 0.4)
    │       └── 加权融合 → 40 候选
    │
    ├─── DashScope Reranker (gte-rerank)
    │       └── 精排 → Top 20 最终结果
    │
    ├─── 构建 Prompt
    │       ├── 检索上下文
    │       ├── 对话历史（最近 5 轮）
    │       └── prompts.yaml 系统提示词
    │
    ├─── MiMo LLM 流式生成
    │       └── SSE 逐 token 推送
    │
    └─── 保存对话 + 写入 SQLite
```

---

## 功能说明

| 功能 | 说明 | 权限 |
|------|------|------|
| 用户注册/登录 | 手机号 + 密码，bcrypt 加密 | 所有人 |
| 智能问答 | 基于 RAG 的法律咨询，流式输出 | 所有人 |
| 多轮对话 | 保留最近 5 轮对话记忆 | 所有人 |
| 对话管理 | 创建/编辑/删除对话 | 所有人 |
| 知识库管理 | 创建/编辑/删除知识库 | 所有人 |
| 文档上传 | PDF/DOCX/TXT，自动向量化 | 专家/管理员 |
| 混合检索 | 向量语义 + BM25 关键词 | 系统自动 |
| 重排序 | DashScope Reranker 精排 | 系统自动 |

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
检查 `.env` 中 `RERANKER_API_KEY` 和 `RERANKER_BASE_URL` 是否正确。DashScope 的 rerank API 地址是：
```
https://dashscope.aliyuncs.com/api/v1/services/reranking/reranking
```

**Q: 端口被占用**
修改 `start.bat` 中的 `PORT` 值，或关闭占用端口的进程：
```bash
netstat -ano | findstr :8080
taskkill /F /PID <进程ID>
```

**Q: 知识库索引损坏，检索结果不对**
删除 `law_faiss/` 目录和 `bm25_index.pkl`，重启服务会自动重建。
