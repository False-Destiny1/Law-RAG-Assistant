"""
RAG 系统基准测试脚本
用于在改进前后对比检索和生成效果。
运行前确保服务已启动: python -m uvicorn app:app --host 127.0.0.1 --port 8080
"""
import requests
import json
import time
import sys

BASE_URL = "http://127.0.0.1:8080"

# ── 测试用例定义 ──────────────────────────────────────────────────────
TEST_CASES = [
    # === 检索质量测试 ===
    {
        "id": "R1",
        "category": "检索-直接法律查询",
        "query": "劳动合同解除的条件有哪些？",
        "expect_keywords": ["第三十九条", "第四十条", "用人单位", "劳动者"],
        "notes": "标准法律术语查询，应精确匹配《劳动合同法》相关条款"
    },
    {
        "id": "R2",
        "category": "检索-口语化查询",
        "query": "老板不给加班费怎么办",
        "expect_keywords": ["加班", "工资报酬", "第四十四条"],
        "notes": "口语化表达，当前系统直接检索，应匹配《劳动法》加班工资条款"
    },
    {
        "id": "R3",
        "category": "检索-模糊查询",
        "query": "离婚财产怎么分",
        "expect_keywords": ["共同财产", "分割", "第一千零八十七条"],
        "notes": "短查询，向量检索可能召回率低"
    },
    {
        "id": "R4",
        "category": "检索-复合问题",
        "query": "离婚时财产怎么分，孩子抚养权归谁？",
        "expect_keywords": ["财产", "抚养", "子女"],
        "notes": "两个独立子问题混合，应分别检索到财产分割和抚养权条款"
    },
    {
        "id": "R5",
        "category": "检索-冷门法律",
        "query": "深海海底资源勘探有什么法律规定",
        "expect_keywords": ["深海海底", "资源勘探"],
        "notes": "冷门法律，测试小众领域检索能力"
    },
    # === 多轮对话测试 ===
    {
        "id": "M1",
        "category": "多轮-指代消解",
        "query": "那试用期呢？",
        "context": "上一轮问的是劳动合同解除条件",
        "expect_keywords": ["试用期", "解除"],
        "notes": "需要理解'那'指代的是劳动合同解除，当前系统可能检索不到"
    },
    {
        "id": "M2",
        "category": "多轮-追问深入",
        "query": "具体要赔偿多少钱？",
        "context": "上一轮问的是违法解除劳动合同的后果",
        "expect_keywords": ["赔偿", "经济补偿", "二倍"],
        "notes": "需要理解上下文才能检索到赔偿标准"
    },
    # === 边界情况测试 ===
    {
        "id": "E1",
        "category": "边界-无关问题",
        "query": "今天天气怎么样？",
        "expect_keywords": [],
        "notes": "应如实回答无法回答，不应编造法律条文"
    },
    {
        "id": "E2",
        "category": "边界-超短查询",
        "query": "借钱",
        "expect_keywords": ["借款", "借贷", "合同"],
        "notes": "极短查询，测试检索鲁棒性"
    },
    {
        "id": "E3",
        "category": "边界-跨法律查询",
        "query": "网络诈骗涉及哪些法律？",
        "expect_keywords": ["诈骗", "刑法", "电信网络诈骗"],
        "notes": "需要检索多部法律（刑法、反电信网络诈骗法等）"
    },
]


def login(session: requests.Session, phone: str = "13333333333", password: str = "123456"):
    """登录获取会话"""
    resp = session.post(f"{BASE_URL}/login", data={
        "identifier": phone,
        "password": password,
        "remember": "on"
    }, allow_redirects=False)
    return resp.status_code in (200, 303)


def create_chat(session: requests.Session) -> str:
    """创建新对话，返回 chat_id"""
    resp = session.post(f"{BASE_URL}/api/chats", json={"title": "基准测试"})
    if resp.status_code == 200:
        return resp.json().get("chat_id") or resp.json().get("id")
    return None


def ask_stream(session: requests.Session, chat_id: str, query: str, timeout: int = 60) -> dict:
    """发送流式问答请求，收集完整响应"""
    start_time = time.time()
    full_answer = ""
    chunks = []
    error = None

    try:
        resp = session.post(f"{BASE_URL}/ask_stream", data={
            "user_input": query,
            "chat_id": chat_id,
        }, stream=True, timeout=timeout)

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                data = json.loads(data_str)
                if "error" in data:
                    error = data["error"]
                    break
                content = data.get("content", "")
                if content:
                    full_answer += content
                    chunks.append(content)
            except json.JSONDecodeError:
                continue
    except Exception as e:
        error = str(e)

    elapsed = time.time() - start_time

    return {
        "answer": full_answer,
        "chunks_count": len(chunks),
        "elapsed_sec": round(elapsed, 2),
        "first_chunk_sec": None,  # TODO: 可以后续精确测量
        "error": error,
        "char_count": len(full_answer),
    }


def check_keywords(answer: str, keywords: list) -> dict:
    """检查回答中是否包含期望关键词"""
    found = []
    missing = []
    for kw in keywords:
        if kw in answer:
            found.append(kw)
        else:
            missing.append(kw)
    hit_rate = len(found) / len(keywords) if keywords else 1.0
    return {
        "found": found,
        "missing": missing,
        "hit_rate": round(hit_rate, 2)
    }


def run_evaluation():
    """运行完整基准测试"""
    print("=" * 60)
    print("RAG 系统基准测试")
    print("=" * 60)

    # 检查服务是否可用
    try:
        resp = requests.get(f"{BASE_URL}/login", timeout=5)
        if resp.status_code != 200:
            print("服务未就绪，请先启动: python -m uvicorn app:app --host 127.0.0.1 --port 8080")
            sys.exit(1)
    except Exception as e:
        print(f"无法连接服务: {e}")
        sys.exit(1)

    # 登录
    session = requests.Session()
    if not login(session):
        print("登录失败，请确认测试账号存在")
        sys.exit(1)
    print("登录成功")

    # 创建测试对话
    chat_id = create_chat(session)
    if not chat_id:
        print("创建对话失败")
        sys.exit(1)
    print(f"创建测试对话: chat_id={chat_id}")

    # 运行测试用例
    results = []
    for tc in TEST_CASES:
        print(f"\n--- [{tc['id']}] {tc['category']} ---")
        print(f"查询: {tc['query']}")

        result = ask_stream(session, chat_id, tc["query"])
        kw_check = check_keywords(result["answer"], tc["expect_keywords"])

        print(f"耗时: {result['elapsed_sec']}s | 字数: {result['char_count']}")
        print(f"关键词命中率: {kw_check['hit_rate'] * 100}% ({len(kw_check['found'])}/{len(tc['expect_keywords'])})")
        if kw_check['missing']:
            print(f"缺失关键词: {kw_check['missing']}")
        if result['error']:
            print(f"错误: {result['error']}")
        print(f"回答前100字: {result['answer'][:100]}...")

        results.append({
            "test_id": tc["id"],
            "category": tc["category"],
            "query": tc["query"],
            "answer_preview": result["answer"][:200],
            "answer_length": result["char_count"],
            "elapsed_sec": result["elapsed_sec"],
            "keyword_hit_rate": kw_check["hit_rate"],
            "keywords_found": kw_check["found"],
            "keywords_missing": kw_check["missing"],
            "error": result["error"],
            "notes": tc["notes"],
        })

    # 汇总统计
    print("\n" + "=" * 60)
    print("汇总统计")
    print("=" * 60)

    successful = [r for r in results if not r["error"]]
    failed = [r for r in results if r["error"]]

    avg_time = sum(r["elapsed_sec"] for r in successful) / len(successful) if successful else 0
    avg_hit = sum(r["keyword_hit_rate"] for r in successful) / len(successful) if successful else 0
    avg_len = sum(r["answer_length"] for r in successful) / len(successful) if successful else 0

    print(f"成功: {len(successful)}/{len(results)}")
    print(f"失败: {len(failed)}/{len(results)}")
    print(f"平均响应时间: {avg_time:.2f}s")
    print(f"平均关键词命中率: {avg_hit * 100:.1f}%")
    print(f"平均回答字数: {avg_len:.0f}")

    # 保存结果
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total": len(results),
            "success": len(successful),
            "failed": len(failed),
            "avg_response_time_sec": round(avg_time, 2),
            "avg_keyword_hit_rate": round(avg_hit, 3),
            "avg_answer_length": round(avg_len),
        },
        "results": results
    }

    output_file = "baseline_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {output_file}")

    return output


if __name__ == "__main__":
    run_evaluation()
