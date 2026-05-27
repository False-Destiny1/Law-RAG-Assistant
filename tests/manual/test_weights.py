"""
混合检索权重自动测试
测试不同 vector/BM25 权重组合，找到最优配置。
运行前确保服务已启动: python -m uvicorn app:app --host 127.0.0.1 --port 8080
"""

import json
import sys
import time

import requests

BASE_URL = "http://127.0.0.1:8080"

# ── 测试用例（与 baseline_eval.py 一致）──
TEST_CASES = [
    {
        "id": "R1",
        "query": "劳动合同解除的条件有哪些？",
        "expect_keywords": ["第三十九条", "第四十条", "用人单位", "劳动者"],
    },
    {"id": "R2", "query": "老板不给加班费怎么办", "expect_keywords": ["加班", "工资报酬", "第四十四条"]},
    {"id": "R3", "query": "离婚财产怎么分", "expect_keywords": ["共同财产", "分割", "第一千零八十七条"]},
    {"id": "R4", "query": "离婚时财产怎么分，孩子抚养权归谁？", "expect_keywords": ["财产", "抚养", "子女"]},
    {"id": "R5", "query": "深海海底资源勘探有什么法律规定", "expect_keywords": ["深海海底", "资源勘探"]},
    {"id": "M1", "query": "那试用期呢？", "expect_keywords": ["试用期", "解除"]},
    {"id": "M2", "query": "具体要赔偿多少钱？", "expect_keywords": ["赔偿", "经济补偿", "二倍"]},
    {"id": "E1", "query": "今天天气怎么样？", "expect_keywords": []},
    {"id": "E2", "query": "借钱", "expect_keywords": ["借款", "借贷", "合同"]},
    {"id": "E3", "query": "网络诈骗涉及哪些法律？", "expect_keywords": ["诈骗", "刑法", "电信网络诈骗"]},
]

# ── 权重组合 ──
WEIGHT_COMBOS = [
    (1.0, 0.0, "纯向量"),
    (0.9, 0.1, "向量为主"),
    (0.8, 0.2, ""),
    (0.7, 0.3, ""),
    (0.6, 0.4, "当前默认"),
    (0.5, 0.5, "均衡"),
    (0.4, 0.6, ""),
    (0.3, 0.7, ""),
    (0.2, 0.8, ""),
    (0.1, 0.9, "BM25为主"),
    (0.0, 1.0, "纯BM25"),
]


def login(session):
    resp = session.post(
        f"{BASE_URL}/login",
        data={"identifier": "admin", "password": "admin123", "remember": "on"},
        allow_redirects=False,
    )
    return resp.status_code in (200, 303)


def create_chat(session):
    resp = session.post(f"{BASE_URL}/api/chats", json={"title": "权重测试"})
    if resp.status_code == 200:
        return resp.json().get("chat_id") or resp.json().get("id")
    return None


def set_weights(session, vw, bw):
    resp = session.post(f"{BASE_URL}/api/retrieval-weights", data={"vector_weight": vw, "bm25_weight": bw})
    return resp.status_code == 200


def ask_and_score(session, chat_id, query, keywords):
    """发送查询并返回关键词命中率"""
    start = time.time()
    full_answer = ""
    try:
        resp = session.post(
            f"{BASE_URL}/ask_stream",
            data={
                "user_input": query,
                "chat_id": chat_id,
            },
            stream=True,
            timeout=90,
        )
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                data = json.loads(data_str)
                if "error" in data:
                    return {"hit_rate": 0, "elapsed": time.time() - start, "error": data["error"]}
                full_answer += data.get("content", "")
            except json.JSONDecodeError:
                continue
    except Exception as e:
        return {"hit_rate": 0, "elapsed": time.time() - start, "error": str(e)}

    elapsed = time.time() - start
    if not keywords:
        hit_rate = 1.0
    else:
        found = sum(1 for kw in keywords if kw in full_answer)
        hit_rate = found / len(keywords)
    return {"hit_rate": hit_rate, "elapsed": round(elapsed, 2), "answer_len": len(full_answer)}


def main():
    print("=" * 70)
    print("混合检索权重自动测试")
    print("=" * 70)

    # 连接检查
    try:
        requests.get(f"{BASE_URL}/login", timeout=5)
    except Exception:
        print("服务未启动，请先运行: python -m uvicorn app:app --host 127.0.0.1 --port 8080")
        sys.exit(1)

    session = requests.Session()
    if not login(session):
        print("登录失败")
        sys.exit(1)
    print("登录成功\n")

    all_results = []

    for vw, bw, label in WEIGHT_COMBOS:
        tag = f"V{vw:.1f}/B{bw:.1f}"
        if label:
            tag += f" ({label})"

        # 设置权重
        if not set_weights(session, vw, bw):
            print(f"[{tag}] 设置权重失败，跳过")
            continue

        # 每轮新建对话（避免多轮记忆干扰）
        chat_id = create_chat(session)
        if not chat_id:
            print(f"[{tag}] 创建对话失败，跳过")
            continue

        hit_rates = []
        times = []
        details = []

        for tc in TEST_CASES:
            result = ask_and_score(session, chat_id, tc["query"], tc["expect_keywords"])
            hit_rates.append(result["hit_rate"])
            times.append(result["elapsed"])
            details.append({"id": tc["id"], "hit": result["hit_rate"], "time": result["elapsed"]})

        avg_hit = sum(hit_rates) / len(hit_rates)
        avg_time = sum(times) / len(times)

        all_results.append(
            {
                "vector_weight": vw,
                "bm25_weight": bw,
                "label": label,
                "avg_hit_rate": round(avg_hit, 3),
                "avg_time": round(avg_time, 2),
                "details": details,
            }
        )

        # 打印每个用例的命中情况
        detail_str = " | ".join([f"{d['id']}:{d['hit'] * 100:.0f}%" for d in details])
        print(f"[{tag}]  命中率={avg_hit * 100:.1f}%  耗时={avg_time:.1f}s  {detail_str}")

    # 汇总排名
    print("\n" + "=" * 70)
    print("排名（按平均关键词命中率）")
    print("=" * 70)
    all_results.sort(key=lambda x: (-x["avg_hit_rate"], x["avg_time"]))
    print(f"{'排名':<4} {'权重(V/B)':<14} {'平均命中率':<10} {'平均耗时':<10} {'标签'}")
    print("-" * 60)
    for i, r in enumerate(all_results, 1):
        tag = f"{r['vector_weight']:.1f}/{r['bm25_weight']:.1f}"
        print(f"{i:<4} {tag:<14} {r['avg_hit_rate'] * 100:.1f}%{'':<6} {r['avg_time']:.1f}s{'':<6} {r['label']}")

    # 最优结果
    best = all_results[0]
    print(
        f"\n最优权重: vector={best['vector_weight']}, bm25={best['bm25_weight']}  "
        f"(命中率 {best['avg_hit_rate'] * 100:.1f}%, 耗时 {best['avg_time']:.1f}s)"
    )

    # 保存完整结果
    output_file = "tests/manual/weight_test_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到 {output_file}")


if __name__ == "__main__":
    main()
