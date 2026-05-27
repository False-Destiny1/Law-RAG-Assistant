"""分批测试混合检索权重"""
import requests
import json
import time
import sys

BASE_URL = "http://127.0.0.1:8080"

TEST_CASES = [
    {"id": "R1", "query": "劳动合同解除的条件有哪些？", "kw": ["第三十九条", "第四十条", "用人单位", "劳动者"]},
    {"id": "R2", "query": "老板不给加班费怎么办", "kw": ["加班", "工资报酬", "第四十四条"]},
    {"id": "R3", "query": "离婚财产怎么分", "kw": ["共同财产", "分割", "第一千零八十七条"]},
    {"id": "R4", "query": "离婚时财产怎么分，孩子抚养权归谁？", "kw": ["财产", "抚养", "子女"]},
    {"id": "R5", "query": "深海海底资源勘探有什么法律规定", "kw": ["深海海底", "资源勘探"]},
    {"id": "M1", "query": "那试用期呢？", "kw": ["试用期", "解除"]},
    {"id": "M2", "query": "具体要赔偿多少钱？", "kw": ["赔偿", "经济补偿", "二倍"]},
    {"id": "E1", "query": "今天天气怎么样？", "kw": []},
    {"id": "E2", "query": "借钱", "kw": ["借款", "借贷", "合同"]},
    {"id": "E3", "query": "网络诈骗涉及哪些法律？", "kw": ["诈骗", "刑法", "电信网络诈骗"]},
]


def login(session):
    r = session.post(f"{BASE_URL}/login", data={"identifier": "admin", "password": "admin123", "remember": "on"}, allow_redirects=False)
    return r.status_code in (200, 303)


def create_chat(session):
    r = session.post(f"{BASE_URL}/api/chats", json={"title": "权重测试"})
    return (r.json().get("chat_id") or r.json().get("id")) if r.status_code == 200 else None


def set_weights(session, vw, bw):
    return session.post(f"{BASE_URL}/api/retrieval-weights", data={"vector_weight": vw, "bm25_weight": bw}).status_code == 200


def ask_and_score(session, chat_id, query, keywords):
    start = time.time()
    full = ""
    try:
        resp = session.post(f"{BASE_URL}/ask_stream", data={"user_input": query, "chat_id": chat_id}, stream=True, timeout=90)
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            if line[6:].strip() == "[DONE]":
                break
            try:
                d = json.loads(line[6:])
                if "error" in d:
                    return 0, time.time() - start
                full += d.get("content", "")
            except json.JSONDecodeError:
                continue
    except Exception:
        return 0, time.time() - start
    elapsed = time.time() - start
    if not keywords:
        return 1.0, elapsed
    return sum(1 for kw in keywords if kw in full) / len(keywords), round(elapsed, 2)


def run_batch(weights):
    s = requests.Session()
    if not login(s):
        print("登录失败"); sys.exit(1)

    results = []
    for vw, bw, label in weights:
        tag = f"V{vw}/B{bw}" + (f" ({label})" if label else "")
        if not set_weights(s, vw, bw):
            print(f"[{tag}] 设置权重失败"); continue
        chat = create_chat(s)
        if not chat:
            print(f"[{tag}] 创建对话失败"); continue

        hits, times = [], []
        details = []
        for tc in TEST_CASES:
            h, t = ask_and_score(s, chat, tc["query"], tc["kw"])
            hits.append(h); times.append(t)
            details.append(f"{tc['id']}:{h*100:.0f}%")

        avg_h = sum(hits) / len(hits)
        avg_t = sum(times) / len(times)
        results.append({"vw": vw, "bw": bw, "label": label, "hit": round(avg_h, 3), "time": round(avg_t, 2)})
        print(f"[{tag}]  命中={avg_h*100:.1f}%  耗时={avg_t:.1f}s  {' | '.join(details)}")

    return results


if __name__ == "__main__":
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    batches = {
        1: [(1.0, 0.0, "纯向量"), (0.9, 0.1, ""), (0.8, 0.2, ""), (0.7, 0.3, "")],
        2: [(0.6, 0.4, "当前默认"), (0.5, 0.5, "均衡"), (0.4, 0.6, ""), (0.3, 0.7, "")],
        3: [(0.2, 0.8, ""), (0.1, 0.9, "BM25为主"), (0.0, 1.0, "纯BM25")],
    }
    if batch not in batches:
        print(f"用法: python test_weights_batch.py [1|2|3]"); sys.exit(1)

    print(f"=== 第 {batch} 组权重测试 ===")
    results = run_batch(batches[batch])
    with open(f"test/weight_batch{batch}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 test/weight_batch{batch}.json")
