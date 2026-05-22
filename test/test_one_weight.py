"""测试单组权重，用法: python test_one_weight.py <vector_weight> <bm25_weight>"""
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

vw, bw = float(sys.argv[1]), float(sys.argv[2])

s = requests.Session()
r = s.post(f"{BASE_URL}/login", data={"identifier": "admin", "password": "admin123", "remember": "on"}, allow_redirects=False)
assert r.status_code in (200, 303), "登录失败"

r = s.post(f"{BASE_URL}/api/retrieval-weights", data={"vector_weight": vw, "bm25_weight": bw})
assert r.status_code == 200, f"设置权重失败: {r.text}"

r = s.post(f"{BASE_URL}/api/chats", json={"title": f"V{vw}/B{bw}"})
chat = r.json().get("id") or r.json().get("chat_id")

hits, times = [], []
for tc in TEST_CASES:
    start = time.time()
    full = ""
    try:
        resp = s.post(f"{BASE_URL}/ask_stream", data={"user_input": tc["query"], "chat_id": chat}, stream=True, timeout=90)
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            if line[6:].strip() == "[DONE]":
                break
            try:
                d = json.loads(line[6:])
                if "error" in d:
                    break
                full += d.get("content", "")
            except json.JSONDecodeError:
                continue
    except Exception:
        pass
    elapsed = round(time.time() - start, 2)
    kw = tc["kw"]
    hit = (sum(1 for k in kw if k in full) / len(kw)) if kw else 1.0
    hits.append(hit); times.append(elapsed)
    print(f"  {tc['id']}: {hit*100:.0f}%  {elapsed}s")

avg_h = sum(hits) / len(hits)
avg_t = sum(times) / len(times)
print(f"\nV{vw}/B{bw}  平均命中={avg_h*100:.1f}%  平均耗时={avg_t:.1f}s")
