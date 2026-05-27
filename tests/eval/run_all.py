"""
测试运行器
运行方式:
  python tests/eval/run_all.py              — 运行所有离线测试
  python tests/eval/run_all.py --online     — 包含需要服务器的在线测试
"""

import os
import subprocess
import sys

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def run_test(script_name, online=False):
    """运行单个测试脚本"""
    script_path = os.path.join(TEST_DIR, script_name)
    if not os.path.exists(script_path):
        print(f"[SKIP] {script_name} — 文件不存在")
        return

    print(f"\n{'=' * 50}")
    print(f"运行: {script_name}")
    print(f"{'=' * 50}")

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=TEST_DIR,
        timeout=300 if online else 60,
    )
    return result.returncode == 0


def main():
    online = "--online" in sys.argv

    # 离线测试（不需要服务器）
    offline_tests = [
        "../manual/test_ocr.py",
    ]

    # 在线测试（需要服务器运行）
    online_tests = [
        "baseline_eval.py",
    ]

    passed = 0
    failed = 0

    for test in offline_tests:
        if run_test(test):
            passed += 1
        else:
            failed += 1

    if online:
        for test in online_tests:
            if run_test(test, online=True):
                passed += 1
            else:
                failed += 1
    else:
        print("\n跳过在线测试（使用 --online 参数运行）")

    print(f"\n{'=' * 50}")
    print(f"结果: {passed} 通过, {failed} 失败")
    print(f"{'=' * 50}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
