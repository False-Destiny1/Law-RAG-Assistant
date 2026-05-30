"""
RAGAS 评估运行器

用法:
    python -m tests.eval.run_ragas                    # 运行全部用例
    python -m tests.eval.run_ragas --category retrieval_direct  # 只运行指定分类
    python -m tests.eval.run_ragas --collect-only     # 只收集数据，不计算指标
    python -m tests.eval.run_ragas --report-only      # 只从已有数据计算指标
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def main():
    parser = argparse.ArgumentParser(description="RAGAS 评估运行器")
    parser.add_argument("--category", help="只运行指定分类的用例")
    parser.add_argument("--limit", type=int, default=0, help="限制用例数量（0=全部）")
    parser.add_argument("--collect-only", action="store_true", help="只收集数据，不计算指标")
    parser.add_argument("--report-only", action="store_true", help="只从已有数据计算指标")
    parser.add_argument("--input", default=None, help="收集数据的输入文件路径")
    parser.add_argument("--output", default=None, help="评估报告输出路径")
    args = parser.parse_args()

    from tests.eval.ragas_dataset import EVAL_DATASET, get_dataset_by_category

    # 确定数据集
    if args.category:
        dataset = get_dataset_by_category(args.category)
        if not dataset:
            print(f"分类 '{args.category}' 不存在。可用分类:")
            for cat in {c["category"] for c in EVAL_DATASET}:
                print(f"  - {cat}")
            return
        print(f"使用分类: {args.category} ({len(dataset)} 个用例)")
    else:
        dataset = EVAL_DATASET
        print(f"使用全部数据集 ({len(dataset)} 个用例)")

    if args.limit > 0:
        dataset = dataset[: args.limit]
        print(f"限制为前 {args.limit} 个用例")

    # 收集阶段
    if not args.report_only:
        print("\n=== 数据收集阶段 ===")
        from tests.eval.ragas_collector import collect_dataset, init_rag, save_collected

        print("初始化 RAG 模型...")
        rag = init_rag()

        print("开始收集 RAG 输出...")
        results = collect_dataset(rag, dataset)
        save_collected(results, args.output and args.output.replace("report", "collected"))

        if args.collect_only:
            print("收集完成（--collect-only 模式，跳过指标计算）")
            return
    else:
        # 从文件加载已有数据
        input_path = args.input or str(Path(__file__).parent / "ragas_collected.json")
        print(f"从文件加载数据: {input_path}")
        with open(input_path, encoding="utf-8") as f:
            results = json.load(f)

    # 评估阶段
    print("\n=== 指标计算阶段 ===")
    from tests.eval.ragas_metrics import evaluate_all, print_report, save_report

    eval_result = evaluate_all(results)
    print_report(eval_result)
    save_report(eval_result, args.output)

    print("\n评估完成!")


if __name__ == "__main__":
    main()
