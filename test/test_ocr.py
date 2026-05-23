"""
OCR 文档预处理测试脚本
测试 DocumentProcessor 的 OCR 回退逻辑，无需启动服务器。
运行: cd test && python test_ocr.py
"""
import sys
import os

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_needs_ocr():
    """测试 _needs_ocr 检测逻辑"""
    from langchain_core.documents import Document
    from law_assistant.processor import DocumentProcessor

    proc = DocumentProcessor()

    # 空页面 → 需要 OCR
    assert proc._needs_ocr([]) is True, "空页面应返回 True"

    # 文本充足 → 不需要 OCR
    pages_ok = [Document(page_content="这是一段足够长的文本" * 10)]
    assert proc._needs_ocr(pages_ok) is False, "文本充足应返回 False"

    # 文本不足 → 需要 OCR
    pages_empty = [Document(page_content="   ")]
    assert proc._needs_ocr(pages_empty) is True, "空白页应返回 True"

    pages_short = [Document(page_content="第1页")]
    assert proc._needs_ocr(pages_short) is True, "极短文本应返回 True"

    # 混合情况：平均不足
    pages_mixed = [
        Document(page_content="a" * 100),  # 100 chars
        Document(page_content=""),           # 0 chars
    ]
    # avg = 50, 不 < 50, 所以不需要 OCR
    assert proc._needs_ocr(pages_mixed) is False, "平均刚好50应返回 False"

    pages_low = [
        Document(page_content="a" * 60),
        Document(page_content=""),
    ]
    # avg = 30, < 50
    assert proc._needs_ocr(pages_low) is True, "平均低于50应返回 True"

    print("[PASS] test_needs_ocr")


def test_ocr_engine_lazy_load():
    """测试 PaddleOCR 引擎延迟加载"""
    from law_assistant.processor import DocumentProcessor

    proc = DocumentProcessor()
    assert proc._ocr_engine is None, "初始化时 OCR 引擎应为 None"

    try:
        engine = proc._get_ocr_engine()
        assert engine is not None, "首次调用后引擎不应为 None"
        # 再次调用应返回同一实例
        engine2 = proc._get_ocr_engine()
        assert engine is engine2, "应返回缓存的同一实例"
        print("[PASS] test_ocr_engine_lazy_load")
    except ImportError as e:
        print(f"[SKIP] test_ocr_engine_lazy_load — PaddleOCR 未安装: {e}")


def test_ocr_text_pdf():
    """测试文本 PDF 走快速路径（不触发 OCR）"""
    from law_assistant.processor import DocumentProcessor

    proc = DocumentProcessor()

    # 找一个知识库中的文本 PDF
    kb_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge_base")
    if not os.path.exists(kb_dir):
        print("[SKIP] test_ocr_text_pdf — knowledge_base 目录不存在")
        return

    txt_files = [f for f in os.listdir(kb_dir) if f.endswith('.txt')]
    if not txt_files:
        print("[SKIP] test_ocr_text_pdf — 无测试文件")
        return

    test_file = os.path.join(kb_dir, txt_files[0])
    docs = proc._load_documents(test_file)
    assert len(docs) > 0, "应加载到文档"
    assert len(docs[0].page_content.strip()) > 0, "文本不应为空"
    print(f"[PASS] test_ocr_text_pdf — {txt_files[0]}: {len(docs)} 页, 首页 {len(docs[0].page_content)} 字符")


def test_legal_detection():
    """测试法律文档检测（含 OCR 回退场景）"""
    from law_assistant.processor import DocumentProcessor

    proc = DocumentProcessor()

    kb_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge_base")
    if not os.path.exists(kb_dir):
        print("[SKIP] test_legal_detection — knowledge_base 目录不存在")
        return

    # 文件名包含"法"的应被检测为法律文档
    law_files = [f for f in os.listdir(kb_dir) if '法' in f]
    if not law_files:
        print("[SKIP] test_legal_detection — 无法律文件")
        return

    test_file = os.path.join(kb_dir, law_files[0])
    is_legal = proc.is_legal_document(test_file)
    assert is_legal is True, f"{law_files[0]} 应被检测为法律文档"
    print(f"[PASS] test_legal_detection — {law_files[0]} 正确识别为法律文档")


def test_process_document():
    """测试完整文档处理流程"""
    from law_assistant.processor import DocumentProcessor

    proc = DocumentProcessor()

    kb_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge_base")
    if not os.path.exists(kb_dir):
        print("[SKIP] test_process_document — knowledge_base 目录不存在")
        return

    txt_files = [f for f in os.listdir(kb_dir) if f.endswith('.txt')]
    if not txt_files:
        print("[SKIP] test_process_document — 无测试文件")
        return

    test_file = os.path.join(kb_dir, txt_files[0])
    result = proc.process_document(test_file)
    assert len(result) > 0, "应处理出至少一个块"
    assert 'full_text' in result[0], "结果应包含 full_text 字段"
    print(f"[PASS] test_process_document — {txt_files[0]}: 处理出 {len(result)} 个块")


def test_image_ocr():
    """测试图片 OCR（需要 PaddleOCR + 测试图片）"""
    from law_assistant.processor import DocumentProcessor

    proc = DocumentProcessor()

    # 创建一个简单的测试图片
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
    except ImportError:
        print("[SKIP] test_image_ocr — Pillow 未安装")
        return

    test_img_path = os.path.join(os.path.dirname(__file__), "test_image.png")
    img = Image.new('RGB', (400, 100), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((10, 30), "中华人民共和国法律测试", fill='black')
    img.save(test_img_path)

    try:
        docs = proc._load_image_with_ocr(test_img_path)
        assert len(docs) == 1, "应返回一个 Document"
        text = docs[0].page_content
        print(f"[PASS] test_image_ocr — 识别文本: {text[:50]}")
    except ImportError as e:
        print(f"[SKIP] test_image_ocr — PaddleOCR 未安装: {e}")
    finally:
        if os.path.exists(test_img_path):
            os.remove(test_img_path)


if __name__ == "__main__":
    print("=" * 50)
    print("OCR 文档预处理测试")
    print("=" * 50)

    test_needs_ocr()
    test_ocr_engine_lazy_load()
    test_ocr_text_pdf()
    test_legal_detection()
    test_process_document()
    test_image_ocr()

    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)
