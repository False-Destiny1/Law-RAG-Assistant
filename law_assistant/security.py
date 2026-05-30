"""Prompt injection defense utilities."""

import re

# ── Legal context detection ──────────────────────────────────────────
# Multi-character legal keywords only (single-char like "法" "条" are too
# easily spoofed). Require at least 2 matches to activate legal-context
# exemption from low-confidence injection patterns.
_LEGAL_KEYWORDS = [
    # Civil / contract
    "合同", "约定", "规定", "条例", "法律", "法规", "司法解释",
    "判决", "裁定", "诉讼", "仲裁", "侵权", "赔偿",
    "债权", "物权", "违约", "免责", "解除",
    # Family / succession
    "婚姻", "继承", "离婚", "抚养", "赡养",
    # Labor
    "劳动", "工资", "加班", "辞退", "工伤", "社保",
    # Criminal
    "刑法", "犯罪", "拘留", "罚款", "量刑", "缓刑", "自首",
    # Administrative
    "行政", "违法", "处罚", "复议", "许可",
    # Procedural
    "起诉", "上诉", "申诉", "执行", "管辖", "举证", "调解",
    # IP
    "专利", "商标", "著作权", "知识产权",
    # Newer laws
    "个人信息", "网络安全", "数据安全",
]


def _is_legal_context(text: str) -> bool:
    """Check whether text likely discusses legal topics.

    Requires at least 2 multi-character legal keywords to reduce bypass risk.
    Single-character keywords are excluded to prevent trivial spoofing.
    """
    matches = sum(1 for kw in _LEGAL_KEYWORDS if kw in text)
    return matches >= 2


# ── Injection patterns ───────────────────────────────────────────────

# High-confidence patterns (always checked, even in legal context)
_HIGH_CONFIDENCE_PATTERNS = [
    # Role/persona hijacking
    r"(你现在|从现在起|从此刻起).{0,20}(是|扮演|角色)",
    r"(你不再|不要|停止).{0,20}(是|扮演|作为)",
    r"you are now.{0,20}(a |an |the )",
    r"act as.{0,20}(a |an |the )",
    r"pretend.{0,20}(you are|to be)",
    r"DAN\s*mode",
    # System prompt extraction
    r"(显示|输出|打印|告诉我|泄露).{0,30}(系统|初始|原始).{0,20}(提示|prompt|指令)",
    r"(system|initial|original)\s*prompt",
    r"show\s*me.{0,20}prompt",
    r"reveal.{0,20}(system|instruction)",
    # Template/variable injection
    r"\{system_prompt\}",
    r"\{instructions?\}",
    # Encoding tricks
    r"(base64|rot13|hex).{0,20}(decode|解码|加密)",
    r"(在|用).{0,10}(编码|加密|base64).{0,10}(回答|输出)",
]

# Low-confidence patterns (skipped when legal context detected)
_LOW_CONFIDENCE_PATTERNS = [
    r"忽略.{0,20}(上面|之前|以上|系统|所有).{0,20}(指令|提示|规则|要求|设定)",
    r"ignore.{0,20}(above|previous|system|all).{0,20}(instruction|prompt|rule)",
    r"disregard.{0,20}(previous|above|system)",
    r"forget.{0,20}(above|previous|all).{0,20}(instruction|rule)",
]

_COMPILED_HIGH = [re.compile(p, re.IGNORECASE) for p in _HIGH_CONFIDENCE_PATTERNS]
_COMPILED_LOW = [re.compile(p, re.IGNORECASE) for p in _LOW_CONFIDENCE_PATTERNS]
_COMPILED_ALL = _COMPILED_HIGH + _COMPILED_LOW

# Max query length (prevent token flooding)
MAX_QUERY_LENGTH = 2000


def check_injection(text: str) -> tuple[bool, str]:
    """Check text for prompt injection patterns.

    Returns (is_safe, reason). If not safe, reason describes the detected pattern.
    Legal-context text is exempt from low-confidence patterns to reduce false positives.
    """
    if not text:
        return True, ""

    if len(text) > MAX_QUERY_LENGTH:
        return False, f"输入过长（{len(text)}字符，上限{MAX_QUERY_LENGTH}）"

    is_legal = _is_legal_context(text)
    patterns = _COMPILED_ALL if not is_legal else _COMPILED_HIGH

    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return False, f"检测到疑似提示词注入: '{match.group()}'"

    return True, ""


def sanitize_context(text: str) -> str:
    """Strip potential injection attempts from retrieved document context.

    This is a lightweight filter -- wraps suspicious segments in markers
    so the LLM treats them as document content, not instructions.
    Legal-context lines are exempt from low-confidence patterns.
    """
    if not text:
        return text

    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        is_legal = _is_legal_context(stripped)
        patterns = _COMPILED_ALL if not is_legal else _COMPILED_HIGH
        if any(p.search(stripped) for p in patterns):
            cleaned.append("[文档内容，已过滤可疑指令]")
        else:
            cleaned.append(line)
    return "\n".join(cleaned)
