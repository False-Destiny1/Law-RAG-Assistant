"""Prompt injection defense utilities."""
import re

# Common injection patterns (Chinese + English)
_INJECTION_PATTERNS = [
    # Direct instruction override attempts
    r'忽略.{0,20}(上面|之前|以上|系统|所有).{0,20}(指令|提示|规则|要求|设定)',
    r'ignore.{0,20}(above|previous|system|all).{0,20}(instruction|prompt|rule)',
    r'disregard.{0,20}(previous|above|system)',
    r'forget.{0,20}(above|previous|all).{0,20}(instruction|rule)',
    # Role/persona hijacking
    r'(你现在|从现在起|从此刻起).{0,20}(是|扮演|角色)',
    r'(你不再|不要|停止).{0,20}(是|扮演|作为)',
    r'you are now.{0,20}(a |an |the )',
    r'act as.{0,20}(a |an |the )',
    r'pretend.{0,20}(you are|to be)',
    r'DAN\s*mode',
    # System prompt extraction
    r'(显示|输出|打印|告诉我|泄露).{0,30}(系统|初始|原始).{0,20}(提示|prompt|指令)',
    r'(system|initial|original)\s*prompt',
    r'show\s*me.{0,20}prompt',
    r'reveal.{0,20}(system|instruction)',
    # Template/variable injection
    r'\{system_prompt\}',
    r'\{instructions?\}',
    # Encoding tricks
    r'(base64|rot13|hex).{0,20}(decode|解码|加密)',
    r'(在|用).{0,10}(编码|加密|base64).{0,10}(回答|输出)',
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

# Max query length (prevent token flooding)
MAX_QUERY_LENGTH = 2000


def check_injection(text: str) -> tuple[bool, str]:
    """Check text for prompt injection patterns.

    Returns (is_safe, reason). If not safe, reason describes the detected pattern.
    """
    if not text:
        return True, ""

    if len(text) > MAX_QUERY_LENGTH:
        return False, f"输入过长（{len(text)}字符，上限{MAX_QUERY_LENGTH}）"

    for pattern in _COMPILED:
        match = pattern.search(text)
        if match:
            return False, f"检测到疑似提示词注入: '{match.group()}'"

    return True, ""


def sanitize_context(text: str) -> str:
    """Strip potential injection attempts from retrieved document context.

    This is a lightweight filter — wraps suspicious segments in markers
    so the LLM treats them as document content, not instructions.
    """
    if not text:
        return text

    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that look like system instructions injected into documents
        if any(p.search(stripped) for p in _COMPILED):
            cleaned.append('[文档内容，已过滤可疑指令]')
        else:
            cleaned.append(line)
    return '\n'.join(cleaned)
