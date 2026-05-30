import logging
import threading
from collections import OrderedDict
from datetime import datetime

logger = logging.getLogger(__name__)

# ── Token 预算配置 ──────────────────────────────────────────────────
HISTORY_TOKEN_BUDGET = 2500   # 对话历史总 token 预算（不含检索上下文）
SUMMARY_TOKEN_BUDGET = 500    # 摘要部分 token 预算
RECENT_TOKEN_BUDGET = 2000    # 滑动窗口（近期消息）token 预算
SUMMARY_MAX_CHARS = 600       # 摘要最大字符数


def _estimate_tokens(text: str) -> int:
    """粗略估算中文文本的 token 数（约 1.5 字符/token）"""
    return max(1, len(text) * 2 // 3)


class ConversationMemory:
    """对话记忆管理类（L1 内存缓存 + L2 Redis + L3 DB 持久化回退，LRU 淘汰 + 自动摘要压缩）"""

    MAX_CACHED_CONVERSATIONS = 500

    def __init__(self, max_history_turns: int = 5, db_session_factory=None, message_model=None):
        self.max_history_turns = max_history_turns
        self.conversations = OrderedDict()
        self._lock = threading.Lock()
        # 依赖注入：DB session 工厂和 Message 模型（避免从 app.py 循环导入）
        self._db_session_factory = db_session_factory
        self._message_model = message_model
        # 摘要生成器（由 rag.py 注入，签名: (messages: list[dict]) -> str）
        self._summarizer = None

    def set_summarizer(self, summarizer):
        """注入摘要生成器（签名: (messages: list[dict]) -> str）"""
        self._summarizer = summarizer

    def _serialize_history(self, history: list) -> list:
        """将历史记录序列化为 JSON 兼容格式"""
        result = []
        for msg in history:
            item = dict(msg)
            ts = item.get("timestamp")
            if isinstance(ts, datetime):
                item["timestamp"] = ts.isoformat()
            elif ts is not None:
                item["timestamp"] = str(ts)
            result.append(item)
        return result

    def _deserialize_history(self, history: list) -> list:
        """将 JSON 格式的历史记录还原为 Python 对象"""
        result = []
        for msg in history:
            item = dict(msg)
            ts = item.get("timestamp")
            if isinstance(ts, str):
                try:
                    item["timestamp"] = datetime.fromisoformat(ts)
                except (ValueError, TypeError):
                    item["timestamp"] = datetime.now()
            result.append(item)
        return result

    def _write_to_redis(self, conversation_id: str):
        """将当前对话数据写入 Redis（在锁外调用，依赖 GIL 保证 dict 读取一致性）"""
        try:
            from law_assistant.redis_utils import cache_set_json

            conversation = self.conversations.get(conversation_id)
            if conversation:
                serializable = {
                    "history": self._serialize_history(conversation["history"]),
                    "summary": conversation.get("summary", ""),
                    "created_at": conversation["created_at"].isoformat(),
                }
                cache_set_json(f"conv:{conversation_id}", serializable, ttl=86400)
        except Exception:
            pass

    def add_message(self, conversation_id: str, role: str, content: str):
        """添加消息到对话历史（内存缓存 + LRU 淘汰 + 自动摘要，线程安全）"""
        # Phase 1: 锁内执行内存操作
        need_summarize = False
        split_at = 0
        with self._lock:
            if conversation_id not in self.conversations:
                # LRU 淘汰：超过上限时移除最旧的对话
                while len(self.conversations) >= self.MAX_CACHED_CONVERSATIONS:
                    self.conversations.popitem(last=False)
                self.conversations[conversation_id] = {"history": [], "summary": "", "created_at": datetime.now()}
            else:
                # 移到末尾（最近使用）
                self.conversations.move_to_end(conversation_id)

            conversation = self.conversations[conversation_id]
            conversation["history"].append({"role": role, "content": content, "timestamp": datetime.now()})

            # 检查是否需要摘要压缩（基于 token 预算，不在锁内执行 LLM 调用）
            total_tokens = sum(_estimate_tokens(m["content"]) for m in conversation["history"])
            if total_tokens > RECENT_TOKEN_BUDGET and self._summarizer and len(conversation["history"]) > 4:
                split_at = len(conversation["history"]) // 2
                if split_at % 2 != 0:
                    split_at += 1
                # 只快照旧消息，不裁剪 history（裁剪推迟到摘要成功后）
                need_summarize = True

            # 兜底：保留最近N轮对话（防止极端情况）
            max_messages = self.max_history_turns * 2
            if len(conversation["history"]) > max_messages:
                conversation["history"] = conversation["history"][-max_messages:]

        # Phase 2: 锁外执行网络/LLM 操作
        if need_summarize:
            # 锁外快照旧消息（conversation 仍在内存中，GIL 保证引用有效）
            old_messages = list(self.conversations.get(conversation_id, {}).get("history", [])[:split_at])
            if not old_messages:
                self._write_to_redis(conversation_id)
                return
            try:
                new_summary = self._summarizer(old_messages)
                # 锁内写回摘要 + 裁剪 history（重新读取 existing_summary 避免并发覆盖）
                with self._lock:
                    if conversation_id in self.conversations:
                        existing = self.conversations[conversation_id].get("summary", "")
                        combined = (existing + "\n" + new_summary) if existing else new_summary
                        if len(combined) > SUMMARY_MAX_CHARS:
                            combined = combined[-SUMMARY_MAX_CHARS:]
                        self.conversations[conversation_id]["summary"] = combined
                        # 摘要成功，才裁剪旧消息
                        self.conversations[conversation_id]["history"] = \
                            self.conversations[conversation_id]["history"][split_at:]
                logger.info(f"对话 {conversation_id} 摘要压缩: {len(old_messages)} 条消息 → 摘要")
            except Exception as e:
                logger.warning(f"对话摘要失败，保留完整历史: {e}")

        self._write_to_redis(conversation_id)

    def get_recent_history(self, conversation_id: str) -> list[dict]:
        """获取最近的对话历史（优先内存 → Redis → DB）"""
        # L1: 内存缓存
        with self._lock:
            if conversation_id in self.conversations:
                return list(self.conversations[conversation_id]["history"])

        # L2: Redis
        try:
            from law_assistant.redis_utils import cache_get_json

            cached = cache_get_json(f"conv:{conversation_id}")
            if cached and "history" in cached:
                history = self._deserialize_history(cached["history"])
                summary = cached.get("summary", "")
                # 回填 L1（放在末尾，标记为最近使用）
                with self._lock:
                    if len(self.conversations) >= self.MAX_CACHED_CONVERSATIONS:
                        self.conversations.popitem(last=False)
                    self.conversations[conversation_id] = {
                        "history": history, "summary": summary, "created_at": datetime.now()
                    }
                return history
        except Exception:
            pass

        # L3: DB 回退
        return self._load_from_db(conversation_id)

    def get_formatted_history(self, conversation_id: str) -> str:
        """获取格式化的对话历史（token-aware 滑动窗口 + 摘要）"""
        # 先确保数据已加载到 L1（含 summary）
        self.get_recent_history(conversation_id)

        with self._lock:
            conversation = self.conversations.get(conversation_id)
            if not conversation:
                return "无对话历史"
            summary = conversation.get("summary", "")
            history = list(conversation["history"])

        if not history and not summary:
            return "无对话历史"

        parts = []
        used_tokens = 0

        # Layer 1: 摘要（固定预算）
        if summary:
            if _estimate_tokens(summary) > SUMMARY_TOKEN_BUDGET:
                summary = summary[:int(SUMMARY_TOKEN_BUDGET * 1.5)]
            parts.append(f"【早期对话摘要】\n{summary}\n")
            used_tokens += _estimate_tokens(summary)
            parts.append("【近期对话】\n")
        else:
            parts.append("最近的对话历史：\n")

        # Layer 2: 滑动窗口（从最新消息向前，token 预算内）
        recent_budget = min(RECENT_TOKEN_BUDGET, HISTORY_TOKEN_BUDGET - used_tokens)
        selected = []
        msg_tokens = 0
        for msg in reversed(history):
            line = f"{'用户' if msg['role'] == 'user' else '助手'}: {msg['content']}"
            line_tokens = _estimate_tokens(line)
            if msg_tokens + line_tokens > recent_budget and selected:
                break
            selected.append(msg)
            msg_tokens += line_tokens
        selected.reverse()

        for i, msg in enumerate(selected):
            speaker = "用户" if msg["role"] == "user" else "助手"
            parts.append(f"{i + 1}. {speaker}: {msg['content']}")

        return "\n".join(parts) + "\n"

    def clear_conversation(self, conversation_id: str):
        """清空特定对话的记忆"""
        with self._lock:
            self.conversations.pop(conversation_id, None)
        # 同步清除 Redis
        try:
            from law_assistant.redis_utils import cache_delete

            cache_delete(f"conv:{conversation_id}")
        except Exception:
            pass

    def _load_from_db(self, conversation_id: str) -> list[dict]:
        """从数据库加载对话历史（回退方案，使用注入的 DB 工厂）"""
        if not self._db_session_factory or not self._message_model:
            return []
        db = None
        try:
            chat_id = int(conversation_id.removeprefix("chat_"))
            db = self._db_session_factory()
            Message = self._message_model
            messages = (
                db.query(Message)
                .filter(Message.chat_id == chat_id)
                .order_by(Message.created_at.desc())
                .limit(self.max_history_turns * 2)
                .all()
            )

            if not messages:
                return []

            messages = list(reversed(messages))
            history = [{"role": msg.role, "content": msg.content, "timestamp": msg.created_at} for msg in messages]

            with self._lock:
                self.conversations[conversation_id] = {"history": history, "created_at": datetime.now()}
                self._write_to_redis(conversation_id)

            logger.info(f"从DB加载对话历史: {conversation_id}, {len(history)} 条消息")
            return history
        except Exception as e:
            logger.warning(f"从DB加载对话历史失败: {e}")
            return []
        finally:
            if db:
                db.close()
