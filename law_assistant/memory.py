import threading
from collections import OrderedDict
from datetime import datetime


class ConversationMemory:
    """对话记忆管理类（L1 内存缓存 + L2 Redis + L3 DB 持久化回退，LRU 淘汰）"""

    MAX_CACHED_CONVERSATIONS = 500

    def __init__(self, max_history_turns: int = 5, db_session_factory=None, message_model=None):
        self.max_history_turns = max_history_turns
        self.conversations = OrderedDict()
        self._lock = threading.Lock()
        # 依赖注入：DB session 工厂和 Message 模型（避免从 app.py 循环导入）
        self._db_session_factory = db_session_factory
        self._message_model = message_model

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
        """将当前对话数据写入 Redis（调用前需已持有 self._lock 或数据已就绪）"""
        try:
            from law_assistant.redis_utils import cache_set_json

            conversation = self.conversations.get(conversation_id)
            if conversation:
                serializable = {
                    "history": self._serialize_history(conversation["history"]),
                    "created_at": conversation["created_at"].isoformat(),
                }
                cache_set_json(f"conv:{conversation_id}", serializable, ttl=86400)
        except Exception:
            pass

    def add_message(self, conversation_id: str, role: str, content: str):
        """添加消息到对话历史（内存缓存 + LRU 淘汰，线程安全）"""
        with self._lock:
            if conversation_id not in self.conversations:
                # LRU 淘汰：超过上限时移除最旧的对话
                while len(self.conversations) >= self.MAX_CACHED_CONVERSATIONS:
                    self.conversations.popitem(last=False)
                self.conversations[conversation_id] = {"history": [], "created_at": datetime.now()}
            else:
                # 移到末尾（最近使用）
                self.conversations.move_to_end(conversation_id)

            conversation = self.conversations[conversation_id]
            conversation["history"].append({"role": role, "content": content, "timestamp": datetime.now()})

            # 保留最近N轮对话
            max_messages = self.max_history_turns * 2
            if len(conversation["history"]) > max_messages:
                conversation["history"] = conversation["history"][-max_messages:]

            # Write-through to Redis（锁内执行，避免与 clear_conversation 的 TOCTOU 竞态）
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
                # 回填 L1（放在末尾，标记为最近使用）
                with self._lock:
                    if len(self.conversations) >= self.MAX_CACHED_CONVERSATIONS:
                        self.conversations.popitem(last=False)
                    self.conversations[conversation_id] = {"history": history, "created_at": datetime.now()}
                return history
        except Exception:
            pass

        # L3: DB 回退
        return self._load_from_db(conversation_id)

    def get_formatted_history(self, conversation_id: str) -> str:
        """获取格式化的对话历史"""
        history = self.get_recent_history(conversation_id)
        if not history:
            return "无对话历史"

        parts = ["最近的对话历史：\n"]
        for i, msg in enumerate(history):
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

            import logging

            logging.getLogger(__name__).info(f"从DB加载对话历史: {conversation_id}, {len(history)} 条消息")
            return history
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"从DB加载对话历史失败: {e}")
            return []
        finally:
            if db:
                db.close()
