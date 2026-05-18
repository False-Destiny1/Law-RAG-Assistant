from datetime import datetime
from typing import List
from collections import OrderedDict


class ConversationMemory:
    """对话记忆管理类（支持内存缓存 + DB 持久化回退，LRU 淘汰）"""

    MAX_CACHED_CONVERSATIONS = 500

    def __init__(self, max_history_turns: int = 5):
        self.max_history_turns = max_history_turns
        self.conversations = OrderedDict()

    def add_message(self, conversation_id: str, role: str, content: str):
        """添加消息到对话历史（内存缓存 + LRU 淘汰）"""
        if conversation_id not in self.conversations:
            # LRU 淘汰：超过上限时移除最旧的对话
            while len(self.conversations) >= self.MAX_CACHED_CONVERSATIONS:
                self.conversations.popitem(last=False)
            self.conversations[conversation_id] = {
                'history': [],
                'created_at': datetime.now()
            }
        else:
            # 移到末尾（最近使用）
            self.conversations.move_to_end(conversation_id)

        conversation = self.conversations[conversation_id]
        conversation['history'].append({
            'role': role,
            'content': content,
            'timestamp': datetime.now()
        })

        # 保留最近N轮对话
        max_messages = self.max_history_turns * 2
        if len(conversation['history']) > max_messages:
            conversation['history'] = conversation['history'][-max_messages:]

    def get_recent_history(self, conversation_id: str) -> List[dict]:
        """获取最近的对话历史（优先内存，回退到DB）"""
        if conversation_id in self.conversations:
            return self.conversations[conversation_id]['history']
        # 内存没有，尝试从DB加载
        return self._load_from_db(conversation_id)

    def get_formatted_history(self, conversation_id: str) -> str:
        """获取格式化的对话历史"""
        history = self.get_recent_history(conversation_id)
        if not history:
            return "无对话历史"

        formatted = "最近的对话历史：\n"
        for i, msg in enumerate(history):
            speaker = "用户" if msg['role'] == 'user' else "助手"
            formatted += f"{i + 1}. {speaker}: {msg['content']}\n"

        return formatted

    def clear_conversation(self, conversation_id: str):
        """清空特定对话的记忆"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]

    def _load_from_db(self, conversation_id: str) -> List[dict]:
        """从数据库加载对话历史（回退方案）"""
        db = None
        try:
            # conversation_id 格式: "chat_{id}"
            chat_id = int(conversation_id.replace("chat_", ""))
            from app import get_db, Message
            db = next(get_db())
            messages = db.query(Message).filter(
                Message.chat_id == chat_id
            ).order_by(Message.created_at.desc()).limit(self.max_history_turns * 2).all()

            if not messages:
                return []

            # 反转为时间正序
            messages = list(reversed(messages))
            history = [{'role': msg.role, 'content': msg.content, 'timestamp': msg.created_at} for msg in messages]

            # 缓存到内存
            self.conversations[conversation_id] = {
                'history': history,
                'created_at': datetime.now()
            }
            print(f"从DB加载对话历史: {conversation_id}, {len(history)} 条消息")
            return history
        except Exception as e:
            print(f"从DB加载对话历史失败: {e}")
            return []
        finally:
            if db:
                db.close()
