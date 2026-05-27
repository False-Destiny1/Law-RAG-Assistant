"""Unit tests for law_assistant.memory module."""

from law_assistant.memory import ConversationMemory


class TestConversationMemory:
    def test_add_and_retrieve_message(self):
        mem = ConversationMemory(max_history_turns=5)
        mem.add_message("chat_1", "user", "你好")
        mem.add_message("chat_1", "assistant", "你好！有什么可以帮您的？")
        history = mem.get_recent_history("chat_1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_lru_eviction(self):
        mem = ConversationMemory(max_history_turns=5)
        mem.MAX_CACHED_CONVERSATIONS = 3
        mem.add_message("chat_1", "user", "msg1")
        mem.add_message("chat_2", "user", "msg2")
        mem.add_message("chat_3", "user", "msg3")
        mem.add_message("chat_4", "user", "msg4")  # should evict chat_1
        assert "chat_1" not in mem.conversations
        assert "chat_4" in mem.conversations

    def test_max_history_turns(self):
        mem = ConversationMemory(max_history_turns=2)
        for i in range(10):
            mem.add_message("chat_1", "user", f"msg{i}")
        history = mem.get_recent_history("chat_1")
        # max_history_turns=2 means max 4 messages (2 user + 2 assistant)
        assert len(history) <= 4

    def test_clear_conversation(self):
        mem = ConversationMemory(max_history_turns=5)
        mem.add_message("chat_1", "user", "hello")
        mem.clear_conversation("chat_1")
        assert "chat_1" not in mem.conversations

    def test_get_formatted_history_empty(self):
        mem = ConversationMemory(max_history_turns=5)
        result = mem.get_formatted_history("chat_999")
        assert result == "无对话历史"

    def test_get_formatted_history_with_messages(self):
        mem = ConversationMemory(max_history_turns=5)
        mem.add_message("chat_1", "user", "什么是合同？")
        mem.add_message("chat_1", "assistant", "合同是当事人之间设立、变更、终止民事法律关系的协议。")
        result = mem.get_formatted_history("chat_1")
        assert "用户" in result
        assert "助手" in result
        assert "什么是合同" in result

    def test_multiple_conversations_isolated(self):
        mem = ConversationMemory(max_history_turns=5)
        mem.add_message("chat_1", "user", "msg from chat 1")
        mem.add_message("chat_2", "user", "msg from chat 2")
        h1 = mem.get_recent_history("chat_1")
        h2 = mem.get_recent_history("chat_2")
        assert len(h1) == 1
        assert len(h2) == 1
        assert h1[0]["content"] == "msg from chat 1"
        assert h2[0]["content"] == "msg from chat 2"
