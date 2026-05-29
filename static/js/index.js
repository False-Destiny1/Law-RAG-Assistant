function getCsrfToken() {
    const match = document.cookie.match(/csrf_token=([^;]+)/);
    return match ? match[1] : '';
}

document.addEventListener('DOMContentLoaded', function() {
    // DOM
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const chatBox = document.getElementById('chatBox');
    const newChatBtn = document.getElementById('newChatBtn');
    const chatList = document.getElementById('chatList');
    const kbSelector = document.getElementById('kbSelector');
    const chatTitle = document.getElementById('chatTitle');
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebarToggle');
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');

    // State
    let currentChatId = null;
    let chats = [];
    let knowledgeBases = [];
    let isStreaming = false;

    // ── Init ──
    function init() {
        fetchChats();
        fetchKnowledgeBases();
        setupAutoResize();
    }

    // ── Auto-resize textarea ──
    function setupAutoResize() {
        userInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });
    }

    // ── Sidebar toggle ──
    if (mobileMenuBtn) {
        mobileMenuBtn.addEventListener('click', function() {
            sidebar.classList.toggle('open');
        });
    }

    // ── Knowledge bases ──
    function fetchKnowledgeBases() {
        fetch('/api/knowledge-bases')
            .then(r => {
                if (!r.ok) throw new Error(`HTTP ${r.status}`);
                return r.json();
            })
            .then(data => {
                knowledgeBases = data;
                renderKBSelector();
            })
            .catch(e => console.error('获取知识库失败:', e));
    }

    function renderKBSelector() {
        kbSelector.innerHTML = '<option value="">全部知识库</option>';
        knowledgeBases.forEach(kb => {
            const opt = document.createElement('option');
            opt.value = kb.id;
            opt.textContent = kb.name;
            kbSelector.appendChild(opt);
        });
    }

    // ── Chats ──
    function fetchChats() {
        fetch('/api/chats')
            .then(r => {
                if (!r.ok) throw new Error('获取对话失败');
                return r.json();
            })
            .then(data => {
                chats = data;
                renderChatList();
                if (chats.length > 0) {
                    loadChat(chats[0].id);
                } else {
                    createNewChat();
                }
            })
            .catch(e => {
                console.error('获取对话失败:', e);
                createNewChat();
            });
    }

    function createNewChat() {
        return fetch('/api/chats', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': getCsrfToken() },
            body: JSON.stringify({ title: '新对话' })
        })
        .then(r => r.json())
        .then(chat => {
            currentChatId = chat.id;
            chats.unshift(chat);
            renderChatList();
            clearChatBox();
            addWelcomeMessage();
            updateTopbarTitle('新对话');
            return chat.id;
        })
        .catch(e => console.error('创建对话失败:', e));
    }

    function loadChat(chatId) {
        currentChatId = chatId;
        fetch(`/api/chats/${chatId}`)
            .then(r => r.json())
            .then(data => {
                clearChatBox();
                data.messages.forEach(msg => addMessage(msg.role === 'user' ? 'user' : 'bot', msg.content, false));
                updateTopbarTitle(data.title || '新对话');
                if (data.knowledge_base_id) {
                    kbSelector.value = data.knowledge_base_id;
                }
                scrollToBottom();
            })
            .catch(e => console.error('获取消息失败:', e));
        renderChatList();
    }

    function deleteChat(chatId) {
        fetch(`/api/chats/${chatId}`, { method: 'DELETE', headers: { 'X-CSRF-Token': getCsrfToken() } })
            .then(r => r.json())
            .then(() => {
                chats = chats.filter(c => c.id !== chatId);
                renderChatList();
                if (currentChatId === chatId) {
                    if (chats.length > 0) loadChat(chats[0].id);
                    else createNewChat();
                }
            })
            .catch(e => console.error('删除失败:', e));
    }

    function updateChatTitle(chatId, title) {
        fetch(`/api/chats/${chatId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': getCsrfToken() },
            body: JSON.stringify({ title })
        })
        .then(r => r.json())
        .then(updated => {
            const idx = chats.findIndex(c => c.id === chatId);
            if (idx !== -1) chats[idx] = updated;
            renderChatList();
            if (chatId === currentChatId) updateTopbarTitle(updated.title);
        })
        .catch(e => console.error('更新标题失败:', e));
    }

    function updateTopbarTitle(title) {
        chatTitle.textContent = title || '新对话';
    }

    // ── Render chat list ──
    function renderChatList() {
        chatList.innerHTML = '';
        const sorted = [...chats].sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at));

        sorted.forEach(chat => {
            const item = document.createElement('div');
            item.className = 'chat-item' + (chat.id === currentChatId ? ' active' : '');
            item.innerHTML = `
                <span class="chat-item-title" title="${escapeHtml(chat.title || '新对话')}">${escapeHtml(chat.title || '新对话')}</span>
                <div class="chat-item-actions">
                    <button type="button" class="edit-btn" data-id="${chat.id}" title="编辑">
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
                    </button>
                    <button type="button" class="del-btn" data-id="${chat.id}" title="删除">
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg>
                    </button>
                </div>
            `;

            item.addEventListener('click', (e) => {
                if (!e.target.closest('.chat-item-actions')) {
                    if (chat.id !== currentChatId) loadChat(chat.id);
                }
            });

            chatList.appendChild(item);
        });

        // Edit buttons
        chatList.querySelectorAll('.edit-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const id = parseInt(btn.dataset.id);
                const chat = chats.find(c => c.id === id);
                if (chat) {
                    const newTitle = prompt('请输入新标题', chat.title || '新对话');
                    if (newTitle && newTitle.trim()) updateChatTitle(id, newTitle.trim());
                }
            });
        });

        // Delete buttons
        chatList.querySelectorAll('.del-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const id = parseInt(btn.dataset.id);
                if (confirm('确定删除此对话？')) deleteChat(id);
            });
        });
    }

    // ── Messages ──
    function clearChatBox() {
        chatBox.innerHTML = '';
    }

    function addWelcomeMessage() {
        addMessage('bot', '您好！我是智能法律助手，专注于为您提供专业的法律咨询服务。\n\n您可以向我提问任何法律相关的问题，我会基于法律法规为您提供参考意见。', false);
    }

    function addMessage(role, content, animate = true) {
        const row = document.createElement('div');
        row.className = `message-row ${role}` + (animate ? ' fade-in' : '');

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = role === 'bot' ? '&#9878;' : '&#128100;';

        const body = document.createElement('div');
        body.className = 'message-body';

        const name = document.createElement('div');
        name.className = 'message-name';
        name.textContent = role === 'bot' ? '法律助手' : '用户';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = formatMessage(content);

        body.appendChild(name);
        body.appendChild(contentDiv);
        row.appendChild(avatar);
        row.appendChild(body);
        chatBox.appendChild(row);

        scrollToBottom();
        return contentDiv;
    }

    function formatMessage(text) {
        if (!text) return '';
        let html = escapeHtml(text);
        // Convert newlines to paragraphs
        html = html.split(/\n\n+/).map(p => `<p>${p.replace(/\n/g, '<br>')}</p>`).join('');
        return html;
    }

    // ── Send message ──
    function sendMessage(text) {
        const msg = text || userInput.value.trim();
        if (!msg || isStreaming) return;

        if (!text) {
            userInput.value = '';
            userInput.style.height = 'auto';
        }

        if (!currentChatId) {
            createNewChat().then(() => sendMessage(msg));
            return;
        }

        addMessage('user', msg);

        // Update title with first message
        const title = msg.substring(0, 20) + (msg.length > 20 ? '...' : '');
        updateChatTitle(currentChatId, title);

        // Show loading
        const loadingRow = document.createElement('div');
        loadingRow.className = 'message-row bot fade-in';
        loadingRow.innerHTML = `
            <div class="message-avatar">&#9878;</div>
            <div class="message-body">
                <div class="message-name">法律助手</div>
                <div class="message-content">
                    <div class="loading-dots"><span></span><span></span><span></span></div>
                </div>
            </div>
        `;
        chatBox.appendChild(loadingRow);
        scrollToBottom();

        isStreaming = true;
        sendBtn.disabled = true;

        const kbId = kbSelector.value;

        fetch('/ask_stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({
                user_input: msg,
                chat_id: currentChatId,
                knowledge_base_id: kbId
            })
        })
        .then(response => {
            if (!response.ok) throw new Error('发送失败: ' + response.status);
            handleStream(response, loadingRow);
        })
        .catch(error => {
            console.error('发送失败:', error);
            if (loadingRow.parentNode) loadingRow.remove();
            addMessage('bot', '抱歉，发送消息时出现错误：' + error.message);
            finishStreaming();
        });
    }

    function handleStream(response, loadingRow) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let accumulated = '';
        let buffer = '';  // Buffer for incomplete lines across chunks

        // Remove loading, create streaming message
        if (loadingRow.parentNode) loadingRow.remove();

        const streamRow = document.createElement('div');
        streamRow.className = 'message-row bot fade-in';
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = '&#9878;';
        const body = document.createElement('div');
        body.className = 'message-body';
        const name = document.createElement('div');
        name.className = 'message-name';
        name.textContent = '法律助手';
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content streaming-cursor';
        body.appendChild(name);
        body.appendChild(contentDiv);
        streamRow.appendChild(avatar);
        streamRow.appendChild(body);
        chatBox.appendChild(streamRow);

        function read() {
            return reader.read().then(({ done, value }) => {
                if (done) {
                    contentDiv.classList.remove('streaming-cursor');
                    contentDiv.innerHTML = formatMessage(accumulated);
                    finishStreaming();
                    return;
                }

                // Append chunk to buffer and split by newlines
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                // Keep the last incomplete line in buffer
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    const data = line.slice(6).trim();

                    if (data === '[DONE]') {
                        contentDiv.classList.remove('streaming-cursor');
                        contentDiv.innerHTML = formatMessage(accumulated);
                        finishStreaming();
                        return;
                    }

                    // Parse JSON for done signal (precise check)
                    if (data.startsWith('{')) {
                        try {
                            const parsed = JSON.parse(data);
                            if (parsed.done === true) {
                                contentDiv.classList.remove('streaming-cursor');
                                contentDiv.innerHTML = formatMessage(accumulated);
                                finishStreaming();
                                return;
                            }
                            if (parsed.error) {
                                accumulated = parsed.error;
                                continue;
                            }
                            if (parsed.content) {
                                accumulated += parsed.content;
                            }
                        } catch (e) {
                            // Incomplete JSON, skip
                            continue;
                        }
                    }

                    contentDiv.innerHTML = formatMessage(accumulated);
                    scrollToBottom();
                }

                return read();
            });
        }

        read().catch(error => {
            console.error('流式读取失败:', error);
            contentDiv.classList.remove('streaming-cursor');
            if (!accumulated) contentDiv.innerHTML = '<p>抱歉，生成回复时出现错误。</p>';
            finishStreaming();
        });
    }

    function finishStreaming() {
        isStreaming = false;
        sendBtn.disabled = false;
        fetchChats(); // Refresh chat list
    }

    function scrollToBottom() {
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // ── Event listeners ──
    sendBtn.addEventListener('click', () => sendMessage());

    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    newChatBtn.addEventListener('click', () => createNewChat());

    kbSelector.addEventListener('change', () => {
        if (currentChatId) {
            fetch(`/api/chats/${currentChatId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': getCsrfToken() },
                body: JSON.stringify({ knowledge_base_id: kbSelector.value || null })
            }).catch(e => console.error('更新知识库失败:', e));
        }
    });

    // Close sidebar on mobile when clicking outside
    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 768 && sidebar.classList.contains('open')) {
            if (!sidebar.contains(e.target) && !mobileMenuBtn.contains(e.target)) {
                sidebar.classList.remove('open');
            }
        }
    });

    init();
});
