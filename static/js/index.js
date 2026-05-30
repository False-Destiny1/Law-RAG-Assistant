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
    let abortController = null;

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
        if (!kbSelector) return;
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
                if (data.knowledge_base_id && kbSelector) {
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
        contentDiv.innerHTML = formatMessage(content, role);

        body.appendChild(name);
        body.appendChild(contentDiv);
        row.appendChild(avatar);
        row.appendChild(body);
        chatBox.appendChild(row);

        if (role === 'bot') addCopyButton(contentDiv, content);

        scrollToBottom();
        return contentDiv;
    }

    function formatMessage(text, role) {
        if (!text) return '';
        if (role === 'bot' && typeof marked !== 'undefined' && typeof DOMPurify !== 'undefined') {
            const rawHtml = marked.parse(text);
            return DOMPurify.sanitize(rawHtml);
        }
        let html = escapeHtml(text);
        html = html.split(/\n\n+/).map(p => `<p>${p.replace(/\n/g, '<br>')}</p>`).join('');
        return html;
    }

    function addCopyButton(contentDiv, rawText) {
        let wrapper = contentDiv.parentNode.querySelector('.message-actions');
        if (!wrapper) {
            wrapper = document.createElement('div');
            wrapper.className = 'message-actions';
            contentDiv.parentNode.insertBefore(wrapper, contentDiv.nextSibling);
        }
        const btn = document.createElement('button');
        btn.className = 'copy-btn';
        btn.title = '复制';
        btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>';
        btn.addEventListener('click', function() {
            navigator.clipboard.writeText(rawText).then(() => {
                btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>';
                btn.classList.add('copied');
                setTimeout(() => {
                    btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>';
                    btn.classList.remove('copied');
                }, 2000);
            });
        });
        wrapper.appendChild(btn);
    }

    function addFeedbackButtons(contentDiv, messageId, chatId) {
        let wrapper = contentDiv.parentNode.querySelector('.message-actions');
        if (!wrapper) {
            wrapper = document.createElement('div');
            wrapper.className = 'message-actions';
            contentDiv.parentNode.insertBefore(wrapper, contentDiv.nextSibling);
        }
        const upBtn = document.createElement('button');
        upBtn.className = 'feedback-btn';
        upBtn.title = '回答有帮助';
        upBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 9V5a3 3 0 00-3-3l-4 9v11h11.28a2 2 0 002-1.7l1.38-9a2 2 0 00-2-2.3H14z"/><path d="M7 22H4a2 2 0 01-2-2v-7a2 2 0 012-2h3"/></svg>';
        const downBtn = document.createElement('button');
        downBtn.className = 'feedback-btn';
        downBtn.title = '回答不准确';
        downBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 15v4a3 3 0 003 3l4-9V2H5.72a2 2 0 00-2 1.7l-1.38 9a2 2 0 002 2.3H10z"/><path d="M17 2h3a2 2 0 012 2v7a2 2 0 01-2 2h-3"/></svg>';

        function submitFeedback(rating) {
            const formData = new FormData();
            formData.append('message_id', messageId);
            formData.append('chat_id', chatId);
            formData.append('rating', rating);
            fetch('/api/feedback', {
                method: 'POST',
                headers: { 'X-CSRF-Token': getCsrfToken() },
                body: formData
            }).then(r => {
                if (r.ok) {
                    upBtn.classList.toggle('feedback-active', rating === 'up');
                    downBtn.classList.toggle('feedback-active', rating === 'down');
                }
            });
        }

        upBtn.addEventListener('click', () => submitFeedback('up'));
        downBtn.addEventListener('click', () => submitFeedback('down'));
        wrapper.appendChild(upBtn);
        wrapper.appendChild(downBtn);
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

        // Only auto-set title on first message (when still default "新对话")
        const currentChat = chats.find(c => c.id === currentChatId);
        if (currentChat && (!currentChat.title || currentChat.title === '新对话')) {
            const title = msg.substring(0, 20) + (msg.length > 20 ? '...' : '');
            updateChatTitle(currentChatId, title);
        }

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
        showStopButton();

        abortController = new AbortController();
        const kbId = kbSelector ? kbSelector.value : '';

        fetch('/ask_stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            signal: abortController.signal,
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
            if (error.name === 'AbortError') {
                if (loadingRow.parentNode) loadingRow.remove();
                addMessage('bot', '（已停止生成）');
            } else {
                console.error('发送失败:', error);
                if (loadingRow.parentNode) loadingRow.remove();
                addMessage('bot', '抱歉，发送消息时出现错误：' + error.message);
            }
            finishStreaming();
        });
    }

    function handleStream(response, loadingRow) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let accumulated = '';
        let buffer = '';  // Buffer for incomplete lines across chunks
        let jsonRetryCount = 0;  // Track retries for incomplete JSON

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
                    contentDiv.innerHTML = formatMessage(accumulated, 'bot');
                    addCopyButton(contentDiv, accumulated);
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
                        contentDiv.innerHTML = formatMessage(accumulated, 'bot');
                        addCopyButton(contentDiv, accumulated);
                        finishStreaming();
                        return;
                    }

                    // Parse JSON for done signal (precise check)
                    if (data.startsWith('{')) {
                        try {
                            jsonRetryCount = 0;  // Reset on successful parse
                            const parsed = JSON.parse(data);
                            if (parsed.done === true) {
                                contentDiv.classList.remove('streaming-cursor');
                                contentDiv.innerHTML = formatMessage(accumulated, 'bot');
                                addCopyButton(contentDiv, accumulated);
                                if (parsed.message_id) addFeedbackButtons(contentDiv, parsed.message_id, parsed.chat_id);
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
                            // Incomplete JSON (possibly split across chunks), put back in buffer
                            jsonRetryCount++;
                            if (jsonRetryCount > 3) {
                                console.warn('Skipping malformed SSE line after 3 retries');
                                jsonRetryCount = 0;
                            } else {
                                buffer = line + '\n' + buffer;
                                break;
                            }
                        }
                    }

                    // Lightweight rendering during streaming (newlines only, no Markdown parse)
                    contentDiv.innerHTML = escapeHtml(accumulated).replace(/\n/g, '<br>');
                    scrollToBottom();
                }

                return read();
            });
        }

        read().catch(error => {
            contentDiv.classList.remove('streaming-cursor');
            if (error.name === 'AbortError') {
                if (accumulated) {
                    contentDiv.innerHTML = formatMessage(accumulated, 'bot');
                    addCopyButton(contentDiv, accumulated);
                } else {
                    contentDiv.innerHTML = '<p>（已停止生成）</p>';
                }
            } else {
                console.error('流式读取失败:', error);
                if (!accumulated) contentDiv.innerHTML = '<p>抱歉，生成回复时出现错误。</p>';
            }
            finishStreaming();
        });
    }

    function showStopButton() {
        sendBtn.classList.add('hidden');
        let stopBtn = document.getElementById('stopBtn');
        if (!stopBtn) {
            stopBtn = document.createElement('button');
            stopBtn.id = 'stopBtn';
            stopBtn.className = 'stop-btn';
            stopBtn.title = '停止生成';
            stopBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>';
            sendBtn.parentNode.appendChild(stopBtn);
        }
        stopBtn.classList.remove('hidden');
        stopBtn.onclick = () => {
            if (abortController) abortController.abort();
        };
    }

    function hideStopButton() {
        sendBtn.classList.remove('hidden');
        const stopBtn = document.getElementById('stopBtn');
        if (stopBtn) stopBtn.classList.add('hidden');
    }

    function finishStreaming() {
        isStreaming = false;
        sendBtn.disabled = false;
        abortController = null;
        hideStopButton();
        // Update current chat timestamp locally instead of re-fetching all chats
        const currentChat = chats.find(c => c.id === currentChatId);
        if (currentChat) {
            currentChat.updated_at = new Date().toISOString();
            renderChatList();
        }
    }

    function scrollToBottom() {
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function escapeHtml(text) {
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
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

    const exportBtn = document.getElementById('exportBtn');
    if (exportBtn) {
        exportBtn.addEventListener('click', () => {
            if (!currentChatId) return;
            window.open(`/api/chats/${currentChatId}/export`, '_blank');
        });
    }

    // Password change modal
    const changePwdBtn = document.getElementById('changePwdBtn');
    const pwdModal = document.getElementById('pwdModal');
    const pwdForm = document.getElementById('pwdForm');
    const pwdCancel = document.getElementById('pwdCancel');
    const pwdError = document.getElementById('pwdError');

    if (changePwdBtn && pwdModal) {
        changePwdBtn.addEventListener('click', () => {
            pwdModal.style.display = 'flex';
            pwdError.style.display = 'none';
            pwdForm.reset();
        });
        pwdCancel.addEventListener('click', () => { pwdModal.style.display = 'none'; });
        pwdModal.addEventListener('click', (e) => { if (e.target === pwdModal) pwdModal.style.display = 'none'; });
        pwdForm.addEventListener('submit', (e) => {
            e.preventDefault();
            fetch('/api/change-password', {
                method: 'POST',
                headers: { 'X-CSRF-Token': getCsrfToken() },
                body: new FormData(pwdForm)
            }).then(r => r.json().then(d => {
                if (r.ok) {
                    pwdModal.style.display = 'none';
                    alert('密码修改成功，请重新登录');
                    window.location.href = '/login';
                } else {
                    pwdError.textContent = d.error || '修改失败';
                    pwdError.style.display = 'block';
                }
            }));
        });
    }

    if (kbSelector) {
        kbSelector.addEventListener('change', () => {
            if (currentChatId) {
                fetch(`/api/chats/${currentChatId}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': getCsrfToken() },
                    body: JSON.stringify({ knowledge_base_id: kbSelector.value || null })
                }).catch(e => console.error('更新知识库失败:', e));
            }
        });
    }

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
