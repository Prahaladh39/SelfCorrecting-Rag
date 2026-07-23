document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const chatBox = document.getElementById('chatBox');
    const sendBtn = document.getElementById('sendBtn');
    const typingIndicator = document.getElementById('typingIndicator');
    const newChatBtn = document.getElementById('newChatBtn');

    // Generate a secure random thread ID for this session
    let currentThreadId = 'web_session_' + Math.random().toString(36).substr(2, 9);

    newChatBtn.addEventListener('click', () => {
        // Reset chat and clear thread to start fresh
        currentThreadId = 'web_session_' + Math.random().toString(36).substr(2, 9);
        chatBox.innerHTML = `
            <div class="message-wrapper ai-wrapper">
                <div class="avatar ai-avatar">
                    <i class="fa-solid fa-robot"></i>
                </div>
                <div class="message ai-message">
                    Thread reset! Im ready for a new analysis. Ask me a question!
                </div>
            </div>
        `;
    });

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const text = userInput.value.trim();
        if (!text) return;

        // 1. Add User Message
        appendMessage(text, 'user');
        userInput.value = '';
        
        // Disable input while waiting
        userInput.disabled = true;
        sendBtn.disabled = true;
        
        // Show Typing Indicator
        typingIndicator.style.display = 'flex';
        chatBox.scrollTop = chatBox.scrollHeight;

        try {
            // 2. Send POST request to Flask Backend
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    message: text,
                    thread_id: currentThreadId
                })
            });

            const data = await response.json();
            
            // Hide Typing Target
            typingIndicator.style.display = 'none';

            if (response.ok) {
                // Render Bot Message with Markdown parsing, fallback to plain text if CDN fails
                let finalContent = data.response;
                try {
                    if (typeof marked !== 'undefined') {
                        finalContent = marked.parse(data.response);
                    }
                } catch (parseErr) {
                    console.error("Markdown parse error:", parseErr);
                }
                appendMessage(finalContent, 'ai', true);
            } else {
                appendMessage('Oops! The internal reasoning engine encountered an error: ' + (data.error || 'Unknown Error'), 'ai');
            }
        } catch (error) {
            typingIndicator.style.display = 'none';
            appendMessage('Client/Network error: ' + error.message, 'ai');
            console.error('Error:', error);
        } finally {
            // Re-enable input
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.focus();
            chatBox.scrollTop = chatBox.scrollHeight;
        }
    });

    function appendMessage(content, sender, isHtml = false) {
        const wrapper = document.createElement('div');
        wrapper.className = `message-wrapper ${sender}-wrapper`;

        const avatar = document.createElement('div');
        avatar.className = `avatar ${sender}-avatar`;
        avatar.innerHTML = sender === 'user' ? '<i class="fa-regular fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        if (isHtml) {
            messageDiv.innerHTML = content;
        } else {
            messageDiv.textContent = content; // Safely escape user input
        }

        wrapper.appendChild(avatar);
        wrapper.appendChild(messageDiv);
        
        // Add right before typing indicator to maintain bottom placement
        chatBox.appendChild(wrapper);
        chatBox.scrollTop = chatBox.scrollHeight;
    }
});
