// // All model/user text is inserted with textContent (never innerHTML) to avoid XSS.
// const chatBox = document.getElementById('chat-box');
// const input = document.getElementById('user-input');
// const sendBtn = document.getElementById('send-button');
// const statusEl = document.getElementById('upload-status');
// const docList = document.getElementById('doc-list');
// const fileInput = document.getElementById('file-input');
// const dropzone = document.getElementById('dropzone');
// const MAX_BYTES = 5 * 1024 * 1024;

// function el(tag, cls, text) {
//   const e = document.createElement(tag);
//   if (cls) e.className = cls;
//   if (text !== undefined) e.textContent = text;
//   return e;
// }
// function addMsg(cls, text) {
//   const m = el('div', 'msg ' + cls);
//   const p = el('p', null, text);
//   m.appendChild(p);
//   chatBox.appendChild(m);
//   chatBox.scrollTop = chatBox.scrollHeight;
//   return { m, p };
// }

// async function refreshDocs() {
//   const r = await fetch('/api/documents');
//   const { documents } = await r.json();
//   docList.replaceChildren();
//   for (const [name, n] of Object.entries(documents)) docList.appendChild(el('li', null, `${name} (${n} chunks)`));
// }

// async function upload(file) {
//   if (!file) return;
//   const ext = file.name.toLowerCase().split('.').pop();
//   if (!['pdf', 'txt'].includes(ext)) { statusEl.textContent = 'Only PDF and TXT files are accepted.'; return; }
//   if (file.size > MAX_BYTES) { statusEl.textContent = 'File exceeds the 5 MB limit.'; return; }
//   statusEl.textContent = 'Uploading and indexing…';
//   const fd = new FormData();
//   fd.append('file', file);
//   try {
//     const r = await fetch('/api/upload', { method: 'POST', body: fd });
//     const j = await r.json();
//     statusEl.textContent = r.ok ? `Indexed ${j.file}: ${j.pages} page(s), ${j.chunks} chunks.` : `Error: ${j.error}`;
//     if (r.ok) refreshDocs();
//   } catch { statusEl.textContent = 'Upload failed (network error).'; }
// }
// fileInput.addEventListener('change', () => { upload(fileInput.files[0]); fileInput.value = ''; });
// dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('over'); });
// dropzone.addEventListener('dragleave', () => dropzone.classList.remove('over'));
// dropzone.addEventListener('drop', e => { e.preventDefault(); dropzone.classList.remove('over'); upload(e.dataTransfer.files[0]); });

// function renderSources(msgEl, sources, cited) {
//   if (!sources.length) return;
//   const box = el('details', 'sources');
//   box.appendChild(el('summary', null, `Sources (${cited.length} cited of ${sources.length} retrieved)`));
//   for (const s of sources) {
//     const row = el('div', 'src' + (cited.includes(s.n) ? ' cited' : ''));
//     row.appendChild(el('strong', null, `[${s.n}] ${s.label}`));
//     row.appendChild(el('span', 'score', ` score ${s.score}`));
//     row.appendChild(el('div', 'snippet', s.snippet + (s.snippet.length >= 300 ? '…' : '')));
//     box.appendChild(row);
//   }
//   msgEl.appendChild(box);
// }

// async function ask(message) {
//   addMsg('user', message);
//   const { m, p } = addMsg('ai', '');
//   sendBtn.disabled = true;
//   let sources = [];
//   try {
//     const r = await fetch('/api/chat/stream', {
//       method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message }),
//     });
//     if (!r.ok) { const j = await r.json().catch(() => ({})); p.textContent = `Error: ${j.error || r.status}`; m.classList.add('error'); return; }
//     const reader = r.body.getReader();
//     const dec = new TextDecoder();
//     let buf = '';
//     for (;;) {
//       const { value, done } = await reader.read();
//       if (done) break;
//       buf += dec.decode(value, { stream: true });
//       let i;
//       while ((i = buf.indexOf('\n\n')) >= 0) {
//         const line = buf.slice(0, i); buf = buf.slice(i + 2);
//         if (!line.startsWith('data:')) continue;
//         const ev = JSON.parse(line.slice(5));
//         if (ev.type === 'sources') sources = ev.sources;
//         else if (ev.type === 'token') { p.textContent += ev.text; chatBox.scrollTop = chatBox.scrollHeight; }
//         else if (ev.type === 'done') renderSources(m, sources, ev.cited);
//         else if (ev.type === 'error') { p.textContent += `\n[${ev.error}]`; m.classList.add('error'); }
//       }
//     }
//   } catch { p.textContent = 'Error communicating with the server.'; m.classList.add('error'); }
//   finally { sendBtn.disabled = false; chatBox.scrollTop = chatBox.scrollHeight; }
// }

// document.getElementById('chat-form').addEventListener('submit', e => {
//   e.preventDefault();
//   const msg = input.value.trim();
//   if (!msg) return;
//   input.value = '';
//   ask(msg);
// });
// input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); document.getElementById('chat-form').requestSubmit(); } });
// document.getElementById('reset-btn').addEventListener('click', async () => {
//   await fetch('/api/session/reset', { method: 'POST' });
//   chatBox.replaceChildren(); addMsg('ai', 'Session cleared. Upload a document to begin.');
//   statusEl.textContent = ''; refreshDocs();
// });
// refreshDocs();

document.addEventListener("DOMContentLoaded", () => {

    const chatBox = document.getElementById("chat-box");
    const chatForm = document.getElementById("chat-form");
    const userInput = document.getElementById("user-input");
    const sendButton = document.getElementById("send-button");
    const fileInput = document.getElementById("file-input");
    const dropzone = document.getElementById("dropzone");
    const uploadStatus = document.getElementById("upload-status");
    const docList = document.getElementById("doc-list");
    const resetButton = document.getElementById("reset-btn");
    const documentCount = document.getElementById("document-count");

    // IMPORTANT: Match these URLs with your Flask routes.
    const API = {
        upload: "/upload",
        ask: "/ask",
        reset: "/reset"
    };

    let isLoading = false;
    let documents = [];

    // -----------------------------
    // UTILITIES
    // -----------------------------

    function getTime() {
        return new Date().toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit"
        });
    }

    function scrollToBottom() {
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function hideWelcome() {
        const welcome = document.getElementById("welcome-screen");
        if (welcome) welcome.remove();
    }

    function showStatus(message, error = false) {
        uploadStatus.textContent = message;
        uploadStatus.style.color = error ? "#ff858e" : "#a58aff";
    }

    function setLoading(state) {
        isLoading = state;
        sendButton.disabled = state || !userInput.value.trim();

        sendButton.querySelector("span:first-child").textContent =
            state ? "Thinking..." : "Send message";
    }

    // -----------------------------
    // CHAT MESSAGES
    // -----------------------------

    function addMessage(role, message, sources = []) {

        hideWelcome();

        const messageElement = document.createElement("div");
        messageElement.className = `msg ${role}`;

        const avatar = document.createElement("div");
        avatar.className = "msg-avatar";
        avatar.textContent = role === "ai" ? "✧" : "DT";

        const content = document.createElement("div");
        content.className = "msg-content";

        const author = document.createElement("div");
        author.className = "msg-author";
        author.textContent = role === "ai" ? "DocuMind AI" : "You";

        const text = document.createElement("div");
        text.className = "msg-text";

        if (role === "user") {
            const bubble = document.createElement("div");
            bubble.className = "user-bubble";
            bubble.textContent = message;
            text.appendChild(bubble);
        } else {
            // Safely render plain text without interpreting HTML.
            const paragraphs = String(message || "").split("\n");

            paragraphs.forEach(line => {
                const p = document.createElement("p");
                p.textContent = line;
                text.appendChild(p);
            });
        }

        const time = document.createElement("div");
        time.className = "msg-time";
        time.textContent = getTime();

        content.append(author, text);

        if (Array.isArray(sources) && sources.length) {
            const sourceBox = document.createElement("div");
            sourceBox.className = "msg-sources";

            const heading = document.createElement("strong");
            heading.textContent = "▤ Sources";
            sourceBox.appendChild(heading);

            sources.forEach(source => {
                const item = document.createElement("div");
                item.textContent = "• " +
                    (typeof source === "string"
                        ? source
                        : source.filename || source.source || JSON.stringify(source));

                sourceBox.appendChild(item);
            });

            content.appendChild(sourceBox);
        }

        content.appendChild(time);
        messageElement.append(avatar, content);

        chatBox.appendChild(messageElement);
        scrollToBottom();

        return messageElement;
    }

    function showTyping() {

        const typing = document.createElement("div");
        typing.className = "msg ai";
        typing.id = "typing-indicator";

        typing.innerHTML = `
            <div class="msg-avatar">✧</div>
            <div class="msg-content">
                <div class="msg-author">DocuMind AI</div>
                <div class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;

        chatBox.appendChild(typing);
        scrollToBottom();
    }

    function removeTyping() {
        const typing = document.getElementById("typing-indicator");
        if (typing) typing.remove();
    }

    // -----------------------------
    // CHAT SUBMISSION
    // -----------------------------

    chatForm.addEventListener("submit", async (event) => {

        event.preventDefault();

        const question = userInput.value.trim();

        if (!question || isLoading) return;

        addMessage("user", question);

        userInput.value = "";
        userInput.style.height = "auto";

        setLoading(true);
        showTyping();

        try {

            const response = await fetch(API.ask, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    question: question
                })
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();

            removeTyping();

            if (data.error) {
                throw new Error(data.error);
            }

            const answer =
                data.answer ||
                data.response ||
                data.message ||
                "I couldn't generate an answer.";

            const sources =
                data.sources ||
                data.citations ||
                [];

            addMessage("ai", answer, sources);

        } catch (error) {

            removeTyping();

            addMessage(
                "ai",
                "Sorry, something went wrong. Please check your connection and try again.\n\n" +
                error.message
            );

            console.error("Chat error:", error);

        } finally {
            setLoading(false);
            userInput.focus();
        }
    });

    // -----------------------------
    // INPUT CONTROLS
    // -----------------------------

    userInput.addEventListener("input", () => {

        userInput.style.height = "auto";

        userInput.style.height =
            Math.min(userInput.scrollHeight, 130) + "px";

        sendButton.disabled =
            !userInput.value.trim() || isLoading;
    });

    userInput.addEventListener("keydown", event => {

        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            chatForm.requestSubmit();
        }
    });

    // -----------------------------
    // SUGGESTED QUESTIONS
    // -----------------------------

    document.querySelectorAll(".suggestion").forEach(button => {

        button.addEventListener("click", () => {

            userInput.value = button.dataset.question;

            userInput.dispatchEvent(new Event("input"));

            chatForm.requestSubmit();
        });
    });

    // -----------------------------
    // DOCUMENT UPLOAD
    // -----------------------------

    async function uploadFile(file) {

        if (!file) return;

        const extension = file.name.split(".").pop().toLowerCase();

        if (!["pdf", "txt"].includes(extension)) {
            showStatus("Only PDF and TXT files are supported.", true);
            return;
        }

        if (file.size > 5 * 1024 * 1024) {
            showStatus("File exceeds the 5 MB limit.", true);
            return;
        }

        showStatus(`Uploading ${file.name}...`);

        const formData = new FormData();
        formData.append("file", file);

        try {

            const response = await fetch(API.upload, {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.status}`);
            }

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            if (!documents.includes(file.name)) {
                documents.push(file.name);
            }

            renderDocuments();

            showStatus(`✓ ${file.name} uploaded successfully.`);

        } catch (error) {

            showStatus(error.message, true);
            console.error("Upload error:", error);

        }
    }

    fileInput.addEventListener("change", async () => {

        for (const file of fileInput.files) {
            await uploadFile(file);
        }

        fileInput.value = "";
    });

    // -----------------------------
    // DRAG AND DROP
    // -----------------------------

    ["dragenter", "dragover"].forEach(eventName => {

        dropzone.addEventListener(eventName, event => {
            event.preventDefault();
            dropzone.classList.add("dragging");
        });
    });

    ["dragleave", "drop"].forEach(eventName => {

        dropzone.addEventListener(eventName, event => {
            event.preventDefault();
            dropzone.classList.remove("dragging");
        });
    });

    dropzone.addEventListener("drop", async event => {

        const files = event.dataTransfer.files;

        for (const file of files) {
            await uploadFile(file);
        }
    });

    // -----------------------------
    // DOCUMENT LIST
    // -----------------------------

    function renderDocuments() {

        docList.innerHTML = "";

        documents.forEach(filename => {

            const li = document.createElement("li");

            const icon = document.createElement("span");
            icon.className = "doc-icon";
            icon.textContent = filename.toLowerCase().endsWith(".pdf")
                ? "▤"
                : "▧";

            const name = document.createElement("span");
            name.className = "doc-name";
            name.textContent = filename;
            name.title = filename;

            li.append(icon, name);
            docList.appendChild(li);
        });

        documentCount.textContent = documents.length;
    }

    // -----------------------------
    // NEW SESSION
    // -----------------------------

    resetButton.addEventListener("click", async () => {

        if (!confirm("Start a new session? This may clear uploaded documents and chat history.")) {
            return;
        }

        try {

            const response = await fetch(API.reset, {
                method: "POST"
            });

            if (!response.ok) {
                throw new Error("Unable to reset session.");
            }

            documents = [];
            renderDocuments();

            chatBox.innerHTML = `
                <div class="msg ai">
                    <div class="msg-avatar">✧</div>
                    <div class="msg-content">
                        <div class="msg-author">DocuMind AI</div>
                        <div class="msg-text">
                            <p>Your new session is ready.</p>
                            <p>Upload a document and start asking questions!</p>
                        </div>
                        <div class="msg-time">${getTime()}</div>
                    </div>
                </div>
            `;

            showStatus("New session started.");

        } catch (error) {

            showStatus(error.message, true);
            console.error("Reset error:", error);

        }
    });

});
