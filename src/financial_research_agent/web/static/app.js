const state = {
  sessionId: null,
  messages: [],
  busy: false,
};

const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const messageList = document.querySelector("#message-list");
const loadingRow = document.querySelector("#loading-row");
const errorBanner = document.querySelector("#error-banner");
const providerPill = document.querySelector("#provider-pill");
const sessionLabel = document.querySelector("#session-label");

function setBusy(value) {
  state.busy = value;
  sendButton.disabled = value;
  input.disabled = value;
  loadingRow.hidden = !value;
}

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.hidden = false;
}

function clearError() {
  errorBanner.textContent = "";
  errorBanner.hidden = true;
}

function renderMessages() {
  messageList.innerHTML = "";
  for (const message of state.messages) {
    const item = document.createElement("li");
    item.className = `message ${message.role}`;

    const meta = document.createElement("div");
    meta.className = "message-meta";
    meta.textContent =
      message.role === "assistant" && message.provider
        ? `${message.provider} / ${message.model}`
        : message.role;

    const content = document.createElement("div");
    content.textContent = message.content;

    item.append(meta, content);
    messageList.append(item);
  }
  messageList.scrollTop = messageList.scrollHeight;
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    const detail = payload.detail || {};
    throw new Error(detail.message || detail.error || `Request failed with ${response.status}`);
  }
  return payload;
}

async function loadStatus() {
  const payload = await requestJson("/api/status");
  providerPill.textContent = `${payload.chat.provider} / ${payload.chat.model}`;
}

async function createSession() {
  const payload = await requestJson("/api/sessions", { method: "POST" });
  state.sessionId = payload.session.id;
  state.messages = payload.session.messages;
  sessionLabel.textContent = payload.session.id;
  renderMessages();
}

async function sendMessage(content) {
  const payload = await requestJson(`/api/sessions/${state.sessionId}/messages`, {
    method: "POST",
    body: JSON.stringify({ content }),
  });
  state.messages = payload.session.messages;
  renderMessages();
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const content = input.value.trim();
  if (!content || state.busy || !state.sessionId) {
    return;
  }
  clearError();
  setBusy(true);
  input.value = "";
  try {
    await sendMessage(content);
  } catch (error) {
    input.value = content;
    showError(error instanceof Error ? error.message : "Chat request failed.");
  } finally {
    setBusy(false);
    input.focus();
  }
});

async function start() {
  setBusy(true);
  try {
    await Promise.all([loadStatus(), createSession()]);
  } catch (error) {
    showError(error instanceof Error ? error.message : "Could not start the chat.");
  } finally {
    setBusy(false);
  }
}

start();
