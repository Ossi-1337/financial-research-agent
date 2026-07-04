const state = {
  sessionId: null,
  messages: [],
  sessions: [],
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
const sessionList = document.querySelector("#session-list");
const newSessionButton = document.querySelector("#new-session-button");
const clearSessionsButton = document.querySelector("#clear-sessions-button");

function setBusy(value) {
  state.busy = value;
  sendButton.disabled = value;
  input.disabled = value;
  newSessionButton.disabled = value;
  clearSessionsButton.disabled = value;
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

function sessionTitle(session) {
  const firstUserMessage = session.messages.find((message) => message.role === "user");
  if (!firstUserMessage) {
    return "New session";
  }
  return firstUserMessage.content.length > 46
    ? `${firstUserMessage.content.slice(0, 43)}...`
    : firstUserMessage.content;
}

function renderSessions() {
  sessionList.innerHTML = "";
  for (const session of state.sessions) {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = session.id === state.sessionId ? "session-button active" : "session-button";
    button.textContent = sessionTitle(session);
    button.addEventListener("click", () => openSession(session.id));
    item.append(button);
    sessionList.append(item);
  }
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
  await loadSessions();
  renderMessages();
}

async function loadSessions() {
  const payload = await requestJson("/api/sessions");
  state.sessions = payload.sessions;
  renderSessions();
}

async function openSession(sessionId, allowBusy = false) {
  if ((!allowBusy && state.busy) || sessionId === state.sessionId) {
    return;
  }
  clearError();
  setBusy(true);
  try {
    const payload = await requestJson(`/api/sessions/${sessionId}`);
    state.sessionId = payload.session.id;
    state.messages = payload.session.messages;
    sessionLabel.textContent = payload.session.id;
    await loadSessions();
    renderMessages();
  } catch (error) {
    showError(error instanceof Error ? error.message : "Could not open the session.");
  } finally {
    setBusy(false);
  }
}

async function clearSessions() {
  if (state.busy) {
    return;
  }
  clearError();
  setBusy(true);
  try {
    await requestJson("/api/sessions", { method: "DELETE" });
    state.sessionId = null;
    state.messages = [];
    await createSession();
  } catch (error) {
    showError(error instanceof Error ? error.message : "Could not clear sessions.");
  } finally {
    setBusy(false);
    input.focus();
  }
}

async function sendMessage(content) {
  const payload = await requestJson(`/api/sessions/${state.sessionId}/messages`, {
    method: "POST",
    body: JSON.stringify({ content }),
  });
  state.messages = payload.session.messages;
  await loadSessions();
  renderMessages();
}

newSessionButton.addEventListener("click", async () => {
  if (state.busy) {
    return;
  }
  clearError();
  setBusy(true);
  try {
    await createSession();
  } catch (error) {
    showError(error instanceof Error ? error.message : "Could not create a session.");
  } finally {
    setBusy(false);
    input.focus();
  }
});

clearSessionsButton.addEventListener("click", () => {
  clearSessions();
});

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
    await loadStatus();
    await loadSessions();
    if (state.sessions.length > 0) {
      await openSession(state.sessions[0].id, true);
    } else {
      await createSession();
    }
  } catch (error) {
    showError(error instanceof Error ? error.message : "Could not start the chat.");
  } finally {
    setBusy(false);
  }
}

start();
