const state = {
  sessionId: null,
  messages: [],
  sessions: [],
  busy: false,
  suggestions: [],
  selectedSuggestionIndex: 0,
  suggestionQuery: "",
  suggestionRequestId: 0,
};

const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const mentionMenu = document.querySelector("#mention-menu");
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
  input.contentEditable = value ? "false" : "true";
  newSessionButton.disabled = value;
  clearSessionsButton.disabled = value;
  loadingRow.hidden = !value;
  if (value) {
    hideMentionMenu();
  }
}

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.hidden = false;
}

function clearError() {
  errorBanner.textContent = "";
  errorBanner.hidden = true;
}

function mentionText(mention) {
  return `@${mention.label}`;
}

function renderMessageContent(container, message) {
  if (message.role !== "user" || !message.mentions?.length) {
    container.textContent = message.content;
    return;
  }

  const mentionsByLabel = new Map(message.mentions.map((mention) => [mention.label, mention]));
  const labels = [...mentionsByLabel.keys()]
    .map((label) => label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    .join("|");
  const matcher = new RegExp(`@(${labels})`, "g");
  let cursor = 0;
  for (const match of message.content.matchAll(matcher)) {
    if (match.index > cursor) {
      container.append(document.createTextNode(message.content.slice(cursor, match.index)));
    }
    container.append(renderMentionChip(mentionsByLabel.get(match[1])));
    cursor = match.index + match[0].length;
  }
  if (cursor < message.content.length) {
    container.append(document.createTextNode(message.content.slice(cursor)));
  }
}

function renderMentionChip(mention) {
  const chip = document.createElement("span");
  chip.className = "mention-chip";
  chip.textContent = mentionText(mention);
  chip.dataset.mention = JSON.stringify(mention);
  chip.contentEditable = "false";
  return chip;
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
    content.className = "message-content";
    renderMessageContent(content, message);

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

function primarySecurity(candidate) {
  return candidate.securities[0] || { ticker: null };
}

function identifierValue(candidate, type) {
  return (
    candidate.company.identifiers.find((identifier) => identifier.type === type)?.value ||
    primarySecurity(candidate).identifiers?.find((identifier) => identifier.type === type)?.value ||
    null
  );
}

function mentionFromCandidate(candidate) {
  const security = primarySecurity(candidate);
  const label =
    security.ticker || candidate.company.display_name || candidate.company.legal_name || "company";
  return {
    id: candidate.company.id,
    label,
    company_id: candidate.company.id,
    legal_name: candidate.company.legal_name,
    ticker: security.ticker || null,
    cik: identifierValue(candidate, "cik"),
    source_provider: candidate.source?.provider || null,
  };
}

function renderMentionMenu() {
  mentionMenu.innerHTML = "";
  mentionMenu.hidden = state.suggestions.length === 0;
  state.suggestions.forEach((candidate, index) => {
    const mention = mentionFromCandidate(candidate);
    const security = primarySecurity(candidate);
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className =
      index === state.selectedSuggestionIndex ? "mention-option active" : "mention-option";

    const title = document.createElement("span");
    title.className = "mention-option-title";
    title.textContent = mentionText(mention);

    const meta = document.createElement("span");
    meta.className = "mention-option-meta";
    meta.textContent = security.ticker
      ? `${candidate.company.legal_name} / ${security.ticker}`
      : candidate.company.legal_name;

    button.append(title, meta);
    button.addEventListener("mousedown", (event) => {
      event.preventDefault();
      insertMention(candidate);
    });
    item.append(button);
    mentionMenu.append(item);
  });
}

function hideMentionMenu() {
  state.suggestions = [];
  state.selectedSuggestionIndex = 0;
  state.suggestionQuery = "";
  mentionMenu.hidden = true;
  mentionMenu.innerHTML = "";
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

async function loadMentionSuggestions(query) {
  const requestId = ++state.suggestionRequestId;
  const payload = await requestJson(
    `/api/company-search?query=${encodeURIComponent(query)}&limit=5`
  );
  if (requestId !== state.suggestionRequestId) {
    return;
  }
  state.suggestions = payload.result.candidates || [];
  state.selectedSuggestionIndex = 0;
  renderMentionMenu();
}

function textBeforeCaret() {
  const selection = window.getSelection();
  if (!selection || selection.rangeCount === 0 || !input.contains(selection.anchorNode)) {
    return "";
  }
  const range = selection.getRangeAt(0).cloneRange();
  range.selectNodeContents(input);
  range.setEnd(selection.anchorNode, selection.anchorOffset);
  return range.toString();
}

function activeMentionQuery() {
  const match = textBeforeCaret().match(/(?:^|\s)@([A-Za-z0-9][A-Za-z0-9 ._-]{1,40})$/);
  if (!match) {
    return null;
  }
  return match[1].trim();
}

function textPosition(root, offset) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let remaining = offset;
  let lastNode = null;
  while (true) {
    const node = walker.nextNode();
    if (!node) {
      break;
    }
    lastNode = node;
    if (remaining <= node.nodeValue.length) {
      return { node, offset: remaining };
    }
    remaining -= node.nodeValue.length;
  }
  if (lastNode) {
    return { node: lastNode, offset: lastNode.nodeValue.length };
  }
  return { node: root, offset: root.childNodes.length };
}

function replaceActiveMentionText(chip) {
  const before = textBeforeCaret();
  const match = before.match(/(?:^|\s)@([A-Za-z0-9][A-Za-z0-9 ._-]{1,40})$/);
  if (!match) {
    return;
  }
  const endOffset = before.length;
  const startOffset = endOffset - match[1].length - 1;
  const start = textPosition(input, startOffset);
  const end = textPosition(input, endOffset);
  const range = document.createRange();
  range.setStart(start.node, start.offset);
  range.setEnd(end.node, end.offset);
  range.deleteContents();
  const spacer = document.createTextNode(" ");
  range.insertNode(spacer);
  range.insertNode(chip);
  range.setStartAfter(spacer);
  range.collapse(true);
  const selection = window.getSelection();
  selection.removeAllRanges();
  selection.addRange(range);
}

function insertMention(candidate) {
  const mention = mentionFromCandidate(candidate);
  replaceActiveMentionText(renderMentionChip(mention));
  hideMentionMenu();
  input.focus();
}

async function updateMentionQuery() {
  if (state.busy) {
    return;
  }
  const query = activeMentionQuery();
  if (!query || query.length < 2) {
    hideMentionMenu();
    return;
  }
  if (query === state.suggestionQuery && state.suggestions.length > 0) {
    return;
  }
  state.suggestionQuery = query;
  try {
    await loadMentionSuggestions(query);
  } catch {
    hideMentionMenu();
  }
}

function editorText() {
  function read(node) {
    if (node.nodeType === Node.TEXT_NODE) {
      return node.nodeValue;
    }
    if (node.nodeType !== Node.ELEMENT_NODE) {
      return "";
    }
    if (node.classList?.contains("mention-chip")) {
      return node.textContent;
    }
    if (node.tagName === "BR") {
      return "\n";
    }
    const childText = [...node.childNodes].map(read).join("");
    if (node.tagName === "DIV" || node.tagName === "P") {
      return `${childText}\n`;
    }
    return childText;
  }
  return read(input).replace(/\u00a0/g, " ").replace(/\n{3,}/g, "\n\n").trim();
}

function editorMentions() {
  const mentions = [];
  const seen = new Set();
  for (const chip of input.querySelectorAll(".mention-chip[data-mention]")) {
    try {
      const mention = JSON.parse(chip.dataset.mention);
      if (!seen.has(mention.id)) {
        seen.add(mention.id);
        mentions.push(mention);
      }
    } catch {
      continue;
    }
  }
  return mentions;
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

async function sendMessage(content, mentions) {
  const payload = await requestJson(`/api/sessions/${state.sessionId}/messages`, {
    method: "POST",
    body: JSON.stringify({ content, mentions }),
  });
  state.messages = payload.session.messages;
  await loadSessions();
  renderMessages();
}

input.addEventListener("input", () => {
  updateMentionQuery();
});

input.addEventListener("keydown", (event) => {
  if (!mentionMenu.hidden && state.suggestions.length > 0) {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      state.selectedSuggestionIndex =
        (state.selectedSuggestionIndex + 1) % state.suggestions.length;
      renderMentionMenu();
      return;
    }
    if (event.key === "ArrowUp") {
      event.preventDefault();
      state.selectedSuggestionIndex =
        (state.selectedSuggestionIndex - 1 + state.suggestions.length) % state.suggestions.length;
      renderMentionMenu();
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      insertMention(state.suggestions[state.selectedSuggestionIndex]);
      return;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      hideMentionMenu();
      return;
    }
  }
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

input.addEventListener("blur", () => {
  window.setTimeout(hideMentionMenu, 120);
});

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
  const content = editorText();
  if (!content || state.busy || !state.sessionId) {
    return;
  }
  const mentions = editorMentions();
  clearError();
  setBusy(true);
  input.innerHTML = "";
  try {
    await sendMessage(content, mentions);
  } catch (error) {
    input.textContent = content;
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
