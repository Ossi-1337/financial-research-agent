const state = {
  sessionId: null,
  messages: [],
  sessions: [],
  busy: false,
  suggestions: [],
  selectedSuggestionIndex: 0,
  suggestionQuery: "",
  suggestionRequestId: 0,
  abortController: null,
  tracesByRunId: {},
};

const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const mentionMenu = document.querySelector("#mention-menu");
const sendButton = document.querySelector("#send-button");
const messageList = document.querySelector("#message-list");
const errorBanner = document.querySelector("#error-banner");
const providerPill = document.querySelector("#provider-pill");
const sessionLabel = document.querySelector("#session-label");
const sessionList = document.querySelector("#session-list");
const contextPanel = document.querySelector("#context-panel");
const contextSourceList = document.querySelector("#context-source-list");
const newSessionButton = document.querySelector("#new-session-button");
const clearSessionsButton = document.querySelector("#clear-sessions-button");

function setBusy(value) {
  state.busy = value;
  input.contentEditable = value ? "false" : "true";
  newSessionButton.disabled = value;
  clearSessionsButton.disabled = value;
  sendButton.textContent = value ? "Stop" : "↑";
  sendButton.setAttribute("aria-label", value ? "Stop response" : "Send message");
  sendButton.classList.toggle("busy", value);
  updateSendButtonState();
  renderMessages();
  if (value) {
    hideMentionMenu();
  }
}

function updateSendButtonState() {
  sendButton.disabled = !state.busy && editorText().length === 0;
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

function safeExternalUrl(value) {
  try {
    const url = new URL(value);
    return url.protocol === "http:" || url.protocol === "https:" ? url.href : null;
  } catch {
    return null;
  }
}

function renderMessageContent(container, message) {
  if (message.role === "assistant" && message.streaming && !message.content) {
    container.classList.add("pending");
    container.textContent = "Thinking...";
    return;
  }

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

function renderMessageCitations(container, message) {
  if (!message.citations?.length) {
    return;
  }
  const snippets = new Map(
    (message.evidence_snippets || []).map((snippet) => [snippet.citation_id, snippet])
  );
  const wrapper = document.createElement("div");
  wrapper.className = "citation-list";
  for (const citation of message.citations) {
    const snippet = snippets.get(citation.id);
    const item = document.createElement("article");
    item.className = "citation-item";

    const header = document.createElement("div");
    header.className = "citation-header";

    const safeUrl = safeExternalUrl(citation.source_url);
    const marker = document.createElement(safeUrl ? "a" : "span");
    marker.className = "citation-marker";
    if (safeUrl) {
      marker.href = safeUrl;
      marker.target = "_blank";
      marker.rel = "noreferrer";
    }
    marker.textContent = citation.marker || `[${citation.id}]`;

    const location = document.createElement("span");
    location.className = "citation-location";
    location.textContent = citation.section || citation.document_id || citation.chunk_id || "Source";

    header.append(marker, location);
    item.append(header);

    if (snippet?.text) {
      const excerpt = document.createElement("p");
      excerpt.className = "evidence-snippet";
      excerpt.textContent = snippet.text;
      item.append(excerpt);
    } else if (citation.quote) {
      const quote = document.createElement("p");
      quote.className = "evidence-snippet";
      quote.textContent = citation.quote;
      item.append(quote);
    }

    wrapper.append(item);
  }
  container.append(wrapper);
}

function titleFromKey(value) {
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function renderSynthesisReport(container, message) {
  const report = message.synthesis_report;
  if (!report) {
    return;
  }
  const wrapper = document.createElement("article");
  wrapper.className = "synthesis-report";

  const header = document.createElement("header");
  header.className = "synthesis-report-header";

  const title = document.createElement("h2");
  title.textContent = report.company_name || report.security_symbol || "Synthesis Report";

  const badges = document.createElement("div");
  badges.className = "synthesis-badges";
  for (const [label, value] of [
    ["Status", report.status],
    ["Coverage", report.evidence_coverage],
    ["Confidence", report.overall_confidence],
  ]) {
    if (!value) {
      continue;
    }
    const badge = document.createElement("span");
    badge.className = "synthesis-badge";
    badge.textContent = `${label}: ${value}`;
    badges.append(badge);
  }

  header.append(title, badges);
  wrapper.append(header);

  const notice = document.createElement("p");
  notice.className = "synthesis-notice";
  notice.textContent = report.no_recommendation_notice || "Not financial advice.";
  wrapper.append(notice);

  const sections = report.sections || {};
  for (const key of [
    "current_situation",
    "strengths",
    "weaknesses",
    "opportunities",
    "risks",
    "unknowns",
  ]) {
    wrapper.append(renderSynthesisSection(titleFromKey(key), sections[key] || []));
  }

  const scenarios = report.scenarios || {};
  const scenarioSection = document.createElement("section");
  scenarioSection.className = "synthesis-section";
  const scenarioTitle = document.createElement("h3");
  scenarioTitle.textContent = "Scenarios";
  scenarioSection.append(scenarioTitle);
  scenarioSection.append(renderScenario("Upside", scenarios.upside));
  scenarioSection.append(renderScenario("Downside", scenarios.downside));
  wrapper.append(scenarioSection);

  container.append(wrapper);
}

function renderTraceControl(container, message) {
  const runId = message.research_run_id || "";
  if (!runId.startsWith("orchestrator_run_")) {
    return;
  }
  const wrapper = document.createElement("article");
  wrapper.className = "trace-card";

  const header = document.createElement("header");
  header.className = "trace-card-header";

  const title = document.createElement("h2");
  title.textContent = "Run Trace";

  const button = document.createElement("button");
  button.type = "button";
  button.className = "trace-button secondary-button";
  button.textContent = state.tracesByRunId[runId]?.trace ? "Refresh" : "Inspect";
  button.addEventListener("click", () => loadRunTrace(runId));

  header.append(title, button);
  wrapper.append(header);

  const stateForRun = state.tracesByRunId[runId];
  if (stateForRun?.loading) {
    const loading = document.createElement("p");
    loading.className = "trace-muted";
    loading.textContent = "Loading trace...";
    wrapper.append(loading);
  } else if (stateForRun?.error) {
    const error = document.createElement("p");
    error.className = "trace-error";
    error.textContent = stateForRun.error;
    wrapper.append(error);
  } else if (stateForRun?.trace) {
    wrapper.append(renderTraceTimeline(stateForRun.trace));
  }

  container.append(wrapper);
}

function renderTraceTimeline(trace) {
  const timeline = document.createElement("ol");
  timeline.className = "trace-timeline";
  for (const event of trace.events || []) {
    const item = document.createElement("li");
    item.className = `trace-event ${event.status}`;

    const title = document.createElement("div");
    title.className = "trace-event-title";
    title.textContent = `${event.sequence}. ${event.title}`;

    const meta = document.createElement("small");
    meta.className = "trace-muted";
    meta.textContent = `${event.kind} / ${event.status} / ${event.duration_ms}ms`;

    item.append(title, meta);
    if (event.error_message) {
      const error = document.createElement("p");
      error.className = "trace-error";
      error.textContent = event.error_message;
      item.append(error);
    }
    const details = [];
    if (event.evidence_ids?.length) {
      details.push(`evidence: ${event.evidence_ids.length}`);
    }
    if (event.warnings?.length) {
      details.push(`warnings: ${event.warnings.length}`);
    }
    if (event.limitations?.length) {
      details.push(`limitations: ${event.limitations.length}`);
    }
    if (details.length) {
      const summary = document.createElement("p");
      summary.className = "trace-muted";
      summary.textContent = details.join(" / ");
      item.append(summary);
    }
    timeline.append(item);
  }
  return timeline;
}

function renderSynthesisSection(titleText, points) {
  const section = document.createElement("section");
  section.className = "synthesis-section";
  const title = document.createElement("h3");
  title.textContent = titleText;
  section.append(title);

  if (!points.length) {
    const empty = document.createElement("p");
    empty.className = "synthesis-empty";
    empty.textContent = "No supported points.";
    section.append(empty);
    return section;
  }

  const list = document.createElement("ul");
  for (const point of points) {
    const item = document.createElement("li");
    const heading = document.createElement("strong");
    heading.textContent = point.title || "Finding";
    const summary = document.createElement("span");
    summary.textContent = ` ${point.summary || ""}`;
    item.append(heading, summary);
    item.append(renderEvidenceMeta(point));
    list.append(item);
  }
  section.append(list);
  return section;
}

function renderScenario(titleText, scenario) {
  const item = document.createElement("div");
  item.className = "synthesis-scenario";
  const title = document.createElement("strong");
  title.textContent = titleText;
  item.append(title);
  if (!scenario) {
    const empty = document.createElement("p");
    empty.textContent = "No scenario available.";
    item.append(empty);
    return item;
  }
  for (const text of [scenario.condition, scenario.potential_development]) {
    if (!text) {
      continue;
    }
    const paragraph = document.createElement("p");
    paragraph.textContent = text;
    item.append(paragraph);
  }
  item.append(renderEvidenceMeta(scenario));
  return item;
}

function renderEvidenceMeta(item) {
  const meta = document.createElement("small");
  meta.className = "synthesis-evidence-meta";
  const evidenceIds = item.evidence_ids || [];
  const handoffIds = item.source_handoff_ids || [];
  const parts = [];
  if (item.confidence) {
    parts.push(`confidence: ${item.confidence}`);
  }
  if (evidenceIds.length) {
    parts.push(`evidence: ${evidenceIds.join(", ")}`);
  }
  if (handoffIds.length) {
    parts.push(`handoffs: ${handoffIds.length}`);
  }
  meta.textContent = parts.join(" / ");
  return meta;
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

    const content = document.createElement("div");
    content.className = "message-content";
    renderMessageContent(content, message);

    if (message.role === "user") {
      const meta = document.createElement("div");
      meta.className = "message-meta";
      meta.textContent = "user";
      item.append(meta);
    }
    item.append(content);
    renderMessageCitations(item, message);
    renderSynthesisReport(item, message);
    renderTraceControl(item, message);
    messageList.append(item);
  }
  const hasStreamingMessage = state.messages.some((message) => message.streaming);
  if (state.busy && !hasStreamingMessage) {
    messageList.append(renderLoadingIndicator());
  }
  renderContextPanel();
  messageList.scrollTop = messageList.scrollHeight;
}

function renderContextPanel() {
  const sources = contextSourcesFromMessages();
  contextSourceList.innerHTML = "";
  contextPanel.hidden = sources.length === 0;
  for (const source of sources) {
    const item = document.createElement("li");
    const link = document.createElement("a");
    link.className = "context-source-link";
    link.href = source.url;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = source.label;
    item.append(link);
    contextSourceList.append(item);
  }
}

function contextSourcesFromMessages() {
  const sources = [];
  const seen = new Set();
  for (const message of [...state.messages].reverse()) {
    for (const citation of message.citations || []) {
      const url = safeExternalUrl(citation.source_url);
      if (!url || seen.has(url)) {
        continue;
      }
      seen.add(url);
      sources.push({
        url,
        label: citation.section || citation.document_id || citation.marker || citation.id,
      });
      if (sources.length >= 8) {
        return sources;
      }
    }
  }
  return sources;
}

function renderLoadingIndicator() {
  const item = document.createElement("li");
  item.className = "loading-row";
  item.textContent = "Thinking...";
  return item;
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

async function errorMessageFromResponse(response) {
  try {
    const payload = await response.json();
    const detail = payload.detail || {};
    return detail.message || detail.error || `Request failed with ${response.status}`;
  } catch {
    return `Request failed with ${response.status}`;
  }
}

function pendingId(prefix) {
  return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function appendOptimisticExchange(content, mentions) {
  const createdAt = new Date().toISOString();
  const userId = pendingId("pending_user");
  const assistantId = pendingId("pending_assistant");
  state.messages = [
    ...state.messages,
    {
      id: userId,
      role: "user",
      content,
      created_at: createdAt,
      provider: null,
      model: null,
      research_run_id: null,
      mentions,
      citations: [],
      evidence_snippets: [],
      pending: true,
    },
    {
      id: assistantId,
      role: "assistant",
      content: "",
      created_at: createdAt,
      provider: null,
      model: null,
      research_run_id: null,
      mentions: [],
      citations: [],
      evidence_snippets: [],
      streaming: true,
    },
  ];
  renderMessages();
  return { userId, assistantId };
}

function removeOptimisticExchange(pending) {
  state.messages = state.messages.filter(
    (message) => message.id !== pending.userId && message.id !== pending.assistantId
  );
  renderMessages();
}

function appendAssistantDelta(assistantId, delta) {
  const message = state.messages.find((item) => item.id === assistantId);
  if (!message) {
    return;
  }
  message.content += delta;
  renderMessages();
}

async function readNdjsonStream(response, onEvent) {
  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("Streaming response is not readable.");
  }
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (line.trim()) {
        await onEvent(JSON.parse(line));
      }
    }
  }
  buffer += decoder.decode();
  if (buffer.trim()) {
    await onEvent(JSON.parse(buffer));
  }
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

async function sendMessage(content, mentions, assistantId) {
  state.abortController = new AbortController();
  const response = await fetch(`/api/sessions/${state.sessionId}/messages/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content, mentions }),
    signal: state.abortController.signal,
  });
  if (!response.ok) {
    throw new Error(await errorMessageFromResponse(response));
  }
  let completed = false;
  await readNdjsonStream(response, async (event) => {
    if (event.type === "delta") {
      appendAssistantDelta(assistantId, event.delta || "");
      return;
    }
    if (event.type === "error") {
      const detail = event.detail || {};
      throw new Error(detail.message || detail.error || "Chat request failed.");
    }
    if (event.type === "completed") {
      completed = true;
      state.messages = event.session.messages;
      await loadSessions();
      renderMessages();
    }
  });
  if (!completed) {
    throw new Error("Chat stream ended before the response completed.");
  }
  state.abortController = null;
}

async function sendSynthesisReport(query) {
  const response = await fetch(`/api/sessions/${state.sessionId}/synthesis-report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      refresh: true,
      context_source_items: [],
    }),
  });
  if (!response.ok) {
    throw new Error(await errorMessageFromResponse(response));
  }
  const payload = await response.json();
  state.messages = payload.session.messages;
  await loadSessions();
  renderMessages();
}

async function loadRunTrace(runId) {
  state.tracesByRunId[runId] = { loading: true };
  renderMessages();
  try {
    const payload = await requestJson(`/api/orchestrator/runs/${runId}/trace`);
    state.tracesByRunId[runId] = { trace: payload.trace };
  } catch (error) {
    state.tracesByRunId[runId] = {
      error: error instanceof Error ? error.message : "Could not load trace.",
    };
  }
  renderMessages();
}

function researchCommand(content) {
  const match = content.match(/^\/research\s+(.+)$/i);
  if (!match) {
    return null;
  }
  const query = match[1].trim();
  return query ? { query } : null;
}

input.addEventListener("input", () => {
  updateSendButtonState();
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

sendButton.addEventListener("click", (event) => {
  if (!state.busy) {
    return;
  }
  event.preventDefault();
  state.abortController?.abort();
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
  if (state.busy) {
    state.abortController?.abort();
    return;
  }
  const content = editorText();
  if (!content || state.busy || !state.sessionId) {
    return;
  }
  const mentions = editorMentions();
  clearError();
  input.innerHTML = "";
  const pending = appendOptimisticExchange(content, mentions);
  setBusy(true);
  try {
    const research = researchCommand(content);
    if (research) {
      await sendSynthesisReport(research.query);
    } else {
      await sendMessage(content, mentions, pending.assistantId);
    }
  } catch (error) {
    removeOptimisticExchange(pending);
    if (error instanceof DOMException && error.name === "AbortError") {
      showError("Response stopped.");
    } else {
      input.textContent = content;
      showError(error instanceof Error ? error.message : "Chat request failed.");
    }
  } finally {
    state.abortController = null;
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
