const state = {
  sessionId: null,
  messages: [],
  sessions: [],
  companyResults: [],
  selectedCompany: null,
  marketData: null,
  financialStatements: null,
  busy: false,
};

const form = document.querySelector("#chat-form");
const companySearchForm = document.querySelector("#company-search-form");
const companySearchInput = document.querySelector("#company-search-input");
const companySearchButton = document.querySelector("#company-search-button");
const companySearchStatus = document.querySelector("#company-search-status");
const companyResults = document.querySelector("#company-results");
const selectedCompany = document.querySelector("#selected-company");
const marketDataStatus = document.querySelector("#market-data-status");
const marketDataSummary = document.querySelector("#market-data-summary");
const financialStatementsStatus = document.querySelector("#financial-statements-status");
const financialStatementsSummary = document.querySelector("#financial-statements-summary");
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
  companySearchInput.disabled = value;
  companySearchButton.disabled = value;
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

function sourceTimestamp(candidate) {
  const retrievedAt = candidate.source?.retrieved_at;
  if (!retrievedAt) {
    return "unknown freshness";
  }
  return new Date(retrievedAt).toLocaleString();
}

function primarySecurity(candidate) {
  return candidate.securities[0] || { ticker: "unknown" };
}

function identifierValue(candidate, type) {
  return (
    candidate.company.identifiers.find((identifier) => identifier.type === type)?.value ||
    primarySecurity(candidate).identifiers?.find((identifier) => identifier.type === type)?.value ||
    null
  );
}

function renderSelectedCompany() {
  selectedCompany.innerHTML = "";
  selectedCompany.hidden = state.selectedCompany === null;
  if (!state.selectedCompany) {
    return;
  }
  const security = primarySecurity(state.selectedCompany);
  const text = document.createElement("div");
  text.textContent = `Selected: ${state.selectedCompany.company.legal_name} / ${security.ticker}`;
  const fetchButton = document.createElement("button");
  fetchButton.type = "button";
  fetchButton.className = "secondary-button fetch-market-button";
  fetchButton.textContent = "Fetch prices";
  fetchButton.addEventListener("click", () => fetchMarketData(state.selectedCompany));
  const statementButton = document.createElement("button");
  statementButton.type = "button";
  statementButton.className = "secondary-button fetch-statements-button";
  statementButton.textContent = "Fetch statements";
  statementButton.disabled = identifierValue(state.selectedCompany, "cik") === null;
  statementButton.addEventListener("click", () => fetchFinancialStatements(state.selectedCompany));
  selectedCompany.append(text, fetchButton, statementButton);
}

function renderMarketData(payload) {
  state.marketData = payload;
  marketDataSummary.innerHTML = "";
  marketDataSummary.hidden = payload === null;
  if (!payload) {
    marketDataStatus.textContent = "";
    return;
  }
  const history = payload.history;
  const latestBar = history.bars[history.bars.length - 1];
  const title = document.createElement("div");
  title.className = "market-data-title";
  title.textContent = `${history.security.symbol} latest close ${history.metrics.latest_close}`;
  const meta = document.createElement("div");
  meta.textContent = `${latestBar?.priced_at || "unknown date"} / ${history.source.provider}`;
  const metrics = document.createElement("div");
  metrics.textContent = `1d return ${history.metrics.return_1d || "n/a"} / max drawdown ${
    history.metrics.max_drawdown || "n/a"
  }`;
  marketDataSummary.append(title, meta, metrics);
  for (const warning of history.warnings || []) {
    const warningRow = document.createElement("div");
    warningRow.className = "market-data-warning";
    warningRow.textContent = warning;
    marketDataSummary.append(warningRow);
  }
  if (history.source.freshness_warning) {
    const freshness = document.createElement("div");
    freshness.className = "market-data-warning";
    freshness.textContent = history.source.freshness_warning;
    marketDataSummary.append(freshness);
  }
  marketDataStatus.textContent = payload.stored ? "Using stored prices" : "Prices fetched";
}

function statementCountByType(statements) {
  return statements.reduce((counts, statement) => {
    counts[statement.statement_type] = (counts[statement.statement_type] || 0) + 1;
    return counts;
  }, {});
}

function renderFinancialStatements(payload) {
  state.financialStatements = payload;
  financialStatementsSummary.innerHTML = "";
  financialStatementsSummary.hidden = payload === null;
  if (!payload) {
    financialStatementsStatus.textContent = "";
    return;
  }
  const result = payload.statements;
  const statements = result.statements || [];
  const latest = statements
    .slice()
    .sort((left, right) => right.period.period_end.localeCompare(left.period.period_end))[0];
  const counts = statementCountByType(statements);
  const title = document.createElement("div");
  title.className = "financial-statements-title";
  title.textContent = `${result.company.legal_name || result.company.cik} / ${
    statements.length
  } statement rows`;
  const meta = document.createElement("div");
  meta.textContent = `${latest?.period.period_end || "unknown period"} / ${result.source.provider}`;
  const types = document.createElement("div");
  types.textContent = Object.entries(counts)
    .map(([type, count]) => `${type}: ${count}`)
    .join(" / ");
  financialStatementsSummary.append(title, meta, types);
  for (const warning of result.warnings || []) {
    const warningRow = document.createElement("div");
    warningRow.className = "financial-statements-warning";
    warningRow.textContent = warning;
    financialStatementsSummary.append(warningRow);
  }
  if (result.source.freshness_warning) {
    const freshness = document.createElement("div");
    freshness.className = "financial-statements-warning";
    freshness.textContent = result.source.freshness_warning;
    financialStatementsSummary.append(freshness);
  }
  financialStatementsStatus.textContent = payload.stored
    ? "Using stored statements"
    : "Statements fetched";
}

function renderCompanyResults(result) {
  state.companyResults = result.candidates || [];
  companyResults.innerHTML = "";
  renderSelectedCompany();
  if (result.status === "no_matches") {
    companySearchStatus.textContent = "No matches";
    return;
  }
  companySearchStatus.textContent =
    state.companyResults.length === 1
      ? "Review 1 candidate"
      : `Review ${state.companyResults.length} candidates`;
  for (const candidate of state.companyResults) {
    const item = document.createElement("li");
    item.className = "company-result";
    const security = primarySecurity(candidate);

    const title = document.createElement("div");
    title.className = "company-result-title";
    title.textContent = candidate.company.legal_name;

    const meta = document.createElement("div");
    meta.className = "company-result-meta";
    meta.textContent = `${security.ticker} / ${candidate.match_reason} / ${sourceTimestamp(candidate)}`;

    const identifiers = document.createElement("div");
    identifiers.className = "company-result-identifiers";
    identifiers.textContent = candidate.company.identifiers
      .map((identifier) => `${identifier.type.toUpperCase()}: ${identifier.value}`)
      .join(" / ");

    const button = document.createElement("button");
    button.type = "button";
    button.className = "secondary-button select-company-button";
    button.textContent = "Select";
    button.addEventListener("click", () => {
      state.selectedCompany = candidate;
      renderMarketData(null);
      renderFinancialStatements(null);
      renderCompanyResults({ ...result, candidates: state.companyResults });
    });

    item.append(title, meta, identifiers, button);
    companyResults.append(item);
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

async function searchCompanies(query) {
  const payload = await requestJson(
    `/api/company-search?query=${encodeURIComponent(query)}&limit=8`
  );
  state.selectedCompany = null;
  renderMarketData(null);
  renderFinancialStatements(null);
  renderCompanyResults(payload.result);
}

async function fetchMarketData(candidate) {
  if (!candidate || state.busy) {
    return;
  }
  const security = primarySecurity(candidate);
  clearError();
  setBusy(true);
  marketDataStatus.textContent = "Fetching prices...";
  try {
    const payload = await requestJson("/api/market-data/history", {
      method: "POST",
      body: JSON.stringify({
        symbol: security.ticker,
        security_id: security.id,
        exchange_mic: security.exchange_mic,
        exchange_name: security.exchange_name,
        currency: security.currency,
        outputsize: "compact",
        refresh: true,
      }),
    });
    renderMarketData(payload);
  } catch (error) {
    marketDataStatus.textContent = "";
    showError(error instanceof Error ? error.message : "Market data request failed.");
  } finally {
    setBusy(false);
  }
}

async function fetchFinancialStatements(candidate) {
  if (!candidate || state.busy) {
    return;
  }
  const cik = identifierValue(candidate, "cik");
  if (!cik) {
    showError("Selected company does not include an SEC CIK.");
    return;
  }
  clearError();
  setBusy(true);
  financialStatementsStatus.textContent = "Fetching statements...";
  try {
    const payload = await requestJson("/api/financial-statements", {
      method: "POST",
      body: JSON.stringify({
        cik,
        company_id: candidate.company.id,
        legal_name: candidate.company.legal_name,
        fiscal_years: 3,
        refresh: true,
      }),
    });
    renderFinancialStatements(payload);
  } catch (error) {
    financialStatementsStatus.textContent = "";
    showError(error instanceof Error ? error.message : "Financial statement request failed.");
  } finally {
    setBusy(false);
  }
}

companySearchForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = companySearchInput.value.trim();
  if (!query || state.busy) {
    return;
  }
  clearError();
  setBusy(true);
  companySearchStatus.textContent = "Searching...";
  try {
    await searchCompanies(query);
  } catch (error) {
    companySearchStatus.textContent = "";
    showError(error instanceof Error ? error.message : "Company search failed.");
  } finally {
    setBusy(false);
    companySearchInput.focus();
  }
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
