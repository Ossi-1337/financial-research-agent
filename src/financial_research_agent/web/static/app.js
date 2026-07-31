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
  activeBackgroundJobId: null,
  exportsByRunId: {},
  evidenceByRunId: {},
  runsByRunId: {},
  settings: null,
  modelOptionsRequestId: 0,
};

const SVG_NAMESPACE = "http://www.w3.org/2000/svg";
const REPORT_EXPORT_CONTENT_VERSION = 3;
const STOCK_CHART_LAYOUT = Object.freeze({
  width: 720,
  height: 300,
  left: 58,
  right: 18,
  top: 18,
  bottom: 44,
});

const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const mentionMenu = document.querySelector("#mention-menu");
const sendButton = document.querySelector("#send-button");
const composerModelSelect = document.querySelector("#composer-model-select");
const messageList = document.querySelector("#message-list");
const errorBanner = document.querySelector("#error-banner");
const sessionList = document.querySelector("#session-list");
const contextPanel = document.querySelector("#context-panel");
const contextSourceList = document.querySelector("#context-source-list");
const newSessionButton = document.querySelector("#new-session-button");
const clearSessionsButton = document.querySelector("#clear-sessions-button");
const settingsButton = document.querySelector("#settings-button");
const settingsPanel = document.querySelector("#settings-panel");
const settingsCloseButton = document.querySelector("#settings-close-button");
const settingsForm = document.querySelector("#settings-form");
const settingsError = document.querySelector("#settings-error");
const settingsSessionLabel = document.querySelector("#settings-session-label");
const settingsAgentRuntimeStatus = document.querySelector("#settings-agent-runtime-status");
const settingsProviderStatus = document.querySelector("#settings-provider-status");
const settingsSecretNote = document.querySelector("#settings-secret-note");
const settingsHealthButton = document.querySelector("#settings-health-button");
const settingsCacheClearButton = document.querySelector("#settings-cache-clear-button");
const settingsResetButton = document.querySelector("#settings-reset-button");

function setBusy(value) {
  state.busy = value;
  input.contentEditable = value ? "false" : "true";
  newSessionButton.disabled = value;
  composerModelSelect.disabled = value || composerModelSelect.options.length === 0;
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

function showSettingsError(message) {
  settingsError.textContent = message;
  settingsError.hidden = false;
}

function clearSettingsError() {
  settingsError.textContent = "";
  settingsError.hidden = true;
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
  if (report.created_at) {
    const timestamp = document.createElement("time");
    timestamp.className = "synthesis-timestamp";
    timestamp.dateTime = report.created_at;
    timestamp.textContent = `Generated ${new Date(report.created_at).toLocaleString()}`;
    wrapper.append(timestamp);
  }
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
    wrapper.append(
      renderSynthesisSection(
        titleFromKey(key),
        sections[key] || [],
        message.research_run_id
      )
    );
  }

  const scenarios = report.scenarios || {};
  const scenarioSection = document.createElement("section");
  scenarioSection.className = "synthesis-section";
  const scenarioTitle = document.createElement("h3");
  scenarioTitle.textContent = "Scenarios";
  scenarioSection.append(scenarioTitle);
  scenarioSection.append(renderScenario("Upside", scenarios.upside, message.research_run_id));
  scenarioSection.append(
    renderScenario("Downside", scenarios.downside, message.research_run_id)
  );
  wrapper.append(scenarioSection);
  wrapper.append(renderStockChart(message.research_run_id));
  wrapper.append(renderRunEvidencePanel(message.research_run_id));
  wrapper.append(renderReportExportControl(message.research_run_id));

  container.append(wrapper);
}

function renderReportExportControl(runId) {
  const wrapper = document.createElement("div");
  wrapper.className = "report-export";
  if (!runId) {
    return wrapper;
  }
  const exportState = state.exportsByRunId[runId];
  const actions = document.createElement("div");
  actions.className = "report-export-actions";
  const artifacts = new Map(
    (exportState?.payload?.export?.artifacts || []).map((artifact) => [
      artifact.format,
      artifact,
    ])
  );
  for (const [format, label] of [
    ["markdown", "Markdown"],
    ["html", "HTML"],
    ["pdf", "PDF"],
  ]) {
    const url = exportState?.payload?.files?.[format];
    const link = document.createElement("a");
    link.className = "report-export-link";
    link.textContent = label;
    if (url) {
      link.href = url;
      link.download = artifacts.get(format)?.filename || "";
    } else {
      link.href = "#";
      link.setAttribute("aria-disabled", String(Boolean(exportState?.loading)));
      link.setAttribute("aria-label", `Generate and download ${label} report`);
      link.addEventListener("click", (event) => {
        event.preventDefault();
        if (!exportState?.loading) {
          createReportExport(runId, format);
        }
      });
    }
    actions.append(link);
  }
  wrapper.append(actions);
  if (exportState?.loading) {
    const status = document.createElement("span");
    status.className = "report-export-status";
    status.setAttribute("role", "status");
    status.textContent = "Preparing files";
    wrapper.append(status);
  }
  if (exportState?.error) {
    const error = document.createElement("span");
    error.className = "report-export-error";
    error.textContent = exportState.error;
    wrapper.append(error);
  }
  return wrapper;
}

function renderSynthesisSection(titleText, points, runId) {
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
    item.append(renderEvidenceMeta(point, runId));
    list.append(item);
  }
  section.append(list);
  return section;
}

function renderScenario(titleText, scenario, runId) {
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
  item.append(renderEvidenceMeta(scenario, runId));
  return item;
}

function renderEvidenceMeta(item, runId) {
  const meta = document.createElement("div");
  meta.className = "synthesis-evidence-meta";
  const evidenceIds = item.evidence_ids || [];
  const handoffIds = item.source_handoff_ids || [];
  const parts = [];
  if (item.confidence) {
    parts.push(`confidence: ${item.confidence}`);
  }
  if (handoffIds.length) {
    parts.push(`handoffs: ${handoffIds.length}`);
  }
  const label = document.createElement("small");
  label.textContent = parts.join(" / ");
  meta.append(label);
  const evidence = state.evidenceByRunId[runId];
  const markers = new Set();
  for (const evidenceId of evidenceIds) {
    for (const marker of evidence?.evidence_markers?.[evidenceId] || []) {
      markers.add(marker);
    }
  }
  for (const handoffId of handoffIds) {
    for (const marker of evidence?.handoff_markers?.[handoffId] || []) {
      markers.add(marker);
    }
  }
  for (const marker of markers) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "source-marker-button";
    button.textContent = marker;
    button.addEventListener("click", () => {
      document.querySelector(sourceElementSelector(runId, marker))?.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    });
    meta.append(button);
  }
  return meta;
}

function renderRunEvidencePanel(runId) {
  const wrapper = document.createElement("details");
  wrapper.className = "run-evidence-panel";
  if (!runId || !state.evidenceByRunId[runId]) {
    wrapper.hidden = true;
    return wrapper;
  }
  const evidence = state.evidenceByRunId[runId];
  const summary = document.createElement("summary");
  summary.textContent = `Sources (${evidence.sources?.length || 0})`;
  wrapper.append(summary);
  const list = document.createElement("ol");
  list.className = "run-evidence-list";
  for (const source of evidence.sources || []) {
    const item = document.createElement("li");
    item.id = sourceElementId(runId, source.marker);
    item.className = source.resolved ? "run-evidence-source" : "run-evidence-source unresolved";
    const header = document.createElement("div");
    header.className = "run-evidence-header";
    const marker = document.createElement("strong");
    marker.textContent = source.marker;
    const safeUrl = safeExternalUrl(source.source_url);
    const sourceName = document.createElement(safeUrl ? "a" : "span");
    sourceName.textContent = source.source_name || "Unresolved source";
    if (safeUrl) {
      sourceName.href = safeUrl;
      sourceName.target = "_blank";
      sourceName.rel = "noreferrer";
    }
    header.append(marker, sourceName);
    item.append(header);
    const metadata = [source.source_date, source.retrieved_at, source.section]
      .filter(Boolean)
      .join(" / ");
    if (metadata) {
      const meta = document.createElement("small");
      meta.textContent = metadata;
      item.append(meta);
    }
    if (source.quote) {
      const quote = document.createElement("p");
      quote.textContent = source.quote;
      item.append(quote);
    }
    const ids = document.createElement("small");
    ids.textContent = `Evidence: ${(source.evidence_ids || []).join(", ")}`;
    item.append(ids);
    list.append(item);
  }
  wrapper.append(list);
  return wrapper;
}

function renderStockChart(runId) {
  const wrapper = document.createElement("section");
  wrapper.className = "stock-chart";
  const run = state.runsByRunId[runId];
  const handoff = run?.handoffs?.find((item) => item.kind === "stock_price_analysis");
  const series = handoff?.output?.analysis?.chart_series || [];
  if (!series.length) {
    wrapper.hidden = true;
    return wrapper;
  }
  const title = document.createElement("h3");
  title.textContent = "Indexed Price Development";
  const note = document.createElement("small");
  const stage = document.createElement("div");
  stage.className = "stock-chart-stage";
  const svg = createSvgElement("svg");
  svg.setAttribute(
    "viewBox",
    `0 0 ${STOCK_CHART_LAYOUT.width} ${STOCK_CHART_LAYOUT.height}`,
  );
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Indexed historical price chart");
  svg.setAttribute("tabindex", "0");
  const normalized = normalizeAlignedChartSeries(series);
  if (!normalized.length) {
    wrapper.hidden = true;
    return wrapper;
  }
  note.textContent =
    `Shared period: ${normalized[0].dates[0]} to ${normalized[0].dates.at(-1)}. ` +
    "Each series starts at 100. Historical data is not a forecast.";
  const values = normalized.flatMap((item) => item.values);
  const axis = chartAxis(Math.min(...values), Math.max(...values));
  renderChartAxes(svg, normalized[0].dates, axis);
  normalized.forEach((item, index) => {
    const polyline = createSvgElement("polyline");
    polyline.setAttribute("class", `stock-chart-line series-${index}`);
    polyline.setAttribute("points", chartPoints(item.values, axis.min, axis.max));
    svg.append(polyline);
  });
  const tooltip = document.createElement("div");
  tooltip.className = "stock-chart-tooltip";
  tooltip.hidden = true;
  tooltip.setAttribute("role", "status");
  tooltip.setAttribute("aria-live", "polite");
  attachChartInteraction(svg, tooltip, normalized, axis);
  stage.append(svg, tooltip);
  const legend = document.createElement("div");
  legend.className = "stock-chart-legend";
  normalized.forEach((item, index) => {
    const label = document.createElement("span");
    label.className = `series-${index}`;
    label.textContent = `${item.symbol}: ${item.values.at(-1).toFixed(1)}`;
    legend.append(label);
  });
  wrapper.append(title, note, stage, legend);
  return wrapper;
}

function normalizeAlignedChartSeries(seriesList) {
  const prepared = seriesList
    .map((series) => {
      const points = new Map();
      for (const point of series.points || []) {
        const value = Number(point.adjusted_close || point.close);
        if (point.priced_at && Number.isFinite(value) && value > 0) {
          points.set(point.priced_at, value);
        }
      }
      return { symbol: series.symbol, points };
    })
    .filter((series) => series.points.size);
  if (prepared.length !== seriesList.length) {
    return [];
  }
  const dates = [...prepared[0].points.keys()]
    .filter((date) => prepared.every((series) => series.points.has(date)))
    .sort();
  if (dates.length < 2) {
    return [];
  }
  return prepared.map((series) => {
    const base = series.points.get(dates[0]);
    return {
      symbol: series.symbol,
      dates,
      values: dates.map((date) => (series.points.get(date) / base) * 100),
    };
  });
}

function chartPoints(values, minValue, maxValue) {
  return values
    .map((value, index) => {
      const x = chartX(index, values.length);
      const y = chartY(value, minValue, maxValue);
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

function chartAxis(minValue, maxValue) {
  const rawRange = Math.max(maxValue - minValue, 1);
  const padding = Math.max(rawRange * 0.08, 0.5);
  const step = niceChartStep(rawRange + padding * 2, 4);
  const min = Math.floor((minValue - padding) / step) * step;
  const max = Math.ceil((maxValue + padding) / step) * step;
  const ticks = [];
  for (let value = min; value <= max + step / 2; value += step) {
    ticks.push(Number(value.toFixed(8)));
  }
  return { min, max, step, ticks };
}

function niceChartStep(range, targetIntervals) {
  const roughStep = range / targetIntervals;
  const magnitude = 10 ** Math.floor(Math.log10(roughStep));
  const normalized = roughStep / magnitude;
  const multiplier = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  return multiplier * magnitude;
}

function renderChartAxes(svg, dates, axis) {
  const grid = createSvgElement("g");
  grid.setAttribute("class", "stock-chart-grid");
  for (const value of axis.ticks) {
    const y = chartY(value, axis.min, axis.max);
    const line = createSvgElement("line");
    line.setAttribute("x1", STOCK_CHART_LAYOUT.left);
    line.setAttribute("x2", STOCK_CHART_LAYOUT.width - STOCK_CHART_LAYOUT.right);
    line.setAttribute("y1", y);
    line.setAttribute("y2", y);
    const label = createSvgElement("text");
    label.setAttribute("class", "stock-chart-axis-label");
    label.setAttribute("x", STOCK_CHART_LAYOUT.left - 9);
    label.setAttribute("y", y + 4);
    label.setAttribute("text-anchor", "end");
    label.textContent = formatAxisValue(value, axis.step);
    grid.append(line, label);
  }
  for (const index of evenlySpacedIndices(dates.length, 5)) {
    const x = chartX(index, dates.length);
    const line = createSvgElement("line");
    line.setAttribute("x1", x);
    line.setAttribute("x2", x);
    line.setAttribute("y1", STOCK_CHART_LAYOUT.top);
    line.setAttribute("y2", STOCK_CHART_LAYOUT.height - STOCK_CHART_LAYOUT.bottom);
    const label = createSvgElement("text");
    label.setAttribute("class", "stock-chart-axis-label");
    label.setAttribute("x", x);
    label.setAttribute("y", STOCK_CHART_LAYOUT.height - 15);
    label.setAttribute("text-anchor", "middle");
    label.textContent = formatChartDate(dates[index]);
    grid.append(line, label);
  }
  svg.append(grid);
}

function attachChartInteraction(svg, tooltip, series, axis) {
  const crosshair = createSvgElement("line");
  crosshair.setAttribute("class", "stock-chart-crosshair");
  crosshair.setAttribute("visibility", "hidden");
  const markers = series.map((_, index) => {
    const marker = createSvgElement("circle");
    marker.setAttribute("class", `stock-chart-marker series-${index}`);
    marker.setAttribute("r", "5");
    marker.setAttribute("visibility", "hidden");
    return marker;
  });
  const overlay = createSvgElement("rect");
  overlay.setAttribute("class", "stock-chart-overlay");
  overlay.setAttribute("x", STOCK_CHART_LAYOUT.left);
  overlay.setAttribute("y", STOCK_CHART_LAYOUT.top);
  overlay.setAttribute(
    "width",
    STOCK_CHART_LAYOUT.width - STOCK_CHART_LAYOUT.left - STOCK_CHART_LAYOUT.right,
  );
  overlay.setAttribute(
    "height",
    STOCK_CHART_LAYOUT.height - STOCK_CHART_LAYOUT.top - STOCK_CHART_LAYOUT.bottom,
  );
  svg.append(crosshair, ...markers, overlay);

  let activeIndex = series[0].dates.length - 1;
  const showPoint = (index) => {
    activeIndex = Math.max(0, Math.min(index, series[0].dates.length - 1));
    const x = chartX(activeIndex, series[0].dates.length);
    crosshair.setAttribute("x1", x);
    crosshair.setAttribute("x2", x);
    crosshair.setAttribute("y1", STOCK_CHART_LAYOUT.top);
    crosshair.setAttribute("y2", STOCK_CHART_LAYOUT.height - STOCK_CHART_LAYOUT.bottom);
    crosshair.setAttribute("visibility", "visible");
    markers.forEach((marker, markerIndex) => {
      marker.setAttribute("cx", x);
      marker.setAttribute("cy", chartY(series[markerIndex].values[activeIndex], axis.min, axis.max));
      marker.setAttribute("visibility", "visible");
    });
    renderChartTooltip(tooltip, series, activeIndex);
    tooltip.style.left = `${(x / STOCK_CHART_LAYOUT.width) * 100}%`;
    tooltip.classList.toggle("align-right", x > STOCK_CHART_LAYOUT.width * 0.5);
    tooltip.hidden = false;
  };
  const hidePoint = () => {
    crosshair.setAttribute("visibility", "hidden");
    markers.forEach((marker) => {
      marker.setAttribute("visibility", "hidden");
    });
    tooltip.hidden = true;
  };
  overlay.addEventListener("pointermove", (event) => {
    const bounds = svg.getBoundingClientRect();
    const svgX = ((event.clientX - bounds.left) / bounds.width) * STOCK_CHART_LAYOUT.width;
    const plotWidth = STOCK_CHART_LAYOUT.width - STOCK_CHART_LAYOUT.left - STOCK_CHART_LAYOUT.right;
    const ratio = (svgX - STOCK_CHART_LAYOUT.left) / plotWidth;
    showPoint(Math.round(ratio * (series[0].dates.length - 1)));
  });
  overlay.addEventListener("pointerleave", hidePoint);
  svg.addEventListener("focus", () => showPoint(activeIndex));
  svg.addEventListener("blur", hidePoint);
  svg.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") {
      return;
    }
    event.preventDefault();
    showPoint(activeIndex + (event.key === "ArrowLeft" ? -1 : 1));
  });
}

function renderChartTooltip(tooltip, series, index) {
  const date = document.createElement("strong");
  date.textContent = series[0].dates[index];
  const values = series.map((item, seriesIndex) => {
    const row = document.createElement("span");
    row.className = "stock-chart-tooltip-value";
    const swatch = document.createElement("i");
    swatch.className = `series-${seriesIndex}`;
    const text = document.createElement("span");
    text.textContent = `${item.symbol}: ${item.values[index].toFixed(2)}`;
    row.append(swatch, text);
    return row;
  });
  tooltip.replaceChildren(date, ...values);
}

function chartX(index, count) {
  const plotWidth = STOCK_CHART_LAYOUT.width - STOCK_CHART_LAYOUT.left - STOCK_CHART_LAYOUT.right;
  return STOCK_CHART_LAYOUT.left + (index / Math.max(count - 1, 1)) * plotWidth;
}

function chartY(value, minValue, maxValue) {
  const plotHeight = STOCK_CHART_LAYOUT.height - STOCK_CHART_LAYOUT.top - STOCK_CHART_LAYOUT.bottom;
  const range = Math.max(maxValue - minValue, 1);
  return STOCK_CHART_LAYOUT.top + ((maxValue - value) / range) * plotHeight;
}

function evenlySpacedIndices(length, count) {
  const indices = new Set();
  for (let index = 0; index < Math.min(length, count); index += 1) {
    indices.add(Math.round((index / Math.max(Math.min(length, count) - 1, 1)) * (length - 1)));
  }
  return [...indices];
}

function formatChartDate(value) {
  const date = new Date(`${value}T00:00:00Z`);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("en", {
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  }).format(date);
}

function formatAxisValue(value, step) {
  return value.toFixed(step < 1 ? 1 : 0);
}

function createSvgElement(name) {
  return document.createElementNS(SVG_NAMESPACE, name);
}

function sourceElementId(runId, marker) {
  return `source-${String(runId).replace(/[^a-zA-Z0-9_-]/g, "-")}-${marker.replace(/\D/g, "")}`;
}

function sourceElementSelector(runId, marker) {
  return `#${sourceElementId(runId, marker)}`;
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

    item.append(content);
    renderMessageCitations(item, message);
    renderSynthesisReport(item, message);
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

async function loadSettingsPanel() {
  clearSettingsError();
  const payload = await requestJson("/api/settings");
  state.settings = payload;
  populateSettingsForm(payload);
  renderSettingsSummary(payload);
  await refreshProviderModels();
}

function populateSettingsForm(payload) {
  const provider = payload.settings.provider;
  const retrieval = payload.settings.retrieval;
  const dataSources = payload.settings.data_sources;
  const background = payload.settings.background;
  setField("llm_provider", provider.llm_provider);
  setModelOptions([provider.llm_model], provider.llm_model);
  setField("llm_base_url", provider.llm_base_url || "");
  setField("llm_local_runtime", provider.llm_local_runtime);
  setField("llm_timeout_seconds", provider.llm_timeout_seconds);
  setField("embedding_provider", provider.embedding_provider);
  setField("embedding_model", provider.embedding_model || "");
  setField("retrieval_top_k", retrieval.top_k);
  setField("retrieval_min_score", retrieval.min_score);
  setField("market_data_cache_ttl_days", dataSources.market_data_cache_ttl_days);
  setField("filing_cache_ttl_days", dataSources.filing_cache_ttl_days);
  setField("background_max_concurrent_research_runs", background.max_concurrent_research_runs);
}

function setField(name, value) {
  const field = settingsForm.elements[name];
  if (field) {
    field.value = value ?? "";
  }
}

function replaceModelOptions(field, models, selectedModel, placeholder) {
  const normalized = [...new Set(models.filter((model) => typeof model === "string" && model))];
  field.replaceChildren();
  if (normalized.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = placeholder;
    field.append(option);
    field.disabled = true;
    field.removeAttribute("title");
    return normalized;
  }
  for (const model of normalized) {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    field.append(option);
  }
  field.disabled = false;
  field.value = normalized.includes(selectedModel) ? selectedModel : normalized[0];
  field.title = field.value;
  return normalized;
}

function setModelOptions(
  models,
  selectedModel,
  placeholder = "No models available",
  { syncComposer = true } = {}
) {
  const normalized = replaceModelOptions(
    settingsForm.elements.llm_model,
    models,
    selectedModel,
    placeholder
  );
  if (syncComposer) {
    replaceModelOptions(composerModelSelect, normalized, selectedModel, placeholder);
    composerModelSelect.disabled = state.busy || normalized.length === 0;
  }
}

function compatibleConfiguredModel(provider) {
  const configured = state.settings?.settings.provider;
  if (!configured || configured.llm_provider !== provider) {
    return provider === "offline-test" ? "offline-test" : null;
  }
  if (provider !== "offline-test" && configured.llm_model === "offline-test") {
    return null;
  }
  return configured.llm_model;
}

async function refreshProviderModels({ syncComposer = true } = {}) {
  const provider = settingsForm.elements.llm_provider.value;
  const configuredModel = compatibleConfiguredModel(provider);
  const requestId = ++state.modelOptionsRequestId;
  if (provider === "offline-test") {
    setModelOptions(["offline-test"], "offline-test", "No models available", { syncComposer });
    settingsProviderStatus.textContent = "offline-test health: ok / deterministic test responses";
    return;
  }

  setModelOptions([], null, "Loading models...", { syncComposer });
  try {
    const payload = await requestJson(
      `/api/settings/provider-health?provider=${encodeURIComponent(provider)}`
    );
    if (requestId !== state.modelOptionsRequestId) {
      return;
    }
    const health = payload.provider_health;
    const models = [...(health.available_models || [])];
    if (configuredModel && !models.includes(configuredModel)) {
      models.push(configuredModel);
    }
    setModelOptions(models, configuredModel, "No models available", { syncComposer });
    settingsProviderStatus.textContent = `${health.provider} health: ${health.status}${
      health.error ? ` / ${health.error}` : ""
    }`;
  } catch (error) {
    if (requestId !== state.modelOptionsRequestId) {
      return;
    }
    setModelOptions(
      configuredModel ? [configuredModel] : [],
      configuredModel,
      "Provider unavailable",
      { syncComposer }
    );
    settingsProviderStatus.textContent =
      error instanceof Error ? error.message : "Provider health check failed.";
  }
}

function renderSettingsSummary(payload) {
  const activeProvider = payload.settings.provider.llm_provider;
  const agentRuntime = payload.research_agent_runtime;
  const provider = payload.providers.find((item) => item.provider === activeProvider);
  const capabilityStatus = provider?.capability_status || {};
  const capabilityRows = ["chat", "streaming", "tool_calls", "structured_output", "embeddings"]
    .map((name) => `${name}: ${capabilityStatus[name] ? "available" : "limited"}`)
    .join(" / ");
  settingsProviderStatus.textContent = provider
    ? `${provider.provider} capabilities: ${capabilityRows}`
    : "Selected provider is not registered.";
  settingsAgentRuntimeStatus.textContent = agentRuntime
    ? `Research agent runtime: ${agentRuntime.provider || "unavailable"} / ${
        agentRuntime.model || "unavailable"
      } (${agentRuntime.compatible ? "ready" : agentRuntime.error_code || "unavailable"})`
    : "Research agent runtime is unavailable.";
  settingsSecretNote.textContent = payload.secrets.message;
}

function settingsPayloadFromForm() {
  const formData = new FormData(settingsForm);
  const payload = {};
  for (const [key, value] of formData.entries()) {
    const text = String(value).trim();
    if (!text) {
      continue;
    }
    if (
      [
        "llm_timeout_seconds",
        "retrieval_min_score",
      ].includes(key)
    ) {
      payload[key] = Number(text);
    } else if (
      [
        "retrieval_top_k",
        "market_data_cache_ttl_days",
        "filing_cache_ttl_days",
        "background_max_concurrent_research_runs",
      ].includes(key)
    ) {
      payload[key] = Number.parseInt(text, 10);
    } else {
      payload[key] = text;
    }
  }
  return payload;
}

function updateAssistantContent(assistantId, content) {
  const message = state.messages.find((item) => item.id === assistantId);
  if (!message) {
    return;
  }
  message.content = content;
  message.streaming = false;
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

async function createSession() {
  const payload = await requestJson("/api/sessions", { method: "POST" });
  state.sessionId = payload.session.id;
  state.messages = payload.session.messages;
  settingsSessionLabel.textContent = payload.session.id;
  await loadSessions();
  renderMessages();
}

async function loadSessions() {
  const payload = await requestJson("/api/sessions");
  state.sessions = payload.sessions;
  renderSessions();
}

async function openSession(sessionId, allowBusy = false) {
  if (!allowBusy && (state.busy || sessionId === state.sessionId)) {
    return;
  }
  clearError();
  setBusy(true);
  try {
    const payload = await requestJson(`/api/sessions/${sessionId}`);
    state.sessionId = payload.session.id;
    state.messages = payload.session.messages;
    settingsSessionLabel.textContent = payload.session.id;
    await loadRunArtifactsForMessages();
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
      return;
    }
    if (event.type === "research") {
      completed = true;
      const job = event.job;
      updateAssistantContent(assistantId, backgroundJobText(job));
      await pollBackgroundResearchJob(job.id, assistantId);
    }
  });
  if (!completed) {
    throw new Error("Chat stream ended before the response completed.");
  }
  state.abortController = null;
}

async function pollBackgroundResearchJob(jobId, assistantId) {
  state.activeBackgroundJobId = jobId;
  while (state.activeBackgroundJobId === jobId) {
    const payload = await requestJson(`/api/background/research-runs/${jobId}`);
    const job = payload.job;
    updateAssistantContent(assistantId, backgroundJobText(job));
    if (["succeeded", "failed", "cancelled"].includes(job.status)) {
      state.activeBackgroundJobId = null;
      if (job.status === "succeeded") {
        await openSession(state.sessionId, true);
        return;
      }
      throw new Error(job.error_message || `Research run ${job.status}.`);
    }
    await sleep(750);
  }
}

async function cancelBackgroundResearchJob() {
  if (!state.activeBackgroundJobId) {
    return;
  }
  await requestJson(`/api/background/research-runs/${state.activeBackgroundJobId}/cancel`, {
    method: "POST",
  });
}

function backgroundJobText(job) {
  const progress = job.progress || {};
  const completed = progress.completed_steps ?? 0;
  const total = progress.total_steps ?? 0;
  const current = progress.current_step ? ` Current step: ${progress.current_step}.` : "";
  if (job.status === "queued") {
    return `Research run queued. ${completed}/${total} steps complete.`;
  }
  if (job.status === "running") {
    return `Research run running. ${completed}/${total} steps complete.${current}`;
  }
  if (job.status === "succeeded") {
    return "Research run completed. Loading results...";
  }
  if (job.status === "cancelled") {
    return "Research run cancelled. Partial results were preserved when available.";
  }
  return job.error_message || "Research run failed.";
}

function sleep(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function createReportExport(runId, requestedFormat = null) {
  state.exportsByRunId[runId] = { loading: true };
  renderMessages();
  try {
    const payload = await requestJson(`/api/orchestrator/runs/${runId}/exports`, {
      method: "POST",
    });
    state.exportsByRunId[runId] = { payload };
    if (requestedFormat) {
      downloadReportExport(payload, requestedFormat);
    }
  } catch (error) {
    state.exportsByRunId[runId] = {
      error: error instanceof Error ? error.message : "Could not export report.",
    };
  }
  renderMessages();
}

function downloadReportExport(payload, format) {
  const url = payload.files?.[format];
  if (!url) {
    return;
  }
  const artifact = (payload.export?.artifacts || []).find((item) => item.format === format);
  const link = document.createElement("a");
  link.href = url;
  link.download = artifact?.filename || "";
  document.body.append(link);
  link.click();
  link.remove();
}

async function loadRunArtifactsForMessages() {
  const runIds = [
    ...new Set(
      state.messages
        .map((message) => message.research_run_id)
        .filter((runId) => String(runId || "").startsWith("orchestrator_run_"))
    ),
  ];
  await Promise.all(runIds.map((runId) => loadRunArtifacts(runId)));
}

async function loadRunArtifacts(runId) {
  try {
    const [runPayload, evidencePayload, exportsPayload] = await Promise.all([
      requestJson(`/api/orchestrator/runs/${runId}`),
      requestJson(`/api/orchestrator/runs/${runId}/evidence`),
      requestJson("/api/report-exports"),
    ]);
    state.runsByRunId[runId] = runPayload.run;
    state.evidenceByRunId[runId] = evidencePayload.evidence;
    const existing = (exportsPayload.exports || []).find(
      (item) =>
        item.export?.run_id === runId &&
        item.export?.content_version === REPORT_EXPORT_CONTENT_VERSION
    );
    if (existing) {
      state.exportsByRunId[runId] = { payload: existing };
    }
  } catch (error) {
    state.evidenceByRunId[runId] = {
      error: error instanceof Error ? error.message : "Could not load run evidence.",
      sources: [],
    };
  }
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
  if (state.abortController) {
    state.abortController.abort();
    return;
  }
  cancelBackgroundResearchJob().catch((error) => {
    showError(error instanceof Error ? error.message : "Could not cancel research run.");
  });
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

settingsButton.addEventListener("click", async () => {
  settingsPanel.hidden = false;
  try {
    await loadSettingsPanel();
  } catch (error) {
    showSettingsError(error instanceof Error ? error.message : "Could not load settings.");
  }
});

settingsCloseButton.addEventListener("click", () => {
  settingsPanel.hidden = true;
});

settingsForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearSettingsError();
  try {
    const payload = await requestJson("/api/settings", {
      method: "PUT",
      body: JSON.stringify(settingsPayloadFromForm()),
    });
    state.settings = payload;
    populateSettingsForm(payload);
    renderSettingsSummary(payload);
    await refreshProviderModels();
  } catch (error) {
    showSettingsError(error instanceof Error ? error.message : "Could not save settings.");
  }
});

settingsHealthButton.addEventListener("click", async () => {
  clearSettingsError();
  try {
    await refreshProviderModels({ syncComposer: false });
  } catch (error) {
    showSettingsError(error instanceof Error ? error.message : "Provider health check failed.");
  }
});

settingsCacheClearButton.addEventListener("click", async () => {
  clearSettingsError();
  try {
    await requestJson("/api/storage/cache", { method: "DELETE" });
    settingsProviderStatus.textContent = "Clearable local provider caches were cleared.";
  } catch (error) {
    showSettingsError(error instanceof Error ? error.message : "Could not clear cache.");
  }
});

settingsResetButton.addEventListener("click", async () => {
  clearSettingsError();
  try {
    const payload = await requestJson("/api/settings", { method: "DELETE" });
    state.settings = payload;
    populateSettingsForm(payload);
    renderSettingsSummary(payload);
    await refreshProviderModels();
  } catch (error) {
    showSettingsError(error instanceof Error ? error.message : "Could not reset settings.");
  }
});

settingsForm.elements.llm_provider.addEventListener("change", () => {
  clearSettingsError();
  refreshProviderModels({ syncComposer: false }).catch((error) => {
    showSettingsError(error instanceof Error ? error.message : "Could not load provider models.");
  });
});

settingsForm.elements.llm_model.addEventListener("change", (event) => {
  event.currentTarget.title = event.currentTarget.value;
});

composerModelSelect.addEventListener("change", async (event) => {
  const selectedModel = event.currentTarget.value;
  if (!selectedModel || state.busy) {
    return;
  }
  clearError();
  event.currentTarget.disabled = true;
  try {
    const payload = await requestJson("/api/settings", {
      method: "PUT",
      body: JSON.stringify({ llm_model: selectedModel }),
    });
    state.settings = payload;
    populateSettingsForm(payload);
    renderSettingsSummary(payload);
    await refreshProviderModels();
  } catch (error) {
    showError(error instanceof Error ? error.message : "Could not change model.");
    await loadSettingsPanel();
  } finally {
    event.currentTarget.disabled = state.busy || event.currentTarget.options.length === 0;
  }
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
    await sendMessage(content, mentions, pending.assistantId);
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
    state.activeBackgroundJobId = null;
    setBusy(false);
    input.focus();
  }
});

async function start() {
  setBusy(true);
  try {
    await loadSettingsPanel();
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
