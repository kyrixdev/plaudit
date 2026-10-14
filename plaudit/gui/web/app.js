"use strict";

/* Plaudit settings window. Talks to Python through window.pywebview.api; when opened in a
   plain browser (design preview) it falls back to an in-memory mock. */

const $ = (selector) => document.querySelector(selector);
const WORDS = { 2: "twice", 3: "3 times", 4: "4 times", 5: "5 times" };
const PAGE_TITLES = { patterns: "Clap Patterns", listening: "Listening", activity: "Activity", about: "About" };

let api = null;
let catalog = null;
let state = null;
let editing = null; // count being edited in the sheet

// ---------------------------------------------------------------------------- startup

function mockApi() {
  const patterns = {
    2: { kind: "media", value: "play_pause", summary: "Play/pause", title: "Media key" },
    3: { kind: "media", value: "next", summary: "Next track", title: "Media key" },
    4: { kind: "screenshot", value: "full", summary: "Screenshot to clipboard", title: "Screenshot" },
    5: { kind: "url", value: "https://example.com", summary: "Open website: https://example.com", title: "Open website" },
  };
  const options = { maxGap: 0.7, ignoreWhileTyping: true, notifyActions: false, startWithWindows: false, paused: false };
  const kinds = [
    [["none", "Feedback only", ""]],
    [["media", "Media key", "which key", [["play_pause", "Play/pause"], ["next", "Next track"], ["previous", "Previous track"], ["volume_up", "Volume up"], ["volume_down", "Volume down"], ["mute", "Mute"]]],
     ["hotkey", "Keyboard shortcut", "keys, e.g. win+d, ctrl+alt+t, alt+tab"]],
    [["open", "Open app, file or folder", "path, app name or URI"], ["url", "Open website", "web address"],
     ["screenshot", "Screenshot", "", [["full", "Whole screen to clipboard"], ["selection", "Select an area (Snipping overlay)"]]]],
    [["sound", "Play Windows sound", "", [["default", "Default"], ["asterisk", "Asterisk"]]], ["speak_text", "Speak text aloud", "text"], ["copy_text", "Copy text to clipboard", "text"]],
    [["command", "Run PowerShell command", "command (runs hidden, with your permissions)"]],
  ];
  const titleOf = (key) => kinds.flat().find((k) => k[0] === key)[1];
  return {
    async get_catalog() {
      return {
        counts: [2, 3, 4, 5], gap: { min: 0.4, max: 1.2, default: 0.7 },
        groups: kinds.map((group) => group.map(([key, title, prompt, choices]) => ({
          key, title, prompt, placeholder: "", choices: (choices || []).map(([value, label]) => ({ value, label })),
        }))),
      };
    },
    async get_state() {
      return {
        patterns: [2, 3, 4, 5].map((count) => ({ count, ...patterns[count] })),
        options: { ...options },
        tray: { running: true, state: options.paused ? "paused" : "listening", text: options.paused ? "Paused - microphone off" : "Listening",
                lastAction: "2× Play/pause (12:41)" },
        mic: { name: "Microphone Array (Intel Smart Sound Technology for Digital Microphones)", mode: "Raw (Windows mic effects off)" },
      };
    },
    async get_activity() {
      const now = Date.now();
      return [
        [2, "2×", "Play/pause", "ran"], [3, "3×", "Next track", "ran"], [1, "1× - ignored (single sounds do nothing)", "", "ignored"],
        [4, "4×", "Screenshot to clipboard", "ran"], [3, "3× - ignored (keyboard/mouse in use)", "", "ignored"],
      ].map(([count, outcome, action, status], i) => ({ time: new Date(now - i * 95000).toISOString(), count, outcome, action, status }));
    },
    async save_pattern(count, kind, value) {
      if (kind !== "none" && kind !== "screenshot" && !value) return { ok: false, error: "Enter a value." };
      const summary = kind === "none" ? "No action" : `${titleOf(kind)}: ${value}`;
      patterns[count] = { kind, value, summary, title: titleOf(kind) };
      return { ok: true, pattern: { count, ...patterns[count] } };
    },
    async try_action() { return { ok: true, message: "Ran (preview)" }; },
    async set_option(name, value) { options[name] = value; return { ok: true }; },
    async start_listening() { options.paused = false; return { ok: true }; },
    async browse() { return "C:\\Program Files\\Example\\app.exe"; },
    async open_logs() {},
  };
}

async function boot() {
  api = window.pywebview ? window.pywebview.api : mockApi();
  catalog = await api.get_catalog();
  buildKindMenu();
  bindEvents();
  await refresh();
  setInterval(refreshStatus, 2000);
}

window.addEventListener("pywebviewready", () => { if (!api) boot(); });
setTimeout(() => { if (!api && !window.pywebview) boot(); }, 600);

// ---------------------------------------------------------------------------- rendering

async function refresh() {
  state = await api.get_state();
  renderPatterns();
  renderStatus();
  renderOptions();
}

async function refreshStatus() {
  if (document.hidden) return;
  const latest = await api.get_state();
  state.tray = latest.tray;
  state.options.paused = latest.options.paused;
  renderStatus();
  if (!$("#page-activity").hidden) renderActivity();
}

function icon(id, cls = "") {
  return `<svg class="${cls}" aria-hidden="true"><use href="#${id}"/></svg>`;
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderPatterns() {
  $("#pattern-list").innerHTML = state.patterns.map((p) => {
    const empty = p.kind === "none";
    return `
      <button class="row pattern-row" type="button" data-count="${p.count}" aria-label="Clap ${WORDS[p.count]}: ${escapeHtml(p.summary)}. Edit">
        <span class="count-badge ${empty ? "empty" : ""}" aria-hidden="true">${p.count}</span>
        <span class="pattern-text">
          <span class="pattern-title">Clap ${WORDS[p.count]}</span>
          <span class="pattern-sub">${icon(`k-${p.kind}`)}<span>${escapeHtml(p.summary)}</span></span>
        </span>
        ${icon("i-chevron", "chevron")}
      </button>`;
  }).join("");
}

function renderStatus() {
  const { tray, options } = state;
  let kind, title, sub, button;
  if (!tray.running) {
    kind = "off"; title = "Plaudit isn't running"; sub = "Start it to listen for claps in the background."; button = "Start";
  } else if (options.paused || tray.state === "paused") {
    kind = "paused"; title = "Paused"; sub = "The microphone is off."; button = "Resume";
  } else if (tray.state === "error") {
    kind = "error"; title = "Not listening"; sub = tray.text; button = "Pause";
  } else if (tray.state === "starting") {
    kind = "paused"; title = "Starting..."; sub = "Learning the room's background noise."; button = "Pause";
  } else {
    kind = "on"; title = "Listening"; sub = tray.lastAction ? `Last: ${tray.lastAction}` : "Clap 2 to 5 times to run an action."; button = "Pause";
  }
  $("#status-icon").className = `status-icon ${kind}`;
  $("#status-title").textContent = title;
  $("#status-sub").textContent = sub;
  $("#status-button").textContent = button;
  $("#status-button").dataset.action = button.toLowerCase();
  $("#side-dot").className = `dot ${kind}`;
  $("#side-status").textContent = title;
}

function renderOptions() {
  const { options, mic } = state;
  const gap = $("#gap");
  gap.min = catalog.gap.min; gap.max = catalog.gap.max;
  gap.value = options.maxGap;
  updateGapLabel();
  $("#opt-typing").checked = options.ignoreWhileTyping;
  $("#opt-startup").checked = options.startWithWindows;
  $("#opt-notify").checked = options.notifyActions;
  $("#mic-name").textContent = mic.name;
  $("#mic-name").title = mic.name;
  $("#mic-mode").textContent = mic.mode;
}

function updateGapLabel() {
  const gap = $("#gap");
  const fill = ((gap.value - gap.min) / (gap.max - gap.min)) * 100;
  gap.style.setProperty("--fill", `${fill}%`);
  $("#gap-value").textContent = `${Number(gap.value).toFixed(2)} s`;
}

async function renderActivity() {
  const entries = await api.get_activity(60);
  const list = $("#activity-list");
  if (!entries.length) {
    list.innerHTML = `<div class="empty-state"><strong>No claps yet</strong>Patterns Plaudit hears will appear here.</div>`;
    return;
  }
  const glyph = { ran: "i-check", ignored: "i-minus", failed: "i-x" };
  const label = { ran: "Ran", ignored: "Ignored", failed: "Failed" };
  list.innerHTML = entries.map((e) => {
    const when = new Date(e.time);
    const reason = e.outcome.split(" - ")[1] || "ignored";
    const detail = e.status === "ignored" ? reason.charAt(0).toUpperCase() + reason.slice(1) : e.action;
    return `
      <div class="row activity-row">
        <span class="state-icon state-${e.status}" title="${label[e.status]}">${icon(glyph[e.status])}</span>
        <span class="pattern-text">
          <span class="pattern-title">${e.count === 1 ? "1 clap" : `${e.count} claps`} <span class="sr-only">${label[e.status]}</span></span>
          <span class="pattern-sub"><span>${escapeHtml(detail)}</span></span>
        </span>
        <span class="when">${when.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</span>
      </div>`;
  }).join("");
}

// ---------------------------------------------------------------------------- the edit sheet

function buildKindMenu() {
  const select = $("#kind");
  select.innerHTML = catalog.groups.map((group, index) =>
    (index ? `<option disabled>──────────</option>` : "") +
    group.map((k) => `<option value="${k.key}">${escapeHtml(k.title)}</option>`).join("")).join("");
}

function kindInfo(key) {
  return catalog.groups.flat().find((k) => k.key === key);
}

function openSheet(count) {
  editing = count;
  const pattern = state.patterns.find((p) => p.count === count);
  $("#sheet-title").textContent = `Clap ${WORDS[count]}`;
  $("#sheet-badge").textContent = count;
  $("#kind").value = pattern.kind;
  configureValue(pattern.kind, pattern.value);
  hideError();
  $("#scrim").hidden = false;
  $("#sheet").hidden = false;
  $("#kind").focus();
}

function closeSheet() {
  $("#sheet").hidden = true;
  $("#scrim").hidden = true;
  const row = document.querySelector(`.pattern-row[data-count="${editing}"]`);
  editing = null;
  if (row) row.focus();
}

function configureValue(kindKey, value = "") {
  const kind = kindInfo(kindKey);
  const row = $("#value-row");
  const hasChoices = kind.choices.length > 0;
  row.hidden = kindKey === "none";
  $("#value-popup").hidden = !hasChoices;
  $("#value-input-wrap").hidden = hasChoices;
  $("#browse").hidden = kindKey !== "open";
  $("#value-label").setAttribute("for", hasChoices ? "value-select" : "value-input");
  $("#value-label").textContent = { media: "Key", screenshot: "Capture", sound: "Sound", hotkey: "Keys", open: "Open",
    url: "Address", command: "Command", copy_text: "Text", speak_text: "Text" }[kindKey] || "Value";
  if (hasChoices) {
    $("#value-select").innerHTML = kind.choices.map((c) => `<option value="${c.value}">${escapeHtml(c.label)}</option>`).join("");
    $("#value-select").value = kind.choices.some((c) => c.value === value) ? value : kind.choices[0].value;
  } else {
    $("#value-input").value = value;
    $("#value-input").placeholder = kind.placeholder || "";
  }
  const help = {
    none: "Plaudit will show this pattern in Activity but won't do anything.",
    command: "Runs hidden in PowerShell with your permissions, every time you clap this pattern.",
    hotkey: "Combine ctrl, alt, shift and win with a key, e.g. ctrl+alt+t.",
    open: "An app name like notepad, a file or folder path, or a link like spotify:.",
  }[kindKey] || "";
  $("#value-help").textContent = help;
  $("#try").disabled = kindKey === "none";
}

function currentValue() {
  return $("#value-popup").hidden ? $("#value-input").value : $("#value-select").value;
}

function showError(message) {
  $("#value-error").textContent = message;
  $("#value-error").hidden = false;
}

function hideError() {
  $("#value-error").hidden = true;
}

async function saveSheet() {
  const button = $("#done");
  button.disabled = true;
  try {
    const result = await api.save_pattern(editing, $("#kind").value, currentValue());
    if (!result.ok) {
      showError(result.error);
      (!$("#value-input-wrap").hidden ? $("#value-input") : $("#kind")).focus();
      return;
    }
    const index = state.patterns.findIndex((p) => p.count === editing);
    state.patterns[index] = result.pattern;
    renderPatterns();
    closeSheet();
    toast("Saved");
  } finally {
    button.disabled = false;
  }
}

async function trySheet() {
  const button = $("#try");
  button.disabled = true;
  hideError();
  try {
    const result = await api.try_action($("#kind").value, currentValue());
    if (result.ok) toast(result.message);
    else showError(result.error);
  } finally {
    button.disabled = $("#kind").value === "none";
  }
}

// ---------------------------------------------------------------------------- events

function showPage(name) {
  document.querySelectorAll(".nav-item").forEach((item) => item.setAttribute("aria-selected", item.dataset.page === name));
  document.querySelectorAll(".page").forEach((page) => { page.hidden = page.id !== `page-${name}`; });
  $("#page-title").textContent = PAGE_TITLES[name];
  $("#scroll").scrollTop = 0;
  if (name === "activity") renderActivity();
}

async function setOption(name, value) {
  const result = await api.set_option(name, value);
  if (!result.ok) toast(result.error);
  else state.options[name] = value;
  return result.ok;
}

function bindEvents() {
  document.querySelectorAll(".nav-item").forEach((item) => item.addEventListener("click", () => showPage(item.dataset.page)));
  $(".nav").addEventListener("keydown", (event) => {
    if (!["ArrowDown", "ArrowUp"].includes(event.key)) return;
    const items = [...document.querySelectorAll(".nav-item")];
    const index = items.indexOf(document.activeElement);
    const next = items[(index + (event.key === "ArrowDown" ? 1 : items.length - 1)) % items.length];
    next.focus();
    showPage(next.dataset.page);
    event.preventDefault();
  });

  $("#pattern-list").addEventListener("click", (event) => {
    const row = event.target.closest(".pattern-row");
    if (row) openSheet(Number(row.dataset.count));
  });

  $("#kind").addEventListener("change", () => { hideError(); configureValue($("#kind").value); });
  $("#value-input").addEventListener("input", hideError);
  $("#done").addEventListener("click", saveSheet);
  $("#cancel").addEventListener("click", closeSheet);
  $("#scrim").addEventListener("click", closeSheet);
  $("#try").addEventListener("click", trySheet);
  $("#browse").addEventListener("click", async () => {
    const path = await api.browse();
    if (path) { $("#value-input").value = path; hideError(); }
  });
  document.addEventListener("keydown", (event) => {
    if ($("#sheet").hidden) return;
    if (event.key === "Escape") closeSheet();
    if (event.key === "Enter" && event.target.tagName !== "BUTTON") { event.preventDefault(); saveSheet(); }
  });

  $("#status-button").addEventListener("click", async () => {
    const action = $("#status-button").dataset.action;
    if (action === "pause") await setOption("paused", true);
    else await api.start_listening();
    await refresh();
  });

  const gap = $("#gap");
  gap.addEventListener("input", updateGapLabel);
  gap.addEventListener("change", async () => { if (await setOption("maxGap", Number(gap.value))) toast("Saved"); });
  $("#opt-typing").addEventListener("change", (e) => setOption("ignoreWhileTyping", e.target.checked));
  $("#opt-notify").addEventListener("change", (e) => setOption("notifyActions", e.target.checked));
  $("#opt-startup").addEventListener("change", async (e) => {
    if (!(await setOption("startWithWindows", e.target.checked))) e.target.checked = !e.target.checked;
  });
  $("#open-logs").addEventListener("click", () => api.open_logs());

  $("#scroll").addEventListener("scroll", () => $(".toolbar").classList.toggle("scrolled", $("#scroll").scrollTop > 2));
}

let toastTimer = null;
function toast(message) {
  const element = $("#toast");
  element.textContent = message;
  element.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { element.hidden = true; }, 2600);
}
