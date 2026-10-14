"use strict";

/* Plaudit install / uninstall screens. Talks to Python through window.pywebview.api;
   in a plain browser (design preview) it uses a mock that simulates the steps. */

const $ = (selector) => document.querySelector(selector);
const STEPS = {
  install: [["welcome", "Welcome"], ["options", "Options"], ["progress", "Install"], ["done", "Done"]],
  uninstall: [["confirm", "Confirm"], ["progress", "Remove"], ["removed", "Done"]],
};

let api = null;
let info = null;
let busy = false;
let retry = null;

function mockApi() {
  const mode = new URLSearchParams(location.search).get("mode") || "install";
  let percent = 0;
  let timer = null;
  const start = () => {
    percent = 0;
    clearInterval(timer);
    timer = setInterval(() => { percent = Math.min(100, percent + 7); }, 120);
    return { ok: true };
  };
  const messages = mode === "install"
    ? ["Closing Plaudit if it's running...", "Copying files...", "Creating shortcuts...", "Registering with Windows..."]
    : ["Closing Plaudit...", "Removing shortcuts and startup entry...", "Removing Plaudit from Installed apps..."];
  return {
    async get_info() {
      return { mode, version: "0.2.0", installDir: "C:\\Users\\you\\AppData\\Local\\Programs\\Plaudit",
               dataDir: "C:\\Users\\you\\AppData\\Roaming\\Plaudit", existingVersion: null, repo: "https://github.com/kyrixdev/plaudit" };
    },
    async install() { return start(); },
    async uninstall() { return start(); },
    async progress() {
      return { percent, message: messages[Math.min(messages.length - 1, Math.floor(percent / (100 / messages.length)))], done: percent >= 100, error: null };
    },
    async launch() {}, async open_repo() { window.open("https://github.com/kyrixdev/plaudit"); },
    async minimize() {}, async close() {},
  };
}

async function boot() {
  api = window.pywebview ? window.pywebview.api : mockApi();
  info = await api.get_info();
  $("#brand-version").textContent = `Version ${info.version}`;
  renderSteps();
  bind();
  if (info.mode === "install") {
    $("#install-dir").textContent = info.installDir;
    $("#install-dir").title = info.installDir;
    if (info.existingVersion) {
      const note = $("#upgrade-note");
      note.hidden = false;
      note.textContent = info.existingVersion === info.version
        ? `Plaudit ${info.version} is already installed. Continuing reinstalls it and keeps your settings.`
        : `This updates Plaudit ${info.existingVersion} to ${info.version}. Your settings are kept.`;
      $("#install").textContent = info.existingVersion === info.version ? "Reinstall" : "Update";
    }
    show("welcome");
  } else {
    $("#data-dir").textContent = `Keep them if you might reinstall later. They're in ${info.dataDir}`;
    show("confirm");
  }
}

window.addEventListener("pywebviewready", () => { if (!api) boot(); });
setTimeout(() => { if (!api && !window.pywebview) boot(); }, 600);

// ---------------------------------------------------------------------------- navigation

function renderSteps() {
  $("#steps").innerHTML = STEPS[info.mode].map(([id, label], i) =>
    `<li data-step="${id}"><span class="marker"><span>${i + 1}</span><svg aria-hidden="true"><use href="#i-check"/></svg></span>${label}</li>`).join("");
}

function show(screen) {
  document.querySelectorAll(".screen").forEach((s) => { s.hidden = s.id !== screen; });
  const order = STEPS[info.mode].map(([id]) => id);
  const index = order.indexOf(screen === "error" ? "progress" : screen);
  document.querySelectorAll("#steps li").forEach((li, i) => {
    li.classList.toggle("current", i === index);
    li.classList.toggle("complete", i < index || (screen === order[order.length - 1] && i === index));
    li.setAttribute("aria-current", i === index ? "step" : "false");
  });
  document.querySelector(".setup").classList.toggle("finished", screen === "done" || screen === "removed");
  const primary = document.querySelector(`#${screen} .btn.primary, #${screen} .btn.destructive`);
  if (primary) primary.focus();
}

function setBusy(value) {
  busy = value;
  $("#close").disabled = value;
}

// ---------------------------------------------------------------------------- running a flow

async function run(start, titles) {
  setBusy(true);
  $("#progress-title").textContent = titles.title;
  setProgress(0, "Preparing...");
  show("progress");
  await start();
  const poll = async () => {
    const p = await api.progress();
    setProgress(p.percent, p.message);
    if (!p.done) return setTimeout(poll, 120);
    setBusy(false);
    if (p.error) {
      $("#error-message").textContent = p.error;
      show("error");
    } else {
      setTimeout(() => show(titles.next), 350);
    }
  };
  poll();
}

function setProgress(percent, message) {
  $("#meter-fill").style.width = `${percent}%`;
  $("#meter").setAttribute("aria-valuenow", percent);
  $("#meter-label").textContent = `${percent}%`;
  if (message) $("#progress-message").textContent = message;
}

function startInstall() {
  const options = { desktopShortcut: $("#opt-desktop").checked, startWithWindows: $("#opt-startup").checked };
  retry = startInstall;
  run(() => api.install(options), { title: "Installing Plaudit", next: "done" });
}

function startUninstall() {
  retry = startUninstall;
  run(() => api.uninstall($("#opt-data").checked), { title: "Removing Plaudit", next: "removed" });
}

// ---------------------------------------------------------------------------- events

function bind() {
  $("#minimize").addEventListener("click", () => api.minimize());
  $("#close").addEventListener("click", () => { if (!busy) api.close(); });
  $("#to-options").addEventListener("click", () => show("options"));
  $("#back").addEventListener("click", () => show("welcome"));
  $("#install").addEventListener("click", startInstall);
  $("#finish").addEventListener("click", async () => {
    if ($("#opt-open").checked) await api.launch();
    api.close();
  });
  $("#star").addEventListener("click", () => api.open_repo());
  $("#cancel").addEventListener("click", () => api.close());
  $("#uninstall").addEventListener("click", startUninstall);
  $("#feedback").addEventListener("click", () => api.open_repo());
  $("#close-done").addEventListener("click", () => api.close());
  $("#error-close").addEventListener("click", () => api.close());
  $("#retry").addEventListener("click", () => retry && retry());
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !busy) api.close();
  });
}
