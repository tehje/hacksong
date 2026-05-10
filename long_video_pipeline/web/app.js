const API_BASE = "";
const THEME_STORAGE_KEY = "longVideoUiTheme";
const MODERN_SIDEBAR_BREAKPOINT = 1240;
const GLOBAL_CHAT_KEY = "__global__";

const state = {
  selectedJobId: null,
  jobsMap: {},
  pollOn: false,
  pollTimer: null,
  theme: "modern",
  modernSidebarCollapsed: false,
  modernMobileSidebarOpen: false,
  chatHistoryByTask: {},
  uploadInProgress: false,
  isAsking: false,
  failureNotifiedJobs: {},
  runtimePreloadDoneKeys: {},
  runtimePreloadInFlightKey: null,
  themeTransitionRunning: false,
  spiderCheerIndex: 0,
  spiderDragActive: false,
  spiderDragPointerId: null,
  spiderDragStartX: 0,
  spiderDragStartOffsetX: 0,
  spiderDragOffsetX: 0,
  jobContextTargetId: null,
  deleteCandidateJobId: null,
  deleteInProgress: false,
};

const els = {
  layoutModern: document.getElementById("layoutModern"),
  layoutPtt: document.getElementById("layoutPtt"),
  modernSidebarOverlay: document.getElementById("modernSidebarOverlay"),
  modernSidebarToggleBtn: document.getElementById("modernSidebarToggleBtn"),
  modernProfileBtn: document.getElementById("modernProfileBtn"),

  modernHeaderSlot: document.getElementById("modernHeaderSlot"),
  modernSidebarSlot: document.getElementById("modernSidebarSlot"),
  modernContentSlot: document.getElementById("modernContentSlot"),
  modernLogSlot: document.getElementById("modernLogSlot"),

  pttHeaderSlot: document.getElementById("pttHeaderSlot"),
  pttSidebarSlot: document.getElementById("pttSidebarSlot"),
  pttContentSlot: document.getElementById("pttContentSlot"),
  pttLogSlot: document.getElementById("pttLogSlot"),
  pttMatrixCanvas: document.getElementById("pttMatrixCanvas"),

  sharedHeader: document.getElementById("sharedHeader"),
  sharedSidebar: document.getElementById("sharedSidebar"),
  sharedContent: document.getElementById("sharedContent"),
  sharedLog: document.getElementById("sharedLog"),

  healthText: document.getElementById("healthText"),
  themeToggleBtn: document.getElementById("themeToggleBtn"),
  consoleLog: document.getElementById("consoleLog"),

  uploadJobForm: document.getElementById("uploadJobForm"),
  uploadFile: document.getElementById("uploadFile"),
  uploadOutputRoot: document.getElementById("uploadOutputRoot"),
  uploadMysteryMode: document.getElementById("uploadMysteryMode"),
  uploadResume: document.getElementById("uploadResume"),
  uploadBtn: document.getElementById("uploadBtn"),

  selectedJobText: document.getElementById("selectedJobText"),
  jobStateText: document.getElementById("jobStateText"),
  qaStateText: document.getElementById("qaStateText"),
  qaProgressPanel: document.getElementById("qaProgressPanel"),
  progressVideoName: document.getElementById("progressVideoName"),
  progressStageText: document.getElementById("progressStageText"),
  progressMetaText: document.getElementById("progressMetaText"),
  progressText: document.getElementById("progressText"),
  progressStageNodes: Array.from(document.querySelectorAll(".qa-stage-node")),

  jobsList: document.getElementById("jobsList"),
  jobsSelect: document.getElementById("jobsSelect"),
  refreshJobsBtn: document.getElementById("refreshJobsBtn"),
  refreshDetailBtn: document.getElementById("refreshDetailBtn"),
  refreshSummaryBtn: document.getElementById("refreshSummaryBtn"),
  refreshSummaryInlineBtn: document.getElementById("refreshSummaryInlineBtn"),
  togglePollBtn: document.getElementById("togglePollBtn"),
  sidebarUploadTriggerBtn: document.getElementById("sidebarUploadTriggerBtn"),
  sidebarUploadInput: document.getElementById("sidebarUploadInput"),
  cancelJobBtn: document.getElementById("cancelJobBtn"),
  fetchResultBtn: document.getElementById("fetchResultBtn"),
  jobResultPre: document.getElementById("jobResultPre"),

  summaryRendered: document.getElementById("summaryRendered"),

  modernHero: document.getElementById("modernHero"),
  botPixelSpider: document.getElementById("botPixelSpider"),

  qaForm: document.getElementById("qaForm"),
  qaMysteryModeSelect: document.getElementById("qaMysteryModeSelect"),
  questionInput: document.getElementById("questionInput"),
  chatFeed: document.getElementById("chatFeed"),
  askBtn: document.getElementById("askBtn"),
  themeTransitionOverlay: document.getElementById("themeTransitionOverlay"),
  themeTransitionCanvas: document.getElementById("themeTransitionCanvas"),
  themeTransitionQuote: document.getElementById("themeTransitionQuote"),
  jobContextMenu: document.getElementById("jobContextMenu"),
  jobContextDeleteBtn: document.getElementById("jobContextDeleteBtn"),
  jobDeleteDialog: document.getElementById("jobDeleteDialog"),
  jobDeleteDialogText: document.getElementById("jobDeleteDialogText"),
  jobDeleteCancelBtn: document.getElementById("jobDeleteCancelBtn"),
  jobDeleteConfirmBtn: document.getElementById("jobDeleteConfirmBtn"),
};

const matrixFx = {
  rafId: 0,
  running: false,
  lastFrameAt: 0,
  width: 0,
  height: 0,
  dpr: 1,
  fontSize: 14,
  cellW: 12,
  cellH: 17,
  ctx: null,
  cells: [],
};

const THEME_TRANSITION_QUOTE_MS = 2000;
const BOT_SWITCH_QUOTE = `开除速度一定要快
有人截图了!                --丁磊`;
const CONSOLE_SWITCH_QUOTE = `The three great virtues of a programmer are laziness, impatience, and hubris                   --Larry Wall`;
const SPIDER_CHEER_TEXTS = ["干巴爹！", "加油哦！", "这是我们热血的组合技啊！"];
const SPIDER_DRAG_EDGE_MARGIN = 12;

const transitionFx = {
  width: 0,
  height: 0,
  dpr: 1,
  ctx: null,
  rafId: 0,
};

function randomMatrixBit() {
  return Math.random() < 0.5 ? "0" : "1";
}

function ensurePttMatrixCanvas() {
  let canvas = els.pttMatrixCanvas;
  if (canvas && canvas.isConnected) return canvas;

  const host = els.layoutPtt || document.getElementById("layoutPtt");
  if (!host) return null;

  canvas = host.querySelector("#pttMatrixCanvas");
  if (!canvas) {
    canvas = document.createElement("canvas");
    canvas.id = "pttMatrixCanvas";
    canvas.className = "ptt-matrix-canvas";
    canvas.setAttribute("aria-hidden", "true");
    host.prepend(canvas);
  }

  els.pttMatrixCanvas = canvas;
  return canvas;
}

function rebuildPttMatrixCells() {
  if (!matrixFx.width || !matrixFx.height) {
    matrixFx.cells = [];
    return;
  }

  const cols = Math.ceil(matrixFx.width / matrixFx.cellW);
  const rows = Math.ceil(matrixFx.height / matrixFx.cellH);
  const cells = [];

  for (let col = 0; col < cols; col += 1) {
    let row = Math.floor(Math.random() * 3);

    while (row < rows) {
      const run = 6 + Math.floor(Math.random() * 10);
      for (let i = 0; i < run && row + i < rows; i += 1) {
        if (Math.random() < 0.05) continue;

        const x = col * matrixFx.cellW + (Math.random() * 1.6 - 0.8);
        const y = (row + i) * matrixFx.cellH + (Math.random() * 1.8 - 0.9);
        const baseAlpha = 0.18 + Math.random() * 0.45;

        cells.push({
          x,
          y,
          bit: randomMatrixBit(),
          alpha: baseAlpha,
          targetAlpha: baseAlpha,
          nextShift: Math.random() * 420,
          phase: Math.random() * Math.PI * 2,
          speed: 0.7 + Math.random() * 1.9,
          head: i === run - 1 && Math.random() < 0.42,
        });
      }

      row += run + 1 + Math.floor(Math.random() * 4);
    }
  }

  matrixFx.cells = cells;
}

function resizePttMatrixFx() {
  const canvas = ensurePttMatrixCanvas();
  if (!canvas) return;

  const width = Math.max(1, Math.floor(window.innerWidth));
  const height = Math.max(1, Math.floor(window.innerHeight));
  const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));

  if (matrixFx.width === width && matrixFx.height === height && matrixFx.dpr === dpr && matrixFx.ctx) {
    return;
  }

  matrixFx.width = width;
  matrixFx.height = height;
  matrixFx.dpr = dpr;
  matrixFx.fontSize = width < 760 ? 12 : 14;
  matrixFx.cellW = width < 760 ? 10 : 12;
  matrixFx.cellH = width < 760 ? 15 : 17;

  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;

  const ctx = canvas.getContext("2d", { alpha: true });
  if (!ctx) return;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  matrixFx.ctx = ctx;

  rebuildPttMatrixCells();

  if (matrixFx.running) {
    drawPttMatrixFrame();
  }
}

function drawPttMatrixFrame(now) {
  if (!matrixFx.running) return;

  if (now - matrixFx.lastFrameAt < 45) {
    matrixFx.rafId = requestAnimationFrame(drawPttMatrixFrame);
    return;
  }

  matrixFx.lastFrameAt = now;

  const ctx = matrixFx.ctx;
  if (!ctx) {
    matrixFx.rafId = requestAnimationFrame(drawPttMatrixFrame);
    return;
  }

  ctx.clearRect(0, 0, matrixFx.width, matrixFx.height);
  ctx.font = `${matrixFx.fontSize}px Consolas, Monaco, Courier New, monospace`;

  for (const cell of matrixFx.cells) {
    if (now >= cell.nextShift) {
      if (Math.random() < 0.66) {
        cell.targetAlpha = 0.14 + Math.random() * 0.6;
      }

      if (Math.random() < 0.44) {
        cell.bit = randomMatrixBit();
      }

      if (Math.random() < 0.14) {
        cell.head = !cell.head;
      }

      cell.nextShift = now + 70 + Math.random() * 380;
    }

    cell.alpha += (cell.targetAlpha - cell.alpha) * 0.18;

    const shimmer = 0.76 + 0.24 * Math.sin(now * 0.001 * cell.speed + cell.phase);
    let alpha = cell.alpha * shimmer;
    if (cell.head) {
      alpha = Math.min(0.9, alpha + 0.22);
    }
    alpha = Math.max(0.08, Math.min(0.9, alpha));

    ctx.fillStyle = `rgba(96, 255, 120, ${alpha})`;
    ctx.fillText(cell.bit, cell.x, cell.y);

    if (alpha > 0.62) {
      const glowAlpha = Math.min(0.64, alpha * 0.52);
      ctx.fillStyle = `rgba(214, 255, 220, ${glowAlpha})`;
      ctx.fillText(cell.bit, cell.x, cell.y);
    }
  }

  matrixFx.rafId = requestAnimationFrame(drawPttMatrixFrame);
}

function startPttMatrixFx() {
  const canvas = ensurePttMatrixCanvas();
  if (!canvas || matrixFx.running) return;

  matrixFx.running = true;
  matrixFx.lastFrameAt = 0;
  resizePttMatrixFx();
  matrixFx.rafId = requestAnimationFrame(drawPttMatrixFrame);
}

function stopPttMatrixFx() {
  if (!matrixFx.running) return;

  matrixFx.running = false;
  if (matrixFx.rafId) {
    cancelAnimationFrame(matrixFx.rafId);
    matrixFx.rafId = 0;
  }

  if (matrixFx.ctx) {
    matrixFx.ctx.clearRect(0, 0, matrixFx.width, matrixFx.height);
  }
}

function syncPttMatrixFx() {
  if (state.theme === "ptt") {
    startPttMatrixFx();
  } else {
    stopPttMatrixFx();
  }
}

function sleepMs(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function ensureTransitionCanvas() {
  const overlay = els.themeTransitionOverlay;
  if (!overlay) return null;

  let canvas = els.themeTransitionCanvas;
  if (!(canvas && canvas.isConnected)) {
    canvas = overlay.querySelector("#themeTransitionCanvas");
  }
  if (!canvas) {
    canvas = document.createElement("canvas");
    canvas.id = "themeTransitionCanvas";
    canvas.className = "theme-transition-canvas";
    canvas.setAttribute("aria-hidden", "true");
    overlay.prepend(canvas);
  }

  let quote = els.themeTransitionQuote;
  if (!(quote && quote.isConnected)) {
    quote = overlay.querySelector("#themeTransitionQuote");
  }
  if (!quote) {
    quote = document.createElement("div");
    quote.id = "themeTransitionQuote";
    quote.className = "theme-transition-quote";
    overlay.appendChild(quote);
  }

  els.themeTransitionCanvas = canvas;
  els.themeTransitionQuote = quote;
  return canvas;
}

function resizeTransitionCanvas() {
  const canvas = ensureTransitionCanvas();
  if (!canvas) return null;

  const width = Math.max(1, Math.floor(window.innerWidth));
  const height = Math.max(1, Math.floor(window.innerHeight));
  const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));

  if (transitionFx.width !== width || transitionFx.height !== height || transitionFx.dpr !== dpr) {
    transitionFx.width = width;
    transitionFx.height = height;
    transitionFx.dpr = dpr;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";
  }

  const ctx = canvas.getContext("2d", { alpha: true });
  if (!ctx) return null;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  transitionFx.ctx = ctx;

  return { ctx, width, height };
}

function setTransitionMode(mode) {
  const overlay = els.themeTransitionOverlay;
  if (!overlay) return;

  overlay.classList.remove("mode-bot", "mode-console");
  if (mode === "bot") {
    overlay.classList.add("mode-bot");
  } else {
    overlay.classList.add("mode-console");
  }
}

function drawBotTransitionBackdrop(ctx, width, height, strength) {
  const alpha = Math.max(0, Math.min(1, strength));
  ctx.clearRect(0, 0, width, height);

  const base = ctx.createLinearGradient(0, 0, 0, height);
  base.addColorStop(0, "rgba(244,248,255," + (0.86 * alpha) + ")");
  base.addColorStop(1, "rgba(235,243,255," + (0.92 * alpha) + ")");
  ctx.fillStyle = base;
  ctx.fillRect(0, 0, width, height);

  const glowA = ctx.createRadialGradient(width * 0.16, height * 0.14, 0, width * 0.16, height * 0.14, width * 0.75);
  glowA.addColorStop(0, "rgba(130,175,255," + (0.34 * alpha) + ")");
  glowA.addColorStop(1, "rgba(130,175,255,0)");
  ctx.fillStyle = glowA;
  ctx.fillRect(0, 0, width, height);

  const glowB = ctx.createRadialGradient(width * 0.84, height * 0.1, 0, width * 0.84, height * 0.1, width * 0.56);
  glowB.addColorStop(0, "rgba(116,222,255," + (0.3 * alpha) + ")");
  glowB.addColorStop(1, "rgba(116,222,255,0)");
  ctx.fillStyle = glowB;
  ctx.fillRect(0, 0, width, height);
}

function drawConsoleTransitionBackdrop(ctx, width, height) {
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, width, height);
}

async function animateBotPrelude() {
  const surface = resizeTransitionCanvas();
  if (!surface) return;

  const { ctx, width, height } = surface;
  const duration = 420;

  await new Promise((resolve) => {
    const start = performance.now();
    const frame = (now) => {
      const t = Math.min(1, (now - start) / duration);
      drawBotTransitionBackdrop(ctx, width, height, t);
      if (t < 1) {
        transitionFx.rafId = requestAnimationFrame(frame);
      } else {
        resolve();
      }
    };
    transitionFx.rafId = requestAnimationFrame(frame);
  });
}

async function animateConsolePrelude() {
  const surface = resizeTransitionCanvas();
  if (!surface) return;

  const { ctx, width, height } = surface;
  const duration = 360;
  const chars = "01ABCDEF[]{}#@&*";

  await new Promise((resolve) => {
    const start = performance.now();
    const frame = (now) => {
      const t = Math.min(1, (now - start) / duration);
      ctx.fillStyle = "rgba(0,0,0," + (0.35 + Math.random() * 0.3) + ")";
      ctx.fillRect(0, 0, width, height);

      const count = Math.floor((width * height) / 5200);
      ctx.font = "16px Lucida Console, Courier New, monospace";

      for (let i = 0; i < count; i += 1) {
        const ch = chars.charAt(Math.floor(Math.random() * chars.length));
        const x = Math.random() * width;
        const y = Math.random() * height;
        const a = 0.25 + Math.random() * 0.7;
        ctx.fillStyle = "rgba(70,255,70," + a + ")";
        ctx.fillText(ch, x, y);
      }

      if (Math.random() < 0.45) {
        ctx.fillStyle = "rgba(120,255,120,0.22)";
        const barY = Math.random() * height;
        ctx.fillRect(0, barY, width, 1 + Math.random() * 3);
      }

      if (t < 1) {
        transitionFx.rafId = requestAnimationFrame(frame);
      } else {
        resolve();
      }
    };

    transitionFx.rafId = requestAnimationFrame(frame);
  });
}

async function showThemeTransitionQuote(nextTheme) {
  const overlay = els.themeTransitionOverlay;
  const quote = els.themeTransitionQuote;
  if (!overlay || !quote) {
    await sleepMs(THEME_TRANSITION_QUOTE_MS);
    return;
  }

  quote.textContent = nextTheme === "modern" ? BOT_SWITCH_QUOTE : CONSOLE_SWITCH_QUOTE;
  overlay.classList.add("show-quote");
  await sleepMs(THEME_TRANSITION_QUOTE_MS);
  overlay.classList.remove("show-quote");
  quote.textContent = "";
}

async function playThemeTransition(nextTheme) {
  const overlay = els.themeTransitionOverlay;
  if (!overlay || !ensureTransitionCanvas()) {
    applyTheme(nextTheme);
    log("已切换界面风格：" + getThemeLabel(nextTheme), "ok");
    return;
  }

  if (state.themeTransitionRunning) return;

  state.themeTransitionRunning = true;
  if (els.themeToggleBtn) {
    els.themeToggleBtn.disabled = true;
  }

  const mode = nextTheme === "modern" ? "bot" : "console";

  try {
    resizeTransitionCanvas();
    overlay.classList.remove("show-quote");
    setTransitionMode(mode);
    overlay.classList.add("active");
    overlay.setAttribute("aria-hidden", "false");

    if (mode === "bot") {
      await animateBotPrelude();
      applyTheme(nextTheme);
      const surface = resizeTransitionCanvas();
      if (surface) {
        drawBotTransitionBackdrop(surface.ctx, surface.width, surface.height, 1);
      }
    } else {
      await animateConsolePrelude();
      applyTheme(nextTheme);
      const surface = resizeTransitionCanvas();
      if (surface) {
        drawConsoleTransitionBackdrop(surface.ctx, surface.width, surface.height);
      }
    }

    await showThemeTransitionQuote(nextTheme);
    log("已切换界面风格：" + getThemeLabel(nextTheme), "ok");
  } finally {
    if (transitionFx.rafId) {
      cancelAnimationFrame(transitionFx.rafId);
      transitionFx.rafId = 0;
    }

    overlay.classList.remove("active", "show-quote", "mode-bot", "mode-console");
    overlay.setAttribute("aria-hidden", "true");

    if (els.themeTransitionQuote) {
      els.themeTransitionQuote.textContent = "";
    }

    if (transitionFx.ctx) {
      transitionFx.ctx.clearRect(0, 0, transitionFx.width, transitionFx.height);
    }

    if (els.themeToggleBtn) {
      els.themeToggleBtn.disabled = false;
    }

    state.themeTransitionRunning = false;
  }
}

function log(msg, type = "info") {
  const now = new Date().toLocaleTimeString();
  const prefix = type === "error" ? "[ERR]" : type === "ok" ? "[OK]" : "[INFO]";
  els.consoleLog.textContent = `${now} ${prefix} ${msg}\n` + els.consoleLog.textContent;
}

function setAskingState(isAsking) {
  state.isAsking = !!isAsking;
  syncAskButtonState();
}

function syncAskButtonState() {
  if (!els.askBtn) return;

  const job = getSelectedJob();
  const ready = isJobReadyForQa(job);
  const asking = !!state.isAsking;

  els.askBtn.disabled = asking || !ready;

  if (asking) {
    els.askBtn.textContent = state.theme === "modern" ? "..." : "回答生成中...";
    return;
  }

  if (ready) {
    els.askBtn.textContent = state.theme === "modern" ? "发送" : "发送问题";
    return;
  }

  if (job && String(job.status || "") === "failed") {
    els.askBtn.textContent = state.theme === "modern" ? "视频失败" : "视频失败";
    return;
  }

  els.askBtn.textContent = state.theme === "modern" ? "等待完成" : "等待视频";
}

function autoResizeQuestionInput() {
  if (!els.questionInput) return;

  const minHeight = state.theme === "modern" ? 56 : 96;
  const maxHeight = state.theme === "modern" ? 220 : 320;

  els.questionInput.style.height = "auto";
  const target = Math.max(minHeight, Math.min(maxHeight, els.questionInput.scrollHeight));
  els.questionInput.style.height = `${target}px`;
  els.questionInput.style.overflowY = els.questionInput.scrollHeight > maxHeight ? "auto" : "hidden";
}

function updateModernHeroVisibility() {
  if (!els.modernHero || !els.chatFeed) return;

  const hasMessages = els.chatFeed.children.length > 0;
  els.modernHero.hidden = hasMessages;
}

function normalizeTheme(theme) {
  return String(theme || "").toLowerCase() === "ptt" ? "ptt" : "modern";
}

function getThemeLabel(theme) {
  return theme === "ptt" ? "PTT 复古风" : "Gemini 聊天风";
}

function getSavedTheme() {
  try {
    return normalizeTheme(localStorage.getItem(THEME_STORAGE_KEY));
  } catch {
    return "modern";
  }
}

function isModernNarrowScreen() {
  return window.matchMedia(`(max-width: ${MODERN_SIDEBAR_BREAKPOINT}px)`).matches;
}

function syncModernSidebarUI() {
  if (!els.modernSidebarToggleBtn) return;

  const isNarrow = isModernNarrowScreen();
  const isOpen = isNarrow ? state.modernMobileSidebarOpen : !state.modernSidebarCollapsed;
  const label = isNarrow
    ? isOpen
      ? "关闭侧边栏"
      : "打开侧边栏"
    : isOpen
      ? "隐藏侧边栏"
      : "显示侧边栏";

  els.modernSidebarToggleBtn.textContent = "☰";
  els.modernSidebarToggleBtn.setAttribute("aria-label", label);
  els.modernSidebarToggleBtn.title = label;
  els.modernSidebarToggleBtn.setAttribute("aria-expanded", String(isOpen));
}

function applyModernSidebarState() {
  if (!els.layoutModern) return;

  const isNarrow = isModernNarrowScreen();
  if (isNarrow) {
    els.layoutModern.classList.remove("sidebar-collapsed");
    els.layoutModern.classList.toggle("mobile-sidebar-open", state.modernMobileSidebarOpen);
  } else {
    state.modernMobileSidebarOpen = false;
    els.layoutModern.classList.remove("mobile-sidebar-open");
    els.layoutModern.classList.toggle("sidebar-collapsed", state.modernSidebarCollapsed);
  }

  if (els.modernSidebarOverlay) {
    const visible = isNarrow && state.modernMobileSidebarOpen;
    els.modernSidebarOverlay.setAttribute("aria-hidden", String(!visible));
  }

  syncModernSidebarUI();
}

function toggleModernSidebar() {
  if (state.theme !== "modern") return;

  if (isModernNarrowScreen()) {
    state.modernMobileSidebarOpen = !state.modernMobileSidebarOpen;
  } else {
    state.modernSidebarCollapsed = !state.modernSidebarCollapsed;
  }

  applyModernSidebarState();
}

function closeModernSidebarOnMobile() {
  if (!isModernNarrowScreen()) return;
  if (!state.modernMobileSidebarOpen) return;

  state.modernMobileSidebarOpen = false;
  applyModernSidebarState();
}

function spawnModernClickHeart(clientX, clientY) {
  if (state.theme !== "modern" || state.themeTransitionRunning) return;
  if (!Number.isFinite(clientX) || !Number.isFinite(clientY)) return;

  const heart = document.createElement("span");
  heart.className = "click-heart";
  heart.style.left = `${clientX}px`;
  heart.style.top = `${clientY}px`;
  heart.style.setProperty("--heart-drift-x", `${(Math.random() * 14 - 7).toFixed(1)}px`);
  heart.style.setProperty("--heart-tilt", `${(Math.random() * 10 - 5).toFixed(1)}deg`);

  const cleanup = () => {
    heart.removeEventListener("animationend", cleanup);
    heart.remove();
  };

  heart.addEventListener("animationend", cleanup);
  document.body.appendChild(heart);
}

function spawnSpiderCheerText() {
  if (state.theme !== "modern" || state.themeTransitionRunning) return;

  const spider = els.botPixelSpider || document.getElementById("botPixelSpider");
  if (!spider || !spider.isConnected) return;

  const phrase = SPIDER_CHEER_TEXTS[state.spiderCheerIndex % SPIDER_CHEER_TEXTS.length];
  state.spiderCheerIndex = (state.spiderCheerIndex + 1) % SPIDER_CHEER_TEXTS.length;

  document.querySelectorAll(".bot-pixel-cheer-text").forEach((node) => node.remove());

  const rect = spider.getBoundingClientRect();
  const gap = 14;
  const maxBubbleWidth = Math.min(window.innerWidth * 0.32, 220);
  const bubbleLeft = Math.min(
    rect.right + gap,
    Math.max(12, window.innerWidth - maxBubbleWidth - 16),
  );
  const bubbleTop = Math.max(12, rect.top + rect.height * 0.42);
  const bubble = document.createElement("span");
  bubble.className = "bot-pixel-cheer-text";
  bubble.textContent = phrase;
  bubble.style.left = `${bubbleLeft}px`;
  bubble.style.top = `${bubbleTop}px`;
  bubble.style.setProperty("--cheer-drift-x", `${(Math.random() * 8 - 4).toFixed(1)}px`);

  const cleanup = () => {
    bubble.removeEventListener("animationend", cleanup);
    bubble.remove();
  };

  bubble.addEventListener("animationend", cleanup);
  document.body.appendChild(bubble);
}

function setSpiderVisualVar(name, value) {
  if (!document.body) return;
  document.body.style.setProperty(name, value);
}

function resetSpiderDragPose() {
  setSpiderVisualVar("--modern-spider-offset-x", `${state.spiderDragOffsetX.toFixed(1)}px`);
  setSpiderVisualVar("--modern-spider-drag-lean", "0deg");
  setSpiderVisualVar("--modern-spider-thread-tilt", "0deg");
  setSpiderVisualVar("--modern-spider-drag-rise", "0px");
}

function getSpiderNodes() {
  const spider = els.botPixelSpider || document.getElementById("botPixelSpider");
  const row = spider?.closest(".bot-pixel-pet-row") || document.querySelector(".bot-pixel-pet-row");
  if (!spider || !row || !spider.isConnected || !row.isConnected) return null;
  return { spider, row };
}

function getSpiderOffsetBounds() {
  const nodes = getSpiderNodes();
  if (!nodes) return null;

  const rect = nodes.row.getBoundingClientRect();
  const baseLeft = rect.left - state.spiderDragOffsetX;
  const minOffset = SPIDER_DRAG_EDGE_MARGIN - baseLeft;
  const maxOffset = window.innerWidth - SPIDER_DRAG_EDGE_MARGIN - rect.width - baseLeft;

  return {
    ...nodes,
    minOffset,
    maxOffset,
  };
}

function applySpiderOffsetX(nextOffset) {
  const bounds = getSpiderOffsetBounds();
  const clamped = bounds
    ? clampNumber(
        nextOffset,
        Math.min(bounds.minOffset, bounds.maxOffset),
        Math.max(bounds.minOffset, bounds.maxOffset),
      )
    : 0;

  state.spiderDragOffsetX = clamped;
  setSpiderVisualVar("--modern-spider-offset-x", `${clamped.toFixed(1)}px`);
  return clamped;
}

function updateSpiderDragPose(deltaX) {
  const lean = clampNumber(deltaX * 0.18, -16, 16);
  const threadTilt = clampNumber(deltaX * 0.22, -12, 12);
  const rise = -clampNumber(Math.abs(deltaX) * 0.14, 0, 10);

  setSpiderVisualVar("--modern-spider-drag-lean", `${lean.toFixed(1)}deg`);
  setSpiderVisualVar("--modern-spider-thread-tilt", `${threadTilt.toFixed(1)}deg`);
  setSpiderVisualVar("--modern-spider-drag-rise", `${rise.toFixed(1)}px`);
}

function setSpiderDragging(active) {
  state.spiderDragActive = active;

  const nodes = getSpiderNodes();
  document.body?.classList.toggle("spider-dragging", active);
  nodes?.row.classList.toggle("is-dragging", active);
  nodes?.spider.classList.toggle("is-dragging", active);
}

function syncSpiderOffsetX() {
  applySpiderOffsetX(state.spiderDragOffsetX);
}

function finishSpiderDrag(event) {
  if (!state.spiderDragActive) return;
  if (event && state.spiderDragPointerId !== null && event.pointerId !== state.spiderDragPointerId) return;

  const nodes = getSpiderNodes();
  if (nodes && typeof nodes.spider.releasePointerCapture === "function" && state.spiderDragPointerId !== null) {
    try {
      if (!nodes.spider.hasPointerCapture || nodes.spider.hasPointerCapture(state.spiderDragPointerId)) {
        nodes.spider.releasePointerCapture(state.spiderDragPointerId);
      }
    } catch {
      // 忽略极少数浏览器下的 capture 释放异常。
    }
  }

  state.spiderDragPointerId = null;
  state.spiderDragStartX = 0;
  state.spiderDragStartOffsetX = state.spiderDragOffsetX;
  setSpiderDragging(false);
  resetSpiderDragPose();
}

function handleSpiderPointerDown(event) {
  if (state.theme !== "modern" || state.themeTransitionRunning) return;
  if (event.button !== 0) return;

  const nodes = getSpiderNodes();
  if (!nodes || event.currentTarget !== nodes.spider) return;

  event.preventDefault();

  state.spiderDragPointerId = event.pointerId;
  state.spiderDragStartX = event.clientX;
  state.spiderDragStartOffsetX = state.spiderDragOffsetX;
  setSpiderDragging(true);
  updateSpiderDragPose(0);

  if (typeof nodes.spider.setPointerCapture === "function") {
    try {
      nodes.spider.setPointerCapture(event.pointerId);
    } catch {
      // 忽略不支持 pointer capture 的环境。
    }
  }
}

function handleSpiderPointerMove(event) {
  if (!state.spiderDragActive || event.pointerId !== state.spiderDragPointerId) return;

  event.preventDefault();

  const deltaX = event.clientX - state.spiderDragStartX;
  const appliedOffset = applySpiderOffsetX(state.spiderDragStartOffsetX + deltaX);
  updateSpiderDragPose(appliedOffset - state.spiderDragStartOffsetX);
}

function handleSpiderPointerUp(event) {
  finishSpiderDrag(event);
}

function handleSpiderPointerCancel(event) {
  finishSpiderDrag(event);
}

function handleModernClickHeart(event) {
  if (state.theme !== "modern") return;
  if (state.themeTransitionRunning) return;
  if (event.button !== 0) return;
  if (event.target === els.themeTransitionOverlay) return;
  if (event.target instanceof Element && event.target.closest(".bot-pixel-pet-row")) return;

  spawnModernClickHeart(event.clientX, event.clientY);
  spawnSpiderCheerText();
}

function handleViewportResize() {
  resizeTransitionCanvas();
  closeJobContextMenu();

  if (state.theme === "modern") {
    applyModernSidebarState();
    syncSpiderOffsetX();
  } else if (state.theme === "ptt") {
    resizePttMatrixFx();
  }
}

function getLayoutSlots(theme) {
  if (theme === "ptt") {
    return {
      header: els.pttHeaderSlot,
      sidebar: els.pttSidebarSlot,
      content: els.pttContentSlot,
      log: els.pttLogSlot,
    };
  }

  return {
    header: els.modernHeaderSlot,
    sidebar: els.modernSidebarSlot,
    content: els.modernContentSlot,
    log: els.modernLogSlot,
  };
}

function mountSharedBlocks(theme) {
  const slots = getLayoutSlots(theme);
  const pairs = [
    [els.sharedHeader, slots.header],
    [els.sharedSidebar, slots.sidebar],
    [els.sharedContent, slots.content],
    [els.sharedLog, slots.log],
  ];

  for (const [node, target] of pairs) {
    if (node && target && node.parentElement !== target) {
      target.appendChild(node);
    }
  }

  if (els.layoutModern) {
    els.layoutModern.hidden = theme === "ptt";
  }
  if (els.layoutPtt) {
    els.layoutPtt.hidden = theme !== "ptt";
  }
}

function applyTheme(theme, options = {}) {
  const { persist = true } = options;
  const normalized = normalizeTheme(theme);

  finishSpiderDrag();
  state.theme = normalized;
  document.body.setAttribute("data-theme", normalized);
  mountSharedBlocks(normalized);

  if (normalized === "modern") {
    applyModernSidebarState();
    syncSpiderOffsetX();
  } else if (els.layoutModern) {
    els.layoutModern.classList.remove("mobile-sidebar-open", "sidebar-collapsed");
  }

  syncPttMatrixFx();

  autoResizeQuestionInput();
  syncAskButtonState();
  updateModernHeroVisibility();

  if (els.themeToggleBtn) {
    // 内部主题键固定为 modern/ptt，按钮文案可独立调整。
    if (normalized === "modern") {
      els.themeToggleBtn.textContent = "控制台";
      els.themeToggleBtn.title = "切换到 命令 控制台";
      els.themeToggleBtn.setAttribute("aria-label", "切换到 命令 控制台");
    } else {
      els.themeToggleBtn.textContent = "BOT";
      els.themeToggleBtn.title = "切换到 交互 BOT";
      els.themeToggleBtn.setAttribute("aria-label", "切换到 交互 BOT");
    }
  }

  if (persist) {
    try {
      localStorage.setItem(THEME_STORAGE_KEY, normalized);
    } catch {
      // 忽略存储异常（例如隐私模式或存储受限）
    }
  }
}

async function toggleTheme() {
  if (state.themeTransitionRunning) return;

  const nextTheme = state.theme === "ptt" ? "modern" : "ptt";
  await playThemeTransition(nextTheme);
}

async function api(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, options);
  const text = await res.text();
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { raw: text };
  }
  if (!res.ok) {
    const detail = data?.detail || data?.raw || JSON.stringify(data);
    throw new Error(`${res.status} ${res.statusText} - ${detail}`);
  }
  return data;
}

function clampPercent(n) {
  const x = Number(n || 0);
  if (!Number.isFinite(x)) return 0;
  return Math.max(0, Math.min(100, x));
}

function clampNumber(value, min, max) {
  const x = Number(value);
  if (!Number.isFinite(x)) return min;
  return Math.max(min, Math.min(max, x));
}

function getJobsSorted() {
  return Object.values(state.jobsMap).sort((a, b) => (b.created_at || "").localeCompare(a.created_at || ""));
}

function isJobReadyForQa(job) {
  return !!job && String(job.status || "") === "succeeded";
}

function getSelectedJob() {
  if (!state.selectedJobId) return null;
  return state.jobsMap[state.selectedJobId] || null;
}

function getJobRequest(job) {
  return job && typeof job.request === "object" && job.request ? job.request : {};
}

function getPathBaseName(pathValue) {
  const raw = String(pathValue || "").trim();
  if (!raw) return "";
  const parts = raw.split(/[\\/]/).filter(Boolean);
  return parts[parts.length - 1] || raw;
}

function getJobDisplayName(job) {
  const request = getJobRequest(job);
  const uploadName = String(request.upload_filename || "").trim();
  const videoName = getPathBaseName(request.video_path);
  return uploadName || videoName || String(job?.job_id || "未命名视频");
}

function formatJobTime(value) {
  const raw = String(value || "").trim();
  if (!raw) return "";

  const normalized = raw.endsWith("Z") ? raw : `${raw}Z`;
  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) return raw;

  return date.toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function canHardDeleteJob(job) {
  const status = String(job?.status || "").trim().toLowerCase();
  return status === "failed" || status === "canceled" || status === "interrupted";
}

function getDeleteActionLabel(job) {
  const request = getJobRequest(job);
  const status = String(job?.status || "").trim().toLowerCase();
  const uploaded = String(request.upload_filename || "").trim();

  if (status === "failed") {
    return uploaded ? "彻底删除失败视频" : "删除失败任务";
  }

  return uploaded ? "删除该视频任务" : "删除该任务";
}

function getDeleteDialogDescription(job) {
  const request = getJobRequest(job);
  const displayName = getJobDisplayName(job);
  const uploaded = String(request.upload_filename || "").trim();
  const status = String(job?.status || "").trim().toLowerCase();
  const statusLabel = status === "failed" ? "失败任务" : "该任务";

  if (uploaded) {
    return `将永久删除“${displayName}”对应的${statusLabel}记录、切片、向量库，以及 DataBase 中的上传源视频。删除后无法恢复。`;
  }

  return `将永久删除“${displayName}”对应的${statusLabel}记录、切片和生成结果；原始视频文件会保留在当前路径。删除后无法恢复。`;
}

function syncSelectedJobWithJobs(jobs = getJobsSorted()) {
  const hasSelected = jobs.some((job) => job.job_id === state.selectedJobId);
  if (!hasSelected) {
    state.selectedJobId = jobs.length > 0 ? jobs[0].job_id : null;
  }
}

function getRuntimePreloadTarget() {
  const job = getSelectedJob();
  if (isJobReadyForQa(job)) {
    return {
      key: `job:${job.job_id}`,
      payload: { job_id: job.job_id },
    };
  }

  return {
    key: "default",
    payload: {},
  };
}

async function triggerRuntimePreload(options = {}) {
  const { silent = true, force = false } = options;
  const target = getRuntimePreloadTarget();
  if (!target) return null;

  const { key, payload } = target;
  if (!force) {
    if (state.runtimePreloadDoneKeys[key]) return null;
    if (state.runtimePreloadInFlightKey) return null;
  }

  state.runtimePreloadInFlightKey = key;
  try {
    const data = await api("/runtime/preload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.runtimePreloadDoneKeys[key] = true;
    if (!silent) {
      log(`问答模型已预热: ${data.qa_model_path || key}`, "ok");
    }
    return data;
  } catch (err) {
    if (!silent) {
      log(`问答模型预热失败: ${err.message}`, "error");
    }
    return null;
  } finally {
    if (state.runtimePreloadInFlightKey === key) {
      state.runtimePreloadInFlightKey = null;
    }
  }
}

function getJobStatusLabel(status) {
  const s = String(status || "").toLowerCase();
  if (s === "succeeded") return "已完成";
  if (s === "failed") return "执行失败";
  if (s === "running") return "进行中";
  if (s === "queued") return "排队中";
  if (s === "canceled") return "已取消";
  if (s === "interrupted") return "重启中断";
  return s || "未知";
}

function getJobStagePresentation(job) {
  const status = String(job?.status || "").trim().toLowerCase();
  const rawStageIndex = Number(job?.stage_index || 0);
  const totalStages = Math.max(1, Number(job?.total_stages || 6) || 6);
  const stageIndex = Math.max(0, Math.min(totalStages, rawStageIndex));
  const stageName = getJobStageDisplay(job);

  if (!job) {
    return {
      tone: "idle",
      stageLabel: "等待选择",
      meta: "选择一个视频后，这里会显示当前阶段和进度。",
      activeStep: 0,
    };
  }

  if (status === "failed") {
    return {
      tone: "failed",
      stageLabel: stageIndex > 0 ? `失败于 ${stageName}` : "启动失败",
      meta: "分析已中断。建议删除无效任务后重新上传。",
      activeStep: stageIndex > 0 ? Math.min(totalStages, stageIndex) : 0,
    };
  }

  if (status === "canceled") {
    return {
      tone: "canceled",
      stageLabel: "任务已取消",
      meta: "当前任务已停止，不会继续处理。",
      activeStep: stageIndex > 0 ? Math.min(totalStages, stageIndex) : 0,
    };
  }

  if (status === "interrupted") {
    return {
      tone: "warning",
      stageLabel: "服务重启中断",
      meta: "任务曾被中断，可重新上传或删除后重建。",
      activeStep: stageIndex > 0 ? Math.min(totalStages, stageIndex) : 0,
    };
  }

  if (status === "succeeded") {
    return {
      tone: "done",
      stageLabel: "分析完成",
      meta: "可以直接提问，系统会只围绕当前视频作答。",
      activeStep: totalStages,
    };
  }

  if (status === "queued") {
    return {
      tone: "queued",
      stageLabel: "排队等待",
      meta: "任务已进入队列，等待处理资源。",
      activeStep: 0,
    };
  }

  const runningTone = `running-${Math.max(1, Math.min(6, stageIndex || 1))}`;
  const activeStep = stageIndex > 0 ? Math.min(totalStages, stageIndex) : 0;
  return {
    tone: runningTone,
    stageLabel: activeStep > 0 ? `阶段 ${activeStep}: ${stageName}` : stageName,
    meta: activeStep > 0 ? `当前共 ${totalStages} 个阶段，正在处理 ${stageName}。` : "任务已启动，正在等待阶段进度。",
    activeStep,
  };
}

function syncProgressPanel(job) {
  const presentation = getJobStagePresentation(job);
  const pct = clampPercent(job?.progress_percent || 0);

  if (els.qaProgressPanel) {
    els.qaProgressPanel.dataset.stageTone = presentation.tone;
  }

  if (els.progressVideoName) {
    els.progressVideoName.textContent = job ? getJobDisplayName(job) : "未选择视频";
  }

  if (els.progressStageText) {
    els.progressStageText.textContent = presentation.stageLabel;
  }

  if (els.progressText) {
    els.progressText.textContent = `${pct.toFixed(1)}%`;
  }

  if (els.progressMetaText) {
    els.progressMetaText.textContent = presentation.meta;
  }

  if (Array.isArray(els.progressStageNodes)) {
    for (const node of els.progressStageNodes) {
      const step = Number(node.dataset.step || 0);
      node.classList.toggle("is-complete", step > 0 && step < presentation.activeStep);
      node.classList.toggle("is-current", step > 0 && step === presentation.activeStep);
      node.classList.toggle("is-pending", step > presentation.activeStep);
      node.classList.toggle("is-empty", presentation.activeStep === 0);
    }
  }
}

function summarizeJobError(raw, maxLen = 560) {
  const text = String(raw || "").trim();
  if (!text) return "视频处理失败，但未返回详细错误。";

  const lines = text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .slice(0, 4);

  const shortText = lines.join("\n");
  if (shortText.length <= maxLen) return shortText;
  return `${shortText.slice(0, maxLen)}...`;
}

function compactText(text, maxLen = 120) {
  const normalized = String(text || "").replace(/\s+/g, " ").trim();
  if (!normalized) return "";
  if (normalized.length <= maxLen) return normalized;
  return `${normalized.slice(0, Math.max(1, maxLen - 3)).trim()}...`;
}

function prettyStageName(rawStage) {
  const stage = String(rawStage || "").trim().toLowerCase();
  const stageLabels = {
    starting: "准备中",
    prepare: "预处理",
    instruct: "结构摘要生成",
    retrieval: "证据检索",
    thinking: "证据复核",
    save: "结果保存",
    done: "分析完成",
    failed: "处理失败",
    canceled: "已取消",
    interrupted: "服务中断",
    asr: "语音转写",
    asr_transcribe: "语音转写",
    transcribe: "语音转写",
    ocr: "文字识别",
    keyframe: "关键帧提取",
    keyframes: "关键帧提取",
    extract_keyframes: "关键帧提取",
    chunk: "视频切片",
    chunking: "视频切片",
    chunk_plan: "分段规划",
    summary: "摘要生成",
    qa: "问答生成",
  };
  return stageLabels[stage] || String(rawStage || "").trim();
}

function normalizeStageDetail(rawText) {
  const text = String(rawText || "")
    .replace(/^\[\d+\s*\/\s*\d+\]\s*/, "")
    .replace(/[。.!…\s]+$/g, "")
    .trim();

  if (!text) return "";

  const lower = text.toLowerCase();
  if (/模型切段/.test(text) || ((/asr/.test(lower) || /转写/.test(text)) && (/ocr/.test(lower) || /关键帧/.test(text)))) {
    return "预处理";
  }
  if (/faster[-_\s]?whisper/i.test(text) || /asr/.test(lower) || /转写/.test(text) || /whisper/i.test(text)) {
    return "ASR 转写";
  }
  if (/ocr/i.test(text) || /文字识别/.test(text)) {
    return "OCR 识别";
  }
  if (/关键帧/.test(text)) {
    return "关键帧抽取";
  }
  if (/切段|切片/.test(text) || /chunk/.test(lower)) {
    return "语义切段";
  }
  if (/instruct/.test(lower) || /section summaries|chunk cards|global summary/.test(lower)) {
    return "结构摘要生成";
  }
  if (/向量库|检索/.test(text) || /retrieval/.test(lower)) {
    return "证据检索";
  }
  if (/thinking/.test(lower) || /复核|推理/.test(text)) {
    return "证据复核";
  }
  if (/落盘|保存|输出/.test(text) || /save/.test(lower)) {
    return "结果保存";
  }
  if (/完成/.test(text) || /done/.test(lower)) {
    return "分析完成";
  }
  return compactText(text, 28);
}

function isGenericJobMessage(rawText) {
  const text = String(rawText || "").trim().toLowerCase();
  return (
    !text ||
    text === "pipeline running" ||
    text === "pipeline failed" ||
    text === "done" ||
    text === "failed" ||
    text === "qa answered" ||
    text === "canceled before start" ||
    text === "service restarted before job finished"
  );
}

function getJobStageDisplay(job) {
  if (!job) return "等待处理";

  if (!isGenericJobMessage(job.message)) {
    const detailFromMessage = normalizeStageDetail(job.message);
    if (detailFromMessage) return detailFromMessage;
  }

  const detailFromStage = prettyStageName(job.stage_name);
  return detailFromStage || "等待处理";
}

function extractUsefulErrorLine(text) {
  const ignoredPatterns = [
    /^traceback\b/i,
    /^file\s+".*?",\s*line\s+\d+/i,
    /^during handling of the above exception/i,
    /^most recent call last/i,
    /^raise\b/i,
    /^return\b/i,
    /^from\s+[a-z0-9_./-]+\s+import\s+/i,
    /^\^+$/,
  ];

  const lines = String(text || "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .filter((line) => !ignoredPatterns.some((pattern) => pattern.test(line)));

  return lines[0] || "";
}

function summarizeJobErrorForChat(raw) {
  const text = String(raw || "").replace(/\r/g, "").trim();
  if (!text) {
    return {
      summary: "视频处理失败，但系统没有返回明确原因。",
      suggestion: "请重新上传视频；如果该任务不再需要，可在左侧列表右键删除。",
    };
  }

  const lower = text.toLowerCase();

  if (/faster[-_\s]?whisper/.test(lower) && /(install|required|missing|no module named)/.test(lower)) {
    return {
      summary: "语音转写依赖 `faster-whisper` 缺失，ASR 没有启动成功。",
      suggestion: "补齐转写依赖后重新上传视频；如果该任务无效，可在左侧列表右键删除。",
    };
  }

  if (/ffmpeg/.test(lower) && /(not found|no such file|command not found|required|failed)/.test(lower)) {
    return {
      summary: "`ffmpeg` 不可用，视频切片或抽帧阶段失败。",
      suggestion: "确认服务环境已安装 ffmpeg 且命令可执行，然后重新上传视频。",
    };
  }

  if (/tesseract/.test(lower) && /(not found|missing|failed|required)/.test(lower)) {
    return {
      summary: "OCR 依赖 `tesseract` 不可用，文字识别阶段失败。",
      suggestion: "补齐 OCR 依赖后重新上传视频。",
    };
  }

  if (/(cuda|gpu|cublas|cudnn).*(out of memory|oom)|out of memory/i.test(text)) {
    return {
      summary: "显存不足，模型加载或推理过程中断。",
      suggestion: "释放显存、降低占用后再重试，或换更轻的运行配置。",
    };
  }

  if (/permission denied|errno 13/i.test(text)) {
    return {
      summary: "服务没有权限访问视频、输出目录或模型文件。",
      suggestion: "检查相关目录权限后重新上传视频。",
    };
  }

  if (/video not found|file not found|no such file or directory/i.test(text)) {
    return {
      summary: "找不到视频文件或依赖文件，任务无法继续。",
      suggestion: "确认源视频和依赖文件仍在原路径后重新上传。",
    };
  }

  if (/(connection|connect|timeout|timed out|refused)/i.test(text)) {
    return {
      summary: "运行过程中连接超时或依赖服务不可用。",
      suggestion: "检查模型服务、网络或本机资源后重新上传视频。",
    };
  }

  const usefulLine = extractUsefulErrorLine(text);
  const stageMatch = usefulLine.match(/^([a-z0-9_]+)\s+failed after\s+\d+\s+retries:\s*(.+)$/i);
  if (stageMatch) {
    const stageName = prettyStageName(stageMatch[1]);
    return {
      summary: `${stageName}失败：${compactText(stageMatch[2], 88)}`,
      suggestion: "修复对应阶段问题后重新上传视频；如果该任务无效，可在左侧列表右键删除。",
    };
  }

  return {
    summary: compactText(usefulLine || summarizeJobError(text, 160), 100) || "视频处理失败。",
    suggestion: "请修复问题后重新上传视频；如果该任务无效，可在左侧列表右键删除。",
  };
}

function buildJobFailureChatMessage(job, options = {}) {
  const { followUp = "请修复后重新上传视频创建会话。" } = options;
  const displayName = getJobDisplayName(job);
  const { summary, suggestion } = summarizeJobErrorForChat(job?.error || job?.message);

  return `视频 **${displayName}** 处理失败，当前无法问答。\n\n失败原因：${summary}\n\n建议操作：${suggestion}\n\n${followUp}`;
}

function getActiveTaskKey() {
  return state.selectedJobId || GLOBAL_CHAT_KEY;
}

function getTaskHistory(taskKey = getActiveTaskKey()) {
  const key = String(taskKey || GLOBAL_CHAT_KEY);
  if (!Array.isArray(state.chatHistoryByTask[key])) {
    state.chatHistoryByTask[key] = [];
  }
  return state.chatHistoryByTask[key];
}

function normalizeClipList(clips) {
  if (!Array.isArray(clips)) return [];
  return clips
    .filter((clip) => clip && clip.clip_id)
    .map((clip) => ({
      clip_id: String(clip.clip_id || ""),
      t_start: String(clip.t_start || ""),
      t_end: String(clip.t_end || ""),
      reason: String(clip.reason || ""),
      clip_job_id: String(clip.clip_job_id || ""),
      source_job_id: String(clip.source_job_id || ""),
      source_video_name: String(clip.source_video_name || ""),
    }));
}

function normalizeSegmentList(segments) {
  if (!Array.isArray(segments)) return [];
  return segments
    .filter((seg) => seg && seg.t_start)
    .map((seg) => ({
      t_start: String(seg.t_start || ""),
      t_end: String(seg.t_end || seg.t_start || ""),
      reason: String(seg.reason || ""),
      source_job_id: String(seg.source_job_id || ""),
      source_video_name: String(seg.source_video_name || ""),
    }));
}

function saveChatEntry(taskKey, role, content, options = {}) {
  const { clips = [], segments = [] } = options;
  const history = getTaskHistory(taskKey);
  history.push({
    role: role === "user" ? "user" : "assistant",
    content: String(content || ""),
    clips: normalizeClipList(clips),
    segments: normalizeSegmentList(segments),
  });
}

function appendChatToTask(taskKey, role, markdownText, options = {}) {
  const key = String(taskKey || GLOBAL_CHAT_KEY);
  const normalizedRole = role === "user" ? "user" : "assistant";
  const text = String(markdownText || "");
  const clips = normalizeClipList(options.clips);
  const segments = normalizeSegmentList(options.segments);

  saveChatEntry(key, normalizedRole, text, { clips, segments });

  if (key === String(getActiveTaskKey())) {
    appendChat(normalizedRole, text, { persist: false, taskKey: key, clips, segments });
  }
}

function notifyJobFailureOnce(job) {
  if (!job || !job.job_id) return;

  const status = String(job.status || "");
  const key = String(job.job_id);

  if (status !== "failed") {
    delete state.failureNotifiedJobs[key];
    return;
  }

  if (state.failureNotifiedJobs[key]) return;
  state.failureNotifiedJobs[key] = true;

  appendChatToTask(
    key,
    "assistant",
    buildJobFailureChatMessage(job)
  );
  log(`视频 ${key} 失败：${summarizeJobErrorForChat(job.error || job.message).summary}`, "error");
}

function renderChatHistoryForSelectedTask(options = {}) {
  const { showEmptyHint = true } = options;
  if (!els.chatFeed) return;

  const taskKey = getActiveTaskKey();
  const history = getTaskHistory(taskKey);

  els.chatFeed.innerHTML = "";
  for (const entry of history) {
    appendChat(entry.role, entry.content, {
      persist: false,
      taskKey,
      clips: entry.clips || [],
      segments: entry.segments || [],
    });
  }

  if (!history.length && showEmptyHint) {
    if (state.selectedJobId) {
      const selectedJob = getSelectedJob();
      const displayName = getJobDisplayName(selectedJob);
      appendChat(
        "assistant",
        `已切换到视频 **${displayName}**。\n\n在下方直接输入你的提示词或问题开始问答，或点击左侧“上传视频”创建新会话。`,
        { persist: false, taskKey }
      );
      return;
    }

    appendChat(
      "assistant",
      "当前尚未选择视频窗口。请先从左侧视频列表切换窗口，或点击左侧“上传视频”创建新会话。",
      { persist: false, taskKey: GLOBAL_CHAT_KEY }
    );
    return;
  }

  updateModernHeroVisibility();
}

function renderJobsSelect() {
  const jobs = getJobsSorted();
  els.jobsSelect.innerHTML = "";

  for (const job of jobs) {
    const opt = document.createElement("option");
    const name = getJobDisplayName(job);
    const statusLabel = getJobStatusLabel(job.status);
    const pct = clampPercent(job.progress_percent).toFixed(0);
    opt.value = job.job_id;
    opt.textContent = `${name} | ${statusLabel} | ${pct}%`;
    opt.title = `${name}\n任务 ID: ${job.job_id}\n阶段: ${job.stage_name || "-"}`;
    if (job.job_id === state.selectedJobId) {
      opt.selected = true;
    }
    els.jobsSelect.appendChild(opt);
  }

  if (state.selectedJobId) {
    els.jobsSelect.value = state.selectedJobId;
  }
}

function renderJobsList() {
  if (!els.jobsList) return;

  const jobs = getJobsSorted();
  els.jobsList.innerHTML = "";

  if (!jobs.length) {
    els.jobsList.innerHTML = `
      <div class="jobs-list-empty">
        <strong>暂无视频</strong>
        <span>点击上方“上传视频”开始新会话。</span>
      </div>
    `;
    return;
  }

  for (const job of jobs) {
    const item = document.createElement("button");
    const displayName = getJobDisplayName(job);
    const statusLabel = getJobStatusLabel(job.status);
    const pct = clampPercent(job.progress_percent).toFixed(0);
    const stageName = getJobStageDisplay(job);
    const updatedText = formatJobTime(job.updated_at || job.created_at) || "刚刚更新";
    const isFailed = String(job.status || "").trim().toLowerCase() === "failed";
    const messageText = String(job.message || "").trim();
    const summaryText = isFailed
      ? summarizeJobError(job.error || job.message, 120).replace(/\s*\n+\s*/g, " / ")
      : (!isGenericJobMessage(messageText) ? messageText : `当前阶段：${stageName}`);
    const statusClass = `status-${String(job.status || "unknown").trim().toLowerCase() || "unknown"}`;

    item.type = "button";
    item.className = `job-card${job.job_id === state.selectedJobId ? " is-selected" : ""}${isFailed ? " is-failed" : ""}`;
    item.dataset.jobId = job.job_id;
    item.dataset.deletable = canHardDeleteJob(job) ? "1" : "0";
    item.title = canHardDeleteJob(job)
      ? "左键切换视频，右键可删除该失败/已取消任务"
      : "左键切换视频";

    item.innerHTML = `
      <div class="job-card-head">
        <div class="job-card-title">${escapeHtml(displayName)}</div>
        <span class="job-status-pill ${statusClass}">${escapeHtml(statusLabel)}</span>
      </div>
      <div class="job-card-subtitle">任务 ID: ${escapeHtml(job.job_id)}</div>
      <div class="job-card-meta">
        <span>${escapeHtml(stageName)}</span>
        <span>${escapeHtml(pct)}%</span>
        <span>${escapeHtml(updatedText)}</span>
      </div>
      <div class="job-card-progress">
        <div class="job-card-progress-fill" style="width: ${pct}%"></div>
      </div>
      <div class="job-card-desc">${escapeHtml(summaryText)}</div>
    `;

    els.jobsList.appendChild(item);
  }
}

async function switchToJob(jobId, options = {}) {
  const {
    silent = true,
    refreshDetail = true,
    refreshSummary = true,
    closeSidebar = true,
  } = options;

  const nextJobId = jobId || null;
  const changed = state.selectedJobId !== nextJobId;

  state.selectedJobId = nextJobId;
  renderJobsSelect();
  renderJobsList();
  updateSelectedJobCard(getSelectedJob());

  if (changed || !nextJobId) {
    renderChatHistoryForSelectedTask();
  }

  if (refreshDetail && nextJobId) {
    await refreshSelectedDetail({ silent });
  }

  if (refreshSummary && nextJobId) {
    await refreshSummaryPreview({ silent: true });
  }

  if (closeSidebar) {
    closeModernSidebarOnMobile();
  }
}

function closeJobContextMenu() {
  if (!els.jobContextMenu) return;
  els.jobContextMenu.hidden = true;
  els.jobContextMenu.setAttribute("aria-hidden", "true");
  state.jobContextTargetId = null;
}

function openJobContextMenu(job, clientX, clientY) {
  if (!els.jobContextMenu || !els.jobContextDeleteBtn || !job) return;

  state.jobContextTargetId = job.job_id;
  els.jobContextDeleteBtn.disabled = !canHardDeleteJob(job);
  els.jobContextDeleteBtn.textContent = canHardDeleteJob(job)
    ? getDeleteActionLabel(job)
    : "仅失败或已取消任务可删除";

  els.jobContextMenu.hidden = false;
  els.jobContextMenu.setAttribute("aria-hidden", "false");
  els.jobContextMenu.style.left = `${clientX}px`;
  els.jobContextMenu.style.top = `${clientY}px`;

  requestAnimationFrame(() => {
    const rect = els.jobContextMenu.getBoundingClientRect();
    const nextLeft = Math.min(clientX, Math.max(12, window.innerWidth - rect.width - 12));
    const nextTop = Math.min(clientY, Math.max(12, window.innerHeight - rect.height - 12));
    els.jobContextMenu.style.left = `${nextLeft}px`;
    els.jobContextMenu.style.top = `${nextTop}px`;
  });
}

function closeDeleteDialog() {
  if (!els.jobDeleteDialog) return;
  if (state.deleteInProgress) return;
  els.jobDeleteDialog.hidden = true;
  els.jobDeleteDialog.setAttribute("aria-hidden", "true");
  state.deleteCandidateJobId = null;
}

function openDeleteDialog(jobId) {
  const job = state.jobsMap[jobId];
  if (!job || !els.jobDeleteDialog || !els.jobDeleteDialogText) return;

  state.deleteCandidateJobId = jobId;
  syncDeleteDialogButtons();
  els.jobDeleteDialogText.textContent = getDeleteDialogDescription(job);
  els.jobDeleteDialog.hidden = false;
  els.jobDeleteDialog.setAttribute("aria-hidden", "false");
}

function syncDeleteDialogButtons() {
  if (els.jobDeleteConfirmBtn) {
    els.jobDeleteConfirmBtn.disabled = state.deleteInProgress;
    els.jobDeleteConfirmBtn.textContent = state.deleteInProgress ? "删除中..." : "彻底删除";
  }
  if (els.jobDeleteCancelBtn) {
    els.jobDeleteCancelBtn.disabled = state.deleteInProgress;
  }
}

async function confirmDeleteJob() {
  const jobId = state.deleteCandidateJobId;
  const job = state.jobsMap[jobId];
  if (!jobId || !job) {
    closeDeleteDialog();
    return;
  }

  state.deleteInProgress = true;
  syncDeleteDialogButtons();

  try {
    const data = await api(`/jobs/${encodeURIComponent(jobId)}`, { method: "DELETE" });
    delete state.jobsMap[jobId];
    delete state.chatHistoryByTask[jobId];
    delete state.failureNotifiedJobs[jobId];
    delete state.runtimePreloadDoneKeys[`job:${jobId}`];

    syncSelectedJobWithJobs();
    renderJobsSelect();
    renderJobsList();
    updateSelectedJobCard(getSelectedJob());
    renderChatHistoryForSelectedTask();
    closeJobContextMenu();

    if (state.selectedJobId) {
      await refreshSelectedDetail({ silent: true });
      await refreshSummaryPreview({ silent: true });
    }

    const keptSource = String(data.retained_video_path || "").trim();
    if (keptSource) {
      log(`已删除失败任务 ${jobId}，原视频保留在 ${keptSource}`, "ok");
    } else {
      log(`已彻底删除失败视频 ${jobId}`, "ok");
    }
  } catch (err) {
    log(`删除失败: ${err.message}`, "error");
  } finally {
    state.deleteInProgress = false;
    syncDeleteDialogButtons();
    closeDeleteDialog();
  }
}

function updateSelectedJobCard(job) {
  if (!job) {
    els.selectedJobText.textContent = "未选择";
    els.jobStateText.textContent = "请先上传视频创建会话";
    els.qaStateText.textContent = "请选择视频，分析完成后可提问";
    syncProgressPanel(null);
    markQuestionReadyState(null);
    syncSummaryHintByJob(null);
    return;
  }

  const stageIndex = Number(job.stage_index || 0);
  const totalStages = Number(job.total_stages || 6);
  const stageName = getJobStageDisplay(job);
  const displayName = getJobDisplayName(job);
  const stageMessage = !isGenericJobMessage(job.message) ? String(job.message || "").trim() : stageName;

  els.selectedJobText.textContent = displayName;

  if (String(job.status || "") === "failed") {
    const errBrief = summarizeJobError(job.error || job.message, 200);
    els.jobStateText.textContent = `任务 ID: ${job.job_id} | 状态: 执行失败 | 阶段: [${stageIndex}/${totalStages}] ${stageName} | ${errBrief}`;
    els.qaStateText.textContent = "视频处理失败，当前无法问答。可右键该视频任务并彻底删除后重新上传。";
  } else {
    els.jobStateText.textContent = `任务 ID: ${job.job_id} | 状态: ${getJobStatusLabel(job.status)} | 阶段: [${stageIndex}/${totalStages}] ${stageName} | ${stageMessage}`;
    els.qaStateText.textContent = isJobReadyForQa(job)
      ? "视频已完成分析，可直接输入提示词或问题"
      : `视频${getJobStatusLabel(job.status)}，暂不可问答`;
  }
  syncProgressPanel(job);
  markQuestionReadyState(job);
  syncSummaryHintByJob(job);
}

function syncSummaryHintByJob(job) {
  if (!els.summaryRendered) return;

  if (!job) {
    els.summaryRendered.innerHTML = markdownToHtml("(请先上传并完成分析)");
    return;
  }

  const status = String(job.status || "");
  if (status === "succeeded") return;

  if (status === "failed") {
    const errBrief = summarizeJobError(job.error || job.message);
    els.summaryRendered.innerHTML = markdownToHtml(
      `### 视频处理失败\n\n当前视频未成功完成分析，因此暂无摘要。\n\n\
\`\`\`\n${errBrief}\n\`\`\``
    );
    return;
  }

  if (status === "canceled") {
    els.summaryRendered.innerHTML = markdownToHtml("### 视频已取消\n\n当前视频已取消，因此暂无摘要。");
    return;
  }

  const pct = clampPercent(job.progress_percent).toFixed(1);
  els.summaryRendered.innerHTML = markdownToHtml(`视频分析进行中（${pct}%），摘要将在完成后生成。`);
}

function markQuestionReadyState(job) {
  if (!els.questionInput) return;

  if (!job) {
    els.questionInput.placeholder = "请先在左侧选择视频窗口，或点击左侧“上传视频”";
    syncAskButtonState();
    return;
  }

  const status = String(job.status || "");
  if (isJobReadyForQa(job)) {
    els.questionInput.placeholder = `围绕“${getJobDisplayName(job)}”输入你的提示词或问题，例如：概括人物动机并标出关键时间段`;
    syncAskButtonState();
    return;
  }

  if (status === "failed") {
    els.questionInput.placeholder = `视频 ${job.job_id} 处理失败，可右键左侧列表删除后重新上传`;
    syncAskButtonState();
    return;
  }

  if (status === "canceled") {
    els.questionInput.placeholder = `视频 ${job.job_id} 已取消，可右键左侧列表删除或重新上传`;
    syncAskButtonState();
    return;
  }

  const pct = clampPercent(job.progress_percent).toFixed(0);
  els.questionInput.placeholder = `视频 ${job.job_id} 分析中（${pct}%），完成后即可问答`;
  syncAskButtonState();
}

function escapeHtml(str) {
  return String(str || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function inlineMarkdownToHtml(text) {
  let s = escapeHtml(text);
  s = s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
  return s;
}

function markdownToHtml(md) {
  const lines = String(md || "").replace(/\r/g, "").split("\n");
  const out = [];
  let inUl = false;
  let inOl = false;
  let inCode = false;

  const closeLists = () => {
    if (inUl) {
      out.push("</ul>");
      inUl = false;
    }
    if (inOl) {
      out.push("</ol>");
      inOl = false;
    }
  };

  for (const line of lines) {
    const trimmed = line.trim();

    if (trimmed.startsWith("```")) {
      closeLists();
      if (!inCode) {
        inCode = true;
        out.push("<pre><code>");
      } else {
        inCode = false;
        out.push("</code></pre>");
      }
      continue;
    }

    if (inCode) {
      out.push(`${escapeHtml(line)}\n`);
      continue;
    }

    if (!trimmed) {
      closeLists();
      out.push("<br />");
      continue;
    }

    const heading = trimmed.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      closeLists();
      const level = heading[1].length;
      out.push(`<h${level}>${inlineMarkdownToHtml(heading[2])}</h${level}>`);
      continue;
    }

    const ol = trimmed.match(/^\d+\.\s+(.*)$/);
    if (ol) {
      if (inUl) {
        out.push("</ul>");
        inUl = false;
      }
      if (!inOl) {
        out.push("<ol>");
        inOl = true;
      }
      out.push(`<li>${inlineMarkdownToHtml(ol[1])}</li>`);
      continue;
    }

    const ul = trimmed.match(/^[-*+]\s+(.*)$/);
    if (ul) {
      if (inOl) {
        out.push("</ol>");
        inOl = false;
      }
      if (!inUl) {
        out.push("<ul>");
        inUl = true;
      }
      out.push(`<li>${inlineMarkdownToHtml(ul[1])}</li>`);
      continue;
    }

    closeLists();
    out.push(`<p>${inlineMarkdownToHtml(trimmed)}</p>`);
  }

  closeLists();
  if (inCode) {
    out.push("</code></pre>");
  }
  return out.join("\n");
}

function timeTextToSeconds(text) {
  const raw = String(text || "").trim();
  if (!raw) return 0;
  const parts = raw.split(":").map((x) => Number(x));
  if (parts.some((n) => !Number.isFinite(n))) return 0;
  if (parts.length === 3) {
    return parts[0] * 3600 + parts[1] * 60 + parts[2];
  }
  if (parts.length === 2) {
    return parts[0] * 60 + parts[1];
  }
  return Number(raw) || 0;
}

function sanitizeQaAnswerMarkdown(markdownText) {
  const raw = String(markdownText || "").replace(/\r/g, "").trim();
  if (!raw) return "证据不足，当前无法给出可靠回答。";

  const lines = raw.split("\n");
  const kept = [];
  let sawAnswerHeading = false;

  for (const line of lines) {
    const trimmed = line.trim();

    if (/^#{1,6}\s*回答\b/.test(trimmed)) {
      sawAnswerHeading = true;
      continue;
    }

    if (/^#{1,6}\s*(对应片段|证据|不确定性)\b/.test(trimmed)) {
      break;
    }

    kept.push(
      sawAnswerHeading
        ? line.replace(/^\s*[-*+]\s+/, "").replace(/^\s*\d+\.\s+/, "")
        : line
    );
  }

  const cleaned = kept.join("\n").replace(/\n{3,}/g, "\n\n").trim();
  return cleaned || "证据不足，当前无法给出可靠回答。";
}

function buildClipCaptionHtml(tStart, tEnd, reason, index, sourceVideoName = "") {
  const start = String(tStart || "").trim();
  const end = String(tEnd || tStart || "").trim();
  const timeLabel = start && end ? `${start}-${end}` : start || end || `相关片段 ${index + 1}`;
  const reasonLabel = String(reason || "").trim() || `与问题相关的片段 ${index + 1}`;
  const sourceLabel = String(sourceVideoName || "").trim();

  return `
    <figcaption class="chat-clip-caption">
      <div class="chat-clip-time">${escapeHtml(timeLabel)}</div>
      <div class="chat-clip-reason">${escapeHtml(reasonLabel)}</div>
      ${sourceLabel ? `<div class="chat-clip-source">来源视频：${escapeHtml(sourceLabel)}</div>` : ""}
    </figcaption>
  `;
}

function buildChatMediaHtml(taskKey, clips, segments) {
  if (!taskKey || taskKey === GLOBAL_CHAT_KEY) return "";
  const normalizedClips = normalizeClipList(clips);
  const normalizedSegments = normalizeSegmentList(segments);

  if (normalizedClips.length) {
    const items = normalizedClips
      .map((clip, index) => {
        const segment = normalizedSegments[index] || {};
        const clipTaskKey = String(clip.clip_job_id || taskKey || "");
        const src = `/jobs/${encodeURIComponent(clipTaskKey)}/qa/clips/${encodeURIComponent(clip.clip_id)}`;
        return `
          <figure class="chat-clip-card">
            <video class="chat-clip-video" controls preload="metadata" playsinline src="${src}"></video>
            ${buildClipCaptionHtml(
              clip.t_start || segment.t_start,
              clip.t_end || segment.t_end,
              clip.reason || segment.reason,
              index,
              clip.source_video_name || segment.source_video_name
            )}
          </figure>
        `;
      })
      .join("");

    return `<div class="chat-clips">${items}</div>`;
  }

  if (!normalizedSegments.length) return "";

  const items = normalizedSegments
    .map((seg, index) => {
      const startSec = Math.max(0, timeTextToSeconds(seg.t_start));
      const endSec = Math.max(startSec + 1, timeTextToSeconds(seg.t_end || seg.t_start));
      const sourceTaskKey = String(seg.source_job_id || taskKey || "");
      const sourceSrc = `/jobs/${encodeURIComponent(sourceTaskKey)}/video`;
      return `
        <figure class="chat-clip-card">
          <video
            class="chat-clip-video chat-range-video"
            controls
            preload="metadata"
            playsinline
            src="${sourceSrc}"
            data-clip-start="${startSec}"
            data-clip-end="${endSec}"
          ></video>
          ${buildClipCaptionHtml(seg.t_start, seg.t_end, seg.reason, index, seg.source_video_name)}
        </figure>
      `;
    })
    .join("");

  return `<div class="chat-clips">${items}</div>`;
}

function bindRangeVideoPlayers(root) {
  if (!root) return;
  const players = root.querySelectorAll(".chat-range-video");
  for (const video of players) {
    if (video.dataset.boundRangePlayer === "1") continue;
    video.dataset.boundRangePlayer = "1";

    const start = Math.max(0, Number(video.dataset.clipStart || 0));
    const end = Math.max(start + 1, Number(video.dataset.clipEnd || start + 1));

    const resetToStart = () => {
      try {
        if (Number.isFinite(start)) {
          video.currentTime = start;
        }
      } catch {
        // ignore seek errors before metadata is ready
      }
    };

    video.addEventListener("loadedmetadata", () => {
      resetToStart();
    });

    video.addEventListener("play", () => {
      if (video.currentTime < start || video.currentTime >= end) {
        resetToStart();
      }
    });

    video.addEventListener("seeking", () => {
      if (video.currentTime < start) {
        video.currentTime = start;
      }
      if (video.currentTime > end) {
        video.currentTime = end;
        video.pause();
      }
    });

    video.addEventListener("timeupdate", () => {
      if (video.currentTime >= end) {
        video.pause();
      }
    });
  }
}

function appendChat(role, markdownText, options = {}) {
  const { persist = true, taskKey = getActiveTaskKey(), clips = [], segments = [] } = options;
  const normalizedRole = role === "user" ? "user" : "assistant";
  const text = String(markdownText || "");
  const normalizedClips = normalizeClipList(clips);
  const normalizedSegments = normalizeSegmentList(segments);
  const item = document.createElement("div");
  item.className = `chat-item ${normalizedRole}`;

  let roleTitle;
  if (state.theme === "ptt") {
    roleTitle = normalizedRole === "user" ? "USER" : "BOT";
  } else {
    roleTitle = normalizedRole === "user" ? "你" : "助手";
  }

  item.innerHTML = `
    <div class="chat-role">${roleTitle}</div>
    <div class="chat-bubble">${normalizedRole === "assistant" ? markdownToHtml(text) : escapeHtml(text)}</div>
    ${normalizedRole === "assistant" ? buildChatMediaHtml(taskKey, normalizedClips, normalizedSegments) : ""}
  `;

  els.chatFeed.appendChild(item);
  bindRangeVideoPlayers(item);
  els.chatFeed.scrollTop = els.chatFeed.scrollHeight;

  if (persist) {
    saveChatEntry(taskKey, normalizedRole, text, {
      clips: normalizedClips,
      segments: normalizedSegments,
    });
  }

  updateModernHeroVisibility();
}

async function refreshHealth() {
  try {
    const data = await api("/health");
    els.healthText.textContent = `[API: ${data.status} @ ${data.time}]`;
  } catch (err) {
    els.healthText.textContent = `[API: DOWN - ${err.message}]`;
  }
}

async function refreshJobs(options = {}) {
  const { silent = true } = options;
  const prevSelectedJobId = state.selectedJobId;

  try {
    const data = await api("/jobs");
    state.jobsMap = data || {};
    syncSelectedJobWithJobs();
    renderJobsSelect();
    renderJobsList();

    if (state.jobContextTargetId && !state.jobsMap[state.jobContextTargetId]) {
      closeJobContextMenu();
    }

    const selectedJob = state.selectedJobId ? state.jobsMap[state.selectedJobId] : null;
    if (selectedJob) {
      notifyJobFailureOnce(selectedJob);
    }
    updateSelectedJobCard(selectedJob || null);

    if (prevSelectedJobId !== state.selectedJobId) {
      renderChatHistoryForSelectedTask();
    }
  } catch (err) {
    if (!silent) log(`刷新视频失败: ${err.message}`, "error");
  }
}

async function refreshSelectedDetail(options = {}) {
  const { silent = true } = options;
  if (!state.selectedJobId) {
    if (!silent) log("请先选择视频", "error");
    return;
  }

  try {
    const job = await api(`/jobs/${state.selectedJobId}`);
    state.jobsMap[job.job_id] = job;
    notifyJobFailureOnce(job);
    updateSelectedJobCard(job);
    renderJobsSelect();
    renderJobsList();
  } catch (err) {
    if (!silent) log(`刷新视频详情失败: ${err.message}`, "error");
  }
}

async function refreshSummaryPreview(options = {}) {
  const { silent = true } = options;
  if (!state.selectedJobId) {
    if (!silent) log("请先选择视频", "error");
    return;
  }

  const job = getSelectedJob();
  const status = String(job?.status || "");

  if (job && status !== "succeeded") {
    syncSummaryHintByJob(job);
    if (!silent) {
      if (status === "failed") {
        log(`视频失败，暂无摘要：${summarizeJobError(job.error || job.message)}`, "error");
      } else if (status === "canceled") {
        log("视频已取消，暂无摘要", "info");
      } else {
        log(`视频状态：${getJobStatusLabel(status)}，摘要尚未生成`, "info");
      }
    }
    return;
  }

  try {
    const data = await api(`/jobs/${state.selectedJobId}/summary`);
    const md = data.final_markdown || data.draft_markdown || "(暂无摘要)";
    els.summaryRendered.innerHTML = markdownToHtml(md);
    if (!silent) log("摘要已刷新", "ok");
  } catch (err) {
    const errMsg = String(err?.message || err || "");
    if (errMsg.includes("404")) {
      if (job) {
        syncSummaryHintByJob(job);
      }
      if (!silent) log("摘要暂不可用（可能仍在生成或写入中），请稍后重试", "info");
      return;
    }

    if (!silent) log(`刷新摘要失败: ${errMsg}`, "error");
  }
}

function setUploadButtonsState(isUploading) {
  state.uploadInProgress = !!isUploading;

  if (els.uploadBtn) {
    els.uploadBtn.disabled = state.uploadInProgress;
    els.uploadBtn.textContent = state.uploadInProgress ? "上传中..." : "上传并开始分析";
  }

  if (els.sidebarUploadTriggerBtn) {
    els.sidebarUploadTriggerBtn.disabled = state.uploadInProgress;
    els.sidebarUploadTriggerBtn.textContent = state.uploadInProgress ? "上传中..." : "上传视频";
  }
}

function buildUploadFormData(file) {
  const form = new FormData();
  form.append("file", file);
  form.append("output_root", els.uploadOutputRoot.value.trim() || "outputs/api_jobs");
  form.append("mystery_mode", String(els.uploadMysteryMode.checked));
  form.append("resume", String(els.uploadResume.checked));
  return form;
}

async function uploadVideoAndCreateJob(file, source = "侧边栏") {
  if (!file) {
    log("请先选择视频文件", "error");
    appendChat("assistant", "请先选择或上传一个视频文件。", {
      taskKey: getActiveTaskKey(),
    });
    return;
  }

  if (state.uploadInProgress) {
    log("已有上传中的视频会话，请稍候", "info");
    return;
  }

  setUploadButtonsState(true);
  try {
    const form = buildUploadFormData(file);
    const job = await api("/jobs/upload", { method: "POST", body: form });

    state.jobsMap[job.job_id] = job;
    state.selectedJobId = job.job_id;
    renderJobsSelect();
    renderJobsList();
    updateSelectedJobCard(job);
    renderChatHistoryForSelectedTask({ showEmptyHint: false });
    closeJobContextMenu();

    appendChat(
      "assistant",
      `已通过${source}创建视频 **${getJobDisplayName(job)}**。\n\n视频上传成功，分析正在进行中，请稍后在该视频窗口继续提问。`,
      { taskKey: job.job_id }
    );

    if (els.uploadJobForm) {
      els.uploadJobForm.reset();
    }
    if (els.uploadResume) {
      els.uploadResume.checked = true;
    }
    if (els.sidebarUploadInput) {
      els.sidebarUploadInput.value = "";
    }

    log(`上传成功，视频ID: ${job.job_id}`, "ok");
  } catch (err) {
    log(`上传失败: ${err.message}`, "error");
    appendChat("assistant", `视频上传失败：${err.message}`);
  } finally {
    if (els.sidebarUploadInput) {
      els.sidebarUploadInput.value = "";
    }
    setUploadButtonsState(false);
  }
}

async function handleUploadJob(event) {
  event.preventDefault();

  const file = els.uploadFile.files?.[0];
  await uploadVideoAndCreateJob(file, "侧边栏");
}

async function handleCancelJob() {
  if (!state.selectedJobId) {
    log("请先选择视频", "error");
    return;
  }

  try {
    const job = await api(`/jobs/${state.selectedJobId}/cancel`, { method: "POST" });
    state.jobsMap[job.job_id] = job;
    updateSelectedJobCard(job);
    renderJobsSelect();
    renderJobsList();
    log("取消请求已发送，若无需保留该任务，可在左侧列表右键删除", "info");
  } catch (err) {
    log(`取消失败: ${err.message}`, "error");
  }
}

async function handleFetchResult() {
  if (!state.selectedJobId) {
    log("请先选择视频", "error");
    return;
  }

  const job = getSelectedJob();
  const status = String(job?.status || "");

  if (job && status !== "succeeded") {
    if (status === "failed") {
      const errBrief = summarizeJobError(job.error || job.message);
      els.jobResultPre.textContent = JSON.stringify(
        {
          job_id: job.job_id,
          status: "failed",
          message: "视频处理失败，暂无结果路径",
          error: errBrief,
        },
        null,
        2
      );
      log(`视频失败，无法获取结果：${errBrief}`, "error");
      return;
    }

    els.jobResultPre.textContent = JSON.stringify(
      {
        job_id: job.job_id,
        status,
        message: `视频状态：${getJobStatusLabel(status)}，完成后可获取结果路径`,
      },
      null,
      2
    );
    log(`视频状态：${getJobStatusLabel(status)}，完成后可获取结果`, "info");
    return;
  }

  try {
    const data = await api(`/jobs/${state.selectedJobId}/result`);
    els.jobResultPre.textContent = JSON.stringify(data, null, 2);
    log("结果路径已更新", "ok");
  } catch (err) {
    const errMsg = String(err?.message || err || "");
    if (errMsg.includes("409")) {
      log("视频尚未成功完成，暂无结果路径", "info");
      return;
    }
    log(`获取结果失败: ${errMsg}`, "error");
  }
}

function parseQaMysteryMode() {
  const v = String(els.qaMysteryModeSelect.value || "");
  if (v === "true") return true;
  if (v === "false") return false;
  return null;
}

function parseThemeCommand(rawInput) {
  const normalized = String(rawInput || "").trim().toUpperCase();
  if (normalized === "ONE_PIECE") return "ptt";
  if (normalized === "BOT") return "modern";
  return "";
}

async function consumeThemeCommand(rawInput) {
  const nextTheme = parseThemeCommand(rawInput);
  if (!nextTheme) return false;

  if (els.questionInput) {
    els.questionInput.value = "";
    autoResizeQuestionInput();
  }

  if (state.themeTransitionRunning) {
    log("界面切换进行中，请稍候", "info");
    return true;
  }

  if (state.theme === nextTheme) {
    log(`当前已是${getThemeLabel(nextTheme)}`, "info");
    return true;
  }

  await playThemeTransition(nextTheme);
  return true;
}

async function handleAskQuestion(event) {
  event.preventDefault();

  const question = String(els.questionInput?.value || "").trim();
  if (await consumeThemeCommand(question)) {
    return;
  }

  if (!question) {
    log("请输入问题", "error");
    return;
  }

  if (!state.selectedJobId) {
    log("请先选择视频", "error");
    appendChat("assistant", "请先在侧边栏选择一个视频窗口，或点击左侧“上传视频”。", {
      taskKey: GLOBAL_CHAT_KEY,
    });
    return;
  }

  const job = getSelectedJob();
  const status = String(job?.status || "");

  if (!job) {
    log("当前视频信息不存在，请刷新视频列表", "error");
    return;
  }

  if (status === "failed") {
    const chatSummary = summarizeJobErrorForChat(job.error || job.message);
    log(`视频处理失败，无法问答：${chatSummary.summary}`, "error");
    appendChatToTask(
      state.selectedJobId,
      "assistant",
      buildJobFailureChatMessage(job, {
        followUp: "请重新上传视频创建新会话。",
      })
    );
    return;
  }

  if (status === "canceled") {
    const displayName = getJobDisplayName(job);
    log("视频已取消，无法问答", "info");
    appendChatToTask(
      state.selectedJobId,
      "assistant",
      `视频 **${displayName}** 已取消，当前无法问答。请重新上传视频创建新会话。`
    );
    return;
  }

  if (!isJobReadyForQa(job)) {
    const displayName = getJobDisplayName(job);
    log(`视频状态：${getJobStatusLabel(status)}，暂不可问答`, "info");
    appendChatToTask(
      state.selectedJobId,
      "assistant",
      `视频 **${displayName}** 当前状态为 **${getJobStatusLabel(status)}**，请等待分析完成后再提问。`
    );
    return;
  }

  const mysteryMode = parseQaMysteryMode();
  closeModernSidebarOnMobile();
  appendChat("user", question);
  els.questionInput.value = "";
  autoResizeQuestionInput();
  setAskingState(true);

  try {
    const payload = { question };
    if (mysteryMode !== null) {
      payload.mystery_mode = mysteryMode;
    }

    const data = await api(`/jobs/${state.selectedJobId}/qa`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const answerMarkdown = sanitizeQaAnswerMarkdown(data.answer_markdown || "(无回答)");

    appendChat("assistant", answerMarkdown, {
      clips: Array.isArray(data.answer_clips) ? data.answer_clips : [],
      segments: Array.isArray(data.answer_segments) ? data.answer_segments : [],
    });
    log("当前视频问答完成", "ok");
  } catch (err) {
    appendChat("assistant", `问答失败：${err.message}`);
    log(`问答失败: ${err.message}`, "error");
  } finally {
    setAskingState(false);
  }
}

function togglePolling() {
  state.pollOn = !state.pollOn;
  els.togglePollBtn.textContent = `自动刷新：${state.pollOn ? "ON" : "OFF"}`;
  log(`自动刷新已${state.pollOn ? "开启" : "关闭"}`);
}

function startPolling() {
  if (state.pollTimer) clearInterval(state.pollTimer);

  state.pollTimer = setInterval(async () => {
    if (!state.pollOn) return;
    await refreshHealth();
    await refreshJobs({ silent: true });
    await refreshSelectedDetail({ silent: true });
  }, 3000);
}

function bindEvents() {
  els.uploadJobForm.addEventListener("submit", handleUploadJob);

  els.refreshJobsBtn.addEventListener("click", async () => {
    await refreshJobs({ silent: false });
    await refreshSelectedDetail({ silent: false });
  });

  els.refreshDetailBtn.addEventListener("click", () => refreshSelectedDetail({ silent: false }));

  els.jobsSelect.addEventListener("change", async (e) => {
    closeJobContextMenu();
    await switchToJob(e.target.value || null, { silent: false });
  });

  if (els.jobsList) {
    els.jobsList.addEventListener("click", async (event) => {
      const card = event.target.closest(".job-card");
      if (!card) return;
      closeJobContextMenu();
      await switchToJob(card.dataset.jobId || null, { silent: false });
    });

    els.jobsList.addEventListener("contextmenu", async (event) => {
      const card = event.target.closest(".job-card");
      if (!card) return;
      event.preventDefault();

      const jobId = card.dataset.jobId || "";
      const job = state.jobsMap[jobId];
      if (!job) return;

      await switchToJob(jobId, {
        silent: true,
        refreshDetail: false,
        refreshSummary: false,
        closeSidebar: false,
      });
      openJobContextMenu(job, event.clientX, event.clientY);
    });
  }

  els.cancelJobBtn.addEventListener("click", handleCancelJob);
  els.fetchResultBtn.addEventListener("click", handleFetchResult);

  els.refreshSummaryBtn.addEventListener("click", () => refreshSummaryPreview({ silent: false }));
  els.refreshSummaryInlineBtn.addEventListener("click", () => refreshSummaryPreview({ silent: false }));

  els.qaForm.addEventListener("submit", handleAskQuestion);
  els.togglePollBtn.addEventListener("click", togglePolling);

  if (els.themeToggleBtn) {
    els.themeToggleBtn.addEventListener("click", toggleTheme);
  }

  if (els.modernSidebarToggleBtn) {
    els.modernSidebarToggleBtn.addEventListener("click", toggleModernSidebar);
  }

  if (els.modernSidebarOverlay) {
    els.modernSidebarOverlay.addEventListener("click", closeModernSidebarOnMobile);
  }

  if (els.modernProfileBtn) {
    els.modernProfileBtn.addEventListener("click", () => {
      log("用户菜单（示意）", "info");
    });
  }

  if (els.sidebarUploadTriggerBtn) {
    els.sidebarUploadTriggerBtn.addEventListener("click", () => {
      if (state.uploadInProgress) {
        log("上传中，请稍候", "info");
        return;
      }
      if (!els.sidebarUploadInput) {
        log("未找到上传输入控件", "error");
        return;
      }
      els.sidebarUploadInput.click();
    });
  }

  if (els.sidebarUploadInput) {
    els.sidebarUploadInput.addEventListener("change", async (event) => {
      const file = event.target.files?.[0];
      await uploadVideoAndCreateJob(file, "左侧工具栏");
    });
  }

  if (els.botPixelSpider) {
    els.botPixelSpider.addEventListener("pointerdown", handleSpiderPointerDown);
    els.botPixelSpider.addEventListener("pointercancel", handleSpiderPointerCancel);
    els.botPixelSpider.addEventListener("lostpointercapture", handleSpiderPointerCancel);
  }

  window.addEventListener("resize", handleViewportResize);
  window.addEventListener("pointermove", handleSpiderPointerMove);
  window.addEventListener("pointerup", handleSpiderPointerUp);
  window.addEventListener("pointercancel", handleSpiderPointerCancel);
  window.addEventListener("blur", () => finishSpiderDrag());

  if (els.questionInput) {
    els.questionInput.addEventListener("input", autoResizeQuestionInput);
    els.questionInput.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        els.qaForm?.requestSubmit();
      }
    });
  }

  if (els.jobContextDeleteBtn) {
    els.jobContextDeleteBtn.addEventListener("click", () => {
      const jobId = state.jobContextTargetId;
      const job = state.jobsMap[jobId];
      if (!job || !canHardDeleteJob(job)) return;
      openDeleteDialog(jobId);
      closeJobContextMenu();
    });
  }

  if (els.jobDeleteCancelBtn) {
    els.jobDeleteCancelBtn.addEventListener("click", closeDeleteDialog);
  }

  if (els.jobDeleteConfirmBtn) {
    els.jobDeleteConfirmBtn.addEventListener("click", confirmDeleteJob);
  }

  document.addEventListener("mousedown", handleModernClickHeart);

  document.addEventListener("click", (event) => {
    if (els.jobContextMenu && !els.jobContextMenu.hidden && !els.jobContextMenu.contains(event.target)) {
      closeJobContextMenu();
    }

    if (event.target === els.jobDeleteDialog) {
      closeDeleteDialog();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeJobContextMenu();
      closeDeleteDialog();
    }
  });
}

async function init() {
  bindEvents();
  applyTheme(getSavedTheme(), { persist: false });
  resetSpiderDragPose();
  updateSelectedJobCard(null);
  setAskingState(false);
  setUploadButtonsState(false);
  syncDeleteDialogButtons();
  autoResizeQuestionInput();
  renderChatHistoryForSelectedTask();
  renderJobsList();
  updateModernHeroVisibility();
  els.summaryRendered.innerHTML = markdownToHtml("(请先上传并完成分析)");
  await refreshHealth();
  await refreshJobs({ silent: true });
  await refreshSelectedDetail({ silent: true });
  renderChatHistoryForSelectedTask();
  startPolling();
  log("前端已就绪");
}

init();
