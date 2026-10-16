"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const state = {
    csrf: null,
    authenticated: false,
    system: null,
    selectedFile: null,
    upload: null,
    events: null,
    poll: null,
    loadingJobs: false,
    cards: new Map(),
    jobs: [],
    busyJobs: new Set(),
    preferredModeApplied: false,
  };
  const terminal = new Set(["completed", "failed", "cancelled"]);
  const statuses = { queued: "Queued", processing: "Processing", completed: "Ready to review", failed: "Failed", cancelled: "Cancelled" };
  const stages = { probing: "Reading video", detecting: "Finding & blurring faces", encoding: "Encoding MP4", finalizing: "Finishing your video" };
  const modeHelp = { auto: "Uses GPU acceleration when available, with CPU fallback.", gpu: "Requires GPU face detection and NVIDIA video encoding.", cpu: "Processes entirely on the CPU. Usually takes longer." };

  function announce(message) { $("live-status").textContent = message; }
  function showError(message) { $("workspace-error-text").textContent = message; $("workspace-error").hidden = false; announce(message); }
  function dismissError() { $("workspace-error").hidden = true; }
  function connection(label, status) { $("connection-label").textContent = label; $("connection-status").dataset.state = status; }
  function size(bytes) {
    if (!Number.isFinite(Number(bytes)) || Number(bytes) < 0) return "";
    const value = Number(bytes);
    if (value < 1024) return `${value} B`;
    if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
    const unit = value >= 1024 ** 3 ? "GB" : "MB";
    return `${(value / (unit === "GB" ? 1024 ** 3 : 1024 ** 2)).toFixed(1)} ${unit}`;
  }
  function duration(seconds) {
    if (!Number.isFinite(Number(seconds)) || seconds === null || Number(seconds) < 0) return "";
    const value = Math.ceil(Number(seconds));
    const hours = Math.floor(value / 3600);
    return hours ? `${hours}:${String(Math.floor(value / 60) % 60).padStart(2, "0")}:${String(value % 60).padStart(2, "0")}` : `${Math.floor(value / 60)}:${String(value % 60).padStart(2, "0")}`;
  }
  function percentage(value) { return Math.max(0, Math.min(100, Math.round(Number(value) || 0))); }
  function jobUrl(id, action = "") { return `/api/jobs/${encodeURIComponent(id)}${action ? `/${action}` : ""}`; }

  async function errorMessage(response, fallback) {
    try {
      const body = await response.json();
      if (typeof body.detail === "string") return body.detail;
      if (typeof body.error === "string") return body.error;
    } catch { /* A proxy may return a non-JSON error. */ }
    return fallback;
  }

  async function api(path, options = {}) {
    const headers = new Headers(options.headers || {});
    if (options.method && options.method !== "GET" && state.csrf) headers.set("X-CSRF-Token", state.csrf);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30000);
    let response;
    try {
      response = await fetch(path, { ...options, headers, credentials: "same-origin", cache: "no-store", signal: controller.signal });
    } catch (error) {
      if (error.name === "AbortError") throw new Error("The server took too long to respond. Please try again.");
      throw new Error("Couldn’t reach Veil. Check your connection and try again.");
    } finally { clearTimeout(timeout); }
    if (response.status === 401 && path !== "/api/login") {
      showLogin("Your session expired. Enter your password to continue.");
      throw new Error("Your session expired.");
    }
    if (!response.ok) throw new Error(await errorMessage(response, `Request failed (${response.status}). Please try again.`));
    if (response.status === 204) return null;
    return response.json();
  }

  function stopUpdates() {
    if (state.events) state.events.close();
    state.events = null;
    if (state.poll) clearInterval(state.poll);
    state.poll = null;
  }

  function showLogin(message = "") {
    state.authenticated = false;
    state.csrf = null;
    stopUpdates();
    if (state.upload) state.upload.abort();
    for (const card of state.cards.values()) {
      card.video.pause();
      card.video.removeAttribute("src");
      card.video.load();
      card.node.remove();
    }
    state.cards.clear();
    state.jobs = [];
    state.system = null;
    state.preferredModeApplied = false;
    selectFile(null);
    $("boot-screen").hidden = true;
    $("workspace").hidden = true;
    $("login-screen").hidden = false;
    $("login-error").textContent = message;
    $("login-error").hidden = !message;
    $("password").value = "";
    $("password").focus();
  }

  async function showWorkspace(session) {
    state.csrf = session.csrf_token;
    state.authenticated = true;
    $("boot-screen").hidden = true;
    $("login-screen").hidden = true;
    $("workspace").hidden = false;
    $("jobs-loading").hidden = false;
    $("empty-state").hidden = true;
    dismissError();
    connection("Connecting", "connecting");
    await Promise.allSettled([loadSystem(), loadJobs()]);
    if (state.authenticated) startUpdates();
  }

  async function loadSystem() {
    $("refresh-system").disabled = true;
    try {
      const system = await api("/api/system");
      if (!state.authenticated) return;
      state.system = system;
      const gpu = Boolean(system.gpu_available);
      const encoder = Boolean(system.nvenc_available);
      $("hardware-status").textContent = gpu && encoder ? "GPU acceleration ready" : gpu ? "GPU detection · CPU encoding" : encoder ? "CPU detection · GPU encoding available" : "CPU processing available";
      $("hardware-description").textContent = gpu && encoder ? "Face detection and video encoding can run on your GPU." : "Automatic mode selects the available hardware for each step.";
      const details = typeof system.details === "string" ? system.details : system.details ? JSON.stringify(system.details, null, 2) : "";
      $("hardware-details").hidden = !details;
      $("hardware-details-text").textContent = details;
      $("model-notice").hidden = system.model_ready !== false;
      $("model-notice").textContent = "The face detection model is missing. Run ‘veil init’ on this machine, then refresh.";
      const max = Number(system.max_upload_bytes);
      $("file-limit").textContent = `iPhone videos, MOV, MP4 & more${max > 0 ? ` · Max ${size(max)}` : ""}`;
      const gpuInput = document.querySelector('input[name="mode"][value="gpu"]');
      gpuInput.disabled = !gpu || !encoder || Boolean(state.upload);
      if (!state.preferredModeApplied && ["auto", "gpu", "cpu"].includes(system.default_mode)) {
        document.querySelector(`input[name="mode"][value="${system.default_mode}"]`).checked = true;
        state.preferredModeApplied = true;
        updateModeHelp();
      }
      if (gpuInput.checked && gpuInput.disabled) {
        document.querySelector('input[name="mode"][value="auto"]').checked = true;
        updateModeHelp();
      }
    } catch (error) {
      if (!state.authenticated) return;
      state.system = null;
      $("hardware-status").textContent = "Couldn’t check this machine";
      $("hardware-description").textContent = "Refresh to check acceleration and upload availability.";
      showError(error.message);
    } finally {
      $("refresh-system").disabled = false;
      updateUploadButton();
    }
  }

  async function loadJobs() {
    if (state.loadingJobs || !state.authenticated) return;
    state.loadingJobs = true;
    try {
      const data = await api("/api/jobs");
      if (state.authenticated) {
        renderJobs(data.jobs || []);
        if (state.poll) connection("Updates every 5s", "connected");
      }
    } catch (error) {
      if (state.authenticated) {
        connection(navigator.onLine ? "Reconnecting" : "Offline", "offline");
        $("jobs-loading").hidden = false;
        $("jobs-loading").textContent = "Couldn’t load videos. Reconnecting…";
      }
    } finally { state.loadingJobs = false; }
  }

  function startPolling() {
    if (!state.poll && state.authenticated) {
      loadJobs();
      state.poll = setInterval(loadJobs, 5000);
    }
  }

  function startUpdates() {
    stopUpdates();
    if (!("EventSource" in window)) {
      connection("Updates every 5s", "connected");
      startPolling();
      return;
    }
    const events = new EventSource("/api/events");
    state.events = events;
    events.addEventListener("open", () => {
      if (!state.authenticated || state.events !== events) return;
      if (state.poll) clearInterval(state.poll);
      state.poll = null;
      connection("Live updates", "connected");
      loadJobs();
    });
    events.addEventListener("jobs", (event) => {
      if (!state.authenticated || state.events !== events) return;
      try {
        const data = JSON.parse(event.data);
        if (Array.isArray(data.jobs)) renderJobs(data.jobs);
      } catch { startPolling(); }
    });
    events.addEventListener("error", () => {
      if (!state.authenticated || state.events !== events) return;
      connection(navigator.onLine ? "Reconnecting" : "Offline", "offline");
      startPolling();
    });
    events.addEventListener("expired", () => {
      if (state.authenticated && state.events === events) showLogin("Your session expired. Enter your password to continue.");
    });
  }

  function createCard(job) {
    const node = $("job-template").content.firstElementChild.cloneNode(true);
    const find = (selector) => node.querySelector(selector);
    const card = {
      node,
      name: find(".job-name"), meta: find(".job-meta"), status: find(".job-status"),
      processing: find(".job-processing"), stage: find(".job-stage"), percent: find(".job-percent"),
      progress: find(".job-progress"), frames: find(".job-frames"), eta: find(".job-eta"),
      message: find(".job-message"), error: find(".job-error"), backend: find(".job-backend"),
      cancel: find(".job-cancel"), remove: find(".job-delete"), previewToggle: find(".job-preview-toggle"),
      download: find(".job-download"), preview: find(".job-preview"), video: find(".result-video"), data: null,
    };
    card.cancel.addEventListener("click", () => mutateJob(job.id, "cancel"));
    card.remove.addEventListener("click", () => mutateJob(job.id, "delete"));
    card.previewToggle.addEventListener("click", () => {
      const open = card.preview.hidden;
      card.preview.hidden = !open;
      card.previewToggle.setAttribute("aria-expanded", String(open));
      card.previewToggle.querySelector("span").textContent = open ? "Close preview" : "Review video";
      if (open && !card.video.hasAttribute("src")) {
        card.video.src = jobUrl(job.id, "preview");
        card.video.load();
      } else if (!open) { card.video.pause(); }
    });
    card.video.addEventListener("error", () => {
      showError("This browser couldn’t play the preview. Try downloading the MP4 to review it.");
    });
    return card;
  }

  function renderJobs(jobs) {
    if (!Array.isArray(jobs)) return;
    state.jobs = jobs;
    $("jobs-loading").hidden = true;
    $("empty-state").hidden = jobs.length > 0;
    $("job-count").textContent = String(jobs.length);
    const present = new Set();
    for (const job of jobs) {
      if (!job || typeof job.id !== "string") continue;
      present.add(job.id);
      let card = state.cards.get(job.id);
      if (!card) { card = createCard(job); state.cards.set(job.id, card); }
      const previous = card.data;
      card.data = job;
      card.node.dataset.jobId = job.id;
      card.node.dataset.status = job.status;
      card.name.textContent = job.filename || "Video";
      card.name.title = job.filename || "Video";
      const media = job.media || {};
      const metadata = [size(job.source_bytes), duration(media.duration)];
      if (media.width && media.height) metadata.push(`${media.width} × ${media.height}`);
      card.meta.textContent = metadata.filter(Boolean).join(" · ");
      card.status.textContent = statuses[job.status] || "Updating";
      const active = !terminal.has(job.status);
      card.processing.hidden = !active;
      const progress = percentage(job.progress);
      card.stage.textContent = job.status === "queued" ? "Waiting for the worker" : stages[job.stage] || "Processing video";
      card.percent.textContent = job.status === "queued" ? "" : `${progress}%`;
      card.progress.value = progress;
      card.progress.setAttribute("aria-label", `${job.filename || "Video"}: ${card.stage.textContent}`);
      const frameDetails = [];
      if (Number(job.total_frames) > 0) frameDetails.push(`${Number(job.frames_processed || 0).toLocaleString()} / ${Number(job.total_frames).toLocaleString()} frames`);
      if (Number(job.fps) > 0 && job.status === "processing") frameDetails.push(`${Number(job.fps).toFixed(1)} fps`);
      card.frames.textContent = frameDetails.join(" · ");
      card.eta.textContent = job.eta_seconds !== null && Number(job.eta_seconds) > 0 ? `About ${duration(job.eta_seconds)} left` : "";
      card.message.textContent = job.message || "";
      card.message.hidden = !job.message || job.status === "failed" || job.message.toLowerCase() === card.status.textContent.toLowerCase();
      card.error.textContent = job.error || (job.status === "failed" ? "Processing failed. Upload the video again to retry." : "");
      card.error.hidden = !card.error.textContent;
      card.backend.textContent = job.backend || (job.status === "queued" ? `${{ auto: "Automatic", gpu: "GPU", cpu: "CPU" }[job.mode] || "Automatic"} processing` : "");
      const busy = state.busyJobs.has(job.id);
      card.cancel.hidden = !active;
      card.cancel.disabled = busy;
      card.cancel.textContent = busy && active ? "Cancelling…" : "Cancel";
      card.cancel.setAttribute("aria-label", `Cancel processing ${job.filename || "video"}`);
      card.remove.hidden = active;
      card.remove.disabled = busy;
      card.remove.textContent = busy && !active ? "Deleting…" : "Delete";
      card.remove.setAttribute("aria-label", `Delete ${job.filename || "video"} and its files`);
      card.download.hidden = job.status !== "completed";
      card.previewToggle.hidden = job.status !== "completed";
      card.download.href = jobUrl(job.id, "download");
      card.download.setAttribute("aria-label", `Download blurred ${job.filename || "video"} as MP4`);
      card.previewToggle.setAttribute("aria-label", `Review blurred ${job.filename || "video"}`);
      if (job.status !== "completed" && !card.preview.hidden) {
        card.video.pause();
        card.preview.hidden = true;
        card.previewToggle.setAttribute("aria-expanded", "false");
      }
      // Existing elements keep their video playback and keyboard focus as progress arrives.
      const targetIndex = Array.from(present).length - 1;
      const currentAtIndex = $("job-list").children[targetIndex];
      if (currentAtIndex !== card.node) $("job-list").insertBefore(card.node, currentAtIndex || null);
      if (previous && previous.status !== job.status && terminal.has(job.status)) announce(`${job.filename || "Video"}: ${statuses[job.status]}.`);
    }
    for (const [id, card] of state.cards) {
      if (!present.has(id)) {
        card.video.pause();
        card.video.removeAttribute("src");
        card.video.load();
        card.node.remove();
        state.cards.delete(id);
      }
    }
  }

  async function mutateJob(id, action) {
    if (state.busyJobs.has(id)) return;
    if (action === "delete" && !window.confirm("Delete this video and all its files? This cannot be undone.")) return;
    state.busyJobs.add(id);
    renderJobs(state.jobs);
    try {
      await api(jobUrl(id, action === "cancel" ? "cancel" : ""), { method: action === "cancel" ? "POST" : "DELETE" });
      await loadJobs();
      announce(action === "cancel" ? "Cancellation requested." : "Video and its files deleted.");
    } catch (error) { if (state.authenticated) showError(error.message); }
    finally {
      state.busyJobs.delete(id);
      if (state.authenticated) renderJobs(state.jobs);
    }
  }

  function updateModeHelp() {
    const selected = document.querySelector('input[name="mode"]:checked');
    $("mode-help").textContent = modeHelp[selected ? selected.value : "auto"];
  }

  function updateUploadButton() {
    const uploading = Boolean(state.upload);
    $("upload-button").disabled = uploading || !state.selectedFile || !state.system || state.system.model_ready === false;
    $("video-file").disabled = uploading;
    $("clear-file").disabled = uploading;
    $("drop-zone").classList.toggle("is-disabled", uploading);
    $("upload-button-label").textContent = uploading ? "Uploading…" : "Blur faces";
    $("abort-upload").hidden = !uploading;
    for (const input of document.querySelectorAll('input[name="mode"]')) {
      input.disabled = uploading || (input.value === "gpu" && (!state.system?.gpu_available || !state.system?.nvenc_available));
    }
  }

  function selectFile(file) {
    if (state.upload) return;
    if (file && !/\.(mov|mp4|m4v|mkv|webm)$/i.test(file.name)) {
      selectFile(null);
      showError("Choose a MOV, MP4, M4V, MKV, or WebM video.");
      return;
    }
    if (file && Number(state.system?.max_upload_bytes) > 0 && file.size > Number(state.system.max_upload_bytes)) {
      selectFile(null);
      showError(`That video is too large. The upload limit is ${size(state.system.max_upload_bytes)}.`);
      return;
    }
    state.selectedFile = file || null;
    $("selected-file").hidden = !file;
    $("selected-filename").textContent = file ? file.name : "";
    $("selected-filename").title = file ? file.name : "";
    $("selected-size").textContent = file ? size(file.size) : "";
    $("file-prompt").textContent = file ? "Choose a different video" : "Choose a video";
    $("video-file").value = "";
    $("upload-progress-wrap").hidden = true;
    updateUploadButton();
  }

  function resetUpload() {
    state.upload = null;
    $("upload-progress-wrap").hidden = true;
    updateUploadButton();
  }

  function uploadVideo(event) {
    event.preventDefault();
    if (!state.selectedFile || state.upload || !state.system || state.system.model_ready === false) return;
    if (Number(state.system.max_upload_bytes) > 0 && state.selectedFile.size > Number(state.system.max_upload_bytes)) {
      showError(`That video is too large. The upload limit is ${size(state.system.max_upload_bytes)}.`);
      return;
    }
    dismissError();
    const file = state.selectedFile;
    const data = new FormData();
    data.append("file", file, file.name);
    data.append("mode", document.querySelector('input[name="mode"]:checked').value);
    const xhr = new XMLHttpRequest();
    state.upload = xhr;
    xhr.open("POST", "/api/jobs");
    xhr.setRequestHeader("X-CSRF-Token", state.csrf);
    xhr.responseType = "json";
    $("upload-progress-wrap").hidden = false;
    $("upload-progress-label").textContent = "Uploading";
    $("upload-percent").textContent = "0%";
    $("upload-progress").value = 0;
    updateUploadButton();
    xhr.upload.addEventListener("progress", (progress) => {
      if (progress.lengthComputable) {
        const value = percentage((progress.loaded / progress.total) * 100);
        $("upload-progress").value = value;
        $("upload-percent").textContent = `${value}%`;
        if (value === 100) $("upload-progress-label").textContent = "Checking upload…";
      } else { $("upload-progress").removeAttribute("value"); $("upload-percent").textContent = ""; }
    });
    xhr.addEventListener("load", () => {
      resetUpload();
      if (xhr.status === 401) { showLogin("Your session expired. Enter your password to continue."); return; }
      if (xhr.status >= 200 && xhr.status < 300) {
        selectFile(null);
        announce(`${file.name} uploaded. Face blurring is queued.`);
        loadJobs().then(() => {
          if (!state.authenticated) return;
          $("jobs-heading").setAttribute("tabindex", "-1");
          $("jobs-heading").focus({ preventScroll: true });
          if (window.matchMedia("(max-width: 760px)").matches) $("jobs-heading").scrollIntoView({ block: "start", behavior: "instant" });
        });
      } else {
        const detail = xhr.response?.detail;
        showError(typeof detail === "string" ? detail : `Upload failed (${xhr.status}). Please try again.`);
      }
    });
    xhr.addEventListener("error", () => { resetUpload(); showError("Upload interrupted. Check your connection, then try again."); });
    xhr.addEventListener("abort", () => { resetUpload(); announce("Upload cancelled."); if (state.authenticated) loadJobs(); });
    xhr.send(data);
  }

  $("login-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    $("login-button").disabled = true;
    $("login-button").querySelector("span").textContent = "Opening…";
    $("login-error").hidden = true;
    try {
      const session = await api("/api/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ password: $("password").value }) });
      $("password").value = "";
      await showWorkspace(session);
    } catch (error) {
      $("login-error").textContent = error.message;
      $("login-error").hidden = false;
      $("password").focus();
    } finally { $("login-button").disabled = false; $("login-button").querySelector("span").textContent = "Open workspace"; }
  });
  $("logout-button").addEventListener("click", async () => {
    $("logout-button").disabled = true;
    try {
      await api("/api/logout", { method: "POST" });
      showLogin();
    } catch (error) { if (state.authenticated) showError(error.message); }
    finally { $("logout-button").disabled = false; }
  });
  $("upload-form").addEventListener("submit", uploadVideo);
  $("video-file").addEventListener("change", (event) => selectFile(event.target.files[0]));
  $("clear-file").addEventListener("click", () => { selectFile(null); $("video-file").focus(); });
  $("abort-upload").addEventListener("click", () => { if (state.upload) state.upload.abort(); });
  $("refresh-system").addEventListener("click", loadSystem);
  $("dismiss-error").addEventListener("click", dismissError);
  for (const input of document.querySelectorAll('input[name="mode"]')) input.addEventListener("change", updateModeHelp);
  let dragDepth = 0;
  $("drop-zone").addEventListener("dragenter", (event) => { event.preventDefault(); dragDepth += 1; if (!state.upload) $("drop-zone").classList.add("is-dragover"); });
  $("drop-zone").addEventListener("dragover", (event) => { event.preventDefault(); if (event.dataTransfer) event.dataTransfer.dropEffect = state.upload ? "none" : "copy"; });
  $("drop-zone").addEventListener("dragleave", () => { dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) $("drop-zone").classList.remove("is-dragover"); });
  $("drop-zone").addEventListener("drop", (event) => {
    event.preventDefault();
    dragDepth = 0;
    $("drop-zone").classList.remove("is-dragover");
    const files = event.dataTransfer?.files;
    if (!files?.length || state.upload) return;
    if (files.length > 1) { showError("Choose one video at a time."); return; }
    selectFile(files[0]);
  });
  // Prevent a dropped file outside the picker from navigating away from the workspace.
  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("drop", (event) => event.preventDefault());
  window.addEventListener("online", () => { if (state.authenticated) { startUpdates(); loadJobs(); loadSystem(); } });
  window.addEventListener("offline", () => { if (state.authenticated) connection("Offline", "offline"); });
  document.addEventListener("visibilitychange", () => { if (!document.hidden && state.authenticated) loadJobs(); });
  window.addEventListener("beforeunload", (event) => { if (state.upload) { event.preventDefault(); event.returnValue = ""; } });

  (async () => {
    try { await showWorkspace(await api("/api/session")); }
    catch (error) { if (!state.authenticated) showLogin(error.message === "Your session expired." ? "" : "Couldn’t connect to Veil. Check the server and try again."); }
  })();
})();
