const form = document.querySelector("#job-form");
const promptInputs = [...document.querySelectorAll(".composer textarea")];
const durationInput = document.querySelector("#duration");
const resolutionInput = document.querySelector("#resolution");
const aspectRatioInput = document.querySelector("#aspect-ratio");
const modelInput = document.querySelector("#model");
const modelHint = document.querySelector("#model-hint");
const durationSpec = document.querySelector("#duration-spec");
const canvasSpec = document.querySelector("#canvas-spec");
const imageField = document.querySelector("#image-field");
const frameInputs = [
  { prefix: "", urlKey: "image_prompt_url", presenceKey: "has_first_frame", label: "start frame" },
  { prefix: "last-", urlKey: "last_image_prompt_url", presenceKey: "has_last_frame", label: "end frame" },
].map(frame => ({
  ...frame,
  input: document.querySelector(`#${frame.prefix}image`),
  preview: document.querySelector(`#${frame.prefix}image-preview`),
  empty: document.querySelector(`#${frame.prefix}upload-empty`),
  box: document.querySelector(`#${frame.prefix}upload-box`),
  change: document.querySelector(`#${frame.prefix}change-image`),
  remove: document.querySelector(`#${frame.prefix}remove-image`),
  previewUrl: null,
}));
const submitButton = document.querySelector("#submit-button");
const formError = document.querySelector("#form-error");
const queueElement = document.querySelector("#queue");
const emptyState = document.querySelector("#empty-state");
const queueCount = document.querySelector("#queue-count");
const enginePill = document.querySelector("#engine-pill");
const engineLabel = document.querySelector("#engine-label");

const cards = new Map();
let refreshInFlight = false;
let currentSnapshot = null;

function resizePromptInput(input) {
  input.style.height = "auto";
  input.style.height = `${input.scrollHeight}px`;
}

function selectedMode() {
  return form.elements.mode.value;
}

function updateMode() {
  const usesImage = selectedMode() === "image";
  imageField.hidden = !usesImage;
}

function canvasDimensions() {
  const resolution = Number(resolutionInput.value);
  const [ratioWidth, ratioHeight] = aspectRatioInput.value.split(":").map(Number);
  const ratio = ratioWidth / ratioHeight;
  let width = ratio >= 1 ? resolution * ratio : resolution;
  let height = ratio >= 1 ? resolution : resolution / ratio;
  const maxPixels = resolution * resolution * 7 / 4;
  if (width * height > maxPixels) {
    const scale = Math.sqrt(maxPixels / (width * height));
    width *= scale;
    height *= scale;
  }
  return [Math.max(32, Math.round(width / 32) * 32), Math.max(32, Math.round(height / 32) * 32)];
}

function updateOutputSpecs() {
  durationSpec.textContent = `${durationInput.value} sec`;
  const [width, height] = canvasDimensions();
  canvasSpec.textContent = `${width}×${height}`;
}

function updateModel() {
  const hints = {
    "minimax-h3": "H3 with the 4-step Turbo LoRA and native stereo audio.",
    "minimax-h3-base": "Regular H3 without the Turbo LoRA; 20 steps for maximum base-model quality.",
  };
  modelHint.textContent = hints[modelInput.value] || "MiniMax H3 generation.";
}

function updatePreview() {
  for (const frame of frameInputs) {
    if (frame.previewUrl) URL.revokeObjectURL(frame.previewUrl);
    const file = frame.input.files[0];
    frame.previewUrl = file ? URL.createObjectURL(file) : null;
    frame.preview.hidden = !file;
    frame.empty.hidden = Boolean(file);
    frame.change.hidden = !file;
    frame.remove.hidden = !file;
    if (file) frame.preview.src = frame.previewUrl;
    else frame.preview.removeAttribute("src");
  }
}

function formatAge(timestamp) {
  if (!timestamp) return "";
  const seconds = Math.max(0, Math.floor(Date.now() / 1000 - timestamp));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${seconds % 60}s`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

function formatDuration(seconds) {
  const rounded = Math.max(0, Math.floor(seconds));
  if (rounded < 60) return `${rounded}s`;
  const minutes = Math.floor(rounded / 60);
  if (minutes < 60) return `${minutes}m ${rounded % 60}s`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

function statusText(job) {
  if (job.status === "running") return "Running now";
  if (job.status === "queued") return `Waiting · #${job.queue_position}`;
  if (job.status === "completed") return "Ready";
  if (job.status === "paused") return "Paused";
  return "Failed";
}

function waitingReason(job, jobs) {
  if (job.status === "running") return "This job is actively generating.";
  if (job.status === "paused") return "Not running · Resume to rejoin the queue.";
  const priorities = { high: 0, medium: 1, low: 2 };
  const higherPriority = jobs.some(peer => ["running", "queued"].includes(peer.status)
    && priorities[peer.priority] < priorities[job.priority]);
  if (higherPriority) return "Not running · Waiting for higher-priority jobs.";
  if (job.queue_position === 1) return "Not running · Next in line.";
  return "Not running · Waiting for earlier jobs in the queue.";
}

function jobCard(job) {
  const card = document.createElement("article");
  card.className = "job panel";
  card.dataset.jobId = job.id;
  card.dataset.status = job.status;

  const top = document.createElement("div");
  top.className = "job-top";
  const copy = document.createElement("div");
  const mode = document.createElement("span");
  mode.className = "job-mode";
  const modeLabel = job.mode === "image" ? "◫ Image to video" : "✦ Text to video";
  const canvas = job.canvas_width && job.canvas_height
    ? `${job.canvas_width}×${job.canvas_height}`
    : `${job.resolution || 512}p ${job.aspect_ratio || "1:1"}`;
  mode.textContent = `${modeLabel} · ${job.model_label || "MiniMax H3"} · ${job.duration_seconds || 5} sec · ${canvas}`;
  const prompt = document.createElement("p");
  prompt.className = "job-prompt";
  prompt.textContent = job.prompt;
  copy.append(mode, prompt);
  const status = document.createElement("span");
  status.className = `status ${job.status}`;
  status.textContent = statusText(job);
  if (["running", "queued", "paused"].includes(job.status)) {
    const state = document.createElement("div");
    state.className = "job-state";
    const description = document.createElement("p");
    description.className = "job-state-description";
    state.append(status, description);
    card.append(state);
    top.append(copy);
  } else {
    top.append(copy, status);
  }
  card.append(top);

  for (const frame of frameInputs) {
    if (job[frame.urlKey]) {
      const imagePrompt = document.createElement("figure");
      imagePrompt.className = "job-image-prompt";
      const imageLabel = document.createElement("figcaption");
      imageLabel.className = "job-image-prompt-label";
      imageLabel.textContent = frame.label;
      const imageLink = document.createElement("a");
      imageLink.className = "job-prompt-image-link";
      imageLink.href = job[frame.urlKey];
      imageLink.target = "_blank";
      imageLink.rel = "noopener";
      imageLink.title = `View ${frame.label} at full size`;
      const image = document.createElement("img");
      image.className = "job-prompt-image";
      image.src = job[frame.urlKey];
      image.alt = frame.label;
      imageLink.append(image);
      imagePrompt.append(imageLabel, imageLink);
      card.append(imagePrompt);
    } else if (job.mode === "image" && job[frame.presenceKey]) {
      const unavailable = document.createElement("p");
      unavailable.className = "job-image-prompt-unavailable";
      unavailable.textContent = `${frame.label} unavailable`;
      card.append(unavailable);
    }
  }

  const meta = document.createElement("div");
  meta.className = "job-meta";
  const seed = document.createElement("span");
  seed.textContent = `Seed ${job.seed_text || job.seed}`;
  const timing = document.createElement("span");
  timing.className = "job-timing";
  if (job.status === "running") {
    timing.dataset.startedAt = job.started_at;
    timing.textContent = `${formatAge(job.started_at)} elapsed`;
  }
  else if (job.finished_at && job.started_at) timing.textContent = `${formatDuration(job.finished_at - job.started_at)} total`;
  else timing.textContent = `Added ${formatAge(job.created_at)} ago`;
  meta.append(seed, timing);
  card.append(meta);

  if (job.media_url) {
    const video = document.createElement("video");
    video.controls = true;
    video.preload = "metadata";
    video.playsInline = true;
    video.src = job.media_url;
    card.append(video);
  }

  if (job.error) {
    const error = document.createElement("p");
    error.className = "job-error";
    error.textContent = job.error;
    card.append(error);
  }

  if (["queued", "running", "paused", "completed", "failed"].includes(job.status)) {
    const actions = document.createElement("div");
    actions.className = "job-actions";
    if (job.media_url) {
      const download = document.createElement("a");
      download.href = job.media_url;
      download.download = `genvideo-${job.id}.mp4`;
      download.textContent = "Download";
      actions.append(download);
    }
    const copyButton = document.createElement("button");
    copyButton.type = "button";
    copyButton.textContent = "Copy to form";
    copyButton.addEventListener("click", () => copyToForm(job, copyButton));
    actions.append(copyButton);
    const pending = ["running", "queued", "paused"].includes(job.status);
    if (pending) {
      const pause = document.createElement("button");
      pause.type = "button";
      pause.textContent = job.status === "paused" ? "Resume" : "Pause";
      pause.addEventListener("click", () => jobAction(job.id,
        job.status === "paused" ? "resume" : "pause", pause));
      actions.append(pause);
      const priorityLabel = document.createElement("label");
      priorityLabel.className = "job-priority";
      priorityLabel.textContent = "Priority";
      const priority = document.createElement("select");
      priority.setAttribute("aria-label", "Job priority");
      for (const value of ["high", "medium", "low"]) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value[0].toUpperCase() + value.slice(1);
        priority.append(option);
      }
      priority.value = job.priority;
      priority.addEventListener("change", () => jobAction(job.id, "", priority, { priority: priority.value }));
      priorityLabel.append(priority);
      actions.append(priorityLabel);
      if (job.status === "queued") {
        const peers = currentSnapshot.jobs.filter(peer => peer.status === "queued" && peer.priority === job.priority);
        for (const direction of ["up", "down"]) {
          const move = document.createElement("button");
          move.type = "button";
          move.textContent = direction === "up" ? "↑" : "↓";
          move.title = `Move ${direction} within ${job.priority} priority`;
          move.setAttribute("aria-label", move.title);
          move.disabled = direction === "up" ? peers[0].id === job.id : peers.at(-1).id === job.id;
          move.addEventListener("click", () => jobAction(job.id, "", move, { move: direction }));
          actions.append(move);
        }
      }
    }
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "delete";
    remove.textContent = pending ? "Cancel" : "Remove";
    remove.addEventListener("click", () => deleteJob(job.id, remove));
    actions.append(remove);
    card.append(actions);
  }

  if (["running", "queued", "paused"].includes(job.status)) {
    const progress = document.createElement("div");
    progress.className = "job-progress";
    const label = document.createElement("p");
    label.className = "progress-label";
    const bar = document.createElement("progress");
    bar.setAttribute("aria-label", "Current stage progress");
    progress.append(label, bar);
    card.append(progress);
  }
  return card;
}

function render(snapshot) {
  currentSnapshot = snapshot;
  const labels = {
    idle: "Idle · model unloaded",
    starting: "Starting engine…",
    working: `${snapshot.engine_model_label || "Model"} · generating`,
    unloading: "Queue clear · unloading…",
  };
  engineLabel.textContent = labels[snapshot.engine_state] || snapshot.engine_state;
  enginePill.className = "engine-pill";
  if (snapshot.engine_state === "working") enginePill.classList.add("working");
  if (["starting", "unloading"].includes(snapshot.engine_state)) enginePill.classList.add("transition");
  const pending = snapshot.jobs.filter(job => ["running", "queued", "paused"].includes(job.status));
  const runningCount = pending.filter(job => job.status === "running").length;
  const pausedCount = pending.filter(job => job.status === "paused").length;
  queueCount.textContent = `${runningCount} running · ${snapshot.queued_count} waiting · ${pausedCount} paused`;
  const history = snapshot.jobs.filter(job => ["completed", "failed"].includes(job.status));
  document.querySelector("#history-count").textContent = `${history.length} jobs`;
  document.querySelector("#queue-tab-count").textContent = pending.length;
  document.querySelector("#history-tab-count").textContent = history.length;
  emptyState.hidden = pending.length > 0;
  document.querySelector("#history-empty").hidden = history.length > 0;
  const ids = new Set(snapshot.jobs.map(job => job.id));
  for (const [id, entry] of cards) {
    if (!ids.has(id)) { entry.card.remove(); cards.delete(id); }
  }
  for (const [container, jobs] of [[queueElement, pending], [document.querySelector("#history"), history]]) {
    jobs.forEach((job, index) => {
      const { progress, ...stable } = job;
      const peers = job.status === "queued" ? pending.filter(peer => peer.status === "queued" && peer.priority === job.priority).map(peer => peer.id) : [];
      const signature = JSON.stringify([stable, peers]);
      let entry = cards.get(job.id);
      if (!entry || entry.signature !== signature) {
        const card = jobCard(job);
        if (entry) entry.card.replaceWith(card);
        entry = { card, signature };
        cards.set(job.id, entry);
      }
      if (container.children[index] !== entry.card) container.insertBefore(entry.card, container.children[index] || null);
      const description = entry.card.querySelector(".job-state-description");
      if (description) description.textContent = waitingReason(job, pending);
      const label = entry.card.querySelector(".progress-label");
      if (label) {
        const info = progress || {};
        const measured = Number.isFinite(info.step) && Number.isFinite(info.total) && info.total > 0;
        const running = job.status === "running";
        const stage = info.stage || "Starting engine";
        label.textContent = `${running ? "" : "Last reported progress · "}${stage}${measured ? ` · ${info.step}/${info.total}` : ""}`;
        label.parentElement.hidden = !running && !info.stage && !measured;
        const bar = entry.card.querySelector("progress");
        bar.hidden = !running;
        if (measured) { bar.max = info.total; bar.value = info.step; }
        else bar.removeAttribute("value");
      }
    });
  }
}

function selectTab(tab, focus = false) {
  document.querySelector(".activity").dataset.tab = tab;
  for (const name of ["queue", "history"]) {
    const button = document.querySelector(`#${name}-tab`);
    button.setAttribute("aria-selected", String(name === tab));
    button.tabIndex = name === tab ? 0 : -1;
    if (focus && name === tab) button.focus();
  }
}
for (const tab of ["queue", "history"]) {
  const button = document.querySelector(`#${tab}-tab`);
  button.addEventListener("click", () => selectTab(tab));
  button.addEventListener("keydown", event => {
    if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
      event.preventDefault();
      selectTab(event.key === "Home" ? "queue" : event.key === "End" ? "history" : tab === "queue" ? "history" : "queue", true);
    }
  });
}

async function refresh() {
  if (refreshInFlight) return;
  refreshInFlight = true;
  try {
    const response = await fetch("/api/jobs", { cache: "no-store" });
    if (!response.ok) throw new Error(await response.text());
    render(await response.json());
  } catch (error) {
    engineLabel.textContent = "Connection lost";
    enginePill.className = "engine-pill transition";
  } finally {
    refreshInFlight = false;
  }
}

async function deleteJob(id, button) {
  button.disabled = true;
  try {
    const response = await fetch(`/api/jobs/${id}`, { method: "DELETE" });
    if (!response.ok) throw new Error(await response.text());
    await refresh();
  } catch (error) {
    alert(error.message);
  } finally {
    button.disabled = false;
  }
}

async function jobAction(id, action, button, changes) {
  const originalText = button.textContent;
  button.disabled = true;
  if (button.tagName === "BUTTON") button.textContent = "Saving…";
  try {
    const response = await fetch(`/api/jobs/${id}${action ? `/${action}` : ""}`, changes ? {
      method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(changes),
    } : { method: "POST" });
    if (!response.ok) throw new Error(await response.text());
    await refresh();
  } catch (error) {
    alert(error.message);
  } finally {
    button.disabled = false;
    if (button.tagName === "BUTTON") button.textContent = originalText;
    else button.value = currentSnapshot?.jobs.find(job => job.id === id)?.priority || button.value;
  }
}

async function copyToForm(job, button) {
  button.disabled = true;
  try {
    const fields = job.structured_prompt || {};
    promptInputs.forEach(input => {
      input.value = fields[input.name] || (input.name === "integrated_multimodal_description" && !Object.keys(fields).length ? job.prompt : "");
      input.dispatchEvent(new Event("input"));
    });
    form.elements.mode.value = job.mode;
    modelInput.value = [...modelInput.options].some(option => option.value === job.model) ? job.model : "minimax-h3";
    durationInput.value = String(job.duration_seconds ?? 5);
    resolutionInput.value = String(job.resolution ?? 512);
    aspectRatioInput.value = job.aspect_ratio || "1:1";
    form.elements.priority.value = job.priority || "medium";
    form.elements.seed.value = job.seed_text || String(job.seed);
    frameInputs.forEach(frame => { frame.input.value = ""; });
    updateMode(); updateModel(); updateOutputSpecs(); updatePreview();
    formError.hidden = true;
    const missing = [];
    for (const frame of frameInputs) {
      if (job[frame.urlKey]) {
        try {
          const response = await fetch(job[frame.urlKey]);
          if (!response.ok) throw new Error("Could not load image.");
          const blob = await response.blob();
          const transfer = new DataTransfer();
          transfer.items.add(new File([blob], frame.label, { type: blob.type }));
          frame.input.files = transfer.files;
        } catch (error) {
          missing.push(frame.label);
        }
      } else if (job[frame.presenceKey]) {
        missing.push(frame.label);
      }
    }
    updatePreview();
    if (missing.length || (job.mode === "image" && !frameInputs.some(frame => frame.input.files.length))) {
      formError.textContent = missing.length
        ? `Copied settings. Could not load the ${missing.join(" and ")}; choose replacements before submitting.`
        : "Copied settings. Choose a start or end frame before submitting.";
      formError.hidden = false;
    }
    document.querySelector(".composer").scrollIntoView({ behavior: "smooth", block: "start" });
    promptInputs[0].focus({ preventScroll: true });
  } catch (error) {
    alert(error.message);
  } finally {
    button.disabled = false;
  }
}

form.addEventListener("submit", async event => {
  event.preventDefault();
  formError.hidden = true;
  if (selectedMode() === "image" && !frameInputs.some(frame => frame.input.files.length)) {
    formError.textContent = "Choose a start frame, an end frame, or both.";
    formError.hidden = false;
    return;
  }
  submitButton.disabled = true;
  submitButton.querySelector("span").textContent = "Adding…";
  try {
    const data = new FormData(form);
    if (selectedMode() === "text") frameInputs.forEach(frame => data.delete(frame.input.name));
    const response = await fetch("/api/jobs", { method: "POST", body: data });
    if (!response.ok) throw new Error(await response.text());
    promptInputs.forEach(input => {
      input.value = "";
      const count = input.nextElementSibling?.querySelector?.(".field-count");
      if (count) count.textContent = "0 / 8000";
      resizePromptInput(input);
    });
    frameInputs.forEach(frame => { frame.input.value = ""; });
    updatePreview();
    await refresh();
    selectTab("queue");
    document.querySelector("#queue-title").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    formError.textContent = error.message;
    formError.hidden = false;
  } finally {
    submitButton.disabled = false;
    submitButton.querySelector("span").textContent = "Add to queue";
  }
});

form.addEventListener("change", event => {
  if (event.target.name === "mode") updateMode();
  if (["duration", "resolution", "aspect_ratio"].includes(event.target.name)) updateOutputSpecs();
  if (event.target.name === "model") updateModel();
});
promptInputs.forEach(input => input.addEventListener("input", () => {
  const count = input.nextElementSibling?.querySelector?.(".field-count");
  if (count) count.textContent = `${input.value.length} / 8000`;
  resizePromptInput(input);
}));
for (const frame of frameInputs) {
  frame.input.addEventListener("change", updatePreview);
  frame.change.addEventListener("click", () => frame.input.click());
  frame.remove.addEventListener("click", () => {
    frame.input.value = "";
    updatePreview();
  });
  ["dragenter", "dragover"].forEach(name => frame.box.addEventListener(name, event => {
    event.preventDefault();
    frame.box.classList.add("dragging");
  }));
  frame.box.addEventListener("dragleave", () => frame.box.classList.remove("dragging"));
  frame.box.addEventListener("drop", event => {
    event.preventDefault();
    frame.box.classList.remove("dragging");
    const file = [...event.dataTransfer.files].find(candidate => candidate.type.startsWith("image/"));
    if (!file) return;
    const transfer = new DataTransfer();
    transfer.items.add(file);
    frame.input.files = transfer.files;
    updatePreview();
  });
}

setInterval(() => {
  document.querySelectorAll(".job-timing[data-started-at]").forEach(element => {
    if (element.dataset.startedAt) element.textContent = `${formatAge(Number(element.dataset.startedAt))} elapsed`;
  });
}, 1000);
setInterval(refresh, 2000);
updateMode();
updateOutputSpecs();
updateModel();
promptInputs.forEach(resizePromptInput);
refresh();
