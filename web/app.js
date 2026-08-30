const form = document.querySelector("#job-form");
const promptInput = document.querySelector("#prompt");
const promptCount = document.querySelector("#prompt-count");
const durationInput = document.querySelector("#duration");
const durationSpec = document.querySelector("#duration-spec");
const imageField = document.querySelector("#image-field");
const imageInput = document.querySelector("#image");
const imagePreview = document.querySelector("#image-preview");
const uploadEmpty = document.querySelector("#upload-empty");
const uploadBox = document.querySelector("#upload-box");
const changeImage = document.querySelector("#change-image");
const submitButton = document.querySelector("#submit-button");
const formError = document.querySelector("#form-error");
const queueElement = document.querySelector("#queue");
const emptyState = document.querySelector("#empty-state");
const queueCount = document.querySelector("#queue-count");
const enginePill = document.querySelector("#engine-pill");
const engineLabel = document.querySelector("#engine-label");

let previewUrl = null;
let lastSignature = "";
let currentSnapshot = null;

function resizePromptInput() {
  promptInput.style.height = "auto";
  promptInput.style.height = `${promptInput.scrollHeight}px`;
}

function selectedMode() {
  return form.elements.mode.value;
}

function updateMode() {
  const usesImage = selectedMode() === "image";
  imageField.hidden = !usesImage;
  imageInput.required = usesImage;
}

function updateDuration() {
  durationSpec.textContent = `${durationInput.value} sec`;
}

function updatePreview() {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  const file = imageInput.files[0];
  previewUrl = file ? URL.createObjectURL(file) : null;
  imagePreview.hidden = !file;
  uploadEmpty.hidden = Boolean(file);
  changeImage.hidden = !file;
  if (file) imagePreview.src = previewUrl;
  else imagePreview.removeAttribute("src");
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
  if (job.status === "running") return "Generating";
  if (job.status === "queued") return `Queued · #${job.queue_position}`;
  if (job.status === "completed") return "Ready";
  return "Failed";
}

function jobCard(job) {
  const card = document.createElement("article");
  card.className = "job panel";
  card.dataset.jobId = job.id;

  const top = document.createElement("div");
  top.className = "job-top";
  const copy = document.createElement("div");
  const mode = document.createElement("span");
  mode.className = "job-mode";
  const modeLabel = job.mode === "image" ? "◫ Image to video" : "✦ Text to video";
  mode.textContent = `${modeLabel} · ${job.duration_seconds || 3} sec`;
  const prompt = document.createElement("p");
  prompt.className = "job-prompt";
  prompt.textContent = job.prompt;
  copy.append(mode, prompt);
  const status = document.createElement("span");
  status.className = `status ${job.status}`;
  status.textContent = statusText(job);
  top.append(copy, status);
  card.append(top);

  if (job.image_prompt_url) {
    const imagePrompt = document.createElement("figure");
    imagePrompt.className = "job-image-prompt";
    const imageLabel = document.createElement("figcaption");
    imageLabel.className = "job-image-prompt-label";
    imageLabel.textContent = "Starting image";
    const imageLink = document.createElement("a");
    imageLink.className = "job-prompt-image-link";
    imageLink.href = job.image_prompt_url;
    imageLink.target = "_blank";
    imageLink.rel = "noopener";
    imageLink.title = "View starting image at full size";
    const image = document.createElement("img");
    image.className = "job-prompt-image";
    image.src = job.image_prompt_url;
    image.alt = "Starting image prompt";
    imageLink.append(image);
    imagePrompt.append(imageLabel, imageLink);
    card.append(imagePrompt);
  } else if (job.mode === "image") {
    const unavailable = document.createElement("p");
    unavailable.className = "job-image-prompt-unavailable";
    unavailable.textContent = "Starting image unavailable";
    card.append(unavailable);
  }

  const meta = document.createElement("div");
  meta.className = "job-meta";
  const seed = document.createElement("span");
  seed.textContent = `Seed ${job.seed}`;
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

  if (["queued", "running", "completed", "failed"].includes(job.status)) {
    const actions = document.createElement("div");
    actions.className = "job-actions";
    if (job.media_url) {
      const download = document.createElement("a");
      download.href = job.media_url;
      download.download = `genvideo-${job.id}.mp4`;
      download.textContent = "Download";
      actions.append(download);
    }
    const regenerate = document.createElement("button");
    regenerate.type = "button";
    regenerate.textContent = "Regenerate";
    if (job.mode === "image" && !job.image_prompt_url) {
      regenerate.disabled = true;
      regenerate.title = "The starting image for this job is no longer available";
    }
    regenerate.addEventListener("click", () => regenerateJob(job.id, regenerate));
    actions.append(regenerate);
    if (job.status !== "running") {
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "delete";
      remove.textContent = job.status === "queued" ? "Cancel" : "Remove";
      remove.addEventListener("click", () => deleteJob(job.id));
      actions.append(remove);
    }
    card.append(actions);
  }

  if (job.status === "running") {
    const line = document.createElement("div");
    line.className = "running-line";
    card.append(line);
  }
  return card;
}

function render(snapshot) {
  currentSnapshot = snapshot;
  const labels = {
    idle: "Idle · model unloaded",
    starting: "Starting engine…",
    working: "LTX loaded · working",
    unloading: "Queue clear · unloading…",
  };
  engineLabel.textContent = labels[snapshot.engine_state] || snapshot.engine_state;
  enginePill.className = "engine-pill";
  if (snapshot.engine_state === "working") enginePill.classList.add("working");
  if (["starting", "unloading"].includes(snapshot.engine_state)) enginePill.classList.add("transition");
  queueCount.textContent = `${snapshot.queued_count} waiting`;

  const signature = JSON.stringify(snapshot.jobs.map(job => [
    job.id, job.status, job.queue_position, job.error, job.media_url,
    job.image_prompt_url, job.prompt, job.mode, job.seed, job.duration_seconds,
  ]));
  if (signature === lastSignature) return;
  lastSignature = signature;
  queueElement.replaceChildren(...snapshot.jobs.map(jobCard));
  emptyState.hidden = snapshot.jobs.length > 0;
}

async function refresh() {
  try {
    const response = await fetch("/api/jobs", { cache: "no-store" });
    if (!response.ok) throw new Error(await response.text());
    render(await response.json());
  } catch (error) {
    engineLabel.textContent = "Connection lost";
    enginePill.className = "engine-pill transition";
  }
}

async function deleteJob(id) {
  try {
    const response = await fetch(`/api/jobs/${id}`, { method: "DELETE" });
    if (!response.ok) throw new Error(await response.text());
    lastSignature = "";
    await refresh();
  } catch (error) {
    alert(error.message);
  }
}

async function regenerateJob(id, button) {
  const originalText = button.textContent;
  button.disabled = true;
  button.textContent = "Adding…";
  try {
    const response = await fetch(`/api/jobs/${id}/regenerate`, { method: "POST" });
    if (!response.ok) throw new Error(await response.text());
    lastSignature = "";
    await refresh();
  } catch (error) {
    alert(error.message);
    button.disabled = false;
    button.textContent = originalText;
  }
}

form.addEventListener("submit", async event => {
  event.preventDefault();
  formError.hidden = true;
  if (selectedMode() === "image" && !imageInput.files.length) {
    formError.textContent = "Choose a starting image first.";
    formError.hidden = false;
    return;
  }
  submitButton.disabled = true;
  submitButton.querySelector("span").textContent = "Adding…";
  try {
    const data = new FormData(form);
    if (selectedMode() === "text") data.delete("image");
    const response = await fetch("/api/jobs", { method: "POST", body: data });
    if (!response.ok) throw new Error(await response.text());
    promptInput.value = "";
    promptCount.textContent = "0 / 8000";
    resizePromptInput();
    imageInput.value = "";
    updatePreview();
    lastSignature = "";
    await refresh();
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
  if (event.target.name === "duration") updateDuration();
});
promptInput.addEventListener("input", () => {
  promptCount.textContent = `${promptInput.value.length} / 8000`;
  resizePromptInput();
});
imageInput.addEventListener("change", updatePreview);
changeImage.addEventListener("click", event => {
  event.preventDefault();
  imageInput.click();
});
["dragenter", "dragover"].forEach(name => uploadBox.addEventListener(name, event => {
  event.preventDefault();
  uploadBox.classList.add("dragging");
}));
["dragleave"].forEach(name => uploadBox.addEventListener(name, () => {
  uploadBox.classList.remove("dragging");
}));
uploadBox.addEventListener("drop", event => {
  event.preventDefault();
  uploadBox.classList.remove("dragging");
  const file = [...event.dataTransfer.files].find(candidate => candidate.type.startsWith("image/"));
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  imageInput.files = transfer.files;
  updatePreview();
});

setInterval(() => {
  document.querySelectorAll(".job-timing[data-started-at]").forEach(element => {
    if (element.dataset.startedAt) element.textContent = `${formatAge(Number(element.dataset.startedAt))} elapsed`;
  });
}, 1000);
setInterval(refresh, 2000);
updateMode();
updateDuration();
resizePromptInput();
refresh();
