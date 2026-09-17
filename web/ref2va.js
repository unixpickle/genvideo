/* Ref2VA labels follow ComfyUI's presentation order, independently per media type. */
(() => {
  const fieldNames = ["subject_definitions", "summary", "retention_analysis", "detailed_description", "overall_soundscape", "non_diegetic_music"];
  const visualRetention = ["fully_preserved", "partially_preserved", "attribute_transfer", "weak_reference"];
  const audioLayers = {shot: "Dialogue / lyrics / shot-synchronized sound", sound: "Ambience / physical sound", music: "Audience-only music", both: "Ambience and audience-only music", complete: "Complete final soundtrack"};
  const defaultLayer = role => ["music", "reuse", "rhythm"].includes(role) ? "music" : role === "effects" ? "sound" : "shot";
  const audioRetention = ["reference", "weak_reference", "fully_copy", "partially_copy"];
  const roles = {
    image: {subject: "Subject / environment / style", first: "First frame", keyframe: "Keyframe", last: "Last frame", edited: "Edited keyframe", composition: "Composition anchor", storyboard: "Storyboard"},
    video: {subject: "Subject / action / effect", editing: "Edit this video", continuation: "Continue this video", camera: "Camera / cuts / pacing"},
    audio: {voice: "Voice timbre", music: "Music style", rhythm: "Beat / rhythm", effects: "Sound effects / ambience", dialogue: "Dialogue / lyric content", reuse: "Reuse audio signal"},
  };
  const templates = {
    identity: {name: "Character, object, or scene reference", refs: [["image", "subject"]], subject: true},
    keyframes: {name: "First and last frames", refs: [["image", "first"], ["image", "last"]]},
    storyboard: {name: "Storyboard / composition", refs: [["image", "storyboard"]]},
    editing: {name: "Edit a source video", refs: [["video", "editing"]]},
    continuation: {name: "Continue a video to a final image", refs: [["video", "continuation"], ["image", "last"]]},
    motion: {name: "Transfer appearance and motion", refs: [["image", "subject"], ["video", "subject"]], subject: true},
    voice: {name: "Character with a reference voice", refs: [["image", "subject"], ["audio", "voice"]], subject: true},
    music: {name: "Generate to a reused soundtrack", refs: [["audio", "reuse"]]},
    custom: {name: "Custom combination", refs: []},
  };
  const root = document.querySelector("#ref2va-builder");
  let references = [], subjects = [], shots = [], urls = [];
  let counter = 0;
  let needsBuild = true;
  const id = () => `r${Date.now().toString(36)}_${++counter}`;
  function el(tag, text, cls) { const node = document.createElement(tag); if (text) node.textContent = text; if (cls) node.className = cls; return node; }
  function button(text, fn) { const node = el("button", text, "builder-button"); node.type = "button"; node.addEventListener("click", fn); return node; }
  function input(label, value, change, multiline = false, {marksDirty = true, placeholder = "", help = ""} = {}) {
    const wrap = el("label", label, "builder-field");
    const node = el(multiline ? "textarea" : "input"); node.value = value || ""; node.maxLength = 8000;
    node.setAttribute("aria-label", label); node.placeholder = placeholder;
    node.addEventListener("input", () => { change(node.value); if (marksDirty) dirty(); });
    wrap.append(node);
    if (help) wrap.append(el("small", help, "builder-help"));
    return wrap;
  }
  function select(label, options, value, change, multiple = false) {
    const wrap = el("label", label, "builder-field"), node = el("select"); node.multiple = multiple; node.setAttribute("aria-label", label);
    for (const [key, name] of Object.entries(options)) { const opt = el("option", name); opt.value = key; opt.selected = multiple ? value.includes(key) : key === value; node.append(opt); }
    node.addEventListener("change", () => { change(multiple ? [...node.selectedOptions].map(o => o.value) : node.value); dirty(); }); wrap.append(node); return wrap;
  }
  function labels() {
    const result = new Map(); let picture = 0, video = 0, audio = 0;
    references.filter(r => r.kind === "image").forEach(r => result.set(r.id, [`<Picture ${++picture}>`]));
    references.filter(r => r.kind === "video").forEach(r => result.set(r.id, [`<Video ${++video}>`, ...(r.use_audio ? [`<Audio ${++audio}>`] : [])]));
    references.filter(r => r.kind === "audio").forEach(r => result.set(r.id, [`<Audio ${++audio}>`]));
    return result;
  }
  root.innerHTML = `
    <p class="builder-help">Combine images, video, and audio. Assign each a role, describe what to keep, then build an editable prompt. <a href="https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/ref-en.txt" target="_blank" rel="noreferrer">Prompt guide ↗</a></p>
    <div id="ref-template" class="builder-toolbar"></div>
    <h3>1. Reference media</h3>
    <p class="builder-help">Up to 9 images, 3 videos and 3 audio tracks; 12 files total. Clips: 2–15 seconds, with at most 15 seconds per media type. Images: 32 MB; clips: 256 MB. Video frames beyond the output duration are omitted. Soundtracks are off until enabled and count toward audio limits.</p>
    <div id="ref-media-list"></div><div id="ref-add-media" class="builder-toolbar"></div>
    <h3>2. Subjects</h3><p class="builder-help">Define people, scenes, objects, style, or motion separately from files. One subject can use several sources; one file can define several subjects. Select multiple sources with Ctrl/⌘.</p>
    <div id="ref-subject-list"></div><div id="ref-add-subject"></div>
    <h3>3. Target video</h3><p class="builder-help">Describe the scene in English, with reference labels where they apply. Aim for 350–500 words across the shots: composition, appearance, lighting, action, camera, and sound. Keep dialogue in its original language.</p><div id="ref-target"></div>
    <div id="ref-shot-list"></div><div id="ref-add-shot"></div>
    <details class="advanced"><summary>Dialogue helper</summary><div id="ref-dialogue"></div></details>
    <div id="ref-sound"></div>
    <div class="builder-toolbar" id="ref-build"></div><p id="ref-build-status" class="builder-help" role="status">Build the prompt when your references and shots are ready.</p>
    <details id="ref-prompt-sections" class="advanced"><summary>Six editable prompt sections</summary><div id="ref-sections"></div></details>
    <details id="ref-preview-panel" class="advanced"><summary>Compiled prompt preview</summary><p id="ref-preview-status" class="builder-help" role="status"></p><pre id="ref-prompt-preview" hidden></pre></details>`;
  const find = selector => root.querySelector(selector);
  const sections = {};
  for (const name of fieldNames) {
    const wrap = input(name.replaceAll("_", " "), "", () => {
      preview();
      if (!needsBuild) status("Prompt edited. These sections will be submitted.");
    }, true, {marksDirty: false});
    const node = wrap.querySelector("textarea"); node.id = `ref-${name}`; sections[name] = node; find("#ref-sections").append(wrap);
  }
  const values = {summary: "", style: "", sound: "", music: "N/A"};
  for (const [key, title] of [["summary", "What should happen?"], ["style", "Visual style and lighting"]])
    find("#ref-target").append(input(title, values[key], v => values[key] = v, true));
  for (const [key, title] of [["sound", "Overall ambience and physical sounds"], ["music", "Audience-only music (N/A for none)"]])
    find("#ref-sound").append(input(title, values[key], v => values[key] = v, true));
  function dirty() { needsBuild = true; status("Builder changed. Rebuild to apply changes to the prompt; rebuilding replaces manual section edits."); preview(); }
  function preview() {
    const hasPrompt = fieldNames.some(n => sections[n].value.trim());
    find("#ref-prompt-preview").hidden = !hasPrompt;
    find("#ref-prompt-preview").textContent = hasPrompt ? fieldNames.map(n => `${n}:\n${sections[n].value}`).join("\n\n") : "";
    find("#ref-preview-status").textContent = !hasPrompt
      ? "No prompt built yet. Fill in your subjects, target description, and shots, then click ‘Build / replace prompt sections’ above."
      : needsBuild ? "This is the previous prompt. Rebuild to include your latest builder changes."
        : "This is the exact prompt that will be submitted. Edits to the six prompt sections appear here immediately.";
  }
  function referenceDefaults(kind, role = Object.keys(roles[kind])[0]) {
    return {id: id(), kind, role, layer: defaultLayer(role), audio_layer: "music", name: "", description: "", retention: kind === "audio" ? (role === "reuse" ? "partially_copy" : "reference") : "fully_preserved", keep: "", where: "[Shot 1]", use_audio: false, audio_role: "music", audio_retention: "reference", audio_description: "", audio_keep: "", speaker: ""};
  }
  function addReference(kind, role) { references.push(referenceDefaults(kind, role)); }
  function renderReferences() {
    urls.forEach(url => URL.revokeObjectURL(url)); urls = [];
    const list = find("#ref-media-list"); list.replaceChildren();
    const tags = labels();
    for (const ref of references) {
      const card = el("div", null, "reference-card"), top = el("div", null, "builder-toolbar");
      top.append(el("strong", tags.get(ref.id).join(" + ")), button("Remove", () => { references = references.filter(r => r !== ref); subjects.forEach(s => s.sources = s.sources.filter(x => x !== ref.id)); renderAll(); dirty(); })); card.append(top);
      const upload = el("input"); upload.type = "file"; upload.accept = ref.kind === "image" ? "image/png,image/jpeg,image/webp" : `${ref.kind}/*`; upload.setAttribute("aria-label", `File for ${tags.get(ref.id)[0]}`);
      upload.addEventListener("change", () => { ref.file = upload.files[0]; ref.name = ref.file?.name || ""; renderAll(); dirty(); }); card.append(upload);
      if (ref.file) {
        const url = URL.createObjectURL(ref.file); urls.push(url);
        const media = el(ref.kind === "image" ? "img" : ref.kind); media.src = url; media.className = "reference-preview";
        if (ref.kind === "image") media.alt = ref.name; else { media.controls = true; media.preload = "metadata"; }
        card.append(el("p", ref.name, "builder-help"), media);
      } else card.append(el("p", "Choose a file for this reference.", "builder-help"));
      card.append(select("Reference role", roles[ref.kind], ref.role, value => { ref.role = value; if (ref.kind === "audio") { ref.retention = value === "reuse" ? "partially_copy" : "reference"; ref.layer = defaultLayer(value); } renderReferences(); }));
      card.append(input("Describe this reference and its role", ref.description, v => ref.description = v, true, {
        placeholder: ref.kind === "audio"
          ? "A calm, low-pitched voice. Match its tone and delivery, but speak the new dialogue."
          : ref.kind === "video" ? "A clip of a woman walking. Use her relaxed gait and arm movement."
            : "A portrait of a woman. Use it for her face, hairstyle, and clothing.",
        help: ref.role === "subject"
          ? "Optional source notes: what this file contributes. Describe the actual person, object, or scene under Subjects below."
          : "Describe what is in the file and how it should guide the result, such as an opening frame, camera movement, or voice.",
      }));
      if (ref.role !== "subject") {
      card.append(input("Where it applies (shot, time, or layer)", ref.where, v => ref.where = v));
      card.append(select("Retention relationship", Object.fromEntries((ref.kind === "audio" ? audioRetention : visualRetention).map(x => [x, x.replaceAll("_", " ")])), ref.retention, v => ref.retention = v));
      card.append(input("What to preserve, change, transfer, or copy", ref.keep, v => ref.keep = v, true));
      }
      if (ref.role === "subject") card.append(el("p", "Define the referenced content and its retention below under Subjects.", "builder-help"));
      if (ref.kind === "audio") card.append(select("Audible layer", audioLayers, ref.layer, v => ref.layer = v));
      if (ref.kind === "audio") card.append(input("Target speaker, if any (e.g. <Subject 1> (S1))", ref.speaker, v => ref.speaker = v));
      if (ref.kind === "video") {
        const label = el("label", null, "builder-checkbox"), checkbox = el("input"); checkbox.type = "checkbox"; checkbox.checked = ref.use_audio;
        checkbox.addEventListener("change", () => { ref.use_audio = checkbox.checked; renderAll(); dirty(); }); label.append(checkbox, document.createTextNode("Use this video's soundtrack")); card.append(label);
        if (ref.use_audio) {
          card.append(select("Soundtrack role", roles.audio, ref.audio_role, v => { ref.audio_role = v; ref.audio_layer = defaultLayer(v); ref.audio_retention = v === "reuse" ? "partially_copy" : "reference"; renderReferences(); }));
          card.append(select("Soundtrack audible layer", audioLayers, ref.audio_layer, v => ref.audio_layer = v));
          card.append(select("Soundtrack relationship", Object.fromEntries(audioRetention.map(x => [x, x.replaceAll("_", " ")])), ref.audio_retention, v => ref.audio_retention = v));
          card.append(input("Soundtrack description / target speaker (include S1 etc. if speaking)", ref.audio_description, v => ref.audio_description = v, true));
          card.append(input("Soundtrack retention details", ref.audio_keep, v => ref.audio_keep = v, true));
        }
      }
      list.append(card);
    }
  }
  function renderSubjects() {
    const list = find("#ref-subject-list"); list.replaceChildren();
    const sources = Object.fromEntries(references.filter(r => r.kind !== "audio").map(r => [r.id, `${labels().get(r.id)[0]} ${r.name || roles[r.kind][r.role]}`]));
    subjects.forEach((subject, index) => {
      const card = el("div", null, "reference-card"); card.append(el("strong", `<Subject ${index + 1}>`));
      card.append(select("Source assets", sources, subject.sources, v => subject.sources = v, true));
      card.append(input("Identity / features and what each source provides", subject.description, v => subject.description = v, true, {
        placeholder: "the woman with chin-length black hair, round glasses, and a yellow raincoat",
        help: "Define the subject to reuse. With several sources, specify the contribution of each: ‘the woman whose face and outfit come from <Picture 1> and whose walking motion comes from <Video 1>’. The builder adds the subject label for you.",
      }));
      card.append(input("Appears in", subject.where, v => subject.where = v));
      card.append(select("Retention relationship", Object.fromEntries(visualRetention.map(x => [x, x.replaceAll("_", " ")])), subject.retention, v => subject.retention = v));
      card.append(input("Retained or changed features", subject.keep, v => subject.keep = v, true));
      card.append(button("Remove subject", () => { subjects.splice(index, 1); renderSubjects(); renderShots(); dirty(); })); list.append(card);
    });
  }
  function addSubject() { subjects.push({sources: references.filter(r => r.kind !== "audio").map(r => r.id), description: "", where: "[Shot 1]", retention: "fully_preserved", keep: ""}); }
  function renderShots() {
    const list = find("#ref-shot-list"); list.replaceChildren();
    shots.forEach((shot, index) => {
      const card = el("div", null, "reference-card"); card.append(el("strong", `[Shot ${index + 1}]`));
      if (index) card.append(input("Cut time (MM:SS.mmm)", shot.time, v => shot.time = v));
      card.append(select("Camera movement", {"": "Describe in the shot", "The camera remains static.": "Static", "The camera slowly pushes in.": "Slow push in", "The camera slowly pulls back.": "Slow pull back", "The camera pans smoothly from left to right.": "Pan right", "The camera tracks alongside the subject.": "Tracking"}, shot.camera, v => shot.camera = v));
      card.append(input("Composition, subjects, action, sound, dialogue, and reference cues", shot.description, v => shot.description = v, true));
      const choices = Object.fromEntries([...subjects.map((s, i) => [`<Subject ${i + 1}>`, `<Subject ${i + 1}>`]), ...[...labels().values()].flat().map(tag => [tag, tag])]);
      if (Object.keys(choices).length) {
        const picker = select("Insert a reference label", choices, Object.keys(choices)[0], () => {});
        card.append(picker, button("Insert label into shot", () => { shot.description += ` ${picker.querySelector("select").value}`; renderShots(); dirty(); }));
      }
      if (shots.length > 1) card.append(button("Remove shot", () => { shots.splice(index, 1); renderShots(); dirty(); })); list.append(card);
    });
  }
  function renderAll() { renderReferences(); renderSubjects(); renderShots(); }
  const templateSelect = select("Task template", Object.fromEntries(Object.entries(templates).map(([k, v]) => [k, v.name])), "identity", () => {});
  find("#ref-template").append(templateSelect, button("Add template", () => {
    const template = templates[templateSelect.querySelector("select").value];
    if (references.length + template.refs.length > 12) { status("At most 12 reference files."); return; }
    const start = references.length;
    template.refs.forEach(([kind, role]) => addReference(kind, role));
    if (template.subject) { addSubject(); subjects.at(-1).sources = references.slice(start).filter(r => r.kind !== "audio").map(r => r.id); }
    renderAll(); dirty();
  }));
  for (const kind of ["image", "video", "audio"]) find("#ref-add-media").append(button(`+ ${kind}`, () => { if (references.length >= 12) return status("At most 12 reference files."); addReference(kind); renderAll(); dirty(); }));
  find("#ref-add-subject").append(button("+ Subject", () => { addSubject(); renderSubjects(); renderShots(); dirty(); }));
  find("#ref-add-shot").append(button("+ Shot", () => { shots.push({time: "00:03.000", camera: "", description: ""}); renderShots(); dirty(); }));
  const dialogue = {speaker: "<Subject 1>", number: "1", language: "English", text: "", delivery: "says", shot: "1"};
  for (const [key, title] of [["speaker", "Subject label or stable narrator description"], ["number", "Speaker ID number (global order of first speech)"], ["language", "Spoken language"], ["delivery", "Delivery / emotion"], ["text", "Exact dialogue or lyrics"], ["shot", "Insert into shot number"]]) find("#ref-dialogue").append(input(title, dialogue[key], v => dialogue[key] = v));
  find("#ref-dialogue").append(button("Insert dialogue", () => {
    const shot = shots[Number(dialogue.shot) - 1];
    if (!shot || !/^[1-9][0-9]*$/.test(dialogue.number) || !dialogue.text.trim()) return status("Choose an existing shot, a positive speaker ID, and dialogue.");
    shot.description += ` ${dialogue.speaker} (S${dialogue.number}) ${dialogue.delivery}, <d>[${dialogue.language}] ${dialogue.text.trim()}</d>`;
    renderShots(); dirty();
  }));
  function status(text) { find("#ref-build-status").textContent = text; }
  function sentence(text) {
    const value = text.trim();
    return /[.!?]$/.test(value) ? value : `${value}.`;
  }
  function build() {
    const tags = labels(), definitions = [], retention = [], cues = [], tasks = new Set();
    const addCue = (where, text) => cues.push({where, text});
    subjects.forEach((s, i) => {
      const label = `<Subject ${i + 1}>`, sources = s.sources.map(x => tags.get(x)?.[0]).filter(Boolean).join(" and ");
      if (!sources || !s.description.trim()) throw new Error(`Describe ${label} and select its source assets.`);
      const description = s.description.trim();
      const sourceLabels = s.sources.map(source => tags.get(source)?.[0]).filter(Boolean);
      const provenance = sourceLabels.every(source => description.includes(source)) ? "" : ` from ${sources}`;
      definitions.push(sentence(`${label} is ${provenance ? description.replace(/[.!?]+$/, "") : description}${provenance}`));
      const sourceNotes = s.sources.map(source => references.find(r => r.id === source))
        .filter(ref => ref?.role === "subject" && ref.description.trim())
        .map(ref => `${tags.get(ref.id)[0]} provides the following reference: ${sentence(ref.description)}`);
      if (sourceNotes.length) definitions[definitions.length - 1] += ` ${sourceNotes.join(" ")}`;
      retention.push(`${label} (appears in ${s.where}): ${s.retention} - ${s.keep || s.description}`);
    });
    const sound = [], music = [];
    function audio(label, role, relation, description, keep, speaker = "", where = "[Shot 1]", layer = defaultLayer(role)) {
      const copied = relation.includes("copy"); tasks.add(copied ? "audio reuse" : "audio reference");
      definitions.push(sentence(`${label} provides ${roles.audio[role].toLowerCase()}${speaker ? ` for ${speaker}` : ""}: ${description || "describe the audible reference"}`));
      retention.push(`${label}: ${relation} - ${keep || (copied ? (relation === "fully_copy" ? "The complete source is reused as the entire final audio track." : "Reuse the specified source audio segment or layers.") : "Follow the described audio characteristics without copying the signal.")}`);
      const sentence = `${label} is ${copied ? "copied" : "referenced"} for ${roles.audio[role].toLowerCase()}. ${keep}`;
      if (layer === "music") music.push(sentence);
      else if (layer === "sound") sound.push(sentence);
      else if (layer === "both" || layer === "complete") {
        sound.push(`The ambience and physical sounds in ${label} are ${copied ? "copied" : "referenced"}. ${keep}`);
        music.push(`Any audience-only music in ${label} is ${copied ? "copied" : "referenced"}. ${keep}`);
        if (layer === "complete") addCue(where, `${label} supplies the complete soundtrack. ${keep}`);
      } else addCue(where, sentence);
    }
    for (const ref of references) {
      const [label, audioLabel] = tags.get(ref.id);
      if (ref.role !== "subject" && !ref.description.trim()) throw new Error(`Describe ${label} before building the prompt.`);
      if (ref.kind === "audio") { audio(label, ref.role, ref.retention, ref.description, ref.keep, ref.speaker, ref.where, ref.layer); continue; }
      if (ref.role === "subject") {
        tasks.add("reference generation");
        if (!subjects.some(s => s.sources.includes(ref.id))) throw new Error(`Add a subject using ${label}.`);
      } else {
        tasks.add(ref.kind === "image" && !["storyboard"].includes(ref.role) ? "keyframe completion" : ref.role === "editing" ? "video editing" : ref.role === "continuation" ? "video continuation" : "reference generation");
        definitions.push(sentence(`${label} is the ${roles[ref.kind][ref.role].toLowerCase()} reference for ${ref.where}: ${ref.description}`));
        retention.push(`${label} (${ref.where}): ${ref.retention} - ${ref.keep || ref.description}`);
        const cue = {first: "The shot begins from", last: "The shot ends on", keyframe: "The shot's keyframe corresponds to", edited: "The shot uses an edited keyframe from", continuation: "The action continues from the ending of", editing: "The target video is an edited version of", camera: "Camera movement and pacing follow", storyboard: "Shot planning follows", composition: "Composition follows"}[ref.role];
        addCue(ref.where, `${cue} ${label}. ${ref.keep}`);
      }
      if (audioLabel) audio(audioLabel, ref.audio_role, ref.audio_retention, `the synchronized soundtrack of ${label}. ${ref.audio_description}`, ref.audio_keep, "", ref.where, ref.audio_layer);
    }
    if (!references.length || !values.summary.trim() || shots.some(s => !s.description.trim())) throw new Error("Add references, a target description, and details for each shot.");
    let previous = 0;
    const timeline = shots.map((s, i) => {
      if (i) {
        if (!/^\d{2}:\d{2}\.\d{3}$/.test(s.time)) throw new Error("Cut times must use MM:SS.mmm.");
        const [m, sec] = s.time.split(":").map(Number), time = m * 60 + sec;
        if (sec >= 60 || time <= previous || time >= Number(document.querySelector("#duration").value)) throw new Error("Cut times must increase and fall inside the output duration.");
        previous = time;
      }
      const shotCues = cues.filter(c => c.where.includes(`[Shot ${i + 1}]`) || (i === 0 && !/\[Shot \d+\]/.test(c.where))).map(c => c.text);
      return `[Shot ${i + 1}] ${i ? `At ${s.time}, ` : ""}${s.description} ${s.camera} ${shotCues.join(" ")}`;
    });
    const edited = references.find(r => r.role === "editing");
    const result = {
      subject_definitions: definitions.join("\n"),
      summary: `[${[...tasks].join(" + ")}] ${edited ? `The target video is an edited version of ${tags.get(edited.id)[0]}. ` : ""}${values.summary}`,
      retention_analysis: retention.join("\n"),
      detailed_description: `${values.style || "Naturalistic imagery and lighting."}\n${timeline.join("\n")}`,
      overall_soundscape: [values.sound, ...sound].filter(Boolean).join(" ") || "N/A",
      non_diegetic_music: [music.length && values.music === "N/A" ? "" : values.music, ...music].filter(Boolean).join(" ") || "N/A",
    };
    fieldNames.forEach(name => sections[name].value = result[name]); needsBuild = false; preview();
    find("#ref-prompt-sections").open = true;
    find("#ref-preview-panel").open = true;
    status("Prompt built. Review the six sections, especially reference timing and audio layers, before adding to the queue.");
  }
  find("#ref-build").append(button("Build / replace prompt sections", () => {
    try { build(); } catch (e) {
      status(e.message);
      preview();
      find("#ref-preview-status").textContent = `Cannot build yet: ${e.message}`;
      find("#ref-preview-panel").open = true;
    }
  }));
  shots.push({time: "", camera: "", description: ""}); renderAll(); preview();
  window.Ref2VA = {
    append(data) {
      if (needsBuild) throw new Error("Rebuild the prompt after changing references or builder fields.");
      if (!references.length || references.some(r => !r.file)) throw new Error("Choose a file for every reference.");
      if (references.some(r => r.file.size > (r.kind === "image" ? 32 : 256) * 1024 * 1024)) throw new Error("Images may be at most 32 MB; clips at most 256 MB.");
      if (fieldNames.some(n => !sections[n].value.trim())) throw new Error("Build and review all six Ref2VA prompt sections first.");
      for (const name of fieldNames) data.set(name, sections[name].value);
      const serial = references.map(({file, ...ref}) => ref);
      data.set("references", JSON.stringify(serial));
      data.set("reference_builder", JSON.stringify({references: serial, subjects, shots, values}));
      references.forEach(ref => data.append(`reference_${ref.id}`, ref.file));
    },
    async restore(job) {
      const state = job.reference_builder || {};
      references = (job.references || []).map(r => ({...referenceDefaults(r.kind), ...r,
        ...(state.references || []).find(s => s.id === r.id), file: null}));
      subjects = state.subjects || []; shots = state.shots || [{time: "", camera: "", description: ""}];
      Object.assign(values, {summary: "", style: "", sound: "", music: "N/A"}, state.values || {});
      const targetInputs = [...find("#ref-target").querySelectorAll("textarea"), ...find("#ref-sound").querySelectorAll("textarea")];
      ["summary", "style", "sound", "music"].forEach((key, index) => targetInputs[index].value = values[key]);
      fieldNames.forEach(n => sections[n].value = job.structured_prompt?.[n] || ""); preview();
      const missing = [];
      await Promise.all(references.map(async ref => {
        try { const response = await fetch(ref.url); if (!response.ok) throw new Error(); const blob = await response.blob(); ref.file = new File([blob], ref.name, {type: blob.type}); }
        catch { missing.push(ref.name); }
      }));
      renderAll(); needsBuild = false; preview(); find("#ref-prompt-sections").open = true; status("Copied saved builder and prompt sections.");
      return missing;
    },
  };
})();
