# genvideo

A private, mobile-friendly web app for generating MiniMax H3 video with
synchronized stereo audio. Create text-to-video or image-plus-text jobs, manage
the generation queue, and play finished videos inline.

## Start the web app

```sh
./genvideo-web
```

Then open [http://10.9.0.8:8080](http://10.9.0.8:8080). The server accepts
`--host`, `--port`, `--output-directory`, and `--memory-limit-gib` options.
Completed MP4s and queue state are stored in `web_outputs/` by default.

## Create a generation

Choose **Text only** or **Use images**, then fill at least one H3 prompt
field: integrated multimodal description, overall soundscape, or non-diegetic
music. Empty fields are omitted, and the app serializes the fields in H3's
required order. In image mode, supply a **Start frame**, an **End frame**, or both. Each image
has its own preview, change, and remove controls. The app automatically adds
frame-alignment instructions: with both images, `<Picture 1>` is the start and
`<Picture 2>` is the end; a lone start or end image is `<Picture 1>`. Images may have any
dimensions; they are scaled to cover the output canvas and center-cropped
without distortion.

MiniMax H3 Turbo is the default, using the 4-step Turbo LoRA. Select
**MiniMax H3 Regular (20 steps)** for the regular schedule without the LoRA.
**MiniMax H3 Turbo v4 — larryvrh (6 steps)** uses
[`minimax_h3_turbo_v4_step600_ema.safetensors`](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
at strength 1.0 with Euler and the simple schedule, matching the six-evaluation
[arena variant](https://huggingface.co/spaces/multimodalart/h3-acceleration-arena/blob/main/validate/variants.json).
Set a seed for repeatable generation.

Videos default to 5 seconds. Every integer duration from 3 through 15 seconds
is available (3 seconds is retained as a compatibility option); H3 snaps the
requested duration to its `17k+5` frame grid at 24 fps. Choose a 512p
memory-saving or native 768p canvas and an aspect ratio of `21:9`, `16:9`,
`4:3`, `1:1`, `3:4`, or `9:16`. Canvases are aligned to 32 pixels and capped
to H3's local 7:4 pixel budget; the app shows the exact dimensions.

Every job shows its full compiled prompt and any start/end images. **Copy to form**
loads any current or historical job's prompt fields, model, duration, resolution,
aspect ratio, priority, exact seed, and both images into the composer for editing.
Settings are still copied if an image cannot be loaded; the form identifies any
missing images so you can replace them before submitting. Submitting creates a
separate job. Older LTX prompts can also be copied into an H3 form; their model
selection defaults to H3 Turbo.

## Manage the queue

Queue / Generations tabs switch between pending jobs and past generations on
desktop and mobile. The running job is highlighted and shows live stage progress.
Waiting jobs explain their queue position or priority, while paused jobs stay
inactive until resumed. Any previous progress on inactive jobs is labeled as
last reported progress. Pending jobs also expose:

- **Priority:** low, medium (default), or high. Higher priority jobs preempt
  running lower priority jobs, including when a queued job's priority is edited.
  Equal priority jobs run in queue order and do not preempt each other.
- **↑ / ↓:** move a waiting job within its priority. Changing priority puts it
  at the end of the destination priority. A preempted job keeps its position.
- **Pause:** stops a running worker or takes a waiting job out of the queue,
  retaining its inputs and checkpoints. Paused jobs stay paused after restart.
- **Resume:** puts a paused job at the end of its priority's queue.
- **Retry:** moves a failed job back to the end of its priority's queue with the
  same settings, seed, images, and saved checkpoints, clearing its previous error.
- **Cancel:** kills an active job and deletes its record, input image, output,
  checkpoints, and temporary files. Completed or failed jobs have Remove instead.

Each generation runs in a disposable subprocess group containing its isolated
ComfyUI instance and media tools. Pause, cancel, preemption, and shutdown kill
that group promptly, including during model loading. A parent watchdog also
stops it if the web server crashes. The next job starts a fresh model process.

Queue state is saved in `web_outputs/queue-state.json`. Generation checkpoints
live in `web_outputs/.jobs/<job-id>/`: conditioning (including image keyframes),
the most recently completed diffusion step with its multistep solver history,
and final audio/video latents. Writes replace the previous checkpoint atomically.
Resumption skips saved conditioning and completed diffusion steps; if final
latents are saved, it only decodes and encodes the output again. An operation
interrupted before its checkpoint commits is repeated, so pausing during the
first conditioning stage may require repeating that stage. Checkpoint writes
add disk I/O to each diffusion step and require free disk space.

Running jobs return to the queue after a server restart. Old queue-state files
are migrated automatically; jobs that predate checkpoint support start from
scratch. Checkpoints are PyTorch files tied to this pipeline and model
installation. Completed jobs automatically discard their checkpoints. Uploaded images remain available
until the corresponding job is removed.

## Local inference runtime

The local stack uses a 10.6 GB Q4 GGUF of the H3 hybrid FL2VA/REF2VA
checkpoint, an NVFP4 Qwen3-VL encoder, and selectable Turbo LoRAs. Its weights are
stored outside the repository under
`/Volumes/MLData3/genvideo/ComfyUI/models/` and linked into the matching
ComfyUI model directories.

On Apple Silicon, the isolated ComfyUI process enables PyTorch's CPU fallback
for operations that MPS does not implement. Model sampling still uses MPS.

The MiniMax H3 installation consists of:

```text
unet/minimax_h3_hybrid_fl2va_ref2va_b25-49-Q4_0.gguf
text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
vae/minimax_h3_video_vae_fp16.safetensors
vae/minimax_h3_audio_vae_fp32.safetensors
loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors
loras/minimax_h3_turbo_v4_step600_ema.safetensors
```

The larryvrh option requires
[`ComfyUI-MiniMax-H3-Turbo`](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo)
(installed revision `4274783a23afcfdbea3b4876cb79effd6c510785`) for its LoRA loader.
The extension and its bundled tensor data live in
`/Volumes/MLData3/genvideo/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Turbo`, symlinked
into `ComfyUI/custom_nodes/`. The LoRA uses the author's default runtime bypass
mode. Our installed ComfyUI handles the audio/video schedules natively through
`ModelSamplingAV`, so the author's sampler is equivalent to Euler; the app's
resumable Euler implementation keeps pause, preemption, and retry working.

The v4 EMA download is pinned to Hugging Face revision
`43a74557ac3f6539db8e0f2a959d03feb7a81480` (779,849,816 bytes), with SHA-256
`5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3`.

Each job launches its own local ComfyUI instance with low-memory settings and
stops if its process tree reaches 56 GiB RSS. Set the web server's
`--memory-limit-gib GIB` option to adjust the limit; values above 64 are rejected.
The worker also preserves a 2 GiB system-memory reserve and stops if a
generation grows swap usage by more than 4 GiB.
MiniMax H3 uses ComfyUI's dynamic low-memory loading and keeps offloaded model
weights disk-backed instead of pinning another copy in unified memory.
Its pipeline-specific conditioning node releases the 32B text encoder before
the diffusion transformer is loaded, avoiding both checkpoints occupying RAM
at the same time.

## Tests

Run the tests with the installed inference environment:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

The tests compare interrupted/resumed sampling at every step of the 4- and
20-step schedules against the installed ComfyUI solver, and exercise priority
scheduling and process termination with real lightweight subprocesses.

## Ref2VA prompt builder

Choose **Ref2VA** in the generation mode tabs. This tab only offers the dedicated
**MiniMax H3 Ref2VA Q4 (20 steps)** model; the text/image tabs keep their own model
selection. Ref2VA uses the regular schedule with no FL2VA Turbo LoRA.

1. Add a task template: character/scene reference, first/last frames, storyboard,
   video editing, video continuation, appearance/motion transfer, reference voice,
   or reused soundtrack. Templates can be combined with additional media.
2. Attach images, videos, and audio; describe their roles and retention relationships.
   Video soundtracks are opt-in. Labels follow the runtime order: pictures, videos
   with enabled soundtracks, then standalone audio. Audio numbering is independent
   of video numbering.
3. Define subjects and select their source assets. Multiple subjects can share one
   asset, and one subject can combine several assets. Describe shots, cut times,
   camera motion, and sound. The dialogue helper inserts speaker IDs and language
   tags; the reference picker inserts labels into shots.
4. Build the six prompt sections, review/edit them, then add the job to the queue.
   Rebuilding replaces manual section edits. Builder changes require rebuilding
   before submission so a changed attachment cannot silently reuse stale labels.
   The compiled prompt uses the ordering and relationship markers from the
   [MiniMax reference guide](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/ref-en.txt).
   Describe each shot in English, retaining the original language for dialogue.

Reference limits: 9 images, 3 videos, 3 audio tracks (including enabled video
soundtracks), and 12 files total. Clips must be 2–15 seconds, with at most 15 seconds
of video and 15 seconds of audio. Images are limited to 32 MB / 40 megapixels;
video/audio files to 256 MB each. Videos are decoded at 24 fps and reference frames
past the generated clip length are omitted by the installed ComfyUI pipeline.
Images retain their aspect ratio and are scaled to the output pixel budget.
The model interprets frame anchors, editing, and audio reuse through conditioning;
these are generative instructions, not a promise of pixel-exact editing or lossless
soundtrack copying. The six sections together allow 24,000 characters.

Reference uploads, the builder, and the edited prompt survive queue restarts,
**Copy to form**, regeneration, retry, and pause/resume. Removing a job removes its
own reference copies. Ref2VA conditioning releases the text encoder before diffusion
and uses the same memory limits and checkpoint mechanism as other jobs.

The dedicated 10.60 GiB Q4 model is installed at:

```text
/Volumes/MLData3/genvideo/ComfyUI/models/unet/minimax_h3_ref2va_pruned-Q4_K.gguf
```

It is linked into `ComfyUI/models/unet/` and shares the existing Qwen encoder and
video/audio VAEs. To reproduce installation:

```sh
.venv/bin/python scripts/install_ref2va.py
```

The installer uses [Unsloth's Ref2VA quantization](https://huggingface.co/unsloth/MiniMax-H3-GGUF)
at revision `d629413c2e5b51b38c453668b75ca3b06ca92703`. The upstream download is
11,381,096,544 bytes, SHA-256
`2fa5840021cf6967843eaeefde9aaa277e540de02986d5ee3d5b0e6a7a8c9dec`.
The installer adds `general.architecture = minimax_h3` to the metadata-free upstream
GGUF for the installed ComfyUI-GGUF loader, preserving tensor data and quantization.
The installed file therefore has a different checksum from the upstream download.
