# genvideo

Generate MiniMax H3 video with synchronized stereo audio while keeping all
working files temporary. Image inputs may have any dimensions; they are scaled
to cover the selected output canvas and center-cropped without distortion.
Only the requested MP4 remains.

```sh
./genvideo input.jpg "The subject comes alive and looks around" output.mp4
```

For text-to-video generation without an initial image:

```sh
./genvideo --text "A tiny sailboat crossing a stormy teacup" output.mp4
```

Turbo is the default. Use `--model minimax-h3-base` for the regular 20-step
schedule without the 4-step Turbo LoRA:

```sh
./genvideo --text "A sailboat crossing a stormy teacup." output.mp4 \
  --overall-soundscape "Rain, rolling thunder, and tiny waves." \
  --non-diegetic-music "Sparse low strings at a slow tempo."
```

The positional prompt becomes H3's `integrated_multimodal_description`. The two
audio options are optional, and empty fields are omitted. Fields are serialized
in H3's required order:

```text
integrated_multimodal_description: ...

overall_soundscape: ...

non_diegetic_music: ...
```

For image-to-video jobs, genvideo automatically prepends H3's required
`<Picture 1>` instruction aligning the supplied image with the first frame.

The local stack uses a 10.6 GB Q4 GGUF of the H3 hybrid FL2VA/REF2VA
checkpoint, an NVFP4 Qwen3-VL encoder, and a 4-step Turbo LoRA. Its weights are
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
```

Use `--seed NUMBER` for a repeatable generation. The command launches its own
local ComfyUI instance with low-memory settings and stops if its process tree
reaches 56 GiB RSS. A lower limit can be selected with
`--memory-limit-gib GIB`; values above 64 are rejected.
It also preserves an 8 GiB system-memory reserve and stops if a generation
grows swap usage by more than 4 GiB.
MiniMax H3 uses ComfyUI's dynamic low-memory loading and keeps offloaded model
weights disk-backed instead of pinning another copy in unified memory.
Its pipeline-specific conditioning node releases the 32B text encoder before
the diffusion transformer is loaded, avoiding both checkpoints occupying RAM
at the same time.

Videos are 5 seconds by default. Every integer duration from 3 through 15
seconds is available (3 seconds is retained as a compatibility option); H3
snaps the requested duration to its `17k+5` frame grid at 24 fps:

```sh
./genvideo --duration 15 input.jpg "The subject keeps moving" output.mp4
```

Choose a 512p memory-saving or native 768p canvas, plus any supported aspect
ratio. Canvases are aligned to 32 pixels and capped to H3's local 7:4 pixel
budget, so the exact dimensions are shown by the website.

```sh
./genvideo --text "A crane crosses the skyline." output.mp4 \
  --resolution 768 --aspect-ratio 9:16
```

Supported aspect ratios are `21:9`, `16:9`, `4:3`, `1:1`, `3:4`, and `9:16`.

## Queue website

Start the private, mobile-friendly queue server with:

```sh
./genvideo-web
```

Then open [http://10.9.0.8:8080](http://10.9.0.8:8080). The page accepts both
text-to-video prompts and image-plus-text jobs, exposes all three H3 structured
prompt fields, offers 3–15-second durations and canvas controls, shows the full
compiled prompt and starting image for every job, and plays finished videos
inline. **Copy to form** loads any current or historical job's prompt fields,
output settings, exact seed, and starting image into the composer for editing.
Submitting creates a separate job. Older LTX prompts can also be copied into
an H3 form; their model selection defaults to H3 Turbo.
Completed MP4s are stored in `web_outputs/`.

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

Run the tests with the installed inference environment:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

The tests compare interrupted/resumed sampling at every step of the 4- and
20-step schedules against the installed ComfyUI solver, and exercise priority
scheduling and process termination with real lightweight subprocesses.
