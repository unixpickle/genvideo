# genvideo

Generate a short LTX-2.5 video from one image while keeping all working files
temporary. The input may have any dimensions; it is scaled to fill a 512x512
frame and center-cropped without distortion. Only the requested MP4 remains.

```sh
./genvideo input.jpg "The subject comes alive and looks around" output.mp4
```

For text-to-video generation without an initial image:

```sh
./genvideo --text "A tiny sailboat crossing a stormy teacup" output.mp4
```

Use `--seed NUMBER` for a repeatable generation. The command launches its own
local ComfyUI instance with low-memory settings and stops if its process tree
reaches 56 GiB RSS. A lower limit can be selected with
`--memory-limit-gib GIB`; values above 64 are rejected.

Videos are 3 seconds by default. Select the tested 5-second, 121-frame mode
with `--duration 5`:

```sh
./genvideo --duration 5 input.jpg "The subject keeps moving" output.mp4
```

## Queue website

Start the private, mobile-friendly queue server with:

```sh
./genvideo-web
```

Then open [http://10.9.0.8:8080](http://10.9.0.8:8080). The page accepts both
text-to-video prompts and image-plus-text jobs, shows their FIFO queue position,
offers tested 3- and 5-second durations, shows the full text and starting image
for every prompt, and plays finished videos inline. A Regenerate button
enqueues an exact copy, including the duration, seed, and starting image.
Completed MP4s are stored in `web_outputs/`.
Queue history and pending jobs are saved there as well, so they survive server
restarts. If the server stops during a generation, that job is queued again the
next time it starts.

The worker starts one isolated ComfyUI process when the first job arrives,
reuses the loaded LTX model for every queued job, and stops ComfyUI as soon as
the queue is empty. Uploaded prompt images are retained with queue history so
jobs can be regenerated, then deleted when their job is removed. All ComfyUI
working files are temporary.
