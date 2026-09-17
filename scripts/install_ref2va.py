"""Install pinned Ref2VA Q4 weights on MLData3; normalize GGUF architecture metadata."""
import hashlib
import os
from pathlib import Path
import urllib.request

import gguf

REVISION = "d629413c2e5b51b38c453668b75ca3b06ca92703"
FILENAME = "minimax_h3_ref2va_pruned-Q4_K.gguf"
SHA256 = "2fa5840021cf6967843eaeefde9aaa277e540de02986d5ee3d5b0e6a7a8c9dec"
DIRECTORY = Path("/Volumes/MLData3/genvideo/ComfyUI/models/unet")
ROOT = Path(__file__).resolve().parents[1]


def install():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    target = DIRECTORY / FILENAME
    source = target.with_suffix(".gguf.partial")
    if not target.exists():
        if not source.exists():
            url = f"https://huggingface.co/unsloth/MiniMax-H3-GGUF/resolve/{REVISION}/{FILENAME}"
            urllib.request.urlretrieve(url, source)
        with source.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        if digest != SHA256:
            raise RuntimeError(f"Download checksum mismatch: {digest}; remove {source} and retry")
        reader = gguf.GGUFReader(str(source))
        # The upstream sd.cpp export has no architecture metadata. ComfyUI-GGUF
        # needs it to recognize H3; preserve all tensor data and quantization.
        temporary = target.with_suffix(".gguf.installing")
        writer = gguf.GGUFWriter(str(temporary), "minimax_h3")
        for tensor in reader.tensors:
            writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()
        os.replace(temporary, target)
        source.unlink()
        print(f"Verified upstream SHA-256: {digest}")
    link = ROOT / "ComfyUI/models/unet" / FILENAME
    if not link.exists():
        link.symlink_to(target)
    elif link.resolve() != target:
        raise RuntimeError(f"An unrelated model already exists at {link}")
    print(f"Installed {target}")


if __name__ == "__main__":
    install()
