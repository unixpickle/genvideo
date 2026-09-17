import sys, json, tempfile, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'ComfyUI'))
import comfy.cli_args
comfy.cli_args.args.cpu = True
import torch
from PIL import Image
from custom_nodes.genvideo_h3 import GenVideoMiniMaxH3ReferenceConditioning
class Clip:
    def tokenize(self, prompt, **kw):
        self.items=kw['minimax_ref_items']; return prompt
    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, 2, 3), {}]]
class VAE:
    def encode(self, image):
        if image.ndim==3: return torch.zeros(1,32,2,80)
        return torch.zeros(1,24,2,image.shape[1]//16,image.shape[2]//16)
with tempfile.TemporaryDirectory() as tmp:
    d=Path(tmp); Image.new('RGB',(64,64),'red').save(d/'image.png')
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=blue:s=64x64:r=24','-f','lavfi','-i','sine=frequency=440:sample_rate=32000','-t','2','-c:v','libx264','-c:a','aac',str(d/'video.mp4')],check=True)
    subprocess.run(['ffmpeg','-v','error','-i',str(d/'video.mp4'),str(d/'audio.wav')],check=True)
    refs=[{'kind':'audio','path':str(d/'audio.wav')}, {'kind':'image','path':str(d/'image.png')}, {'kind':'video','path':str(d/'video.mp4'),'width':64,'height':64,'use_audio':True}]
    clip=Clip(); node=GenVideoMiniMaxH3ReferenceConditioning(); released=[]
    node._release_clip=lambda c: released.append(c)
    cond,latent=node.encode_and_release(clip,VAE(),VAE(),'prompt',64,64,73,json.dumps(refs),str(d))
    assert [x['type'] for x in clip.items]==['image','audio','video','audio'],clip.items
    assert [x['kind'] for x in cond[0][1]['minimax_refs']]==['image','video_audio','audio']
    assert len(released)==1 and (d/'conditioning.pt').exists()
    print('Actual ComfyUI reference node passed: image + synchronized video/audio + standalone audio, ordering, checkpoint save, encoder release')
