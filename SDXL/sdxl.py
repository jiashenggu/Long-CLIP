import sys
sys.path.append('/gpfs/public/vl/gjs/Long-CLIP')
from diffusers import DiffusionPipeline
import torch
from open_clip_long import factory as open_clip
import torch.nn as nn
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from SDXL_pipeline import get_image
from SDXL_img2img import image2image

base = DiffusionPipeline.from_pretrained(
    "/gpfs/public/vl/gjs/model/SPO-SDXL_4k-p_10ep", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.to("cuda")

# refiner = DiffusionPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-refiner-1.0",
#     text_encoder_2=base.text_encoder_2,
#     vae=base.vae,
#     torch_dtype=torch.float16,
#     use_safetensors=True,
#     variant="fp16",
# )
# refiner.to("cuda")

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 30
high_noise_frac = 0.8

# prompt = "The painting captures a serene moment in nature. At the center, a calm lake reflects the sky, its surface rippled only by the gentlest of breezes. The sky above is a brilliant mix of blues and whites, with fluffy clouds drifting leisurely across. On the banks of the lake, tall trees stand gracefully, their leaves rustling in the wind. In the foreground, an old man sits on a rock, seemingly lost in deep thought or meditation. The soft light of the setting sun bathes the entire scene in a warm glow, creating a sense of peace and tranquility. The colors are muted yet vibrant, and the details are captured with precision, giving the painting a sense of realism while still retaining a dreamlike quality."

# image = get_image(
#     pipe=base,
#     prompt=prompt,
#     num_inference_steps=n_steps,
#     denoising_end=high_noise_frac,
#     output_type="latent",
# ).images
    
    
# image = image2image(
#     pipe=refiner,
#     prompt=prompt,
#     num_inference_steps=n_steps,
#     denoising_start=high_noise_frac,
#     image=image,
# ).images[0]
prompts = ["""A beautiful girl with
blonde hair""", """The image shows a person standing in front of his
farm. He wears a blue hat, blue T-shirt with red logo
and black jeans. He is smiling happily and his
wrinkles is stretching.""", """The early morning sun shines, dew sparkles among
the green leaves, birds sing cheerfully, and the gentle
breeze carries the fragrance of flowers, creating a
vivid picture of vitality.""", """The serene lake surface resembled a flawless mirror,
reflecting the soft blue sky and the surrounding greenery.
A gentle breeze played across its expanse, ruffling the
surface into delicate ripples that gradually spread out,
disappearing into the distance. Along the shore, weeping
willows swayed gracefully in the light breeze, their long
branches dipping into the water, creating a soothing
sound as they gently brushed against the surface. In the
midst of this serene scene, a pure white swan floated
gracefully on the lake. Its elegant neck curved into a
graceful arc, giving it an air of dignity and elegance that
resembled a dancer poised for a performance."""]
for prompt in prompts:
    image = get_image(
        pipe=base,
        prompt=prompt,
        num_inference_steps=n_steps,
    ).images[0]
    
    import glob
    num = len(glob.glob('/gpfs/public/vl/gjs/Long-CLIP/SDXL/images/*'))
    
    image_name = f"/gpfs/public/vl/gjs/Long-CLIP/SDXL/images/sdxl_{num}.png"
    print(image_name)
    image.save(image_name)

    # image_ori = base(prompt=prompt, num_inference_steps=n_steps).images[0]
    # num = len(glob.glob('/gpfs/public/vl/gjs/Long-CLIP/SDXL/images_ori/*'))
    # image_ori_name = f"/gpfs/public/vl/gjs/Long-CLIP/SDXL/images_ori/sdxl_ori_{num}.png"
    # print(image_ori_name)
    # image_ori.save(image_ori_name)
