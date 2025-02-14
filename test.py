import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large-turbo", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

width = 1024
height = 1024

prompt="Palette knife painting of an autumn cityscape"

image = pipeline(
    prompt=prompt,
    num_inference_steps=28,
    guidance_scale=7,
    width=width,
    height=height
).images[0]
