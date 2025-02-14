import torch
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import login
import os
from dotenv import load_dotenv

load_dotenv()

token = os.getenv('HUG_TOKEN')
login(token)

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large-turbo", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

width = 1024
height = 1024

prompt="Palette knife painting of an autumn cityscape"

image = pipe(
    prompt=prompt,
    num_inference_steps=2,
    guidance_scale=7,
    width=width,
    height=height
).images[0]
image.save('output.PNG')
