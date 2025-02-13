import torch
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel
from diffusers import BitsAndBytesConfig

torch.cuda.empty_cache()
model_id = "stabilityai/stable-diffusion-3.5-large"

# NF4 configuration
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the transformer with NF4 quantization
transformer = SD3Transformer2DModel.from_single_file(
    "https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/sd3.5_large.safetensors",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16,
)

pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.bfloat16
)
pipeline.enable_model_cpu_offload()

prompt = "A whimsical and creative image depicting a hybrid creature that is a mix of a waffle and a hippopotamus, basking in a river of melted butter amidst a breakfast-themed landscape..."
width = 1024
height = 1024

image = pipeline(
    prompt=prompt,
    num_inference_steps=28,
    guidance_scale=3.5,
    width=width,
    height=height
).images[0]
image.save("output.png")
