import torch
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline
from transformers import T5EncoderModel
import os

def load_model():
    model_id = "stabilityai/stable-diffusion-3.5-large-turbo"
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print("Loading transformer model...")
    model_nf4 = SD3Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16
    )

    print("Loading text encoder model...")
    t5_nf4 = T5EncoderModel.from_pretrained("diffusers/t5-nf4", torch_dtype=torch.bfloat16)

    print("Initializing pipeline...")
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        transformer=model_nf4,
        text_encoder_3=t5_nf4,
        torch_dtype=torch.bfloat16
    )
    pipeline.enable_model_cpu_offload()
    return pipeline

def worker_process(task_queue, result_queue):
    pipeline = load_model()
    while True:
        task = task_queue.get()
        if task is None:
            break
        prompt, num_steps, guidance_scale, max_seq_length, userHeight, userWidth = task
        image = pipeline(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=max_seq_length,
            height=userHeight,
            width=userWidth
        ).images[0]
        output_path = f"generated_image_{os.getpid()}.png"
        image.save(output_path)
        result_queue.put(output_path)
