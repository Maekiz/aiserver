import multiprocessing
from celery import Celery
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline
from transformers import T5EncoderModel
import torch

multiprocessing.set_start_method('spawn', force=True)

global pipeline
celery = Celery('tasks', broker='redis://localhost:6379/0')

def main():
    global pipeline
    model_id = "stabilityai/stable-diffusion-3.5-large-turbo"

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load models
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

@celery.task(bind=True)
def worker(self, prompt, num_steps, guidance_scale, max_seq_length, userHeight, userWidth):
    global pipeline
    try:
        print(f"Generating image for prompt: {prompt}")
        print(f"{userWidth}x{userHeight}")

        image = pipeline(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=max_seq_length,
            height=userHeight,
            width=userWidth
        ).images[0]
        
        output_path = f"generated_image_{self.request.id}.png"
        image.save(output_path)

        return {"status": "completed", "file_path": output_path}

    except Exception as e:
        return {"status": "failed", "error": str(e)}

if __name__ == '__main__':
    main()