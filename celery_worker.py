import multiprocessing
from celery import Celery
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline
from transformers import T5EncoderModel
import torch
from celery.signals import worker_process_init

multiprocessing.set_start_method('spawn', force=True)

celery = Celery('tasks', broker='redis://localhost:6379/0')

# Use a dictionary to store pipeline per thread
pipeline_storage = {}

@worker_process_init.connect
def initialize_pipeline(**kwargs):
    """Ensure the pipeline is initialized inside the worker process."""
    global pipeline_storage
    thread_id = torch.cuda.current_device()  # Get unique ID per worker thread

    if thread_id not in pipeline_storage:  # Initialize pipeline only once per thread
        print(f"Initializing pipeline in worker thread {thread_id}...")

        model_id = "stabilityai/stable-diffusion-3.5-large-turbo"

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Load models inside worker thread
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

        # Store pipeline in the dictionary
        pipeline_storage[thread_id] = pipeline

@celery.task(bind=True)
def worker(self, prompt, num_steps, guidance_scale, max_seq_length, userHeight, userWidth):
    """Generate an image using the preloaded pipeline."""
    global pipeline_storage
    thread_id = torch.cuda.current_device()

    if thread_id not in pipeline_storage:
        return {"status": "failed", "error": "Pipeline was not initialized in worker process!"}

    pipeline = pipeline_storage[thread_id]  # Get pipeline for this worker

    try:
        print(f"Generating image for prompt: {prompt}")
        print(f"Image size: {userWidth}x{userHeight}")

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
