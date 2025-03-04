import torch
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline
from transformers import T5EncoderModel
import os
from celery import Celery

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

app = Flask(__name__)
CORS(app, origins=['https://aleksanderekman.github.io', "https://bakkadiffusion.vercel.app"])

# Celery Configuration
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Model Configuration
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

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Validate prompt input
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in request body"}), 400

        prompt = data['prompt']
        num_steps = data.get('num_inference_steps', 4)
        guidance_scale = data.get('guidance_scale', 0.0)
        max_seq_length = data.get('max_sequence_length', 512)
        userHeight = data.get('height', 1024)
        userWidth = data.get('width', 1024)

        # Start Celery task
        task = worker.delay(prompt, num_steps, guidance_scale, max_seq_length, userHeight, userWidth)

        return jsonify({"message": "Task started", "task_id": task.id}), 202

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/result/<task_id>', methods=['GET'])
def get_result(task_id):
    task = worker.AsyncResult(task_id)

    if task.state == 'PENDING':
        return jsonify({"message": "Processing, please check later"}), 202
    elif task.state == 'SUCCESS':
        result = task.result
        if result["status"] == "completed":
            return send_file(result["file_path"], mimetype='image/png')
        else:
            return jsonify({"error": result["error"]}), 500
    else:
        return jsonify({"status": task.state}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
