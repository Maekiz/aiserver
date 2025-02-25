import torch
import torch.multiprocessing as mp
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline
from transformers import T5EncoderModel
import os
import logging
from gunicorn.app.base import BaseApplication
from functools import partial

mp.set_start_method('spawn', force=True)

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

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

def worker_init(pipeline):
    global model
    model = pipeline

def generate_image(prompt, num_steps, guidance_scale, max_seq_length, userHeight, userWidth):
    image = model(
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        max_sequence_length=max_seq_length,
        height=userHeight,
        width=userWidth
    ).images[0]
    output_path = f"generated_image_{os.getpid()}.png"
    image.save(output_path)
    return output_path

pool = None

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()

        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in request body"}), 400

        prompt = data['prompt']
        num_steps = data.get('num_inference_steps', 4)
        guidance_scale = data.get('guidance_scale', 0.0)
        max_seq_length = data.get('max_sequence_length', 512)
        userHeight = data.get('height', 1024)
        userWidth = data.get('width', 1024)

        app.logger.info(f"Generating image for prompt: {prompt}")
        app.logger.info(f"{userWidth}x{userHeight}")

        output_path = pool.apply(generate_image, (prompt, num_steps, guidance_scale, max_seq_length, userHeight, userWidth))
        return send_file(output_path, mimetype='image/png')

    except Exception as e:
        app.logger.error(f"Error generating image: {str(e)}")
        return jsonify({"error": str(e)}), 500

class StandaloneApplication(BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        config = {key: value for key, value in self.options.items()
                  if key in self.cfg.settings and value is not None}
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application

if __name__ == '__main__':
    pipeline = load_model()
    pool = mp.Pool(processes=2, initializer=worker_init, initargs=(pipeline,))

    options = {
        'bind': '0.0.0.0:5000',
        'workers': 1,
        'timeout': 600,
        'worker_class': 'gevent'
    }
    StandaloneApplication(app, options).run()
