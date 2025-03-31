import torch
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline
from transformers import T5EncoderModel
import os
import threading
import logging
import jwt
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
gen_list = []
load_dotenv()

def get_client_ip():
    # Check for X-Forwarded-For header first
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0].strip()
    return request.remote_addr

# Setter opp logging
logging.basicConfig(
    filename='server.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")

app = Flask(__name__)
lock = threading.Lock()
CORS(app, origins=['https://aleksanderekman.github.io', "https://www.bakkadiffusion.no"])
# Initialize Flask-Limiter
limiter = Limiter(
    key_func=get_client_ip,
    app=app,
    default_limits=["1 per 10 seconds"]  # Default rate limits
)
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

def verify_token(auth_token):
    try:
        decoded_data = jwt.decode(
            auth_token,
            JWT_SECRET_KEY,
            algorithms=['HS256'],
            options={"verify_exp": True} 
        )
        return decoded_data['userData']['username']
    except:
        return None
    
@app.errorhandler(429)
def ratelimit_handler(e):
    logging.warning(f"Rate limit exceeded: {request.remote_addr}")
    return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429

@app.route('/generate', methods=['POST'])
@limiter.limit("1 per 10 seconds")  # Rate limit for this endpoint
def generate():
    with lock:
        # Get JSON data from request
        data = request.get_json()

        # Validate prompt input
        if not data or 'prompt' not in data:
            logging.error("Missing 'prompt' in request body")
            return jsonify({"error": "Missing 'prompt' in request body"}), 400

        prompt = data['prompt']
        num_steps = 8
        guidance_scale = 0.0
        max_seq_length = 512
        userHeight = data.get('height', 1024)
        userWidth = data.get('width', 1024)

        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith("Bearer "):
            logging.error("Missing or invalid Authorization header")
            return jsonify({"error": "Missing or invalid Authorization header"}), 401
        authToken = auth_header.split(" ")[1]
        username = verify_token(authToken)
        if username is None:
            logging.error("Invalid token")
            return jsonify({"error": "Invalid token"}), 401
        if username in gen_list:
            logging.error("User is already generating an image")
            return jsonify({"error": "User is already generating an image"}), 400

        if userHeight > 1024 or userWidth > 1840 or userHeight < 1024 or userWidth < 576:
            logging.error("Invalid image dimensions")
            return jsonify({"error": "Invalid image dimensions"}), 400

        ip = request.headers.get('X-Forwarded-For')
        logging.info(f"{username} on IP {ip}: Generating image ({userWidth}x{userHeight}) for prompt: {prompt}")

        image = pipeline(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=max_seq_length,
            height=userHeight,
            width=userWidth
        ).images[0]
        output_path = "generated_image.png"
        image.save(output_path)
        gen_list.remove(authToken)
        return send_file(output_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)