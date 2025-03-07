from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from celery.result import AsyncResult
from celery_worker import celery, worker  # Import Celery app and worker task

# Enable expandable segments for CUDA memory allocation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Initialize Flask app
app = Flask(__name__)
app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379/0',  # Redis broker URL
    CELERY_RESULT_BACKEND='redis://localhost:6379/0',  # Redis result backend
)

# Enable CORS for specific origins
CORS(app, origins=['https://aleksanderekman.github.io', "https://bakkadiffusion.vercel.app"])

# Route to generate an image based on the prompt and return it immediately
@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Get JSON data from the request
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

        # Start Celery task and wait for completion
        task = worker.apply(args=[prompt, num_steps, guidance_scale, max_seq_length, userHeight, userWidth])

        # Wait for the task to complete (blocking)
        task_result = task.get()  # This will block until the task finishes

        if task_result["status"] == "completed":
            file_path = task_result["file_path"]
            return send_file(file_path, mimetype='image/png')  # Send the image back as a response
        else:
            return jsonify({"error": task_result["error"]}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
