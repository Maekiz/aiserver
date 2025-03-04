import torch
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from celery_worker import make_celery, create_pipeline, create_worker  # Import refactored functions
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

app = Flask(__name__)
app.config.update(
    CELERY_BROKER_URL="redis://localhost:6379/0",
    CELERY_RESULT_BACKEND="redis://localhost:6379/0",
)

# Initialize Celery
celery = make_celery(app)

# Initialize pipeline
pipeline = create_pipeline()

# Create the Celery worker
worker = create_worker(celery)

CORS(app, origins=['https://aleksanderekman.github.io', "https://bakkadiffusion.vercel.app"])

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Get JSON data from request
        data = request.get_json()

        prompt = data['prompt']
        num_steps = data.get('num_inference_steps', 4)
        guidance_scale = data.get('guidance_scale', 0.0)
        max_seq_length = data.get('max_sequence_length', 512)
        userHeight = data.get('height', 1024)
        userWidth = data.get('width', 1024)

        # Start Celery task
        task = worker.delay(prompt, num_steps, guidance_scale, max_seq_length, userHeight, userWidth, pipeline)

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
